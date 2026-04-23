import numpy as np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.pos_emb = layers.Embedding(input_dim=sequence_length, output_dim=output_dim)
        self.supports_masking = True

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        return inputs + self.pos_emb(positions)


SEQUENCE_LENGTH = 30
NUM_LANDMARKS = 21
NUM_FEATURES = 63
CONFIDENCE_THRESHOLD = 0.60

app = FastAPI(title="ASL Sequence Transformer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Booting up AI models...")

transformer_model = load_model('asl_sequence_transformer.keras',
                               custom_objects={'PositionalEmbedding': PositionalEmbedding})
transformer_labels = {}
with open('combined_label_map.txt', 'r') as f:
    for line in f:
        idx, word = line.strip().split(':')
        transformer_labels[int(idx)] = word


class SequencePayload(BaseModel):
    coordinates: list


@app.get("/")
def health_check():
    return {"status": "Online", "models_loaded": "SequenceTransformer"}

def normalize_sequence(sequence):
    seq = sequence.reshape(SEQUENCE_LENGTH, NUM_LANDMARKS, 3).copy()
    wrist = seq[:, 0:1, :]
    seq = seq - wrist
    scale = np.linalg.norm(seq[:, 12, :], axis=-1, keepdims=True)
    scale = np.where(scale < 1e-6, 1.0, scale)
    seq = seq / scale[:, np.newaxis]
    return seq.reshape(SEQUENCE_LENGTH, NUM_FEATURES)

@app.post("/predict/transformer")
def predict_sequence(payload: SequencePayload):
    try:
        seq_array = np.asarray(payload.coordinates, dtype=np.float32)
        if seq_array.shape != (SEQUENCE_LENGTH, NUM_FEATURES):
            return {
                "error": (
                    f"Expected coordinates with shape "
                    f"({SEQUENCE_LENGTH}, {NUM_FEATURES}), got {seq_array.shape}"
                )
            }

        normalized_seq = normalize_sequence(seq_array)
        model_input = np.expand_dims(normalized_seq, axis=0)

        predictions = transformer_model.predict(model_input, verbose=0)[0]
        best_index = int(np.argmax(predictions))
        confidence = float(predictions[best_index])
        prediction = transformer_labels.get(best_index, "Unknown")

        if confidence < CONFIDENCE_THRESHOLD:
            prediction = "---"

        return {
            "prediction": prediction,
            "confidence": confidence
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
