import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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


app = FastAPI(title="ASL Translation API")

# VERY IMPORTANT: CORS allows your web browser to talk to this Python server.
# Without this, the browser will block the connection for security reasons.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, you'd restrict this to your Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Booting up AI models...")

# Load Your Transformer
transformer_model = load_model('asl_sequence_transformer.keras',
                               custom_objects={'PositionalEmbedding': PositionalEmbedding})
transformer_labels = {}
with open('combined_label_map.txt', 'r') as f:
    for line in f:
        idx, word = line.strip().split(':')
        transformer_labels[int(idx)] = word


try:
    fingerspell_model = load_model('fingerspelling_model.keras')
    # Standard A-Z alphabet array for mapping their output
    alphabet_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
except Exception as e:
    print(f"Warning: Could not load fingerspelling model yet. Error: {e}")
    fingerspell_model = None


class SequencePayload(BaseModel):
    coordinates: list


class FramePayload(BaseModel):
    coordinates: list


@app.get("/")
def health_check():
    return {"status": "Online", "models_loaded": "transformer, fingerspelling"}



@app.post("/predict/transformer")
def predict_sequence(payload: SequencePayload):
    try:
        seq_array = np.array(payload.coordinates)

        model_input = np.expand_dims(seq_array, axis=0)

        predictions = transformer_model.predict(model_input, verbose=0)[0]
        best_index = int(np.argmax(predictions))
        confidence = float(predictions[best_index])

        return {
            "prediction": transformer_labels.get(best_index, "Unknown"),
            "confidence": confidence
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/predict/fingerspelling")
def predict_letter(payload: FramePayload):
    if fingerspell_model is None:
        return {"error": "Model not loaded on server."}

    try:
        frame_array = np.array(payload.coordinates)
        model_input = np.expand_dims(frame_array, axis=0)

        predictions = fingerspell_model.predict(model_input, verbose=0)[0]
        best_index = int(np.argmax(predictions))
        confidence = float(predictions[best_index])

        return {
            "prediction": alphabet_labels[best_index],
            "confidence": confidence
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn


    uvicorn.run(app, host="0.0.0.0", port=8000)