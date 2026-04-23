import base64
import threading
import time
from collections import deque
from io import BytesIO

import mediapipe as mp
import numpy as np
from PIL import Image

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
LOW_CONFIDENCE_THRESHOLD = 0.30

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

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

hands_lock = threading.Lock()
sessions_lock = threading.Lock()
session_state = {}


class SequencePayload(BaseModel):
    coordinates: list


class FramePayload(BaseModel):
    session_id: str
    image_base64: str


@app.get("/")
def health_check():
    return {"status": "Online", "models_loaded": "SequenceTransformer"}


def get_session(session_id):
    now = time.time()
    with sessions_lock:
        for existing_session_id in list(session_state.keys()):
            if now - session_state[existing_session_id]["last_seen"] > 900:
                del session_state[existing_session_id]

        if session_id not in session_state:
            session_state[session_id] = {
                "sequence": deque(maxlen=SEQUENCE_LENGTH),
                "current_word": "Waiting...",
                "confidence": 0.0,
                "last_seen": now,
            }

        session_state[session_id]["last_seen"] = now
        return session_state[session_id]


def decode_base64_image(image_base64):
    image_data = image_base64.split(",", 1)[-1]
    image_bytes = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return np.array(image)


def extract_hand_coordinates(image_rgb):
    with hands_lock:
        results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        coords = []
        for pt in results.multi_hand_landmarks[0].landmark:
            coords.extend([pt.x, pt.y, pt.z])
        return np.asarray(coords, dtype=np.float32)

    return np.zeros(NUM_FEATURES, dtype=np.float32)


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


@app.post("/predict/frame")
def predict_frame(payload: FramePayload):
    try:
        session = get_session(payload.session_id)
        image_rgb = decode_base64_image(payload.image_base64)
        coords = extract_hand_coordinates(image_rgb)

        session["sequence"].append(coords)

        prediction = session["current_word"]
        confidence = float(session["confidence"])

        if len(session["sequence"]) == SEQUENCE_LENGTH:
            seq_array = np.asarray(session["sequence"], dtype=np.float32)
            normalized_seq = normalize_sequence(seq_array)
            model_input = np.expand_dims(normalized_seq, axis=0)

            predictions = transformer_model.predict(model_input, verbose=0)[0]
            best_index = int(np.argmax(predictions))
            confidence = float(predictions[best_index])

            if confidence > CONFIDENCE_THRESHOLD:
                prediction = transformer_labels.get(best_index, "Unknown")
            elif confidence < LOW_CONFIDENCE_THRESHOLD:
                prediction = "---"

            session["current_word"] = prediction
            session["confidence"] = confidence

        return {
            "prediction": prediction,
            "confidence": confidence
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
