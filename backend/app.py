import base64
import json
from io import BytesIO
from PIL import Image
import numpy as np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.pos_emb = layers.Embedding(input_dim=sequence_length, output_dim=output_dim)
        self.supports_masking = True

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        return inputs + self.pos_emb(positions)


class ImprovedCNN(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


app = FastAPI(title="ASL Multi-Modal API")

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

with open("asl_classes.json", "r") as f:
    cnn_labels = json.load(f)

fingerspell_model = ImprovedCNN(len(cnn_labels))
fingerspell_model.load_state_dict(torch.load('asl_cnn_model.pth', map_location=torch.device('cpu')))
fingerspell_model.eval()  # Set to evaluation mode

cnn_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class SequencePayload(BaseModel):
    coordinates: list


class ImagePayload(BaseModel):
    image_base64: str


@app.get("/")
def health_check():
    return {"status": "Online", "models_loaded": "SequenceTransformer, ImprovedCNN"}

def normalize_sequence(sequence):
    seq = sequence.reshape(30, 21, 3).copy()
    wrist = seq[:, 0:1, :]
    seq = seq - wrist
    scale = np.linalg.norm(seq[:, 12, :], axis=-1, keepdims=True)
    scale = np.where(scale < 1e-6, 1.0, scale)
    seq = seq / scale[:, np.newaxis]
    return seq.reshape(30, 63)

@app.post("/predict/transformer")
def predict_sequence(payload: SequencePayload):
    try:
        seq_array = np.array(payload.coordinates)
        normalized_seq = normalize_sequence(seq_array)
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
def predict_letter(payload: ImagePayload):
    try:
        b64_string = payload.image_base64
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]

        image_bytes = base64.b64decode(b64_string)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        input_tensor = cnn_transform(image).unsqueeze(0)

        # 4. Predict
        with torch.no_grad():
            outputs = fingerspell_model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        return {
            "prediction": cnn_labels[predicted_idx.item()],
            "confidence": confidence.item()
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)