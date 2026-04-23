import React, { useEffect, useRef, useState } from 'react';
import { Hands, Results } from '@mediapipe/hands';
import { Camera } from '@mediapipe/camera_utils';
import { drawConnectors, drawLandmarks } from '@mediapipe/drawing_utils';
import { HAND_CONNECTIONS } from '@mediapipe/hands';

const SEQUENCE_LENGTH = 30;
const NUM_FEATURES = 63;

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sequenceRef = useRef<number[][]>([]);
  const lastCallTimeRef = useRef<number>(0);

  const [activeModel, setActiveModel] = useState<'transformer' | 'cnn'>('transformer');
  const [prediction, setPrediction] = useState<string>('Waiting...');
  const [confidence, setConfidence] = useState<number>(0);

  useEffect(() => {
    if (!videoRef.current || !canvasRef.current) return;

    const hands = new Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
    });

    hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 1,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    hands.onResults(onResults);

    const camera = new Camera(videoRef.current, {
      onFrame: async () => {
        if (videoRef.current) {
          await hands.send({ image: videoRef.current });
        }
      },
      width: 640,
      height: 480,
    });

    camera.start();

    return () => {
      hands.close();
    };
  }, [activeModel]); // Re-bind if model changes just in case

  const onResults = async (results: Results) => {
    if (!canvasRef.current || !videoRef.current) return;
    const canvasCtx = canvasRef.current.getContext('2d');
    if (!canvasCtx) return;

    // 1. Draw the video feed and skeleton
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

    // Mirror the canvas for a selfie-view
    canvasCtx.translate(canvasRef.current.width, 0);
    canvasCtx.scale(-1, 1);
    canvasCtx.drawImage(results.image, 0, 0, canvasRef.current.width, canvasRef.current.height);

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      for (const landmarks of results.multiHandLandmarks) {
        drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, { color: '#00FF00', lineWidth: 3 });
        drawLandmarks(canvasCtx, landmarks, { color: '#FF0000', lineWidth: 2 });
      }
    }
    canvasCtx.restore();

    // Throttle API calls so we don't crash our local server (max 4 calls per second)
    const now = Date.now();
    if (now - lastCallTimeRef.current < 250) return;

    // --- TRANSFORMER LOGIC ---
    if (activeModel === 'transformer') {
      let coords = Array(NUM_FEATURES).fill(0.0);

      if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        const landmarks = results.multiHandLandmarks[0];
        coords = [];
        for (let i = 0; i < landmarks.length; i++) {
          coords.push(landmarks[i].x, landmarks[i].y, landmarks[i].z);
        }
      }

      sequenceRef.current.push(coords);
      if (sequenceRef.current.length > SEQUENCE_LENGTH) {
        sequenceRef.current.shift(); // Keep rolling window at 30
      }

      if (sequenceRef.current.length === SEQUENCE_LENGTH) {
        lastCallTimeRef.current = now;
        sendToTransformer(sequenceRef.current);
      }
    }

    // --- CNN LOGIC ---
    if (activeModel === 'cnn') {
        lastCallTimeRef.current = now;

        // Grab the raw frame from the hidden video element (not the canvas with the green lines!)
        const hiddenCanvas = document.createElement('canvas');
        hiddenCanvas.width = videoRef.current.videoWidth;
        hiddenCanvas.height = videoRef.current.videoHeight;
        const ctx = hiddenCanvas.getContext('2d');
        if (ctx) {
            ctx.drawImage(videoRef.current, 0, 0);
            const base64Image = hiddenCanvas.toDataURL('image/jpeg', 0.8);
            sendToCNN(base64Image);
        }
    }
  };

  const sendToTransformer = async (sequence: number[][]) => {
    try {
      const response = await fetch('http://localhost:8000/predict/transformer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ coordinates: sequence }),
      });
      const data = await response.json();
      if (data.confidence > 0.6) {
        setPrediction(data.prediction);
        setConfidence(data.confidence);
      } else {
        setPrediction('---');
      }
    } catch (err) {
      console.error(err);
    }
  };

  const sendToCNN = async (base64Image: string) => {
    try {
      const response = await fetch('http://localhost:8000/predict/fingerspelling', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_base64: base64Image }),
      });
      const data = await response.json();
      if (data.confidence > 0.5) {
        setPrediction(data.prediction);
        setConfidence(data.confidence);
      } else {
        setPrediction('---');
      }
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div style={{ padding: '20px', fontFamily: 'sans-serif', maxWidth: '800px', margin: '0 auto' }}>
      <h1 style={{ textAlign: 'center' }}>Multi-Modal ASL Translator</h1>

      {/* The Tabs */}
      <div style={{ display: 'flex', justifyContent: 'center', gap: '10px', marginBottom: '20px' }}>
        <button
          onClick={() => { setActiveModel('transformer'); sequenceRef.current = []; setPrediction('Waiting...'); }}
          style={{ padding: '10px 20px', backgroundColor: activeModel === 'transformer' ? '#2563eb' : '#e5e7eb', color: activeModel === 'transformer' ? 'white' : 'black', border: 'none', borderRadius: '5px', cursor: 'pointer', fontWeight: 'bold' }}
        >
          Your Sequence Transformer (201 Words)
        </button>
        <button
          onClick={() => { setActiveModel('cnn'); setPrediction('Waiting...'); }}
          style={{ padding: '10px 20px', backgroundColor: activeModel === 'cnn' ? '#16a34a' : '#e5e7eb', color: activeModel === 'cnn' ? 'white' : 'black', border: 'none', borderRadius: '5px', cursor: 'pointer', fontWeight: 'bold' }}
        >
          Teammate's CNN (A-Z Fingerspelling)
        </button>
      </div>

      {/* The UI & Video */}
      <div style={{ position: 'relative', width: '640px', height: '480px', margin: '0 auto', backgroundColor: 'black', borderRadius: '10px', overflow: 'hidden' }}>

        {/* Prediction UI Overlay */}
        <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', padding: '15px', backgroundColor: 'rgba(0,0,0,0.6)', color: 'white', display: 'flex', justifyContent: 'space-between', zIndex: 10 }}>
          <h2 style={{ margin: 0 }}>Word: {prediction.toUpperCase()}</h2>
          <h2 style={{ margin: 0 }}>Conf: {Math.round(confidence * 100)}%</h2>
        </div>

        {/* Hidden Video element for MediaPipe to read */}
        <video ref={videoRef} style={{ display: 'none' }} playsInline></video>

        {/* Visible Canvas for us to draw on */}
        <canvas ref={canvasRef} width="640" height="480" style={{ width: '100%', height: '100%' }}></canvas>
      </div>
    </div>
  );
}