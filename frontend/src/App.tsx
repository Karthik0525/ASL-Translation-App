import { useEffect, useRef, useState } from 'react';
import './App.css';

const { Hands, HAND_CONNECTIONS } = window as any;
const { Camera } = window as any;
const { drawConnectors, drawLandmarks } = window as any;

type Results = any;
type ModelType = 'transformer' | 'cnn';

const SEQUENCE_LENGTH = 30;
const NUM_FEATURES = 63;

const MODEL_COPY: Record<
  ModelType,
  {
    title: string;
    subtitle: string;
    outputLabel: string;
    notes: string;
    tip: string;
  }
> = {
  transformer: {
    title: 'Kinematic Transformer',
    subtitle: 'Temporal sequence model for 201 signed words',
    outputLabel: 'Word',
    notes: 'Tracks 30 frames of hand landmarks before sending a prediction.',
    tip: 'Best for full signed words with motion over time.',
  },
  cnn: {
    title: 'Static CNN',
    subtitle: 'Image classifier for A-Z fingerspelling',
    outputLabel: 'Letter',
    notes: 'Captures the current frame and classifies a static hand shape.',
    tip: 'Best for single hand shapes held still for a moment.',
  },
};

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sequenceRef = useRef<number[][]>([]);
  const lastCallTimeRef = useRef<number>(0);

  const [activeModel, setActiveModel] = useState<ModelType>('transformer');
  const [prediction, setPrediction] = useState<string>('---');
  const [confidence, setConfidence] = useState<number>(0);

  useEffect(() => {
    if (!videoRef.current || !canvasRef.current) return;

    const hands = new Hands({
      locateFile: (file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
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
      camera.stop();
    };
  }, [activeModel]);

const onResults = async (results: Results) => {
    if (!canvasRef.current || !videoRef.current) return;
    const canvasCtx = canvasRef.current.getContext('2d');
    if (!canvasCtx) return;

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    canvasCtx.translate(canvasRef.current.width, 0);
    canvasCtx.scale(-1, 1);
    canvasCtx.drawImage(results.image, 0, 0, canvasRef.current.width, canvasRef.current.height);

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      for (const landmarks of results.multiHandLandmarks) {
        drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
          color: '#22c55e',
          lineWidth: 3,
        });
        drawLandmarks(canvasCtx, landmarks, {
          color: '#f8fafc',
          lineWidth: 1.5,
          radius: 3,
        });
      }
    }
    canvasCtx.restore();

    const now = Date.now();

    // --- TRANSFORMER LOGIC ---
    if (activeModel === 'transformer') {
      let coords = Array(NUM_FEATURES).fill(0.0);

      // 1. ALWAYS collect the frame data at full speed
      if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        const landmarks = results.multiHandLandmarks[0];
        coords = [];
        for (let i = 0; i < landmarks.length; i += 1) {
          coords.push(landmarks[i].x, landmarks[i].y, landmarks[i].z);
        }
      }

      sequenceRef.current.push(coords);
      if (sequenceRef.current.length > SEQUENCE_LENGTH) {
        sequenceRef.current.shift();
      }

      // 2. ONLY throttle the network request, not the tracking
      if (sequenceRef.current.length === SEQUENCE_LENGTH) {
        if (now - lastCallTimeRef.current >= 250) {
          lastCallTimeRef.current = now;
          // Send a copy of the array to prevent mutation issues during the fetch
          sendToTransformer([...sequenceRef.current]);
        }
      }
    }

    // --- CNN LOGIC ---
    if (activeModel === 'cnn') {
      // 3. Throttle the CNN network requests here
      if (now - lastCallTimeRef.current >= 250) {
        lastCallTimeRef.current = now;
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
    }
  };

  const sendToTransformer = async (sequence: number[][]) => {
    try {
      const response = await fetch('https://bekfast-asl-multi-modal-api.hf.space/predict/transformer', {
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
        setConfidence(0);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const sendToCNN = async (base64Image: string) => {
    try {
      const response = await fetch('https://bekfast-asl-multi-modal-api.hf.space/predict/fingerspelling', {
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
        setConfidence(0);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const switchModel = (model: ModelType) => {
    setActiveModel(model);
    sequenceRef.current = [];
    lastCallTimeRef.current = 0;
    setPrediction('---');
    setConfidence(0);
  };

  const isTransformer = activeModel === 'transformer';
  const modelCopy = MODEL_COPY[activeModel];
  const confidencePercent = Math.round(confidence * 100);
  const predictionDisplay = prediction === '---' ? 'Waiting...' : prediction.toUpperCase();
  const confidenceTone =
    confidence >= 0.8 ? 'high' : confidence >= 0.5 ? 'medium' : 'low';
  const thresholdText = isTransformer ? '60%' : '50%';

  return (
    <div className="app-shell">
      <main className="app-layout">
        <header className="page-header">
          <p className="page-label">Student AI Demo</p>
          <h1>ASL Translation Demo</h1>
          <p className="page-subtitle">
            A simple webcam demo that compares a transformer for dynamic ASL words
            with a CNN for static fingerspelling.
          </p>
        </header>

        <section className="simple-card controls-card">
          <h2>Model Switch</h2>
          <div className="model-switcher" role="tablist" aria-label="Model selector">
            <button
              type="button"
              className={`model-tab ${isTransformer ? 'active' : ''}`}
              onClick={() => switchModel('transformer')}
            >
              Kinematic Transformer
            </button>
            <button
              type="button"
              className={`model-tab ${!isTransformer ? 'active' : ''}`}
              onClick={() => switchModel('cnn')}
            >
              Static CNN
            </button>
          </div>
          <div className="mode-summary">
            <h3>Current Mode: {modelCopy.title}</h3>
            <p>{modelCopy.subtitle}</p>
            <p>{modelCopy.tip}</p>
          </div>
        </section>

        <section className="simple-card camera-section">
          <div className="section-heading">
            <div>
              <h2>Camera Feed</h2>
              <p>Live webcam input with hand landmark tracking.</p>
            </div>
          </div>

          <div className="camera-frame">
            <div className="prediction-overlay">
              <div className="prediction-block">
                <span className="prediction-label">{modelCopy.outputLabel}</span>
                <strong className="prediction-value">{predictionDisplay}</strong>
              </div>
              <div className="prediction-divider" />
              <div className="prediction-block">
                <span className="prediction-label">Confidence</span>
                <strong className={`prediction-value confidence-${confidenceTone}`}>
                  {confidencePercent}%
                </strong>
              </div>
            </div>

            <video ref={videoRef} className="hidden-video" playsInline />
            <canvas ref={canvasRef} width="640" height="480" className="camera-canvas" />
          </div>
        </section>

        <section className="info-grid">
          <div className="simple-card info-card">
            <h2>Prediction Status</h2>
            <p className="status-line">
              Current {modelCopy.outputLabel.toLowerCase()}: <strong>{predictionDisplay}</strong>
            </p>
            <p className="status-line">
              Confidence: <strong>{confidencePercent}%</strong>
            </p>
            <p className="info-text">
              The app only shows a result after the confidence passes {thresholdText} to
              reduce noisy predictions.
            </p>
          </div>

          <div className="simple-card info-card">
            <h2>Demo Tips</h2>
            <ul className="info-list">
              <li>Keep one hand centered in the frame.</li>
              <li>Use good lighting if possible.</li>
              <li>Hold the final sign still for a moment before switching signs.</li>
            </ul>
          </div>
        </section>
      </main>
    </div>
  );
}
