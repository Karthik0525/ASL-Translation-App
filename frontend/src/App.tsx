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
    shortLabel: string;
    subtitle: string;
    outputLabel: string;
    notes: string;
    threshold: string;
  }
> = {
  transformer: {
    title: 'Kinematic Transformer',
    shortLabel: 'Word Model',
    subtitle: 'Temporal sequence model for 201 signed words',
    outputLabel: 'Word',
    notes: 'Tracks 30 frames of hand landmarks before sending a prediction.',
    threshold: 'Confidence gate: 60%',
  },
  cnn: {
    title: 'Static CNN',
    shortLabel: 'Letter Model',
    subtitle: 'Image classifier for A-Z fingerspelling',
    outputLabel: 'Letter',
    notes: 'Captures the current frame and classifies a static hand shape.',
    threshold: 'Confidence gate: 50%',
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
    if (now - lastCallTimeRef.current < 250) return;

    if (activeModel === 'transformer') {
      let coords = Array(NUM_FEATURES).fill(0.0);

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

      if (sequenceRef.current.length === SEQUENCE_LENGTH) {
        lastCallTimeRef.current = now;
        sendToTransformer(sequenceRef.current);
      }
    }

    if (activeModel === 'cnn') {
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

  return (
    <div className="app-shell">
      <main className="app-layout">
        <section className="hero-panel">
          <div className="hero-copy">
            <div className="hero-kicker">AI Class Demo | Computer Vision + ASL Recognition</div>
            <h1>ASL Translation Demo</h1>
            <p className="hero-subtitle">
              A webcam-based prototype that compares a temporal transformer for signed
              words with a CNN for static fingerspelling. Built as a student demo to
              test real-time gesture recognition in the browser.
            </p>
            <div className="hero-tags">
              <span>MediaPipe Hands</span>
              <span>React + TypeScript</span>
              <span>Hugging Face API</span>
            </div>
          </div>

          <div className="overview-card">
            <p className="overview-label">Current mode</p>
            <h2>{modelCopy.title}</h2>
            <p className="overview-text">{modelCopy.subtitle}</p>
            <div className="overview-stats">
              <div>
                <span className="stat-label">Output</span>
                <strong>{modelCopy.outputLabel}</strong>
              </div>
              <div>
                <span className="stat-label">Throttle</span>
                <strong>250 ms</strong>
              </div>
              <div>
                <span className="stat-label">Filter</span>
                <strong>{modelCopy.threshold}</strong>
              </div>
            </div>
          </div>
        </section>

        <section className="workspace-grid">
          <div className="camera-panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">Live Inference</p>
                <h2>Camera Feed</h2>
              </div>
              <div className={`mode-chip ${isTransformer ? 'transformer' : 'cnn'}`}>
                {modelCopy.shortLabel}
              </div>
            </div>

            <div className="model-switcher" role="tablist" aria-label="Model selector">
              <button
                type="button"
                className={`model-tab ${isTransformer ? 'active transformer' : ''}`}
                onClick={() => switchModel('transformer')}
              >
                <span className="model-tab-title">Kinematic Transformer</span>
                <span className="model-tab-text">201-word dynamic recognition</span>
              </button>
              <button
                type="button"
                className={`model-tab ${!isTransformer ? 'active cnn' : ''}`}
                onClick={() => switchModel('cnn')}
              >
                <span className="model-tab-title">Static CNN</span>
                <span className="model-tab-text">A-Z fingerspelling classifier</span>
              </button>
            </div>

            <div className="camera-frame">
              <div className="camera-overlay-top">
                <span className="camera-badge">Webcam active</span>
                <span className="camera-note">{modelCopy.notes}</span>
              </div>

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
          </div>

          <aside className="side-panel">
            <div className="info-card">
              <p className="eyebrow">Prediction Status</p>
              <h2>{prediction === '---' ? 'No confident output yet' : predictionDisplay}</h2>
              <p className="info-text">
                The app only shows a result when the model confidence passes the current
                threshold, which helps reduce noisy predictions during movement.
              </p>
            </div>

            <div className="info-card">
              <p className="eyebrow">Project Notes</p>
              <ul className="info-list">
                <li>Transformer mode uses hand landmark sequences across multiple frames.</li>
                <li>CNN mode works better for single-frame alphabet poses.</li>
                <li>The webcam feed is mirrored to feel natural during signing.</li>
              </ul>
            </div>

            <div className="info-card">
              <p className="eyebrow">Demo Tips</p>
              <ul className="info-list">
                <li>Keep one hand centered in the frame for the cleanest skeleton tracking.</li>
                <li>Pause briefly at the end of a gesture so the confidence can stabilize.</li>
                <li>Use the model switcher to compare dynamic word recognition vs. letters.</li>
              </ul>
            </div>
          </aside>
        </section>
      </main>
    </div>
  );
}
