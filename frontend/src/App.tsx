import { useEffect, useEffectEvent, useRef, useState } from 'react';
import './App.css';

const DEFAULT_API_BASE_URL = 'https://bekfast-asl-multi-modal-api.hf.space';
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? DEFAULT_API_BASE_URL).replace(/\/$/, '');
const CAPTURE_INTERVAL_MS = 250;

const MODEL_COPY = {
  title: 'Python Sequence Pipeline',
  subtitle: 'Website now uses the same backend hand-tracking flow as the working local script.',
  outputLabel: 'Word',
  notes: 'Frames are sent to the backend, which runs MediaPipe and the sequence transformer in Python.',
  tip: 'This removes browser landmark differences and should behave much closer to the local webcam demo.',
};

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const intervalRef = useRef<number | null>(null);
  const requestInFlightRef = useRef<boolean>(false);
  const sessionIdRef = useRef<string>(crypto.randomUUID());

  const [prediction, setPrediction] = useState<string>('Waiting...');
  const [confidence, setConfidence] = useState<number>(0);
  const [status, setStatus] = useState<string>('Starting camera...');
  const [statusDetail, setStatusDetail] = useState<string>('');
  const modelCopy = MODEL_COPY;

  const sendFrameToBackend = useEffectEvent(async () => {
    const captureCanvas = captureCanvasRef.current;
    if (!captureCanvas || requestInFlightRef.current) {
      return;
    }

    requestInFlightRef.current = true;

    try {
      const imageBase64 = captureCanvas.toDataURL('image/jpeg', 0.9);
      const response = await fetch(`${API_BASE_URL}/predict/frame`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionIdRef.current,
          image_base64: imageBase64,
        }),
      });
      const data = await response.json();

      if (!response.ok || data.error) {
        throw new Error(data.error ?? `Prediction request failed with status ${response.status}`);
      }

      setPrediction(data.prediction === '---' ? 'Waiting...' : String(data.prediction).toUpperCase());
      setConfidence(typeof data.confidence === 'number' ? data.confidence : 0);
      setStatus('Live');
      setStatusDetail('');
    } catch (err) {
      console.error(err);
      const message = err instanceof Error ? err.message : 'Unknown backend error';
      setStatus('Backend error');
      setStatusDetail(message);
    } finally {
      requestInFlightRef.current = false;
    }
  });

  const drawVideoFrame = useEffectEvent(function drawVideoFrameImpl() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const captureCanvas = captureCanvasRef.current;

    if (!video || !canvas || !captureCanvas) {
      animationFrameRef.current = requestAnimationFrame(() => {
        drawVideoFrameImpl();
      });
      return;
    }

    const displayCtx = canvas.getContext('2d');
    const captureCtx = captureCanvas.getContext('2d');

    if (!displayCtx || !captureCtx) {
      animationFrameRef.current = requestAnimationFrame(() => {
        drawVideoFrameImpl();
      });
      return;
    }

    if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
      displayCtx.save();
      displayCtx.clearRect(0, 0, canvas.width, canvas.height);
      displayCtx.translate(canvas.width, 0);
      displayCtx.scale(-1, 1);
      displayCtx.drawImage(video, 0, 0, canvas.width, canvas.height);
      displayCtx.restore();

      captureCtx.save();
      captureCtx.clearRect(0, 0, captureCanvas.width, captureCanvas.height);
      captureCtx.translate(captureCanvas.width, 0);
      captureCtx.scale(-1, 1);
      captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
      captureCtx.restore();
    }

    animationFrameRef.current = requestAnimationFrame(() => {
      drawVideoFrameImpl();
    });
  });

  useEffect(() => {
    let cancelled = false;

    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480 },
          audio: false,
        });

        if (cancelled) {
          for (const track of stream.getTracks()) {
            track.stop();
          }
          return;
        }

        streamRef.current = stream;

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }

        if (!captureCanvasRef.current) {
          captureCanvasRef.current = document.createElement('canvas');
          captureCanvasRef.current.width = 640;
          captureCanvasRef.current.height = 480;
        }

        setStatus('Camera ready');
        setStatusDetail('');
        void drawVideoFrame();

        intervalRef.current = window.setInterval(() => {
          void sendFrameToBackend();
        }, CAPTURE_INTERVAL_MS);
      } catch (err) {
        console.error(err);
        setStatus('Camera access failed');
        setStatusDetail(err instanceof Error ? err.message : 'Unable to access camera');
      }
    };

    void startCamera();

    return () => {
      cancelled = true;

      if (intervalRef.current !== null) {
        window.clearInterval(intervalRef.current);
      }

      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current);
      }

      if (streamRef.current) {
        for (const track of streamRef.current.getTracks()) {
          track.stop();
        }
      }
    };
  }, []);

  const confidencePercent = Math.round(confidence * 100);
  const confidenceTone =
    confidence >= 0.8 ? 'high' : confidence >= 0.5 ? 'medium' : 'low';
  const thresholdText = '60%';

  return (
    <div className="app-shell">
      <main className="app-layout">
        <header className="page-header">
          <p className="page-label">Student AI Demo</p>
          <h1>ASL Translation Demo</h1>
          <p className="page-subtitle">
            A webcam demo for translating signed words using the exact Python
            backend tracking pipeline instead of browser-side landmarks.
          </p>
        </header>

        <section className="section-block intro-section">
          <div className="mode-summary">
            <p className="section-label">Current Mode</p>
            <h2>{modelCopy.title}</h2>
            <p>{modelCopy.subtitle}</p>
            <p>{modelCopy.tip}</p>
          </div>
        </section>

        <section className="section-block camera-section">
          <div className="section-heading">
            <div>
              <p className="section-label">Live Demo</p>
              <h2>Camera Feed</h2>
              <p>{modelCopy.notes}</p>
            </div>
          </div>

          <div className="camera-frame">
            <div className="prediction-overlay">
              <div className="prediction-block">
                <span className="prediction-label">{modelCopy.outputLabel}</span>
                <strong className="prediction-value">{prediction}</strong>
              </div>
              <div className="prediction-divider" />
              <div className="prediction-block">
                <span className="prediction-label">Confidence</span>
                <strong className={`prediction-value confidence-${confidenceTone}`}>
                  {confidencePercent}%
                </strong>
              </div>
              <div className="prediction-divider" />
              <div className="prediction-block">
                <span className="prediction-label">Status</span>
                <strong className="prediction-value">{status.toUpperCase()}</strong>
              </div>
            </div>

            <video ref={videoRef} className="hidden-video" playsInline muted />
            <canvas ref={canvasRef} width="640" height="480" className="camera-canvas" />
          </div>
        </section>

        <section className="info-grid">
          <div className="section-block info-card">
            <h2>Prediction Status</h2>
            <p className="status-line">
              Current {modelCopy.outputLabel.toLowerCase()}: <strong>{prediction}</strong>
            </p>
            <p className="status-line">
              Confidence: <strong>{confidencePercent}%</strong>
            </p>
            <p className="info-text">
              The app only shows a stable result after the backend confidence passes {thresholdText}.
            </p>
            {statusDetail ? <p className="info-text">Backend detail: <strong>{statusDetail}</strong></p> : null}
          </div>

          <div className="section-block info-card">
            <h2>Demo Tips</h2>
            <ul className="info-list">
              <li>Keep one hand centered in the frame.</li>
              <li>Hold the sign long enough for the backend to collect 30 frames.</li>
              <li>Pause briefly between signs so the sequence can settle.</li>
            </ul>
          </div>
        </section>
      </main>
    </div>
  );
}
