import { useEffect, useRef, useState } from 'react';

// Grab the global Google variables from the browser window directly
const { Hands, HAND_CONNECTIONS } = window as any;
const { Camera } = window as any;
const { drawConnectors, drawLandmarks } = window as any;

// Define the Results type since we aren't importing it anymore
type Results = any;

const SEQUENCE_LENGTH = 30;
const NUM_FEATURES = 63;

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sequenceRef = useRef<number[][]>([]);
  const lastCallTimeRef = useRef<number>(0);

  const [activeModel, setActiveModel] = useState<'transformer' | 'cnn'>('transformer');
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
      if (camera) camera.stop();
    };
  }, [activeModel]);

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
        drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, { color: '#10B981', lineWidth: 3 }); // Modern Emerald Green
        drawLandmarks(canvasCtx, landmarks, { color: '#ffffff', lineWidth: 2, radius: 3 }); // Clean White Dots
      }
    }
    canvasCtx.restore();

    // Throttle API calls
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
        sequenceRef.current.shift();
      }

      if (sequenceRef.current.length === SEQUENCE_LENGTH) {
        lastCallTimeRef.current = now;
        sendToTransformer(sequenceRef.current);
      }
    }

    // --- CNN LOGIC ---
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

  // UI Theme Variables
  const isTransformer = activeModel === 'transformer';

  return (
    <div style={{
      minHeight: '100vh',
      backgroundColor: '#0F172A', // Slate 900
      color: '#F8FAFC', // Slate 50
      fontFamily: 'system-ui, -apple-system, sans-serif',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      padding: '40px 20px'
    }}>

      {/* Header */}
      <div style={{ textAlign: 'center', marginBottom: '32px' }}>
        <h1 style={{
          margin: '0 0 12px 0',
          fontSize: '2.5rem',
          fontWeight: '800',
          background: 'linear-gradient(to right, #60A5FA, #34D399)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent'
        }}>
          ASL Vision Hub
        </h1>
        <p style={{ margin: 0, color: '#94A3B8', fontSize: '1.1rem' }}>
          Multi-Modal Neural Translation Engine
        </p>
      </div>

      {/* Modern Tabs */}
      <div style={{
        display: 'flex',
        backgroundColor: '#1E293B',
        padding: '6px',
        borderRadius: '12px',
        marginBottom: '40px',
        boxShadow: 'inset 0 2px 4px rgba(0,0,0,0.2)'
      }}>
        <button
          onClick={() => { setActiveModel('transformer'); sequenceRef.current = []; setPrediction('---'); setConfidence(0); }}
          style={{
            padding: '12px 24px',
            backgroundColor: isTransformer ? '#3B82F6' : 'transparent',
            color: isTransformer ? 'white' : '#94A3B8',
            border: 'none',
            borderRadius: '8px',
            cursor: 'pointer',
            fontWeight: '600',
            fontSize: '0.95rem',
            transition: 'all 0.2s ease',
            boxShadow: isTransformer ? '0 4px 6px -1px rgba(59, 130, 246, 0.5)' : 'none'
          }}
        >
          Kinematic Transformer (201 Words)
        </button>
        <button
          onClick={() => { setActiveModel('cnn'); setPrediction('---'); setConfidence(0); }}
          style={{
            padding: '12px 24px',
            backgroundColor: !isTransformer ? '#10B981' : 'transparent',
            color: !isTransformer ? 'white' : '#94A3B8',
            border: 'none',
            borderRadius: '8px',
            cursor: 'pointer',
            fontWeight: '600',
            fontSize: '0.95rem',
            transition: 'all 0.2s ease',
            boxShadow: !isTransformer ? '0 4px 6px -1px rgba(16, 185, 129, 0.5)' : 'none'
          }}
        >
          Static CNN (A-Z Alphabet)
        </button>
      </div>

      {/* The Video Card */}
      <div style={{
        position: 'relative',
        width: '640px',
        height: '480px',
        backgroundColor: '#000',
        borderRadius: '24px',
        overflow: 'hidden',
        boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)',
        border: '1px solid #334155'
      }}>

        {/* Modern Floating Prediction Pill */}
        <div style={{
          position: 'absolute',
          bottom: '32px',
          left: '50%',
          transform: 'translateX(-50%)',
          backgroundColor: 'rgba(15, 23, 42, 0.85)', // Glassmorphism background
          backdropFilter: 'blur(12px)',
          padding: '16px 32px',
          borderRadius: '50px',
          display: 'flex',
          alignItems: 'center',
          gap: '24px',
          zIndex: 10,
          border: '1px solid rgba(255,255,255,0.1)',
          boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.2)'
        }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px' }}>
            <span style={{ color: '#94A3B8', fontSize: '0.9rem', fontWeight: '500', textTransform: 'uppercase', letterSpacing: '1px' }}>
              {isTransformer ? 'Word' : 'Letter'}
            </span>
            <span style={{ fontSize: '1.5rem', fontWeight: '800', color: '#fff' }}>
              {prediction.toUpperCase()}
            </span>
          </div>

          <div style={{ width: '2px', height: '32px', backgroundColor: 'rgba(255,255,255,0.1)' }}></div>

          <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px' }}>
            <span style={{ color: '#94A3B8', fontSize: '0.9rem', fontWeight: '500', textTransform: 'uppercase', letterSpacing: '1px' }}>
              Conf
            </span>
            <span style={{
              fontSize: '1.5rem',
              fontWeight: '800',
              color: confidence > 0.8 ? '#34D399' : (confidence > 0.5 ? '#FBBF24' : '#F87171')
            }}>
              {Math.round(confidence * 100)}%
            </span>
          </div>
        </div>

        {/* Hidden Video element for MediaPipe to read */}
        <video ref={videoRef} style={{ display: 'none' }} playsInline></video>

        {/* Visible Canvas for us to draw on */}
        <canvas ref={canvasRef} width="640" height="480" style={{ width: '100%', height: '100%', objectFit: 'cover' }}></canvas>
      </div>

    </div>
  );
}