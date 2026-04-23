import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  // ADD THIS BLOCK: Tell Vite to leave MediaPipe alone!
  optimizeDeps: {
    exclude: ['@mediapipe/hands', '@mediapipe/camera_utils', '@mediapipe/drawing_utils']
  }
})