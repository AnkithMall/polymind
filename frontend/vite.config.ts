import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/v1': 'http://backend:8000',
      '/api': 'http://backend:8000',
      '/health': 'http://backend:8000',
    }
  }
})
