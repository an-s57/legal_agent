import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  // 相对路径：本地（localhost:8000 根路径）与 GitHub Pages（/legal_agent/ 子路径）都能正确加载资源
  base: './',
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/legal': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
