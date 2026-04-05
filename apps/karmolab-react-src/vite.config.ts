import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: '../karmolab/react-dist',
    emptyOutDir: true,
    rollupOptions: {
      output: {
        entryFileNames: 'assets/planner.js',
        chunkFileNames: 'assets/planner-[name]-[hash].js',
        assetFileNames: 'assets/planner.[ext]',
      }
    }
  },
  base: '/apps/karmolab/react-dist/'
})
