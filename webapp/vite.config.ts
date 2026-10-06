import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'
import { VitePWA } from 'vite-plugin-pwa'

// Relative base so the build works on GitHub Pages under /<repo>/ as well as at a root URL.
export default defineConfig({
  base: './',
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['favicon.png', 'apple-touch-icon.png'],
      manifest: {
        name: 'Hieroglyph Translator',
        short_name: 'Hieroglyphs',
        description: 'Photograph a hieroglyph inscription and translate it to English.',
        start_url: './',
        scope: './',
        display: 'standalone',
        orientation: 'portrait',
        background_color: '#1a1207',
        theme_color: '#1a1207',
        icons: [
          { src: 'pwa-192.png', sizes: '192x192', type: 'image/png' },
          { src: 'pwa-512.png', sizes: '512x512', type: 'image/png' },
          { src: 'pwa-512.png', sizes: '512x512', type: 'image/png', purpose: 'maskable' },
        ],
      },
      workbox: {
        // Leave room for the ONNX model and onnxruntime-web .wasm files that get added later.
        globPatterns: ['**/*.{js,css,html,png,svg,wasm,onnx}'],
        maximumFileSizeToCacheInBytes: 60 * 1024 * 1024,
      },
    }),
  ],
})
