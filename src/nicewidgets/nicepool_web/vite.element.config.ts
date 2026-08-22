import { resolve } from 'node:path'

import vue from '@vitejs/plugin-vue'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [vue()],
  define: { 'process.env.NODE_ENV': JSON.stringify('production') },
  build: {
    outDir: 'dist-element',
    lib: {
      entry: resolve(import.meta.dirname, 'src/element/auto-register.ts'),
      formats: ['es'],
      fileName: 'nicepool-element',
    },
  },
})
