import { resolve } from 'node:path'

import vue from '@vitejs/plugin-vue'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [vue()],
  build: {
    outDir: 'dist-lib',
    lib: {
      entry: resolve(import.meta.dirname, 'src/public-api.ts'),
      formats: ['es'],
      fileName: 'nicepool-web',
    },
    rollupOptions: {
      external: ['vue', 'plotly.js-dist-min'],
    },
  },
})
