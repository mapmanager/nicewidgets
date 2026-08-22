<script setup lang="ts">
import { ref } from 'vue'

import type { NicePoolSelection, NicePoolTheme } from '../core/types'
import NicePoolWidget from '../vue/NicePoolWidget.vue'
import { sampleDataset } from './sampleData'

const dataset = sampleDataset()
const lastSelection = ref<NicePoolSelection>({ primaryRowId: null, selectedRowIds: [] })
const theme = ref<NicePoolTheme>('dark')
</script>

<template>
  <div class="demo-page" :class="`demo-page-${theme}`">
    <header>
      <div>
        <p class="eyebrow">Slice 1 development host</p>
        <h1>NicePool Web</h1>
      </div>
      <p>{{ dataset.rows.length }} rows · primary {{ lastSelection.primaryRowId ?? 'none' }}</p>
    </header>
    <NicePoolWidget :dataset="dataset" :theme="theme" preset-storage-key="nicepool-web-demo-presets-v3" @selection-change="lastSelection = $event" @theme-change="theme = $event" />
  </div>
</template>

<style>
html, body, #app { min-height: 100%; margin: 0; }
.demo-page { min-height: 100vh; box-sizing: border-box; margin: 0 auto; padding: 24px max(24px, calc((100vw - 1280px) / 2)); transition: background 120ms ease, color 120ms ease; }
.demo-page-dark { color: #e5e7eb; background: #0b1120; color-scheme: dark; }
.demo-page-light { color: #172033; background: #e2e8f0; color-scheme: light; }
.demo-page > header { display: flex; align-items: end; justify-content: space-between; gap: 16px; margin-bottom: 16px; color: inherit; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
.demo-page h1, .demo-page p { margin: 0; }
.demo-page .eyebrow { margin-bottom: 4px; color: #94a3b8; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }
</style>
