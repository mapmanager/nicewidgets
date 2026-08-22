<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'

import type { NicePoolSelection, NicePoolTheme, RowId } from '../core/types'
import { buildPlotlySpecification, rowIdsFromPlotlyEvent } from '../plots/plotly'
import type { PreparedPlotData } from '../plots/types'

const props = defineProps<{ data: PreparedPlotData; selection: NicePoolSelection; theme: NicePoolTheme }>()
const emit = defineEmits<{ selection: [rowIds: RowId[], primaryRowId: RowId | null, additive: boolean] }>()
const host = ref<HTMLDivElement | null>(null)
let plotly: typeof import('plotly.js') | null = null
let resizeObserver: ResizeObserver | null = null
let specificationUpdates = 0
let shiftPressed = false
let boundHost: (HTMLDivElement & {
  on: (name: string, callback: (event: unknown) => void) => void
  removeAllListeners: (name: string) => void
}) | null = null

/** Make Plotly's document-level runtime CSS available inside a Custom Element shadow root. */
function installPlotlyShadowStyles(): void {
  const root = host.value?.getRootNode()
  if (!(root instanceof ShadowRoot) || root.querySelector('style[data-nicepool-plotly]')) return
  const source = [...document.styleSheets].find((sheet) => (sheet.ownerNode as HTMLElement | null)?.id === 'plotly.js-style-global')
  if (!source) return
  const style = document.createElement('style')
  style.dataset.nicepoolPlotly = ''
  style.textContent = [...source.cssRules].map((rule) => rule.cssText).join('\n')
  root.prepend(style)
}

function selectionHandler(event: unknown): void {
  if (specificationUpdates > 0) return
  const rowIds = rowIdsFromPlotlyEvent(event as never)
  emit('selection', rowIds, rowIds[0] ?? null, shiftPressed)
}

function clickHandler(event: unknown): void {
  const rowIds = rowIdsFromPlotlyEvent(event as never)
  const primary = rowIds[0] ?? null
  emit('selection', primary ? [primary] : [], primary, shiftPressed)
}

function keyChanged(event: KeyboardEvent): void {
  if (event.key === 'Shift') shiftPressed = event.type === 'keydown'
}

function resetModifiers(): void {
  shiftPressed = false
}

async function render(): Promise<void> {
  if (!host.value) return
  plotly ??= (await import('plotly.js-dist-min')).default
  installPlotlyShadowStyles()
  const specification = buildPlotlySpecification(props.data, props.selection, props.theme)
  specificationUpdates += 1
  try {
    await plotly.react(host.value, specification.traces, specification.layout, specification.config)
  } finally {
    specificationUpdates -= 1
  }
  const nextHost = host.value as typeof boundHost
  if (boundHost !== nextHost) {
    boundHost?.removeAllListeners('plotly_selected')
    boundHost?.removeAllListeners('plotly_click')
    nextHost?.on('plotly_selected', selectionHandler)
    nextHost?.on('plotly_click', clickHandler)
    boundHost = nextHost
  }
}

onMounted(() => {
  window.addEventListener('keydown', keyChanged)
  window.addEventListener('keyup', keyChanged)
  window.addEventListener('blur', resetModifiers)
  resizeObserver = new ResizeObserver(() => {
    if (plotly && host.value) void plotly.Plots.resize(host.value)
  })
  if (host.value) resizeObserver.observe(host.value)
  void render()
})
watch(() => [props.data, props.selection, props.theme] as const, () => void render(), { deep: true })
onBeforeUnmount(() => {
  window.removeEventListener('keydown', keyChanged)
  window.removeEventListener('keyup', keyChanged)
  window.removeEventListener('blur', resetModifiers)
  resizeObserver?.disconnect()
  boundHost?.removeAllListeners('plotly_selected')
  boundHost?.removeAllListeners('plotly_click')
  if (host.value) plotly?.purge(host.value)
})
</script>

<template><div ref="host" class="nicepool-plot" /></template>
