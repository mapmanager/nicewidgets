<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, shallowRef, watch } from 'vue'
import { NicePoolEngine } from '../core/engine'
import { validatePlotPreset, visiblePlotCount } from '../core/state'
import type { DatasetInput, NicePoolSelection, NicePoolState, NicePoolTheme, NicePoolValue, PlotLayout, PlotPreset, PlotState, RowId } from '../core/types'
import type { PreparedPlot, PlotSummary } from '../plots/types'
import { formatPlotSummaryToTsv } from '../plots/summary-format'
import PlotlyView from './PlotlyView.vue'
import './widget.css'

const props = withDefaults(defineProps<{ dataset?: DatasetInput; presetStorageKey?: string; theme?: NicePoolTheme }>(), {
  presetStorageKey: '',
  theme: 'dark',
})
const emit = defineEmits<{
  'selection-change': [selection: NicePoolSelection]
  'state-change': [state: NicePoolState]
  'presets-change': [presets: readonly PlotPreset[]]
  'theme-change': [theme: NicePoolTheme]
  'data-reset': []
}>()
const engine = new NicePoolEngine()
const preparedPlots = shallowRef<readonly PreparedPlot[]>([])
const selection = ref<NicePoolSelection>({ primaryRowId: null, selectedRowIds: [] })
const presets = ref<PlotPreset[]>([])
const selectedPresetName = ref('')
const presetName = ref('')
const error = ref<string | null>(null)
const revision = ref(0)
const activeTheme = ref<NicePoolTheme>(props.theme)
const controlsWidth = ref(290)
const plotHeight = ref(620)
const summaryCopyStatus = ref('')
const showSummaryPlotState = ref(true)
const showSummaryRawData = ref(true)
let stopPointerResize: (() => void) | null = null

function jsonClone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T
}

function setTheme(theme: NicePoolTheme, userInitiated = false): void {
  activeTheme.value = theme
  if (userInitiated) emit('theme-change', theme)
}

function getTheme(): NicePoolTheme {
  return activeTheme.value
}

function startResize(axis: 'horizontal' | 'vertical', event: PointerEvent): void {
  event.preventDefault()
  stopPointerResize?.()
  const startPosition = axis === 'vertical' ? event.clientX : event.clientY
  const startValue = axis === 'vertical' ? controlsWidth.value : plotHeight.value
  const onMove = (moveEvent: PointerEvent): void => {
    const position = axis === 'vertical' ? moveEvent.clientX : moveEvent.clientY
    const next = startValue + position - startPosition
    if (axis === 'vertical') controlsWidth.value = Math.min(520, Math.max(0, next))
    else plotHeight.value = Math.min(1400, Math.max(320, next))
  }
  const stop = (): void => {
    window.removeEventListener('pointermove', onMove)
    window.removeEventListener('pointerup', stop)
    window.removeEventListener('pointercancel', stop)
    document.body.classList.remove('nicepool-resizing')
    stopPointerResize = null
  }
  stopPointerResize = stop
  document.body.classList.add('nicepool-resizing')
  window.addEventListener('pointermove', onMove)
  window.addEventListener('pointerup', stop)
  window.addEventListener('pointercancel', stop)
}

function resizeWithKeyboard(axis: 'horizontal' | 'vertical', event: KeyboardEvent): void {
  const step = event.shiftKey ? 50 : 10
  if (axis === 'vertical' && ['ArrowLeft', 'ArrowRight'].includes(event.key)) {
    event.preventDefault()
    controlsWidth.value = Math.min(520, Math.max(0, controlsWidth.value + (event.key === 'ArrowRight' ? step : -step)))
  }
  if (axis === 'horizontal' && ['ArrowUp', 'ArrowDown'].includes(event.key)) {
    event.preventDefault()
    plotHeight.value = Math.min(1400, Math.max(320, plotHeight.value + (event.key === 'ArrowDown' ? step : -step)))
  }
}

const state = computed(() => { revision.value; return preparedPlots.value.length ? engine.state : null })
const plotState = computed(() => state.value?.plots[state.value.activePlotIndex] ?? null)
const activeSummary = computed(() => state.value ? preparedPlots.value[state.value.activePlotIndex]?.summary ?? null : null)
const visibleIndexes = computed(() => state.value ? Array.from({ length: visiblePlotCount(state.value.layout) }, (_, index) => index) : [])
const numericColumns = computed(() => { revision.value; return preparedPlots.value.length ? engine.dataset.numericColumns() : [] })
const scatterXColumns = computed(() => {
  revision.value
  return preparedPlots.value.length ? engine.dataset.schema.map(({ name }) => name).filter((name) => name !== engine.dataset.rowIdColumn) : []
})
const categoricalColumns = computed(() => { revision.value; return preparedPlots.value.length ? engine.dataset.categoricalColumns() : [] })
const filterColumns = computed(() => { revision.value; return preparedPlots.value.length ? engine.dataset.preFilterColumns() : [] })
const isHistogram = computed(() => plotState.value?.plotType === 'histogram' || plotState.value?.plotType === 'cumulativeHistogram')
const isDistribution = computed(() => plotState.value?.plotType === 'swarm' || plotState.value?.plotType === 'box' || plotState.value?.plotType === 'violin')
const supportsPointSize = computed(() => plotState.value?.plotType === 'scatter' || plotState.value?.plotType === 'swarm')
const showsQuartiles = computed(() => activeSummary.value?.plotType === 'box' || activeSummary.value?.plotType === 'violin')
const availableXColumns = computed(() => isHistogram.value ? numericColumns.value : scatterXColumns.value)

function refresh(): void {
  try {
    preparedPlots.value = engine.prepareVisiblePlots()
    selection.value = engine.selection
    error.value = null
    revision.value += 1
  } catch (reason) { error.value = reason instanceof Error ? reason.message : String(reason) }
}

function persistedPresets(): PlotPreset[] {
  if (!props.presetStorageKey) return []
  try {
    const value: unknown = JSON.parse(localStorage.getItem(props.presetStorageKey) ?? '[]')
    if (!Array.isArray(value)) return []
    return value.flatMap((item) => {
      try { return [validatePlotPreset(engine.dataset, item as PlotPreset)] }
      catch { return [] }
    })
  } catch { return [] }
}

function persistPresets(): void {
  if (props.presetStorageKey) localStorage.setItem(props.presetStorageKey, JSON.stringify(presets.value))
  emit('presets-change', getPlotPresets())
}

/** Fully replace the dataset and reset every dataset-dependent view state. */
function setData(input: DatasetInput): void {
  engine.setData(input)
  presets.value = persistedPresets()
  selectedPresetName.value = ''
  refresh()
  emit('data-reset')
}

function updatePlotState(patch: Partial<PlotState>): void {
  if (!plotState.value) return
  engine.setPlotState({ ...plotState.value, ...patch })
  refresh()
  emit('state-change', getState())
}

function changePlotType(plotType: PlotState['plotType']): void {
  const requiresGroup = ['swarm', 'box', 'violin'].includes(plotType)
  const histogram = ['histogram', 'cumulativeHistogram'].includes(plotType)
  const groupColumn = requiresGroup && plotState.value?.groupColumn === null
    ? categoricalColumns.value[0] ?? null
    : histogram
      ? null
      : plotState.value?.groupColumn
  const xColumn = histogram && plotState.value && !numericColumns.value.includes(plotState.value.xColumn)
    ? numericColumns.value[0] ?? plotState.value.xColumn
    : plotState.value?.xColumn
  updatePlotState({ plotType, ...(groupColumn !== undefined ? { groupColumn } : {}), ...(xColumn !== undefined ? { xColumn } : {}) })
}

function updateFilter(column: string, value: string): void {
  if (!plotState.value) return
  const preFilters = { ...plotState.value.preFilters }
  if (value === '') delete preFilters[column]
  else preFilters[column] = value as NicePoolValue
  updatePlotState({ preFilters })
}

function setLayout(layout: PlotLayout): void { engine.setLayout(layout); refresh(); emit('state-change', getState()) }
function setActivePlot(index: number): void { engine.setActivePlot(index); refresh(); selectedPresetName.value = ''; emit('state-change', getState()) }

function selectRows(rowIds: RowId[], primaryRowId: RowId | null, additive = false, userInitiated = true): void {
  if (additive) engine.extendSelection(rowIds, primaryRowId)
  else engine.setSelection({ primaryRowId, selectedRowIds: rowIds })
  refresh()
  if (userInitiated) emit('selection-change', engine.selection)
}

function savePreset(): void {
  if (!plotState.value) return
  const name = presetName.value.trim()
  if (!name) { error.value = 'Enter a preset name before saving.'; return }
  const preset = validatePlotPreset(engine.dataset, { schemaVersion: 1, name, plotState: structuredClone(plotState.value) })
  const index = presets.value.findIndex((item) => item.name === name)
  if (index >= 0) presets.value.splice(index, 1, preset)
  else presets.value.push(preset)
  presets.value.sort((a, b) => a.name.localeCompare(b.name))
  selectedPresetName.value = name
  persistPresets()
  error.value = null
}

function applyPreset(name: string): void {
  selectedPresetName.value = name
  if (!name) return
  const preset = presets.value.find((item) => item.name === name)
  if (!preset) return
  try { engine.applyPlotPreset(jsonClone(preset)); refresh(); emit('state-change', getState()) }
  catch (reason) { error.value = reason instanceof Error ? reason.message : String(reason) }
}

function deletePreset(): void {
  if (!selectedPresetName.value) return
  presets.value = presets.value.filter(({ name }) => name !== selectedPresetName.value)
  selectedPresetName.value = ''
  persistPresets()
}

function setState(next: NicePoolState): void { engine.setState(next); refresh() }
function getState(): NicePoolState { return structuredClone(engine.state) }
function setPlotPresets(next: readonly PlotPreset[]): void { presets.value = next.map((preset) => validatePlotPreset(engine.dataset, preset)); persistPresets() }
function getPlotPresets(): PlotPreset[] { return jsonClone(presets.value) }
function setSelection(next: NicePoolSelection): void { engine.setSelection(next); refresh() }
function setPrimarySelection(rowId: RowId | null): void { engine.setPrimarySelection(rowId); refresh() }
function clearSelection(userInitiated = false): void {
  engine.clearSelection()
  refresh()
  if (userInitiated) emit('selection-change', engine.selection)
}
function handleGlobalKeydown(event: KeyboardEvent): void {
  if (event.key !== 'Escape' || !state.value) return
  event.preventDefault()
  clearSelection(true)
}
function getSelection(): NicePoolSelection { return engine.selection }
function getPlotSummary(plotIndex = engine.state.activePlotIndex): PlotSummary | null { return preparedPlots.value[plotIndex]?.summary ?? null }

async function copySummary(): Promise<void> {
  if (!activeSummary.value) return
  try {
    await navigator.clipboard.writeText(formatPlotSummaryToTsv(activeSummary.value, {
      includeParameters: showSummaryPlotState.value,
      includeRepresentedRows: showSummaryRawData.value,
    }))
    summaryCopyStatus.value = 'Copied'
  } catch (reason) {
    summaryCopyStatus.value = 'Copy failed'
    error.value = reason instanceof Error ? reason.message : String(reason)
  }
}

function displayStatistic(value: number | null): string {
  return value === null ? '' : new Intl.NumberFormat(undefined, { maximumSignificantDigits: 6 }).format(value)
}

function displaySummaryValue(value: unknown): string {
  if (value === null || value === undefined) return ''
  return typeof value === 'object' ? JSON.stringify(value) : String(value)
}

defineExpose({ setData, setState, getState, setPlotPresets, getPlotPresets, setSelection, setPrimarySelection, clearSelection, getSelection, getPlotSummary, setTheme, getTheme })
watch(() => props.dataset, (dataset) => { if (dataset) setData(dataset) }, { immediate: true })
watch(() => props.theme, (theme) => setTheme(theme))
onMounted(() => window.addEventListener('keydown', handleGlobalKeydown))
onBeforeUnmount(() => {
  stopPointerResize?.()
  window.removeEventListener('keydown', handleGlobalKeydown)
})
</script>

<template>
  <section class="nicepool-shell" :class="`nicepool-theme-${activeTheme}`" :style="{ '--nicepool-controls-width': `${controlsWidth}px`, '--nicepool-plot-height': `${plotHeight}px` }">
    <aside v-if="state && plotState" class="nicepool-controls" :class="{ 'nicepool-controls-collapsed': controlsWidth === 0 }">
      <div class="nicepool-control-row">
        <label>Layout<select :value="state.layout" @change="setLayout(($event.target as HTMLSelectElement).value as PlotLayout)">
          <option value="1x1">1×1</option><option value="1x2">1×2</option><option value="2x1">2×1</option><option value="2x2">2×2</option>
        </select></label>
        <label>Edit plot<select :value="state.activePlotIndex" @change="setActivePlot(Number(($event.target as HTMLSelectElement).value))">
          <option v-for="index in visibleIndexes" :key="index" :value="index">Plot {{ index + 1 }}</option>
        </select></label>
      </div>
      <label>Plot type<select :value="plotState.plotType" @change="changePlotType(($event.target as HTMLSelectElement).value as PlotState['plotType'])">
        <option value="scatter">Scatter</option><option value="swarm">Swarm</option><option value="box">Box</option><option value="violin">Violin</option><option value="histogram">Histogram</option><option value="cumulativeHistogram">Cumulative histogram</option>
      </select></label>
      <label :class="{ 'nicepool-control-disabled': isDistribution }">X column<select :disabled="isDistribution" :value="plotState.xColumn" @change="updatePlotState({ xColumn: ($event.target as HTMLSelectElement).value })">
        <option v-for="column in availableXColumns" :key="column">{{ column }}</option>
      </select></label>
      <label :class="{ 'nicepool-control-disabled': isHistogram }">Y column<select :disabled="isHistogram" :value="plotState.yColumn" @change="updatePlotState({ yColumn: ($event.target as HTMLSelectElement).value })">
        <option v-for="column in numericColumns" :key="column">{{ column }}</option>
      </select></label>
      <label :class="{ 'nicepool-control-disabled': plotState.plotType === 'scatter' }">Group<select :disabled="plotState.plotType === 'scatter'" :value="plotState.groupColumn ?? ''" @change="updatePlotState({ groupColumn: ($event.target as HTMLSelectElement).value || null })">
        <option value="">Choose a group</option><option v-for="column in categoricalColumns" :key="column">{{ column }}</option>
      </select></label>
      <label>Color by<select :value="plotState.colorColumn ?? ''" @change="updatePlotState({ colorColumn: ($event.target as HTMLSelectElement).value || null })">
        <option value="">None</option><option v-for="column in categoricalColumns" :key="column">{{ column }}</option>
      </select></label>
      <fieldset v-if="filterColumns.length"><legend>Filters</legend>
        <label v-for="column in filterColumns" :key="column">{{ column }}<select :value="String(plotState.preFilters[column] ?? '')" @change="updateFilter(column, ($event.target as HTMLSelectElement).value)">
          <option value="">All</option><option v-for="value in engine.dataset.uniqueValues(column)" :key="String(value)" :value="String(value)">{{ value }}</option>
        </select></label>
      </fieldset>
      <label><span>Absolute values</span><input type="checkbox" :checked="plotState.useAbsoluteValue" @change="updatePlotState({ useAbsoluteValue: ($event.target as HTMLInputElement).checked })" /></label>
      <label>Keep within ±<span class="nicepool-inline-option"><input type="checkbox" aria-label="Exclude extreme values" :checked="plotState.removeValuesThreshold !== null" @change="updatePlotState({ removeValuesThreshold: ($event.target as HTMLInputElement).checked ? 10 : null })" /><input type="number" min="0" step="0.5" aria-label="Extreme-value threshold" :disabled="plotState.removeValuesThreshold === null" :value="plotState.removeValuesThreshold ?? 10" @change="updatePlotState({ removeValuesThreshold: Number(($event.target as HTMLInputElement).value) })" /></span></label>
      <label :class="{ 'nicepool-control-disabled': !supportsPointSize }">Point size<input type="number" min="1" max="30" step="1" :disabled="!supportsPointSize" :value="plotState.pointSize" @change="updatePlotState({ pointSize: Number(($event.target as HTMLInputElement).value) })" /></label>
      <label :class="{ 'nicepool-control-disabled': !isHistogram }">Histogram bins<input type="number" min="1" max="200" step="1" :disabled="!isHistogram" :value="plotState.histogramBins" @change="updatePlotState({ histogramBins: Number(($event.target as HTMLInputElement).value) })" /></label>
      <div class="nicepool-control-group">
        <label :class="{ 'nicepool-control-disabled': !isDistribution }"><span>Raw points</span><input type="checkbox" :disabled="!isDistribution" :checked="plotState.showRaw" @change="updatePlotState({ showRaw: ($event.target as HTMLInputElement).checked })" /></label>
        <label :class="{ 'nicepool-control-disabled': plotState.plotType !== 'swarm' }"><span>Show mean</span><input type="checkbox" :disabled="plotState.plotType !== 'swarm'" :checked="plotState.showMean" @change="updatePlotState({ showMean: ($event.target as HTMLInputElement).checked })" /></label>
        <label :class="{ 'nicepool-control-disabled': plotState.plotType !== 'swarm' }">Error type<span class="nicepool-inline-option"><input type="checkbox" aria-label="Show error bars" :disabled="plotState.plotType !== 'swarm'" :checked="plotState.showErrorBars" @change="updatePlotState({ showErrorBars: ($event.target as HTMLInputElement).checked })" /><select aria-label="Error type" :disabled="plotState.plotType !== 'swarm' || !plotState.showErrorBars" :value="plotState.errorBarType" @change="updatePlotState({ errorBarType: ($event.target as HTMLSelectElement).value as PlotState['errorBarType'] })">
          <option value="sem">Standard error (SE)</option><option value="std">Standard deviation (SD)</option>
        </select></span></label>
      </div>
      <details class="nicepool-display-options">
        <summary><span aria-hidden="true">☰</span> Display options</summary>
        <div class="nicepool-display-options-panel">
          <label>Legend position<span class="nicepool-inline-option"><input type="checkbox" aria-label="Show legend" :checked="plotState.showLegend" @change="updatePlotState({ showLegend: ($event.target as HTMLInputElement).checked })" /><select aria-label="Legend position" :disabled="!plotState.showLegend" :value="plotState.legendPosition" @change="updatePlotState({ legendPosition: ($event.target as HTMLSelectElement).value as PlotState['legendPosition'] })">
            <option value="bottom">Bottom</option><option value="right">Right</option><option value="top">Top</option><option value="left">Left</option>
          </select></span></label>
          <label><span>Plotly toolbar</span><input type="checkbox" :checked="plotState.showPlotlyToolbar" @change="updatePlotState({ showPlotlyToolbar: ($event.target as HTMLInputElement).checked })" /></label>
          <label><span>Hover</span><input type="checkbox" :checked="plotState.showHover" @change="updatePlotState({ showHover: ($event.target as HTMLInputElement).checked })" /></label>
          <label><span>Axes</span><input type="checkbox" :checked="plotState.showAxes" @change="updatePlotState({ showAxes: ($event.target as HTMLInputElement).checked })" /></label>
          <label><span>Horizontal grid</span><input type="checkbox" :checked="plotState.showHorizontalGrid" @change="updatePlotState({ showHorizontalGrid: ($event.target as HTMLInputElement).checked })" /></label>
          <label><span>Vertical grid</span><input type="checkbox" :checked="plotState.showVerticalGrid" @change="updatePlotState({ showVerticalGrid: ($event.target as HTMLInputElement).checked })" /></label>
        </div>
      </details>
      <fieldset><legend>Saved plot</legend>
        <label>Preset<select :value="selectedPresetName" @change="applyPreset(($event.target as HTMLSelectElement).value)"><option value="">None</option><option v-for="preset in presets" :key="preset.name" :value="preset.name">{{ preset.name }}</option></select></label>
        <label>Name<input v-model="presetName" type="text" placeholder="Preset name" /></label>
        <div class="nicepool-control-row"><button type="button" @click="savePreset">Save</button><button type="button" :disabled="!selectedPresetName" @click="deletePreset">Delete</button></div>
      </fieldset>
      <button type="button" @click="clearSelection(true)">Clear selection</button><p class="nicepool-status">Selected {{ selection.selectedRowIds.length }}</p>
    </aside>
    <div class="nicepool-splitter nicepool-splitter-vertical" role="separator" aria-label="Resize controls" aria-orientation="vertical" tabindex="0" @pointerdown="startResize('vertical', $event)" @keydown="resizeWithKeyboard('vertical', $event)" />
    <section class="nicepool-plot-region">
      <main v-if="state" class="nicepool-main nicepool-grid" :class="`nicepool-layout-${state.layout}`">
        <section v-for="(prepared, index) in preparedPlots" :key="index" class="nicepool-plot-cell" :class="{ 'nicepool-plot-active': index === state.activePlotIndex }" @click="setActivePlot(index)">
          <span class="nicepool-plot-number">Plot {{ index + 1 }}</span><PlotlyView :data="prepared.data" :selection="selection" :theme="activeTheme" @selection="selectRows" />
        </section>
        <p v-if="error" class="nicepool-error">{{ error }}</p>
      </main>
      <main v-else class="nicepool-main"><p class="nicepool-empty">Set a dataset to begin.</p></main>
      <div class="nicepool-splitter nicepool-splitter-horizontal" role="separator" aria-label="Resize plots" aria-orientation="horizontal" tabindex="0" @pointerdown="startResize('horizontal', $event)" @keydown="resizeWithKeyboard('horizontal', $event)" />
      <details v-if="activeSummary" class="nicepool-summary-panel">
        <summary>Plot {{ state!.activePlotIndex + 1 }} summary · {{ activeSummary.representedRows.length }} rows</summary>
        <div class="nicepool-summary-content">
          <div class="nicepool-summary-actions">
            <button type="button" @click="copySummary">Copy Summary</button>
            <label><input v-model="showSummaryPlotState" type="checkbox" /> Plot State</label>
            <label><input v-model="showSummaryRawData" type="checkbox" /> Raw Data</label>
            <span role="status">{{ summaryCopyStatus }}</span>
          </div>
          <div class="nicepool-summary-table-scroll">
            <template v-if="showSummaryPlotState">
              <h3>Plot State</h3>
              <table><tbody><tr v-for="(value, key) in activeSummary.parameters" :key="key"><th>{{ key }}</th><td>{{ displaySummaryValue(value) }}</td></tr></tbody></table>
            </template>
            <h3>Summary</h3>
            <table>
              <thead><tr><th>Group</th><th>Color</th><th>Count</th><th>Min</th><th v-if="showsQuartiles">Q1</th><th>Median</th><th v-if="showsQuartiles">Q3</th><th>Max</th><th v-if="showsQuartiles">IQR</th><th>Mean</th><th>SD</th><th>SE</th><th>CV</th></tr></thead>
              <tbody><tr v-for="(row, index) in activeSummary.aggregateRows" :key="index">
                <td>{{ row.groupValue ?? 'Overall' }}</td><td>{{ row.colorValue ?? '' }}</td><td>{{ row.statistics.count }}</td><td>{{ displayStatistic(row.statistics.min) }}</td><td v-if="showsQuartiles">{{ displayStatistic(row.statistics.q1 ?? null) }}</td><td>{{ displayStatistic(row.statistics.median) }}</td><td v-if="showsQuartiles">{{ displayStatistic(row.statistics.q3 ?? null) }}</td><td>{{ displayStatistic(row.statistics.max) }}</td><td v-if="showsQuartiles">{{ displayStatistic(row.statistics.iqr ?? null) }}</td><td>{{ displayStatistic(row.statistics.mean) }}</td><td>{{ displayStatistic(row.statistics.std) }}</td><td>{{ displayStatistic(row.statistics.sem) }}</td><td>{{ displayStatistic(row.statistics.cv) }}</td>
              </tr></tbody>
            </table>
            <template v-if="activeSummary.bins">
              <h3>Bins</h3>
              <table>
                <thead><tr><th>Group</th><th>Color</th><th>Lower</th><th>Upper</th><th>Center</th><th>Count</th><th>Cumulative count</th><th>Cumulative proportion</th></tr></thead>
                <tbody><tr v-for="(bin, index) in activeSummary.bins" :key="index"><td>{{ bin.groupValue ?? 'Overall' }}</td><td>{{ bin.colorValue ?? '' }}</td><td>{{ displayStatistic(bin.lower) }}</td><td>{{ displayStatistic(bin.upper) }}</td><td>{{ displayStatistic(bin.center) }}</td><td>{{ bin.count }}</td><td>{{ bin.cumulativeCount }}</td><td>{{ displayStatistic(bin.cumulativeProportion) }}</td></tr></tbody>
              </table>
            </template>
            <template v-if="showSummaryRawData">
              <h3>Raw Data</h3>
              <table>
                <thead><tr><th>Row ID</th><th>X</th><th>Y</th><th>Group</th><th>Color</th></tr></thead>
                <tbody><tr v-for="row in activeSummary.representedRows" :key="row.rowId"><td>{{ row.rowId }}</td><td>{{ displaySummaryValue(row.x) }}</td><td>{{ displayStatistic(row.y) }}</td><td>{{ row.groupValue ?? '' }}</td><td>{{ row.colorValue ?? '' }}</td></tr></tbody>
              </table>
            </template>
          </div>
        </div>
      </details>
    </section>
  </section>
</template>
