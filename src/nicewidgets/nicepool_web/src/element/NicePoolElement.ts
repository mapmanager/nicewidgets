import { createApp, h, ref, type App, type ComponentPublicInstance } from 'vue'

import type { DatasetInput, NicePoolSelection, NicePoolState, NicePoolTheme, PlotPreset, RowId } from '../core/types'
import type { PlotSummary } from '../plots/types'
import NicePoolWidget from '../vue/NicePoolWidget.vue'
import widgetStyles from '../vue/widget.css?inline'

interface WidgetApi {
  setData(input: DatasetInput): void
  setState(state: NicePoolState): void
  getState(): NicePoolState
  setPlotPresets(presets: readonly PlotPreset[]): void
  getPlotPresets(): PlotPreset[]
  setSelection(selection: NicePoolSelection): void
  setPrimarySelection(rowId: RowId | null): void
  clearSelection(): void
  getSelection(): NicePoolSelection
  getPlotSummary(): PlotSummary | null
  setTheme(theme: NicePoolTheme): void
  getTheme(): NicePoolTheme
}

/** Framework-neutral browser element backed by the same Vue view and pure engine. */
export class NicePoolElement extends HTMLElement {
  #app: App<Element> | null = null
  #widget = ref<(ComponentPublicInstance & WidgetApi) | null>(null)
  #pendingDataset: DatasetInput | null = null

  connectedCallback(): void {
    if (this.#app) return
    const shadow = this.shadowRoot ?? this.attachShadow({ mode: 'open' })
    const style = document.createElement('style')
    style.textContent = widgetStyles
    const mount = document.createElement('div')
    shadow.replaceChildren(style, mount)
    this.#app = createApp({
      render: () => h(NicePoolWidget, {
        ref: this.#widget,
        onSelectionChange: (selection: NicePoolSelection) => {
          this.dispatchEvent(new CustomEvent('nicepool-selection-change', {
            detail: selection,
            bubbles: true,
            composed: true,
          }))
        },
        onDataReset: () => {
          this.dispatchEvent(new CustomEvent('nicepool-data-reset', { bubbles: true, composed: true }))
        },
        onStateChange: (state: NicePoolState) => {
          this.dispatchEvent(new CustomEvent('nicepool-state-change', {
            detail: state,
            bubbles: true,
            composed: true,
          }))
        },
        onPresetsChange: (presets: PlotPreset[]) => {
          this.dispatchEvent(new CustomEvent('nicepool-presets-change', {
            detail: presets,
            bubbles: true,
            composed: true,
          }))
        },
        onThemeChange: (theme: NicePoolTheme) => {
          this.dispatchEvent(new CustomEvent('nicepool-theme-change', {
            detail: theme,
            bubbles: true,
            composed: true,
          }))
        },
      }),
    })
    this.#app.mount(mount)
    if (this.#pendingDataset) {
      const dataset = this.#pendingDataset
      this.#pendingDataset = null
      queueMicrotask(() => this.#widget.value?.setData(dataset))
    }
  }

  disconnectedCallback(): void {
    this.#app?.unmount()
    this.#app = null
    this.#widget.value = null
  }

  /** Replace data and reset selection, plot state, filters, and derived views. */
  setData(input: DatasetInput): void {
    if (!this.#widget.value) {
      this.#pendingDataset = input
      return
    }
    this.#widget.value.setData(input)
  }

  setSelection(selection: NicePoolSelection): void {
    this.#widget.value?.setSelection(selection)
  }

  setState(state: NicePoolState): void { this.#widget.value?.setState(state) }
  getState(): NicePoolState {
    if (!this.#widget.value) throw new Error('NicePool element is not connected')
    return this.#widget.value.getState()
  }
  setPlotPresets(presets: readonly PlotPreset[]): void { this.#widget.value?.setPlotPresets(presets) }
  getPlotPresets(): PlotPreset[] { return this.#widget.value?.getPlotPresets() ?? [] }
  setTheme(theme: NicePoolTheme): void { this.#widget.value?.setTheme(theme) }
  getTheme(): NicePoolTheme { return this.#widget.value?.getTheme() ?? 'dark' }

  setPrimarySelection(rowId: RowId | null): void {
    this.#widget.value?.setPrimarySelection(rowId)
  }

  clearSelection(): void {
    this.#widget.value?.clearSelection()
  }

  getSelection(): NicePoolSelection {
    return this.#widget.value?.getSelection() ?? { primaryRowId: null, selectedRowIds: [] }
  }

  getPlotSummary(): PlotSummary | null {
    return this.#widget.value?.getPlotSummary() ?? null
  }
}
