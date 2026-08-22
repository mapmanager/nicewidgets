import type { NicePoolSelection, RowId } from './types'

const EMPTY_SELECTION: NicePoolSelection = Object.freeze({ primaryRowId: null, selectedRowIds: Object.freeze([]) })

/** Authoritative primary and multi-row selection independent of any visual view. */
export class SelectionModel {
  #selection: NicePoolSelection = EMPTY_SELECTION

  get value(): NicePoolSelection {
    return this.#selection
  }

  reset(): void {
    this.#selection = EMPTY_SELECTION
  }

  set(selection: NicePoolSelection, validIds: ReadonlySet<RowId>): void {
    const unique = [...new Set(selection.selectedRowIds)]
    for (const rowId of unique) {
      if (!validIds.has(rowId)) throw new RangeError(`Unknown row ID ${JSON.stringify(rowId)}`)
    }
    if (selection.primaryRowId !== null && !validIds.has(selection.primaryRowId)) {
      throw new RangeError(`Unknown primary row ID ${JSON.stringify(selection.primaryRowId)}`)
    }
    if (selection.primaryRowId !== null && !unique.includes(selection.primaryRowId)) unique.unshift(selection.primaryRowId)
    this.#selection = Object.freeze({
      primaryRowId: selection.primaryRowId,
      selectedRowIds: Object.freeze(unique),
    })
  }
}
