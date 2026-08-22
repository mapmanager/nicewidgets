import { NicePoolElement } from './NicePoolElement'

/** Register the default framework-neutral `<nice-pool>` browser element. */
export function registerNicePoolElement(tagName = 'nice-pool'): void {
  if (!customElements.get(tagName)) customElements.define(tagName, NicePoolElement)
}
