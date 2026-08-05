/** Accessible, viewer-owned tooltips for controls inside one raster viewer. */

export class ViewerTooltip {
  constructor(root, instanceId) {
    this.root = root;
    this.tooltip = document.createElement('div');
    this.tooltip.id = `rv-tooltip-${instanceId}`;
    this.tooltip.className = 'rv-tooltip';
    this.tooltip.setAttribute('role', 'tooltip');
    this.tooltip.hidden = true;
    this.showTimer = null;
    this.root.append(this.tooltip);
    this.showHandler = event => this.showFor(event.target);
    this.hideHandler = event => {
      const next = event.relatedTarget instanceof Element
        ? event.relatedTarget.closest('[data-rv-tooltip]')
        : null;
      if (!next || !this.root.contains(next)) this.hide();
    };
    this.root.addEventListener('pointerover', this.showHandler);
    this.root.addEventListener('pointerout', this.hideHandler);
    this.root.addEventListener('focusin', this.showHandler);
    this.root.addEventListener('focusout', this.hideHandler);
  }

  /** Connect only explicitly opted-in controls to this tooltip surface. */
  refresh() {
    for (const target of this.root.querySelectorAll('[data-rv-tooltip]')) {
      target.setAttribute('aria-describedby', this.tooltip.id);
    }
  }

  showFor(origin) {
    const target = origin instanceof Element ? origin.closest('[data-rv-tooltip]') : null;
    if (!target || !this.root.contains(target)) return;
    clearTimeout(this.showTimer);
    this.showTimer = setTimeout(() => this.reveal(target), 700);
  }

  reveal(target) {
    if (!target.isConnected || !this.root.contains(target)) return;
    this.tooltip.textContent = target.dataset.rvTooltip;
    this.tooltip.hidden = false;
    const targetRect = target.getBoundingClientRect();
    const tooltipRect = this.tooltip.getBoundingClientRect();
    const left = Math.max(6, Math.min(
      window.innerWidth - tooltipRect.width - 6,
      targetRect.left + (targetRect.width - tooltipRect.width) / 2,
    ));
    this.tooltip.style.left = `${left}px`;
    this.tooltip.style.top = `${Math.max(6, targetRect.bottom + 7)}px`;
  }

  hide() {
    clearTimeout(this.showTimer);
    this.showTimer = null;
    this.tooltip.hidden = true;
  }

  destroy() {
    clearTimeout(this.showTimer);
    this.root.removeEventListener('pointerover', this.showHandler);
    this.root.removeEventListener('pointerout', this.hideHandler);
    this.root.removeEventListener('focusin', this.showHandler);
    this.root.removeEventListener('focusout', this.hideHandler);
    this.tooltip.remove();
  }
}
