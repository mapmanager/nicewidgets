/** Resolve viewer-chrome colors from scoped CSS theme variables. */

export const ViewerTheme = Object.freeze({
  LIGHT: 'light',
  DARK: 'dark',
});

export function normalizeViewerTheme(theme) {
  if (theme !== ViewerTheme.LIGHT && theme !== ViewerTheme.DARK) {
    throw new Error(`unsupported viewer theme: ${theme}`);
  }
  return theme;
}

export function readChromeTheme(root) {
  const style = getComputedStyle(root);
  const value = name => style.getPropertyValue(name).trim();
  return {
    canvasBackground: value('--rv-canvas-bg'),
    axisLine: value('--rv-axis-line'),
    axisText: value('--rv-axis-text'),
    histogramBar: value('--rv-histogram-bar'),
    histogramMinimum: value('--rv-histogram-min'),
    histogramMaximum: value('--rv-histogram-max'),
    histogramHandleLine: value('--rv-histogram-handle-line'),
  };
}
