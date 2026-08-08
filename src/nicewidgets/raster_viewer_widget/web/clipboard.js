/** Compose one visible viewport and copy it as a PNG image. */

function canvasToPngBlob(canvas) {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      blob => blob ? resolve(blob) : reject(new Error('PNG encoding failed')),
      'image/png',
    );
  });
}

function blobToDataUrl(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error('PNG data URL encoding failed'));
    reader.readAsDataURL(blob);
  });
}

/**
 * True when the browser Clipboard API can write a PNG image.
 *
 * @returns {boolean} Browser PNG clipboard support.
 */
export function clipboardImageSupported() {
  if (typeof navigator === 'undefined' || typeof window === 'undefined') return false;
  return Boolean(
    navigator.clipboard?.write
    && window.ClipboardItem
    && (!ClipboardItem.supports || ClipboardItem.supports('image/png')),
  );
}

/**
 * True when Copy view can succeed via browser API and/or a native host bridge.
 *
 * @param {boolean} [hostClipboardBridge=false] Python native clipboard bridge.
 * @returns {boolean} Whether the Copy view control should be enabled.
 */
export function clipboardCopyAvailable(hostClipboardBridge = false) {
  return Boolean(hostClipboardBridge) || clipboardImageSupported();
}

/**
 * Draw raster + overlays into one offscreen canvas and encode a PNG blob.
 *
 * @param {import('./viewport.js').RasterViewport} viewport Active viewport.
 * @returns {Promise<Blob>} PNG blob.
 */
export async function composeViewportPngBlob(viewport) {
  viewport.draw();
  const output = document.createElement('canvas');
  output.width = viewport.canvas.width;
  output.height = viewport.canvas.height;
  const context = output.getContext('2d');
  context.drawImage(viewport.canvas, 0, 0);
  if (viewport.plotOverlay) {
    context.drawImage(viewport.plotOverlay.canvas, 0, 0);
  }
  if (viewport.interactionCanvas !== viewport.canvas) {
    context.drawImage(viewport.interactionCanvas, 0, 0);
  }
  return canvasToPngBlob(output);
}

/**
 * Compose the visible viewport as a ``data:image/png;base64,...`` URL.
 *
 * Used when the native NiceGUI host must write the OS clipboard in Python.
 *
 * @param {import('./viewport.js').RasterViewport} viewport Active viewport.
 * @returns {Promise<string>} PNG data URL.
 */
export async function composeViewportPngDataUrl(viewport) {
  const blob = await composeViewportPngBlob(viewport);
  return blobToDataUrl(blob);
}

/**
 * Compose the visible viewport and write it through the browser Clipboard API.
 *
 * @param {import('./viewport.js').RasterViewport} viewport Active viewport.
 * @returns {Promise<void>} Resolves after the clipboard write completes.
 */
export async function copyViewportToClipboard(viewport) {
  if (!clipboardImageSupported()) {
    throw new Error('This browser does not support copying PNG images to the clipboard');
  }
  const blob = await composeViewportPngBlob(viewport);
  await navigator.clipboard.write([new ClipboardItem({'image/png': blob})]);
}
