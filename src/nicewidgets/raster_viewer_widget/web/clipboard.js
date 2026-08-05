/** Compose one visible viewport and copy it as a PNG image. */

function canvasToPngBlob(canvas) {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      blob => blob ? resolve(blob) : reject(new Error('PNG encoding failed')),
      'image/png',
    );
  });
}

export function clipboardImageSupported() {
  return Boolean(
    navigator.clipboard?.write
    && window.ClipboardItem
    && (!ClipboardItem.supports || ClipboardItem.supports('image/png')),
  );
}

export async function copyViewportToClipboard(viewport) {
  if (!clipboardImageSupported()) {
    throw new Error('This browser does not support copying PNG images to the clipboard');
  }
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
  const blob = await canvasToPngBlob(output);
  await navigator.clipboard.write([new ClipboardItem({'image/png': blob})]);
}
