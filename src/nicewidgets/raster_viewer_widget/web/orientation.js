/** Explicit NumPy-source to transposed, bottom-origin display orientation. */

const TRANSPOSE_TILE = 32;

export const DISPLAY_ORIENTATION = Object.freeze({
  transpose: true,
  flip_y: true,
});

export function transposedShape(sourceHeight, sourceWidth) {
  return {width: sourceHeight, height: sourceWidth};
}

export function transposedAxes(sourceAxes) {
  if (!sourceAxes) return null;
  return {x: sourceAxes.y, y: sourceAxes.x};
}

export function transposePlane(source, sourceHeight, sourceWidth) {
  if (source.length !== sourceHeight * sourceWidth) {
    throw new Error('source plane sample count mismatch');
  }
  const result = new source.constructor(source.length);
  for (let rowBlock = 0; rowBlock < sourceHeight; rowBlock += TRANSPOSE_TILE) {
    const rowEnd = Math.min(sourceHeight, rowBlock + TRANSPOSE_TILE);
    for (let colBlock = 0; colBlock < sourceWidth; colBlock += TRANSPOSE_TILE) {
      const colEnd = Math.min(sourceWidth, colBlock + TRANSPOSE_TILE);
      for (let row = rowBlock; row < rowEnd; row += 1) {
        const sourceOffset = row * sourceWidth;
        for (let col = colBlock; col < colEnd; col += 1) {
          result[col * sourceHeight + row] = source[sourceOffset + col];
        }
      }
    }
  }
  return result;
}

export function sourceToDisplay(row, col) {
  return {x: row, y: col};
}

export function displayToSource(x, y) {
  return {row: x, col: y};
}
