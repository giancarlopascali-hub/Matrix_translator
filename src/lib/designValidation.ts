import { ReconstructedMatrix } from './types';

export type DesignCorner = 'top_left' | 'top_right' | 'bottom_left' | 'bottom_right';

export interface DesignValidationResult {
  matrixId: string;
  validation: 0 | 1;
  binaryValuesOnly: boolean;
  activeCellCount: number;
  componentCount: number;
  significantComponentCount: number;
  ignoredSmallComponentCount: number;
  connectivityPassed: boolean;
  qualifyingCorners: DesignCorner[];
}

type BinaryMatrixLike = Pick<ReconstructedMatrix, 'id' | 'rows' | 'cols' | 'data'>;

/**
 * Validate a thresholded binary design.
 *
 * A design passes when no more than one orthogonally connected component is
 * larger than a 2x2 bounding box and at least two distinct, edge-anchored
 * corner 2x2 blocks are all active. Small disconnected islands remain in the
 * matrix but are ignored for the connectivity verdict.
 */
export function validateBinaryDesign(matrix: BinaryMatrixLike): DesignValidationResult {
  const { id, rows, cols, data } = matrix;
  const cellCount = rows * cols;
  let binaryValuesOnly = data.length === cellCount;
  let activeCellCount = 0;

  for (let index = 0; index < data.length; index++) {
    const value = data[index];
    if (value !== 0 && value !== 1) binaryValuesOnly = false;
    if (value === 1) activeCellCount++;
  }

  const visited = new Uint8Array(cellCount);
  const queue = new Int32Array(cellCount);
  const components: Array<{ minRow: number; maxRow: number; minCol: number; maxCol: number }> = [];

  if (binaryValuesOnly) {
    for (let start = 0; start < cellCount; start++) {
      if (data[start] !== 1 || visited[start]) continue;

      let head = 0;
      let tail = 0;
      queue[tail++] = start;
      visited[start] = 1;
      let minRow = rows;
      let maxRow = -1;
      let minCol = cols;
      let maxCol = -1;

      const visit = (index: number): void => {
        if (!visited[index] && data[index] === 1) {
          visited[index] = 1;
          queue[tail++] = index;
        }
      };

      while (head < tail) {
        const index = queue[head++];
        const row = Math.floor(index / cols);
        const col = index % cols;
        minRow = Math.min(minRow, row);
        maxRow = Math.max(maxRow, row);
        minCol = Math.min(minCol, col);
        maxCol = Math.max(maxCol, col);

        if (row > 0) visit(index - cols);
        if (row + 1 < rows) visit(index + cols);
        if (col > 0) visit(index - 1);
        if (col + 1 < cols) visit(index + 1);
      }

      components.push({ minRow, maxRow, minCol, maxCol });
    }
  }

  const ignoredSmallComponentCount = components.filter(component =>
    component.maxRow - component.minRow + 1 <= 2 &&
    component.maxCol - component.minCol + 1 <= 2,
  ).length;
  const significantComponentCount = components.length - ignoredSmallComponentCount;

  const qualifyingCorners: DesignCorner[] = [];
  if (binaryValuesOnly && rows >= 2 && cols >= 2) {
    const candidates: Array<{ corner: DesignCorner; row: number; col: number }> = [
      { corner: 'top_left', row: 0, col: 0 },
      { corner: 'top_right', row: 0, col: cols - 2 },
      { corner: 'bottom_left', row: rows - 2, col: 0 },
      { corner: 'bottom_right', row: rows - 2, col: cols - 2 },
    ];
    const distinctBlocks = new Set<string>();

    for (const candidate of candidates) {
      const blockKey = `${candidate.row}:${candidate.col}`;
      if (distinctBlocks.has(blockKey)) continue;

      const topLeft = candidate.row * cols + candidate.col;
      if (
        data[topLeft] === 1 &&
        data[topLeft + 1] === 1 &&
        data[topLeft + cols] === 1 &&
        data[topLeft + cols + 1] === 1
      ) {
        distinctBlocks.add(blockKey);
        qualifyingCorners.push(candidate.corner);
      }
    }
  }

  const connectivityPassed = binaryValuesOnly && significantComponentCount <= 1;
  const validation: 0 | 1 = connectivityPassed && qualifyingCorners.length >= 2 ? 1 : 0;

  return {
    matrixId: id,
    validation,
    binaryValuesOnly,
    activeCellCount,
    componentCount: components.length,
    significantComponentCount,
    ignoredSmallComponentCount,
    connectivityPassed,
    qualifyingCorners,
  };
}
