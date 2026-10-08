/**
 * Matrix format parsers and serialization helpers
 */
import { MatrixItem, ReconstructedMatrix, LatentRow } from './types';

// Standard 20 amino acids + gap character for 200x21 matrices
export const AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-'];
export const AMINO_ACID_MAP = new Map<string, number>(AMINO_ACIDS.map((aa, idx) => [aa, idx]));

/**
 * Convert a sequence string (e.g. "MKWVT...") to a 200x21 one-hot Float32Array
 */
export function sequenceToOneHot(seq: string, targetLength: number = 200): Float32Array {
  const clean = seq.trim().toUpperCase().replace(/[^A-Z\-]/g, '');
  const data = new Float32Array(targetLength * 21);
  
  for (let i = 0; i < targetLength; i++) {
    const char = i < clean.length ? clean[i] : '-';
    const colIdx = AMINO_ACID_MAP.has(char) ? AMINO_ACID_MAP.get(char)! : 20; // Default to gap '-'
    data[i * 21 + colIdx] = 1.0;
  }
  return data;
}

/**
 * Decode a 200x21 one-hot or probability matrix back to sequence string
 */
export function oneHotToSequence(data: Float32Array, rows: number = 200, cols: number = 21): string {
  let seq = '';
  for (let r = 0; r < rows; r++) {
    let bestCol = -1;
    let maxVal = -1;
    for (let c = 0; c < cols; c++) {
      const val = data[r * cols + c];
      if (val > maxVal) {
        maxVal = val;
        bestCol = c;
      }
    }
    if (bestCol >= 0 && maxVal > 0) {
      seq += AMINO_ACIDS[bestCol] || '?';
    } else {
      seq += '-';
    }
  }
  return seq;
}

/**
 * Parse text from FASTA or raw sequence lines
 */
export function parseFastaOrSequences(text: string, targetLength: number = 200): MatrixItem[] {
  const lines = text.split(/\r?\n/).map(l => l.trim()).filter(l => l.length > 0);
  const items: MatrixItem[] = [];
  
  let currentHeader = '';
  let currentSeq = '';
  let count = 0;

  for (const line of lines) {
    if (line.startsWith('>')) {
      if (currentSeq.length > 0) {
        count++;
        const id = currentHeader || `seq_${count}`;
        const data = sequenceToOneHot(currentSeq, targetLength);
        items.push(createMatrixItem(id, id, targetLength, 21, data, currentSeq));
        currentSeq = '';
      }
      currentHeader = line.substring(1).trim().split(/\s+/)[0] || `seq_${count + 1}`;
    } else {
      // Could be raw sequence lines
      if (!currentHeader) {
        count++;
        currentHeader = `seq_${count}`;
      }
      currentSeq += line;
    }
  }

  if (currentSeq.length > 0) {
    count++;
    const id = currentHeader || `seq_${count}`;
    const data = sequenceToOneHot(currentSeq, targetLength);
    items.push(createMatrixItem(id, id, targetLength, 21, data, currentSeq));
  }

  return items;
}

/**
 * Parse a dense 2D CSV/TSV table into a single matrix
 */
export function parseDense2DTable(text: string, id: string = 'matrix_1', name: string = 'Matrix 1'): MatrixItem | null {
  const lines = text.split(/\r?\n/).map(l => l.trim()).filter(l => l.length > 0);
  if (lines.length === 0) return null;

  // Detect delimiter
  const firstLine = lines[0];
  const delim = firstLine.includes('\t') ? '\t' : (firstLine.includes(';') ? ';' : ',');
  
  // Check if first line is a header (contains non-numeric characters besides delimiters and signs)
  const tokens0 = firstLine.split(delim).map(t => t.trim());
  const isHeader = tokens0.some(t => isNaN(Number(t)) && t !== '');
  const dataLines = isHeader ? lines.slice(1) : lines;

  if (dataLines.length === 0) return null;

  const rows = dataLines.length;
  const cols = dataLines[0].split(delim).map(t => t.trim()).filter(t => t.length > 0).length;

  const data = new Float32Array(rows * cols);
  let ptr = 0;

  for (let r = 0; r < rows; r++) {
    const cells = dataLines[r].split(delim);
    for (let c = 0; c < cols; c++) {
      const val = parseFloat(cells[c]);
      data[ptr++] = isNaN(val) ? 0 : val;
    }
  }

  return createMatrixItem(id, name, rows, cols, data);
}

/**
 * Parse COO (Coordinate Format) CSV:
 * Can have columns: row_index, col_index, value
 * Or: matrix_id, row_index, col_index, value
 */
export function parseSparseCoo(
  text: string, 
  targetRows: number = 200, 
  targetCols: number = 21
): MatrixItem[] {
  const lines = text.split(/\r?\n/).map(l => l.trim()).filter(l => l.length > 0);
  if (lines.length === 0) return [];

  const delim = lines[0].includes('\t') ? '\t' : ',';
  let hasIdCol = false;
  let startIdx = 0;

  // Check header
  const header = lines[0].toLowerCase();
  if (header.includes('row') || header.includes('col') || header.includes('matrix')) {
    startIdx = 1;
    if (header.includes('matrix') || header.includes('id')) {
      hasIdCol = true;
    }
  } else {
    // Check first data row token count
    const parts = lines[0].split(delim);
    if (parts.length >= 4) {
      hasIdCol = true;
    }
  }

  const grouped = new Map<string, { r: number; c: number; v: number }[]>();

  for (let i = startIdx; i < lines.length; i++) {
    const parts = lines[i].split(delim).map(p => p.trim());
    if (parts.length < (hasIdCol ? 4 : 3)) continue;

    let mId = 'matrix_1';
    let rIdx = 0;
    let cIdx = 0;
    let val = 1.0;

    if (hasIdCol) {
      mId = parts[0] || 'matrix_1';
      rIdx = parseInt(parts[1], 10);
      cIdx = parseInt(parts[2], 10);
      val = parseFloat(parts[3]);
    } else {
      rIdx = parseInt(parts[0], 10);
      cIdx = parseInt(parts[1], 10);
      val = parseFloat(parts[2]);
    }

    if (isNaN(rIdx) || isNaN(cIdx)) continue;
    if (isNaN(val)) val = 1.0;

    if (!grouped.has(mId)) {
      grouped.set(mId, []);
    }
    grouped.get(mId)!.push({ r: rIdx, c: cIdx, v: val });
  }

  const result: MatrixItem[] = [];
  grouped.forEach((entries, mId) => {
    // Find max rows/cols if target not strictly met
    let maxR = targetRows;
    let maxC = targetCols;
    for (const e of entries) {
      if (e.r >= maxR) maxR = e.r + 1;
      if (e.c >= maxC) maxC = e.c + 1;
    }

    const data = new Float32Array(maxR * maxC);
    for (const e of entries) {
      if (e.r >= 0 && e.r < maxR && e.c >= 0 && e.c < maxC) {
        data[e.r * maxC + e.c] = e.v;
      }
    }
    result.push(createMatrixItem(mId, mId, maxR, maxC, data));
  });

  return result;
}

/**
 * Parse flattened CSV where each row is 1 complete matrix (e.g. 4200 values per row)
 */
export function parseFlattenedCsv(
  text: string, 
  rows: number = 200, 
  cols: number = 21
): MatrixItem[] {
  const lines = text.split(/\r?\n/).map(l => l.trim()).filter(l => l.length > 0);
  const result: MatrixItem[] = [];
  const expectedDim = rows * cols;

  let rowCount = 0;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.startsWith('#')) continue; // Skip comments

    const delim = line.includes('\t') ? '\t' : ',';
    const tokens = line.split(delim).map(t => t.trim());

    // Check if first token is an ID string (e.g., "matrix_1, 0, 1, 0...")
    let id = `matrix_${++rowCount}`;
    let valuesTokens = tokens;
    if (isNaN(Number(tokens[0])) && tokens[0] !== '') {
      id = tokens[0];
      valuesTokens = tokens.slice(1);
    }

    // Must have at least a substantial amount of numbers
    if (valuesTokens.length === 0) continue;

    const data = new Float32Array(expectedDim);
    for (let k = 0; k < expectedDim && k < valuesTokens.length; k++) {
      const val = parseFloat(valuesTokens[k]);
      data[k] = isNaN(val) ? 0 : val;
    }

    result.push(createMatrixItem(id, id, rows, cols, data));
  }

  return result;
}

/**
 * Universal auto-detection parser for multiple formats
 */
export function parseMatrixPayload(
  text: string, 
  fileName?: string,
  targetRows: number = 200, 
  targetCols: number = 21,
  forcedFormat?: string
): MatrixItem[] {
  const trimmed = text.trim();
  if (!trimmed) return [];

  // Forced format checks
  if (forcedFormat === 'fasta_sequence') {
    return parseFastaOrSequences(trimmed, targetRows);
  }
  if (forcedFormat === 'sparse_coo') {
    return parseSparseCoo(trimmed, targetRows, targetCols);
  }
  if (forcedFormat === 'flattened_csv') {
    return parseFlattenedCsv(trimmed, targetRows, targetCols);
  }
  if (forcedFormat === 'dense_csv') {
    const item = parseDense2DTable(trimmed, fileName || 'matrix_1', fileName || 'Matrix 1');
    return item ? [item] : [];
  }

  // Auto-detection:
  // 1. JSON
  if (trimmed.startsWith('[') || trimmed.startsWith('{')) {
    try {
      const parsed = JSON.parse(trimmed);
      if (Array.isArray(parsed)) {
        if (parsed.length > 0 && Array.isArray(parsed[0]) && Array.isArray(parsed[0][0])) {
          // Array of 2D matrices: [ [[...], [...]], ... ]
          return parsed.map((m: number[][], idx: number) => {
            const r = m.length;
            const c = m[0]?.length || targetCols;
            const data = new Float32Array(r * c);
            for (let i = 0; i < r; i++) {
              for (let j = 0; j < c; j++) {
                data[i * c + j] = m[i][j] || 0;
              }
            }
            return createMatrixItem(`matrix_${idx + 1}`, `Matrix ${idx + 1}`, r, c, data);
          });
        }
        if (parsed.length > 0 && Array.isArray(parsed[0]) && typeof parsed[0][0] === 'number') {
          // Single 2D matrix
          const r = parsed.length;
          const c = parsed[0].length;
          const data = new Float32Array(r * c);
          for (let i = 0; i < r; i++) {
            for (let j = 0; j < c; j++) {
              data[i * c + j] = parsed[i][j] || 0;
            }
          }
          return [createMatrixItem(fileName || 'matrix_1', fileName || 'Matrix 1', r, c, data)];
        }
      }
    } catch {
      // not JSON, continue
    }
  }

  // 2. FASTA or sequence characters
  if (trimmed.startsWith('>') || /^[ACDEFGHIKLMNPQRSTVWY\-\s\r\n]+$/i.test(trimmed)) {
    return parseFastaOrSequences(trimmed, targetRows);
  }

  // 3. Sparse COO check (keywords row, col, value or 3-4 comma-separated numbers)
  const firstLines = trimmed.split(/\r?\n/).slice(0, 5).join(' ').toLowerCase();
  if (firstLines.includes('row') && firstLines.includes('col')) {
    return parseSparseCoo(trimmed, targetRows, targetCols);
  }

  // Check token counts of lines
  const lines = trimmed.split(/\r?\n/).filter(l => l.trim().length > 0);
  if (lines.length > 0) {
    const delim = lines[0].includes('\t') ? '\t' : ',';
    const firstLineParts = lines[0].split(delim);
    
    // If lines have 3 or 4 columns and there are many lines, likely COO
    if ((firstLineParts.length === 3 || firstLineParts.length === 4) && lines.length > targetRows) {
      return parseSparseCoo(trimmed, targetRows, targetCols);
    }

    // If first line has thousands of columns (e.g. 4200), it's flattened
    if (firstLineParts.length >= 500) {
      return parseFlattenedCsv(trimmed, targetRows, targetCols);
    }

    // Otherwise, parse as 2D dense table
    const dense = parseDense2DTable(trimmed, fileName || 'matrix_1', fileName || 'Matrix 1');
    if (dense) return [dense];
  }

  return [];
}

/**
 * Parse CSV file with 8 latent dimensions per row
 * Format:
 * [# dimensions: RxC]
 * [id,] z0, z1, z2, z3, z4, z5, z6, z7
 */
export function parseLatentCsv(text: string): LatentRow[] {
  const lines = text.split(/\r?\n/).map(l => l.trim()).filter(l => l.length > 0);
  const rows: LatentRow[] = [];

  let detectedRows: number | undefined;
  let detectedCols: number | undefined;
  let detectedK: number | undefined;
  let detectedValueType: 'binary_01' | 'continuous_numeric' | undefined;

  // Parse comment-line metadata
  for (const line of lines) {
    if (line.startsWith('#')) {
      const dimMatch = line.match(/(?:dimensions|shape|size):\s*(\d+)\s*[xX,]\s*(\d+)/i);
      if (dimMatch) {
        detectedRows = parseInt(dimMatch[1], 10);
        detectedCols = parseInt(dimMatch[2], 10);
      }
      const kMatch = line.match(/k:\s*(\d+)/i);
      if (kMatch) detectedK = parseInt(kMatch[1], 10);
      const vtMatch = line.match(/value_type:\s*(binary_01|continuous_numeric)/i);
      if (vtMatch) detectedValueType = vtMatch[1] as 'binary_01' | 'continuous_numeric';
    }
  }

  let count = 0;
  for (const line of lines) {
    if (line.startsWith('#')) continue;
    const delim = line.includes('\t') ? '\t' : ',';
    const tokens = line.split(delim).map(t => t.trim());

    // Skip header lines — a true header has ALL tokens non-numeric
    // (e.g. "matrix_id,z1,z2,..."). A data row always has numeric z values.
    if (tokens.every(t => isNaN(Number(t)))) continue;

    // Determine which tokens are numbers
    const numericStart = isNaN(Number(tokens[0])) ? 1 : 0;
    let id = numericStart === 1 ? tokens[0] : `latent_${++count}`;
    if (numericStart === 0) count++;

    const numbers = tokens.slice(numericStart).map(Number).filter(n => !isNaN(n));

    // Accept any K >= 2 (not just exactly 8)
    if (numbers.length >= 2) {
      rows.push({
        id,
        z: numbers,
        detectedRows,
        detectedCols,
        detectedK: numbers.length,
        detectedValueType,
      });
    }
  }

  return rows;
}

/**
 * Format latent vectors into CSV string ready for download
 */
export function formatLatentCsv(
  items: { id: string; latent: number[] }[],
  includeIdHeader = true,
  metadata?: { rows: number; cols: number; k?: number; valueType?: 'binary_01' | 'continuous_numeric' }
): string {
  const K = items[0]?.latent?.length ?? 8;
  let header = '# Matrix PCA Latent Vectors — normalized [0,1] per component\n';
  if (metadata) {
    header += `# dimensions: ${metadata.rows}x${metadata.cols}\n`;
    header += `# k: ${metadata.k ?? K}\n`;
    if (metadata.valueType) header += `# value_type: ${metadata.valueType}\n`;
  }
  header += '# z values are normalized to [0,1] — use the pca_basis.json for decoding\n';
  const colNames = Array.from({ length: K }, (_, i) => `z${i + 1}`).join(',');
  header += includeIdHeader ? `matrix_id,${colNames}\n` : `${colNames}\n`;

  const rows = items.map(item => {
    const zFormatted = item.latent.map(v => {
      const rounded = Number(v.toFixed(8));
      return isNaN(rounded) ? '0' : rounded.toString();
    }).join(',');
    return includeIdHeader ? `${item.id},${zFormatted}` : zFormatted;
  });
  return header + rows.join('\n');
}

/**
 * Format reconstructed matrices as Sparse COO CSV
 */
export function formatSparseCooCsv(matrices: ReconstructedMatrix[], decimalPrecision: number = 4): string {
  let csv = 'matrix_id,row_index,col_index,value\n';
  for (const m of matrices) {
    const { id, rows, cols, data, valueType } = m;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const idx = r * cols + c;
        const v = data[idx];
        if (v !== 0) {
          const valStr = valueType === 'binary_01' 
            ? '1' 
            : (decimalPrecision < 0 ? v.toString() : Number(v.toFixed(decimalPrecision)).toString());
          csv += `${id},${r},${c},${valStr}\n`;
        }
      }
    }
  }
  return csv;
}

/**
 * Format reconstructed matrix as a single 2D Dense CSV table
 */
export function formatSingleDenseCsv(m: ReconstructedMatrix, decimalPrecision: number | boolean = 4): string {
  const { rows, cols, data, valueType } = m;
  const prec = typeof decimalPrecision === 'number' ? decimalPrecision : 4;
  const lines: string[] = [];
  
  // Header with amino acids or column index
  const header = cols === 21 
    ? AMINO_ACIDS.join(',') 
    : Array.from({ length: cols }, (_, i) => `c${i}`).join(',');
  lines.push(header);

  for (let r = 0; r < rows; r++) {
    const rowVals: string[] = [];
    for (let c = 0; c < cols; c++) {
      const idx = r * cols + c;
      const v = data[idx];
      if (valueType === 'binary_01') {
        rowVals.push(v > 0 ? '1' : '0');
      } else {
        rowVals.push(prec < 0 ? v.toString() : Number(v.toFixed(prec)).toString());
      }
    }
    lines.push(rowVals.join(','));
  }

  return lines.join('\n');
}

/**
 * Format multiple reconstructed matrices as a concatenated dense CSV
 */
export function formatCombinedDenseCsv(matrices: ReconstructedMatrix[], decimalPrecision: number | boolean = 4): string {
  const prec = typeof decimalPrecision === 'number' ? decimalPrecision : 4;
  return matrices.map(m => {
    return `# MATRIX: ${m.id} (Rows: ${m.rows}, Cols: ${m.cols}, Active: ${m.nonZeroCount})\n` + 
           formatSingleDenseCsv(m, prec);
  }).join('\n\n');
}

/**
 * Format reconstructed matrices as FASTA sequences
 */
export function formatFastaSequences(matrices: ReconstructedMatrix[]): string {
  return matrices.map(m => {
    const seq = oneHotToSequence(m.data, m.rows, m.cols);
    return `>${m.id} length=${m.rows} active_positions=${m.nonZeroCount}\n${seq}`;
  }).join('\n');
}

/**
 * Format reconstructed matrices as Flattened CSV (1 row per matrix)
 */
export function formatFlattenedCsv(matrices: ReconstructedMatrix[], decimalPrecision: number = 4): string {
  let csv = 'matrix_id,' + Array.from({ length: matrices[0]?.rows * matrices[0]?.cols || 4200 }, (_, i) => `f${i}`).join(',') + '\n';
  for (const m of matrices) {
    const vals = Array.from(m.data).map(v => {
      if (m.valueType === 'binary_01') return v > 0 ? '1' : '0';
      return decimalPrecision < 0 ? v.toString() : Number(v.toFixed(decimalPrecision)).toString();
    }).join(',');
    csv += `${m.id},${vals}\n`;
  }
  return csv;
}

/**
 * Helper to build MatrixItem
 */
export function createMatrixItem(
  id: string, 
  name: string, 
  rows: number, 
  cols: number, 
  data: Float32Array,
  originalSequence?: string
): MatrixItem {
  let nonZeroCount = 0;
  let minVal = Infinity;
  let maxVal = -Infinity;
  let isBinary = true;

  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (v !== 0) nonZeroCount++;
    if (v < minVal) minVal = v;
    if (v > maxVal) maxVal = v;
    if (v !== 0 && v !== 1) isBinary = false;
  }
  if (minVal === Infinity) {
    minVal = 0;
    maxVal = 0;
  }
  const total = rows * cols;
  const sparsityPercent = total > 0 ? ((total - nonZeroCount) / total) * 100 : 0;

  return {
    id,
    name,
    rows,
    cols,
    data,
    minVal,
    maxVal,
    isBinary,
    nonZeroCount,
    sparsityPercent,
    originalSequence
  };
}
