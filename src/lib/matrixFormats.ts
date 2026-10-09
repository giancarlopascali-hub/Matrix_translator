/**
 * Matrix format parsers and serialization helpers
 */
import { MatrixItem, ReconstructedMatrix, LatentRow } from './types';

// Standard 20 amino acids + gap character for 200x21 matrices
export const AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-'];
export const AMINO_ACID_MAP = new Map<string, number>(AMINO_ACIDS.map((aa, idx) => [aa, idx]));

function formatLatentCoordinate(value: number): string {
  if (!Number.isFinite(value)) return '0';

  // PCA arithmetic can leave values such as 4e-28 or
  // 0.49999999999999994. Snap only numerical noise around the three common
  // reference points, then use a readable non-scientific decimal format.
  const anchors = [0, 0.5, 1];
  let cleanValue = Math.max(0, Math.min(1, value));
  for (const anchor of anchors) {
    if (Math.abs(cleanValue - anchor) <= 1e-10) {
      cleanValue = anchor;
      break;
    }
  }
  const fixed = cleanValue.toFixed(10).replace(/0+$/, '').replace(/\.$/, '');
  return fixed === '' || fixed === '-0' ? '0' : fixed;
}

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
  const lines = text.split(/\r?\n/).map(l => l.trim()).filter(l => l.length > 0 && !l.startsWith('#') && !l.startsWith('//'));
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
 * Parse one or multiple dense 2D matrices from a single CSV/TSV payload.
 * Supports:
 *  1. Matrices separated by blank lines
 *  2. Matrices separated by comment/ID headers (e.g. "# Matrix 1" or ">mat1")
 *  3. Vertically stacked matrices where total lines is a multiple of targetRows (e.g. 90 lines = 2 x 45x45 matrices)
 *  4. Single standard 2D matrix
 */
export function parseDense2DTables(
  text: string, 
  baseId: string = 'matrix',
  targetRows = 45,
  targetCols = 45
): MatrixItem[] {
  const trimmed = text.trim();
  if (!trimmed) return [];

  // 1. Check for blank-line separation between blocks
  const rawBlocks = trimmed.split(/\r?\n\s*\r?\n+/).map(b => b.trim()).filter(b => b.length > 0);
  if (rawBlocks.length > 1) {
    const items: MatrixItem[] = [];
    for (let i = 0; i < rawBlocks.length; i++) {
      const bText = rawBlocks[i];
      const firstLine = bText.split(/\r?\n/)[0]?.trim() || '';
      let id = `${baseId}_${i + 1}`;
      if (firstLine.startsWith('#') || firstLine.startsWith('>')) {
        const cleaned = firstLine.replace(/^[#>/\-\s]+/, '').trim();
        if (cleaned) id = cleaned;
      }
      const item = parseDense2DTable(bText, id, id);
      if (item) items.push(item);
    }
    if (items.length > 0) return items;
  }

  // 2. Check for comments / header lines like "# Matrix 1" or ">mat1"
  const lines = trimmed.split(/\r?\n/).map(l => l.trim()).filter(l => l.length > 0);
  const headerIndices: number[] = [];
  for (let i = 0; i < lines.length; i++) {
    const l = lines[i];
    if (l.startsWith('#') || l.startsWith('>') || l.startsWith('//') || l.startsWith('---')) {
      headerIndices.push(i);
    }
  }
  if (headerIndices.length > 1) {
    const items: MatrixItem[] = [];
    for (let h = 0; h < headerIndices.length; h++) {
      const start = headerIndices[h] + 1;
      const end = h + 1 < headerIndices.length ? headerIndices[h + 1] : lines.length;
      const subText = lines.slice(start, end).join('\n');
      const headerText = lines[headerIndices[h]].replace(/^[#>/\-\s]+/, '').trim();
      const id = headerText || `${baseId}_${h + 1}`;
      const item = parseDense2DTable(subText, id, id);
      if (item) items.push(item);
    }
    if (items.length > 0) return items;
  }

  // 3. Single block — check if stacked multiple of targetRows
  const single = parseDense2DTable(trimmed, baseId, baseId);
  if (single) {
    if (
      targetRows > 0 && 
      targetCols > 0 && 
      single.cols === targetCols && 
      single.rows > targetRows && 
      single.rows % targetRows === 0
    ) {
      const numMatrices = single.rows / targetRows;
      const items: MatrixItem[] = [];
      const cellsPerMat = targetRows * targetCols;
      for (let m = 0; m < numMatrices; m++) {
        const subData = new Float32Array(cellsPerMat);
        subData.set(single.data.subarray(m * cellsPerMat, (m + 1) * cellsPerMat));
        const id = `${baseId}_${m + 1}`;
        items.push(createMatrixItem(id, id, targetRows, targetCols, subData));
      }
      return items;
    }
    return [single];
  }

  return [];
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
  const firstDataIndex = lines.findIndex(line => !line.startsWith('#'));
  const firstDataLine = firstDataIndex >= 0 ? lines[firstDataIndex] : undefined;
  if (!firstDataLine) return [];
  const initialDelimiter = firstDataLine.includes('\t') ? '\t' : ',';
  const initialTokens = firstDataLine.split(initialDelimiter).map(token => token.trim());
  const firstToken = initialTokens[0]?.toLowerCase();
  const hasHeader = firstToken === 'matrix_id' || firstToken === 'id' ||
    initialTokens.slice(1).some(token => /^f\d+$/i.test(token));
  const headerHasId = hasHeader && (firstToken === 'matrix_id' || firstToken === 'id');

  let rowCount = 0;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.startsWith('#')) continue; // Skip comments
    if (hasHeader && i === firstDataIndex) continue;

    const delim = line.includes('\t') ? '\t' : ',';
    const tokens = line.split(delim).map(t => t.trim());

    const rowHasId = headerHasId || tokens.length === expectedDim + 1;
    let id = `matrix_${rowCount + 1}`;
    let valuesTokens = tokens;
    if (rowHasId) {
      id = tokens[0];
      valuesTokens = tokens.slice(1);
    }

    // A malformed row must not be silently padded or truncated.
    if (!id || valuesTokens.length !== expectedDim) continue;
    const values = valuesTokens.map(Number);
    if (values.some(value => !Number.isFinite(value))) continue;

    const data = Float32Array.from(values);

    result.push(createMatrixItem(id, id, rows, cols, data));
    rowCount++;
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
    return parseDense2DTables(trimmed, fileName || 'matrix_1', targetRows, targetCols);
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

    // If first line has many columns (e.g. flattened 1D row per matrix)
    if (firstLineParts.length >= Math.min(targetRows * targetCols, 50)) {
      const flattened = parseFlattenedCsv(trimmed, targetRows, targetCols);
      if (flattened.length > 0) return flattened;
    }

    // Otherwise, parse as 2D dense table(s)
    const dense = parseDense2DTables(trimmed, fileName || 'matrix', targetRows, targetCols);
    if (dense.length > 0) return dense;
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
  let detectedModelId: string | undefined;
  let detectedSchemaVersion: number | undefined;

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
      const modelMatch = line.match(/model_id:\s*([^\s]+)/i);
      if (modelMatch) detectedModelId = modelMatch[1];
      const schemaMatch = line.match(/schema_version:\s*(\d+)/i);
      if (schemaMatch) detectedSchemaVersion = parseInt(schemaMatch[1], 10);
    }
  }

  let headerLine: string | undefined;
  let idColumn = -1;
  let latentColumns: number[] = [];
  for (const line of lines) {
    if (line.startsWith('#')) continue;
    const delim = line.includes('\t') ? '\t' : ',';
    const tokens = line.split(delim).map(token => token.trim());
    const candidates = tokens
      .map((token, index) => ({ token, index }))
      .filter(({ token }) => /^z\d+$/i.test(token));
    if (candidates.length > 0) {
      headerLine = line;
      idColumn = tokens.findIndex(token => /^(matrix_)?id$/i.test(token));
      latentColumns = candidates
        .sort((a, b) => parseInt(a.token.slice(1), 10) - parseInt(b.token.slice(1), 10))
        .map(candidate => candidate.index);
      break;
    }
  }

  let count = 0;
  for (const line of lines) {
    if (line.startsWith('#')) continue;
    if (line === headerLine) continue;
    const delim = line.includes('\t') ? '\t' : ',';
    const tokens = line.split(delim).map(t => t.trim());

    let id: string;
    let numbers: number[];
    if (latentColumns.length > 0) {
      id = idColumn >= 0 && tokens[idColumn] ? tokens[idColumn] : `latent_${count + 1}`;
      numbers = latentColumns.map(index => Number(tokens[index]));
    } else {
      // Legacy headerless files: a non-numeric first token is treated as an ID.
      const numericStart = isNaN(Number(tokens[0])) ? 1 : 0;
      id = numericStart === 1 ? tokens[0] : `latent_${count + 1}`;
      numbers = tokens.slice(numericStart).map(Number);
    }

    if (id && numbers.length >= 1 && numbers.every(Number.isFinite)) {
      rows.push({
        id,
        z: numbers,
        detectedRows,
        detectedCols,
        detectedK: detectedK ?? numbers.length,
        detectedValueType,
        detectedModelId,
        detectedSchemaVersion,
      });
      count++;
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
  metadata?: {
    rows: number;
    cols: number;
    k?: number;
    valueType?: 'binary_01' | 'continuous_numeric';
    modelId?: string;
    schemaVersion?: number;
    dataComponentCount?: number;
    syntheticAnchorUsed?: boolean;
  }
): string {
  const K = items[0]?.latent?.length ?? 8;
  let header = '# Matrix PCA Latent Vectors — normalized [0,1] per component\n';
  if (metadata) {
    header += `# dimensions: ${metadata.rows}x${metadata.cols}\n`;
    header += `# k: ${metadata.k ?? K}\n`;
    if (metadata.valueType) header += `# value_type: ${metadata.valueType}\n`;
    if (metadata.modelId) header += `# model_id: ${metadata.modelId}\n`;
    if (metadata.schemaVersion) header += `# schema_version: ${metadata.schemaVersion}\n`;
    if (metadata.dataComponentCount !== undefined) {
      const dataCount = Math.max(0, Math.min(K, Math.floor(metadata.dataComponentCount)));
      const dimensionNames = Array.from({ length: K }, (_, index) => `z${index + 1}`);
      const dataDimensions = dimensionNames.slice(0, dataCount);
      const completionDimensions = dimensionNames.slice(dataCount);
      const referenceLatent = items[0]?.latent ?? [];

      header += `# data_informed_count: ${dataCount}\n`;
      header += `# completion_count: ${K - dataCount}\n`;
      header += `# dimension_roles: ${dimensionNames.map((name, index) => `${name}=${index < dataCount ? 'data_informed' : 'completion'}`).join(',')}\n`;
      header += `# optimize_by_default: ${dataDimensions.length > 0 ? dataDimensions.join(',') : 'none'}\n`;
      header += `# completion_dimensions: ${completionDimensions.length > 0 ? completionDimensions.join(',') : 'none'}\n`;
      if (completionDimensions.length > 0) {
        const references = completionDimensions.map((name, offset) => {
          const index = dataCount + offset;
          return `${name}=${formatLatentCoordinate(referenceLatent[index] ?? 0.5)}`;
        });
        header += `# completion_reference_values: ${references.join(',')}\n`;
      }
      if (metadata.syntheticAnchorUsed) header += '# bootstrap_direction: z1\n';
    }
  }
  header += '# z values are normalized to [0,1] — use the pca_basis.json for decoding\n';
  const colNames = Array.from({ length: K }, (_, i) => `z${i + 1}`).join(',');
  header += includeIdHeader ? `matrix_id,${colNames}\n` : `${colNames}\n`;

  const rows = items.map(item => {
    const zFormatted = item.latent.map(formatLatentCoordinate).join(',');
    return includeIdHeader ? `${item.id},${zFormatted}` : zFormatted;
  });
  return header + rows.join('\n');
}

/**
 * Preserve a supplied latent/optimizer CSV and add one validation value per
 * parsed latent row. Existing non-z columns (for example objective values) are
 * retained. Headerless legacy files are converted to the canonical format.
 */
export function formatValidatedLatentCsv(
  sourceText: string,
  latentRows: LatentRow[],
  validations: ReadonlyArray<0 | 1>,
  threshold: number,
): string {
  if (latentRows.length !== validations.length) {
    throw new Error('Validation results do not match the latent CSV row count.');
  }
  if (latentRows.length === 0) {
    throw new Error('No latent rows are available for validation export.');
  }

  const validationMetadata = [
    '# validation_rule: design_validation_v1',
    '# validation_connectivity: orthogonal_4',
    '# validation_corner_requirement: at_least_2_distinct_edge_anchored_2x2',
    '# validation_values: 1=passed,0=failed',
    `# validation_threshold: ${formatLatentCoordinate(threshold)}`,
  ];
  const isValidationMetadata = (line: string): boolean =>
    /^#\s*validation_(?:rule|connectivity|corner_requirement|values|threshold):/i.test(line.trim());

  const lines = sourceText.split(/\r?\n/);
  let headerIndex = -1;
  let delimiter = ',';
  let headerTokens: string[] = [];
  let latentColumns: number[] = [];

  for (let index = 0; index < lines.length; index++) {
    const trimmed = lines[index].trim();
    if (!trimmed || trimmed.startsWith('#')) continue;
    const candidateDelimiter = lines[index].includes('\t') ? '\t' : ',';
    const tokens = lines[index].split(candidateDelimiter).map(token => token.trim());
    const candidates = tokens
      .map((token, tokenIndex) => ({ token, tokenIndex }))
      .filter(({ token }) => /^z\d+$/i.test(token));
    if (candidates.length > 0) {
      headerIndex = index;
      delimiter = candidateDelimiter;
      headerTokens = tokens;
      latentColumns = candidates.map(candidate => candidate.tokenIndex);
      break;
    }
  }

  if (headerIndex < 0) {
    const preservedComments = lines
      .map(line => line.trim())
      .filter(line => line.startsWith('#') && !isValidationMetadata(line));
    const k = latentRows[0].z.length;
    const canonicalHeader = ['matrix_id', ...Array.from({ length: k }, (_, index) => `z${index + 1}`), 'validation'];
    const canonicalRows = latentRows.map((row, index) => [
      row.id,
      ...row.z.map(formatLatentCoordinate),
      String(validations[index]),
    ].join(','));
    return [...preservedComments, ...validationMetadata, canonicalHeader.join(','), ...canonicalRows].join('\n');
  }

  const validationColumn = headerTokens.findIndex(token => token.toLowerCase() === 'validation');
  const output: string[] = [];
  let validationRowIndex = 0;

  for (let lineIndex = 0; lineIndex < lines.length; lineIndex++) {
    const line = lines[lineIndex];
    const trimmed = line.trim();
    if (isValidationMetadata(trimmed)) continue;

    if (lineIndex === headerIndex) {
      output.push(...validationMetadata);
      output.push(
        validationColumn >= 0
          ? headerTokens.join(delimiter)
          : [...headerTokens, 'validation'].join(delimiter),
      );
      continue;
    }

    if (!trimmed || trimmed.startsWith('#')) {
      output.push(line);
      continue;
    }

    const rawTokens = line.split(delimiter);
    const hasValidLatents = latentColumns.every(column =>
      column < rawTokens.length && Number.isFinite(Number(rawTokens[column].trim())),
    );
    if (!hasValidLatents) {
      output.push(line);
      continue;
    }
    if (validationRowIndex >= validations.length) {
      throw new Error('The source CSV contains more latent rows than the decoded collection.');
    }

    const value = String(validations[validationRowIndex++]);
    if (validationColumn >= 0) {
      while (rawTokens.length <= validationColumn) rawTokens.push('');
      rawTokens[validationColumn] = value;
    } else {
      rawTokens.push(value);
    }
    output.push(rawTokens.join(delimiter));
  }

  if (validationRowIndex !== validations.length) {
    throw new Error('The source CSV contains fewer latent rows than the decoded collection.');
  }
  return output.join('\n');
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

/**
 * Safely trigger a browser file download for a Blob
 * Works reliably across all browsers (Chrome, Edge, Firefox, Safari)
 */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.style.display = 'none';
  document.body.appendChild(link);
  link.click();
  setTimeout(() => {
    if (document.body.contains(link)) {
      document.body.removeChild(link);
    }
    URL.revokeObjectURL(url);
  }, 1000);
}

