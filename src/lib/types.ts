/**
 * Matrix PCA Studio - Data Types
 */

export type MatrixValueType = 'binary_01' | 'continuous_numeric';

export interface MatrixItem {
  id: string;
  name: string;
  rows: number;
  cols: number;
  data: Float32Array;
  minVal: number;
  maxVal: number;
  isBinary: boolean;
  nonZeroCount: number;
  sparsityPercent: number;
  latent?: number[];          // Normalized [0,1]^K latent coordinates
  originalSequence?: string;  // If imported from FASTA
}

export type InputFormatMode =
  | 'auto'
  | 'dense_csv'
  | 'sparse_coo'
  | 'fasta_sequence'
  | 'flattened_csv'
  | 'json';

export type OutputFormatMode =
  | 'dense_zip'
  | 'dense_combined_csv'
  | 'sparse_coo_csv'
  | 'fasta_sequence'
  | 'flattened_csv'
  | 'json';

export type DiscretizationMode = 'threshold' | 'argmax_categorical';

export interface DecoderOptions {
  threshold?: number;
  discretizationMode?: DiscretizationMode;
  valueType?: MatrixValueType;
  decimalPrecision?: number;
}

export interface LatentRow {
  id: string;
  z: number[];          // Normalized [0,1]^K latent coordinates
  detectedRows?: number;
  detectedCols?: number;
  detectedK?: number;
  detectedValueType?: MatrixValueType;
  detectedMinVal?: number;
  detectedMaxVal?: number;
  detectedModelId?: string;
  detectedSchemaVersion?: number;
}

/**
 * Exported PCA basis — everything needed to decode latent vectors
 * without re-fitting. Upload this JSON in Module 2 for standalone use.
 */
export interface PCABasis {
  schemaVersion: number;
  algorithm: 'pca-linear-v2' | string;
  modelId: string;
  valueType: MatrixValueType;
  rows: number;
  cols: number;
  k: number;
  mean: number[];             // D-dimensional mean vector
  components: number[][];     // K × D orthonormal basis vectors
  zMin: number[];             // K-dim raw projection lower bounds (for [0,1] normalisation)
  zMax: number[];             // K-dim raw projection upper bounds
  explainedVarianceRatios: number[];  // per-component explained variance fraction
  totalVariance: number;
  fittedCount: number;        // number of matrices used to fit
}

export interface ReconstructedMatrix {
  id: string;
  rows: number;
  cols: number;
  data: Float32Array;         // Final output (binary or continuous)
  continuous: Float32Array;   // Pre-threshold continuous reconstruction
  valueType: MatrixValueType;
  minVal: number;
  maxVal: number;
  nonZeroCount: number;
  sparsityPercent: number;
  groundTruth?: Float32Array;
  metrics?: {
    hammingAccuracy?: number;   // binary: fraction of cells correctly reconstructed
    hammingDistance?: number;   // binary: number of differing cells
    precision?: number;
    recall?: number;
    f1?: number;
    mae?: number;               // continuous: mean absolute error
    rmse?: number;              // continuous: root mean square error
  };
}

export interface FitResult {
  numComponents: number;
  requestedComponents?: number;
  effectiveRank?: number;
  explainedVarianceRatios: number[];   // per-component fractions, sum ≤ 1
  cumulativeVarianceRatios: number[];  // cumulative fractions
}
