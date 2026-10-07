/**
 * Matrix Latent Autoencoder Studio - Data Types
 */

export type MatrixValueType = 'continuous_numeric' | 'binary_01';

export interface MatrixItem {
  id: string;
  name: string;
  rows: number;
  cols: number;
  data: Float32Array; // Flattened rows * cols (any real numerical values or binary 0/1)
  minVal: number;
  maxVal: number;
  isBinary: boolean;
  nonZeroCount: number;
  sparsityPercent: number;
  latent?: number[]; // 8 latent dimensions [z0, ..., z7]
  originalSequence?: string; // If imported from FASTA/amino acids
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

export type DiscretizationMode =
  | 'threshold'   // p >= threshold ? 1 : 0 (Binary mode)
  | 'argmax'      // 1 at max column per row (categorical/one-hot)
  | 'continuous'; // raw probabilities [0.0 - 1.0]

export interface LatentRow {
  id: string;
  z: number[]; // 8 dimensions
  detectedRows?: number;
  detectedCols?: number;
  detectedMinVal?: number;
  detectedMaxVal?: number;
  detectedValueType?: MatrixValueType;
}

export interface ReconstructedMatrix {
  id: string;
  rows: number;
  cols: number;
  data: Float32Array; // Reconstructed numerical values (real numbers or binary 0/1)
  probabilities?: Float32Array; // Sigmoid outputs [0, 1] if in binary/prob mode
  valueType: MatrixValueType;
  minVal: number;
  maxVal: number;
  nonZeroCount: number;
  sparsityPercent: number;
  decodedSequence?: string;
  groundTruth?: Float32Array;
  metrics?: {
    accuracy?: number;
    precision?: number;
    recall?: number;
    f1?: number;
    mae?: number;
    rmse?: number;
  };
}

export interface AutoencoderConfig {
  inputRows: number;   // default 200
  inputCols: number;   // default 21
  inputDim: number;    // rows * cols (default 4200)
  latentDim: 8;        // fixed 8
  hidden1: number;     // 512
  hidden2: number;     // 128
}

export interface DecoderOptions {
  matrixValueType: MatrixValueType; // 'continuous_numeric' | 'binary_01'
  threshold: number;                // for binary_01 mode (default 0.5)
  discretizationMode: DiscretizationMode;
  continuousRangeMode: 'raw' | 'custom_range' | 'positive_only';
  customMin: number;
  customMax: number;
  sparsityCutoff: number;           // zero out values if |val| < cutoff (0 = no cutoff)
  decimalPrecision: number;         // 0 to 6
}
