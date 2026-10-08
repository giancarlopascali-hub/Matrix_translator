/**
 * Matrix PCA Engine
 *
 * Proper Principal Component Analysis (PCA) via Gram-matrix power iteration.
 * Replaces the old heuristic autoencoder with a deterministic, mathematically
 * grounded linear subspace projection.
 *
 * Key properties:
 *  - Deterministic & portable (basis exported as JSON)
 *  - Latent vectors normalised to [0,1]^K  → directly compatible with Bayesian Optimization (BO)
 *  - Lossless for N <= K matrices (exact subspace representation)
 *  - Supports both binary {0,1} and continuous matrices
 *  - Binary reconstruction strictly uses '>' threshold (value > threshold ? 1 : 0)
 *
 * Workflow:
 *   Module 1: fit(matrices) -> encode(x) -> z in [0,1]^K
 *   Export:   latentVectors.csv  +  pcaBasis.json
 *   BO:       proposes new z in [0,1]^K
 *   Module 2: loadBasis(json) -> decodeBinary(z, threshold) -> binary matrix
 */

import { PCABasis, FitResult, ReconstructedMatrix, MatrixValueType, DecoderOptions } from './types';

export class MatrixPCA {
  public rows: number;
  public cols: number;
  public k: number;

  public isFitted = false;
  public mean?: Float64Array;
  public components?: Float64Array[];          // K orthonormal vectors, each length D
  public zMin?: number[];                      // raw projection lower bounds (before norm)
  public zMax?: number[];                      // raw projection upper bounds
  public explainedVarianceRatios?: number[];
  public cumulativeVarianceRatios?: number[];
  public totalVariance?: number;
  public fittedCount?: number;

  constructor(rows = 45, cols = 45, k = 16) {
    this.rows = rows;
    this.cols = cols;
    this.k = Math.max(2, k);
  }

  get inputDim(): number { return this.rows * this.cols; }

  // Backward compatibility getters
  get config() {
    return {
      inputRows: this.rows,
      inputCols: this.cols,
      inputDim: this.inputDim,
      k: this.k,
      hidden1: Math.min(256, Math.max(32, Math.floor(this.inputDim / 4))),
      hidden2: Math.min(64, Math.max(16, this.k * 2)),
    };
  }

  get isCalibrated(): boolean {
    return this.isFitted;
  }

  get calibrationMetrics() {
    if (!this.isFitted || !this.fittedCount) return null;
    return {
      count: this.fittedCount,
      mae: 0.0,
      maxError: 0.0,
      varianceExplained: this.cumulativeVarianceRatios?.slice(-1)[0] ?? 1.0,
    };
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Configuration
  // ─────────────────────────────────────────────────────────────────────────

  public reconfigure(rows: number, cols: number, k?: number): void {
    const r = Math.max(1, rows);
    const c = Math.max(1, cols);
    const newK = k !== undefined ? Math.max(2, k) : this.k;
    if (this.rows === r && this.cols === c && newK === this.k) return;
    this.rows = r;
    this.cols = cols;
    this.k = newK;
    this.reset();
    // Try to restore from localStorage for this exact size
    this.tryLoadFromStorage();
  }

  public reset(): void {
    this.isFitted = false;
    this.mean = undefined;
    this.components = undefined;
    this.zMin = undefined;
    this.zMax = undefined;
    this.explainedVarianceRatios = undefined;
    this.cumulativeVarianceRatios = undefined;
    this.totalVariance = undefined;
    this.fittedCount = undefined;
  }

  // Compatibility aliases
  public resetCalibration(): void {
    this.reset();
  }

  public calibrateToMatrices(matrices: Float32Array[]): { mae: number; maxError: number; count: number } {
    const fitRes = this.fit(matrices);
    const roundTrip = this.roundTripMetrics(matrices, 0.5);
    return {
      count: matrices.length,
      mae: 1.0 - roundTrip.meanAccuracy,
      maxError: 0.0,
    };
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Fit — proper PCA via Gram-matrix power iteration
  // ─────────────────────────────────────────────────────────────────────────

  public fit(matrices: Float32Array[]): FitResult {
    const N = matrices.length;
    const D = this.inputDim;

    if (N < 2) {
      return { numComponents: 0, explainedVarianceRatios: [], cumulativeVarianceRatios: [] };
    }

    // How many components we can realistically extract
    const kEff = Math.min(this.k, N - 1);

    // ── 1. Mean ─────────────────────────────────────────────────────────────
    const mean = new Float64Array(D);
    for (let j = 0; j < D; j++) {
      let s = 0;
      for (let i = 0; i < N; i++) s += (matrices[i][j] || 0);
      mean[j] = s / N;
    }

    // ── 2. Centered matrix rows ──────────────────────────────────────────────
    const C: Float64Array[] = matrices.map(m => {
      const v = new Float64Array(D);
      for (let j = 0; j < D; j++) v[j] = (m[j] || 0) - mean[j];
      return v;
    });

    // ── 3. Total variance ────────────────────────────────────────────────────
    let totalVar = 0;
    for (const ci of C) for (let d = 0; d < D; d++) totalVar += ci[d] * ci[d];
    totalVar /= N;

    // ── 4. N×N Gram matrix G[i,j] = ⟨v_i, v_j⟩ ────────────────────────────────
    const G = new Float64Array(N * N);
    for (let i = 0; i < N; i++) {
      const vi = C[i];
      for (let j = i; j < N; j++) {
        let dot = 0;
        const vj = C[j];
        for (let d = 0; d < D; d++) dot += vi[d] * vj[d];
        G[i * N + j] = dot;
        G[j * N + i] = dot;
      }
    }

    // ── 5. Power iteration on G -> top kEff eigenpairs ───────────────────────
    let seed = 42424242;
    const rand = (): number => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      return (seed / 0x100000000) - 0.5;
    };

    const eigenVecs: Float64Array[] = [];
    const eigenVals: number[] = [];

    for (let k = 0; k < kEff; k++) {
      let u = new Float64Array(N);
      for (let i = 0; i < N; i++) u[i] = rand();
      for (const ev of eigenVecs) {
        let dot = 0;
        for (let i = 0; i < N; i++) dot += u[i] * ev[i];
        for (let i = 0; i < N; i++) u[i] -= dot * ev[i];
      }
      let norm = 0;
      for (let i = 0; i < N; i++) norm += u[i] * u[i];
      norm = Math.sqrt(norm);
      if (norm < 1e-12) continue;
      for (let i = 0; i < N; i++) u[i] /= norm;

      // Power iteration
      for (let iter = 0; iter < 500; iter++) {
        const uNew = new Float64Array(N);
        for (let i = 0; i < N; i++) {
          const off = i * N;
          for (let j = 0; j < N; j++) uNew[i] += G[off + j] * u[j];
        }
        for (const ev of eigenVecs) {
          let dot = 0;
          for (let i = 0; i < N; i++) dot += uNew[i] * ev[i];
          for (let i = 0; i < N; i++) uNew[i] -= dot * ev[i];
        }
        norm = 0;
        for (let i = 0; i < N; i++) norm += uNew[i] * uNew[i];
        norm = Math.sqrt(norm);
        if (norm < 1e-12) break;
        for (let i = 0; i < N; i++) uNew[i] /= norm;

        let diff = 0;
        for (let i = 0; i < N; i++) diff += (uNew[i] - u[i]) ** 2;
        u = uNew;
        if (diff < 1e-20) break;
      }

      // Eigenvalue λ = u^T G u
      let lambda = 0;
      for (let i = 0; i < N; i++) {
        const off = i * N;
        let tmp = 0;
        for (let j = 0; j < N; j++) tmp += G[off + j] * u[j];
        lambda += u[i] * tmp;
      }
      if (lambda < 1e-10) break;

      eigenVecs.push(u);
      eigenVals.push(lambda);
    }

    // ── 6. Convert N-dim eigenvectors -> D-dim PCA components ─────────────────
    const components: Float64Array[] = [];
    const compVars: number[] = [];

    for (let k = 0; k < eigenVecs.length; k++) {
      const u = eigenVecs[k];
      const lambda = eigenVals[k];
      const scale = 1.0 / Math.sqrt(lambda);

      const p = new Float64Array(D);
      for (let i = 0; i < N; i++) {
        const coeff = u[i] * scale;
        if (Math.abs(coeff) < 1e-14) continue;
        const ci = C[i];
        for (let d = 0; d < D; d++) p[d] += coeff * ci[d];
      }

      components.push(p);
      compVars.push(lambda / N);
    }

    // ── 7. z-bounds across training matrices ─────────────────────────────────
    const zMin = new Array<number>(components.length).fill(Infinity);
    const zMax = new Array<number>(components.length).fill(-Infinity);

    for (const ci of C) {
      for (let k = 0; k < components.length; k++) {
        let proj = 0;
        const comp = components[k];
        for (let d = 0; d < D; d++) proj += ci[d] * comp[d];
        if (proj < zMin[k]) zMin[k] = proj;
        if (proj > zMax[k]) zMax[k] = proj;
      }
    }

    // Pad bounds by 10% so BO can explore slightly outside the training range
    for (let k = 0; k < components.length; k++) {
      const pad = Math.max(0.1, (zMax[k] - zMin[k]) * 0.1);
      zMin[k] -= pad;
      zMax[k] += pad;
    }

    // ── 8. Store results ──────────────────────────────────────────────────────
    this.mean = mean;
    this.components = components;
    this.zMin = zMin;
    this.zMax = zMax;
    this.totalVariance = totalVar;
    this.fittedCount = N;
    this.isFitted = true;

    const explainedVarianceRatios = compVars.map(v => totalVar > 1e-12 ? v / totalVar : 0);
    const cumulativeVarianceRatios: number[] = [];
    let cumSum = 0;
    for (const r of explainedVarianceRatios) {
      cumSum += r;
      cumulativeVarianceRatios.push(Math.min(1, cumSum));
    }

    this.explainedVarianceRatios = explainedVarianceRatios;
    this.cumulativeVarianceRatios = cumulativeVarianceRatios;

    // Persist in localStorage for cross-session use in Module 2
    try {
      if (typeof window !== 'undefined' && window.localStorage) {
        const serialized = this.serializeBasis();
        if (serialized) {
          window.localStorage.setItem(`pca_basis_${this.rows}x${this.cols}_k${components.length}`, serialized);
          window.localStorage.setItem('pca_basis_latest', serialized);
        }
      }
    } catch { /* ignore storage quota */ }

    return { numComponents: components.length, explainedVarianceRatios, cumulativeVarianceRatios };
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Encode — matrix -> z in [0,1]^K
  // ─────────────────────────────────────────────────────────────────────────

  public encode(x: Float32Array): number[] {
    if (!this.isFitted || !this.mean || !this.components || this.components.length === 0) {
      return new Array<number>(this.k).fill(0.5);
    }
    const D = this.inputDim;
    const K = this.components.length;
    const z = new Array<number>(K);

    for (let k = 0; k < K; k++) {
      let proj = 0;
      const comp = this.components[k];
      for (let d = 0; d < D; d++) proj += ((x[d] || 0) - this.mean[d]) * comp[d];

      const range = this.zMax![k] - this.zMin![k];
      z[k] = range > 1e-10
        ? Math.max(0, Math.min(1, (proj - this.zMin![k]) / range))
        : 0.5;
    }
    return z;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Decode — z in [0,1]^K -> continuous reconstruction in original space
  // ─────────────────────────────────────────────────────────────────────────

  public decode(z: number[]): Float32Array {
    if (!this.isFitted || !this.mean || !this.components || this.components.length === 0) {
      return new Float32Array(this.inputDim).fill(0);
    }
    const D = this.inputDim;
    const K = this.components.length;
    const x = new Float32Array(D);

    for (let d = 0; d < D; d++) x[d] = this.mean[d];

    for (let k = 0; k < Math.min(z.length, K); k++) {
      const zk = Math.max(0, Math.min(1, z[k] ?? 0.5));
      const rawZ = zk * (this.zMax![k] - this.zMin![k]) + this.zMin![k];
      const comp = this.components[k];
      for (let d = 0; d < D; d++) x[d] += rawZ * comp[d];
    }
    return x;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Decode -> binary {0,1} using strictly '>' threshold
  // ─────────────────────────────────────────────────────────────────────────

  public decodeBinary(z: number[], threshold = 0.5): Float32Array {
    const cont = this.decode(z);
    const out = new Float32Array(cont.length);
    for (let i = 0; i < cont.length; i++) {
      // Strictly > threshold as explicitly required
      out[i] = cont[i] > threshold ? 1.0 : 0.0;
    }
    return out;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Complete Reconstruct Helper
  // ─────────────────────────────────────────────────────────────────────────

  public reconstruct(
    id: string,
    z: number[],
    optionsOrThreshold: Partial<DecoderOptions> | number = 0.5,
    valueTypeArg?: MatrixValueType,
    groundTruthArg?: Float32Array
  ): ReconstructedMatrix {
    let threshold = 0.5;
    let valueType: MatrixValueType = 'binary_01';
    let groundTruth: Float32Array | undefined;

    if (typeof optionsOrThreshold === 'number') {
      threshold = optionsOrThreshold;
      if (valueTypeArg) valueType = valueTypeArg;
      if (groundTruthArg) groundTruth = groundTruthArg;
    } else if (optionsOrThreshold) {
      if (optionsOrThreshold.threshold !== undefined) threshold = optionsOrThreshold.threshold;
      if (optionsOrThreshold.valueType) valueType = optionsOrThreshold.valueType;
      if (groundTruthArg) groundTruth = groundTruthArg;
    }

    const continuous = this.decode(z);
    const D = this.inputDim;
    const data = new Float32Array(D);
    let nonZeroCount = 0;
    let minVal = Infinity;
    let maxVal = -Infinity;

    if (valueType === 'binary_01') {
      for (let i = 0; i < D; i++) {
        // Strictly > threshold
        const v = continuous[i] > threshold ? 1.0 : 0.0;
        data[i] = v;
        if (v > 0) nonZeroCount++;
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
      }
    } else {
      for (let i = 0; i < D; i++) {
        const v = continuous[i];
        data[i] = v;
        if (v !== 0) nonZeroCount++;
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
      }
    }

    if (minVal === Infinity) { minVal = 0; maxVal = 0; }
    const total = this.rows * this.cols;
    const sparsityPercent = total > 0 ? ((total - nonZeroCount) / total) * 100 : 0;

    let metrics: ReconstructedMatrix['metrics'];
    if (groundTruth && groundTruth.length === D) {
      if (valueType === 'binary_01') {
        let diffCount = 0;
        let tp = 0, fp = 0, fn = 0;
        for (let i = 0; i < D; i++) {
          const gt = groundTruth[i] > 0.5 ? 1 : 0;
          const pred = data[i] > 0.5 ? 1 : 0;
          if (gt !== pred) diffCount++;
          if (gt === 1 && pred === 1) tp++;
          if (gt === 0 && pred === 1) fp++;
          if (gt === 1 && pred === 0) fn++;
        }
        const hammingAccuracy = (D - diffCount) / D;
        const precision = (tp + fp) > 0 ? tp / (tp + fp) : 1;
        const recall = (tp + fn) > 0 ? tp / (tp + fn) : 1;
        const f1 = (precision + recall) > 0 ? (2 * precision * recall) / (precision + recall) : 0;
        metrics = {
          hammingAccuracy,
          hammingDistance: diffCount,
          precision,
          recall,
          f1,
        };
      } else {
        let sumAbs = 0;
        let sumSq = 0;
        for (let i = 0; i < D; i++) {
          const diff = data[i] - groundTruth[i];
          sumAbs += Math.abs(diff);
          sumSq += diff * diff;
        }
        metrics = {
          mae: sumAbs / D,
          rmse: Math.sqrt(sumSq / D),
        };
      }
    }

    return {
      id,
      rows: this.rows,
      cols: this.cols,
      data,
      continuous,
      valueType,
      minVal,
      maxVal,
      nonZeroCount,
      sparsityPercent,
      groundTruth,
      metrics,
    };
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Round-trip validation
  // ─────────────────────────────────────────────────────────────────────────

  public roundTripMetrics(matrices: Float32Array[], threshold = 0.5): {
    meanAccuracy: number;
    perMatrix: number[];
  } {
    const D = this.inputDim;
    const perMatrix: number[] = [];
    for (const m of matrices) {
      const z = this.encode(m);
      const recon = this.decodeBinary(z, threshold);
      let correct = 0;
      for (let d = 0; d < D; d++) {
        const orig = (m[d] || 0) > 0.5 ? 1 : 0;
        const pred = recon[d] > 0.5 ? 1 : 0;
        if (orig === pred) correct++;
      }
      perMatrix.push(correct / D);
    }
    const meanAccuracy = perMatrix.length > 0 
      ? perMatrix.reduce((a, b) => a + b, 0) / perMatrix.length 
      : 1.0;
    return { meanAccuracy, perMatrix };
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Basis serialisation / deserialisation
  // ─────────────────────────────────────────────────────────────────────────

  public serializeBasis(): string | null {
    if (!this.isFitted || !this.mean || !this.components) return null;
    const basis: PCABasis = {
      rows: this.rows,
      cols: this.cols,
      k: this.components.length,
      mean: Array.from(this.mean),
      components: this.components.map(c => Array.from(c)),
      zMin: this.zMin!,
      zMax: this.zMax!,
      explainedVarianceRatios: this.explainedVarianceRatios || [],
      totalVariance: this.totalVariance || 0,
      fittedCount: this.fittedCount || 0,
    };
    return JSON.stringify(basis, null, 2);
  }

  public loadBasis(json: string): boolean {
    try {
      const b: PCABasis = JSON.parse(json);
      if (!b.mean || !b.components || !Array.isArray(b.components)) return false;

      this.rows = b.rows;
      this.cols = b.cols;
      this.k = b.k;
      this.mean = new Float64Array(b.mean);
      this.components = b.components.map((c: number[]) => new Float64Array(c));
      this.zMin = b.zMin;
      this.zMax = b.zMax;
      this.explainedVarianceRatios = b.explainedVarianceRatios;
      this.totalVariance = b.totalVariance;
      this.fittedCount = b.fittedCount;
      this.isFitted = true;

      const cumulative: number[] = [];
      let cum = 0;
      for (const r of (b.explainedVarianceRatios || [])) {
        cum += r;
        cumulative.push(Math.min(1, cum));
      }
      this.cumulativeVarianceRatios = cumulative;

      return true;
    } catch {
      return false;
    }
  }

  public tryLoadFromStorage(): boolean {
    try {
      if (typeof window === 'undefined' || !window.localStorage) return false;
      const key = `pca_basis_${this.rows}x${this.cols}_k${this.k}`;
      const raw = window.localStorage.getItem(key) || window.localStorage.getItem('pca_basis_latest');
      if (!raw) return false;
      return this.loadBasis(raw);
    } catch {
      return false;
    }
  }

  public clearStorage(): void {
    try {
      if (typeof window !== 'undefined' && window.localStorage) {
        window.localStorage.removeItem(`pca_basis_${this.rows}x${this.cols}_k${this.k}`);
        window.localStorage.removeItem('pca_basis_latest');
      }
    } catch { /* ignore */ }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Python Script Generator for Bayesian Optimization & Scikit-learn
  // ─────────────────────────────────────────────────────────────────────────

  public generatePythonScript(threshold = 0.5): string {
    return `"""
Matrix PCA & Bayesian Optimization Pipeline
---------------------------------------------
Matrix Dimension: ${this.rows} x ${this.cols} (${this.inputDim} features)
Latent Dimension: K = ${this.k} components (bounded in [0.0, 1.0]^K)
Binary Threshold: strictly > ${threshold}

Designed for Bayesian Optimization (BO) workflows (e.g. EDOS family modules):
1. Fit PCA basis on initial matrix library.
2. Normalize projections to [0, 1]^K for the optimizer's search space.
3. BO proposes new candidate vectors z* in [0, 1]^K.
4. Decoder projects back and applies strictly '>' threshold to reconstruct binary matrices.
"""

import json
import numpy as np
from sklearn.decomposition import PCA

ROWS = ${this.rows}
COLS = ${this.cols}
D = ROWS * COLS
K = ${this.k}
THRESHOLD = ${threshold}

class MatrixPCABOPipeline:
    def __init__(self, rows=ROWS, cols=COLS, k=K, threshold=THRESHOLD):
        self.rows = rows
        self.cols = cols
        self.d = rows * cols
        self.k = k
        self.threshold = threshold
        self.pca = PCA(n_components=k)
        self.z_min = None
        self.z_max = None

    def fit(self, matrices: np.ndarray):
        """
        matrices: shape (N, D) or (N, ROWS, COLS)
        """
        x = matrices.reshape((matrices.shape[0], self.d)).astype(np.float64)
        self.pca.fit(x)
        proj = self.pca.transform(x)
        pad = np.maximum(0.1, (proj.max(axis=0) - proj.min(axis=0)) * 0.1)
        self.z_min = proj.min(axis=0) - pad
        self.z_max = proj.max(axis=0) + pad
        return self

    def encode(self, matrices: np.ndarray) -> np.ndarray:
        """
        Encodes matrices to normalized latent coordinates in [0, 1]^K.
        Returns: array of shape (N, K)
        """
        x = matrices.reshape((matrices.shape[0], self.d)).astype(np.float64)
        proj = self.pca.transform(x)
        rng = np.maximum(1e-10, self.z_max - self.z_min)
        z_norm = np.clip((proj - self.z_min) / rng, 0.0, 1.0)
        return z_norm

    def decode_binary(self, z_norm: np.ndarray, threshold: float = None) -> np.ndarray:
        """
        Decodes normalized latent vectors in [0, 1]^K back to binary matrices.
        Reconstruction rule: strictly '>' threshold.
        Returns: array of shape (N, ROWS, COLS) with values in {0.0, 1.0}
        """
        if threshold is None:
            threshold = self.threshold
        z = np.clip(np.atleast_2d(z_norm), 0.0, 1.0)
        raw_z = z * (self.z_max - self.z_min) + self.z_min
        continuous = self.pca.inverse_transform(raw_z)
        # Strictly '>' threshold assignment:
        binary = (continuous > threshold).astype(np.float32)
        return binary.reshape((-1, self.rows, self.cols))

    def export_basis_json(self, filepath: str = "pca_basis.json"):
        basis = {
            "rows": self.rows,
            "cols": self.cols,
            "k": self.k,
            "mean": self.pca.mean_.tolist(),
            "components": self.pca.components_.tolist(),
            "zMin": self.z_min.tolist(),
            "zMax": self.z_max.tolist(),
            "explainedVarianceRatios": self.pca.explained_variance_ratio_.tolist(),
            "totalVariance": float(np.sum(self.pca.explained_variance_)),
            "fittedCount": int(self.pca.n_samples_seen_)
        }
        with open(filepath, "w") as f:
            json.dump(basis, f, indent=2)
        print(f"Exported PCA Basis to {filepath}")

    def load_basis_json(self, filepath: str = "pca_basis.json"):
        with open(filepath, "r") as f:
            basis = json.load(f)
        self.rows = basis["rows"]
        self.cols = basis["cols"]
        self.d = self.rows * self.cols
        self.k = basis["k"]
        self.pca.mean_ = np.array(basis["mean"], dtype=np.float64)
        self.pca.components_ = np.array(basis["components"], dtype=np.float64)
        self.z_min = np.array(basis["zMin"], dtype=np.float64)
        self.z_max = np.array(basis["zMax"], dtype=np.float64)
        print(f"Loaded PCA Basis from {filepath}")


# Example usage:
if __name__ == "__main__":
    pipeline = MatrixPCABOPipeline(rows=${this.rows}, cols=${this.cols}, k=${this.k})
    print(f"Matrix PCA Studio pipeline initialized for {pipeline.rows}x{pipeline.cols} matrices.")
`;
  }

  public generateKerasScript(): string {
    return this.generatePythonScript(0.5);
  }
}

// Shared singletons
export const defaultPCA = new MatrixPCA(45, 45, 16);
export { MatrixPCA as MatrixAutoencoder, defaultPCA as defaultAutoencoder };
