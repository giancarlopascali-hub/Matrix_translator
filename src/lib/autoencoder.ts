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
 *  - Exact on fitted samples only when their affine rank is <= K
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

export const PCA_BASIS_SCHEMA_VERSION = 2;
export const PCA_BASIS_ALGORITHM = 'pca-linear-v2';

type ModelSource = 'preview' | 'trained' | 'imported';

function modelFingerprint(payload: object): string {
  const text = JSON.stringify(payload);
  let h1 = 0x811c9dc5;
  let h2 = 0x9e3779b9;
  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i);
    h1 = Math.imul(h1 ^ code, 0x01000193) >>> 0;
    h2 = Math.imul(h2 ^ code, 0x85ebca6b) >>> 0;
  }
  return `pca2-${h1.toString(16).padStart(8, '0')}${h2.toString(16).padStart(8, '0')}`;
}

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
  public usedSyntheticAnchor = false;
  public modelId?: string;
  public valueType: MatrixValueType = 'binary_01';
  public modelSource: ModelSource = 'preview';
  public lastError?: string;

  constructor(rows = 45, cols = 45, k = 16) {
    this.rows = Math.max(1, Math.floor(rows));
    this.cols = Math.max(1, Math.floor(cols));
    this.k = Math.max(1, Math.min(Math.floor(k), this.rows * this.cols));
    if (!this.tryLoadFromStorage()) {
      this.initCanonicalBasis();
    }
  }

  get inputDim(): number { return this.rows * this.cols; }

  get hasDecodingBasis(): boolean {
    return this.isFitted && this.modelSource !== 'preview' && !!this.modelId;
  }

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
  // Canonical 2D-DCT Orthonormal Basis & Bounds Generation
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Generates a 2D Discrete Cosine Transform (DCT-II) orthonormal basis for R x C matrices.
   * Sorted by spatial frequency (u^2 + v^2) so early dimensions capture broad patterns
   * (DC density, horizontal/vertical gradients, quadrants) and later capture fine details.
   */
  public generateDctPool(rows: number, cols: number, count: number): Float64Array[] {
    const D = rows * cols;
    const coords: { dist: number; u: number; v: number }[] = [];
    for (let u = 0; u < rows; u++) {
      for (let v = 0; v < cols; v++) {
        coords.push({ dist: u * u + v * v, u, v });
      }
    }
    coords.sort((a, b) => a.dist - b.dist || a.u - b.u || a.v - b.v);

    const pool: Float64Array[] = [];
    const maxK = Math.min(coords.length, Math.max(count, 64));

    for (let k = 0; k < maxK; k++) {
      const { u, v } = coords[k];
      const phi = new Float64Array(D);
      const alphaU = u === 0 ? Math.sqrt(1.0 / rows) : Math.sqrt(2.0 / rows);
      const alphaV = v === 0 ? Math.sqrt(1.0 / cols) : Math.sqrt(2.0 / cols);

      for (let r = 0; r < rows; r++) {
        const cosU = Math.cos((Math.PI * (2 * r + 1) * u) / (2 * rows));
        const rOff = r * cols;
        for (let c = 0; c < cols; c++) {
          const cosV = Math.cos((Math.PI * (2 * c + 1) * v) / (2 * cols));
          phi[rOff + c] = alphaU * alphaV * cosU * cosV;
        }
      }
      pool.push(phi);
    }
    return pool;
  }

  /**
   * Computes exact theoretical projection bounds for binary {0, 1} matrices on given components,
   * with optional sample calibration padding for Bayesian Optimization hypercube exploration.
   */
  public computeBinaryBounds(
    components: Float64Array[], 
    mean: Float64Array, 
    sampleMatrices?: (Float64Array | Float32Array)[]
  ): { zMin: number[]; zMax: number[] } {
    const D = this.inputDim;
    const K = components.length;
    const zMin = new Array<number>(K);
    const zMax = new Array<number>(K);

    for (let k = 0; k < K; k++) {
      const comp = components[k];
      let bMax = 0;
      let bMin = 0;
      let muDot = 0;

      for (let d = 0; d < D; d++) {
        const cd = comp[d];
        muDot += mean[d] * cd;
        if (cd > 0) bMax += cd;
        else if (cd < 0) bMin += cd;
      }
      bMax -= muDot;
      bMin -= muDot;

      if (sampleMatrices && sampleMatrices.length > 1) {
        let sMin = Infinity;
        let sMax = -Infinity;
        for (const sm of sampleMatrices) {
          let proj = 0;
          for (let d = 0; d < D; d++) proj += (sm[d] - mean[d]) * comp[d];
          if (proj < sMin) sMin = proj;
          if (proj > sMax) sMax = proj;
        }

        if (sMax - sMin > 1e-4) {
          const pad = Math.max(0.05, (sMax - sMin) * 0.20);
          zMin[k] = Math.max(bMin, sMin - pad);
          zMax[k] = Math.min(bMax, sMax + pad);
        } else {
          zMin[k] = bMin;
          zMax[k] = bMax;
        }
      } else {
        zMin[k] = bMin;
        zMax[k] = bMax;
      }

      // Safeguard against zero range: binary bounds span is >= 1.0
      if (zMax[k] - zMin[k] < 1e-4) {
        zMin[k] = bMin;
        zMax[k] = bMax;
      }
    }

    return { zMin, zMax };
  }

  /**
   * Initializes a canonical DCT preview. It is deliberately not considered a
   * trained decoding basis and must never be used for standalone CSV decoding.
   */
  public initCanonicalBasis(): void {
    const D = this.inputDim;
    const dctPool = this.generateDctPool(this.rows, this.cols, this.k);
    const components = dctPool.slice(0, this.k);
    const mean = new Float64Array(D).fill(0.0);

    const bounds = this.computeBinaryBounds(components, mean);
    this.mean = mean;
    this.components = components;
    this.zMin = bounds.zMin;
    this.zMax = bounds.zMax;
    this.isFitted = false;
    this.modelSource = 'preview';
    this.modelId = undefined;
    this.valueType = 'binary_01';
    this.usedSyntheticAnchor = false;
    this.lastError = undefined;
    this.fittedCount = 0;
    this.totalVariance = 1.0;
    this.explainedVarianceRatios = components.map((_, i) => 1.0 / (i + 1));
    this.cumulativeVarianceRatios = components.map((_, i) => Math.min(1.0, (i + 1) / components.length));
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Configuration
  // ─────────────────────────────────────────────────────────────────────────

  public reconfigure(rows: number, cols: number, k?: number): void {
    const r = Math.max(1, Math.floor(rows));
    const c = Math.max(1, Math.floor(cols));
    const newK = Math.max(1, Math.min(Math.floor(k ?? this.k), r * c));
    if (this.rows === r && this.cols === c && newK === this.k) {
      return;
    }
    this.rows = r;
    this.cols = c;
    this.k = newK;
    if (!this.tryLoadFromStorage()) {
      this.initCanonicalBasis();
    }
  }

  public reset(): void {
    this.clearStorage();
    this.initCanonicalBasis();
  }

  // Compatibility aliases
  public resetCalibration(): void {
    this.reset();
  }

  public calibrateToMatrices(matrices: Float32Array[]): { mae: number; maxError: number; count: number } {
    this.fit(matrices);
    const roundTrip = this.roundTripMetrics(matrices, 0.5);
    return {
      count: matrices.length,
      mae: 1.0 - roundTrip.meanAccuracy,
      maxError: 0.0,
    };
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Fit — PCA using only data-supported, non-zero-variance components
  // ─────────────────────────────────────────────────────────────────────────

  private createSyntheticAnchor(matrix: Float32Array, valueType: MatrixValueType): Float32Array {
    const anchor = new Float32Array(matrix.length);

    if (valueType === 'binary_01') {
      // An all-zero baseline is the least opinionated binary anchor. For an
      // all-zero input, use all ones so the fitted direction is non-degenerate.
      let hasOne = false;
      for (let d = 0; d < matrix.length; d++) {
        if (matrix[d] === 1) {
          hasOne = true;
          break;
        }
      }
      if (!hasOne) anchor.fill(1);
      return anchor;
    }

    // Continuous matrices also use zero as the baseline. If the matrix is too
    // close to zero for a stable component, perturb one cell deterministically.
    let distanceSq = 0;
    for (let d = 0; d < matrix.length; d++) distanceSq += matrix[d] * matrix[d];
    if (distanceSq <= 1e-6) {
      anchor.set(matrix);
      anchor[0] = matrix[0] + 1;
    }
    return anchor;
  }

  public fit(matrices: Float32Array[]): FitResult {
    const realCount = matrices.length;
    const D = this.inputDim;
    const requestedK = this.k;

    // A failed refit must never leave the previous basis active for new data.
    this.isFitted = false;
    this.modelSource = 'preview';
    this.modelId = undefined;
    this.usedSyntheticAnchor = false;

    if (realCount < 1) {
      throw new Error('At least one matrix is required to learn a BO latent representation.');
    }

    for (let i = 0; i < realCount; i++) {
      if (matrices[i].length !== D) {
        throw new Error(`Matrix ${i + 1} has ${matrices[i].length} cells; expected exactly ${D} (${this.rows}×${this.cols}).`);
      }
      for (let d = 0; d < D; d++) {
        if (!Number.isFinite(matrices[i][d])) {
          throw new Error(`Matrix ${i + 1} contains a non-finite value at flattened cell ${d}.`);
        }
      }
    }
    const fittedValueType: MatrixValueType = matrices.every(matrix => {
      for (let d = 0; d < matrix.length; d++) {
        if (matrix[d] !== 0 && matrix[d] !== 1) return false;
      }
      return true;
    }) ? 'binary_01' : 'continuous_numeric';

    const usedSyntheticAnchor = realCount === 1;
    const trainingMatrices = usedSyntheticAnchor
      ? [matrices[0], this.createSyntheticAnchor(matrices[0], fittedValueType)]
      : matrices;
    const N = trainingMatrices.length;

    const kEff = Math.min(requestedK, N - 1, D);

    // ── 1. Empirical Mean ───────────────────────────────────────────────────
    const mean = new Float64Array(D);
    for (let j = 0; j < D; j++) {
      let s = 0;
      for (let i = 0; i < N; i++) s += trainingMatrices[i][j];
      mean[j] = s / N;
    }

    // ── 2. Centered matrix rows ──────────────────────────────────────────────
    const C: Float64Array[] = trainingMatrices.map(m => {
      const v = new Float64Array(D);
      for (let j = 0; j < D; j++) v[j] = m[j] - mean[j];
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
      if (lambda < 1e-8) break;

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

      let pNorm = 0;
      for (let d = 0; d < D; d++) pNorm += p[d] * p[d];
      pNorm = Math.sqrt(pNorm);
      if (pNorm > 1e-10) {
        for (let d = 0; d < D; d++) p[d] /= pNorm;
        components.push(p);
        compVars.push(lambda / N);
      }
    }

    if (components.length === 0 || totalVar <= 1e-12) {
      throw new Error('The uploaded matrices contain no measurable variation, so a BO latent representation cannot be fitted.');
    }

    // Projection bounds are the observed training bounds. Padding would expose
    // BO dimensions that have never been represented by the fitted data.
    const zMin = new Array<number>(components.length).fill(Infinity);
    const zMax = new Array<number>(components.length).fill(-Infinity);
    for (const centered of C) {
      for (let k = 0; k < components.length; k++) {
        let projection = 0;
        for (let d = 0; d < D; d++) projection += centered[d] * components[k][d];
        zMin[k] = Math.min(zMin[k], projection);
        zMax[k] = Math.max(zMax[k], projection);
      }
    }
    for (let k = 0; k < components.length; k++) {
      if (zMax[k] - zMin[k] <= 1e-10) {
        throw new Error(`PCA component ${k + 1} has zero usable range.`);
      }
    }

    // Store only the empirically supported components. In particular, do not
    // pad a low-rank data set with arbitrary DCT dimensions.
    this.mean = mean;
    this.components = components;
    this.zMin = zMin;
    this.zMax = zMax;
    this.k = components.length;
    this.totalVariance = totalVar;
    this.fittedCount = realCount;
    this.usedSyntheticAnchor = usedSyntheticAnchor;
    this.isFitted = true;
    this.modelSource = 'trained';
    this.valueType = fittedValueType;
    this.lastError = undefined;

    const explainedVarianceRatios = compVars.map(v => totalVar > 1e-12 ? Math.min(1.0, v / totalVar) : 1.0 / components.length);
    const cumulativeVarianceRatios: number[] = [];
    let cumSum = 0;
    for (const r of explainedVarianceRatios) {
      cumSum += r;
      cumulativeVarianceRatios.push(Math.min(1.0, cumSum));
    }

    this.explainedVarianceRatios = explainedVarianceRatios;
    this.cumulativeVarianceRatios = cumulativeVarianceRatios;
    this.modelId = this.computeModelId();

    this.persistToStorage();

    return {
      numComponents: components.length,
      requestedComponents: requestedK,
      effectiveRank: components.length,
      usedSyntheticAnchor,
      explainedVarianceRatios,
      cumulativeVarianceRatios,
    };
  }

  private computeModelId(): string {
    if (!this.mean || !this.components || !this.zMin || !this.zMax) {
      throw new Error('Cannot fingerprint an incomplete PCA basis.');
    }
    return modelFingerprint({
      schemaVersion: PCA_BASIS_SCHEMA_VERSION,
      algorithm: PCA_BASIS_ALGORITHM,
      valueType: this.valueType,
      rows: this.rows,
      cols: this.cols,
      k: this.components.length,
      mean: Array.from(this.mean),
      components: this.components.map(component => Array.from(component)),
      zMin: this.zMin,
      zMax: this.zMax,
    });
  }

  private persistToStorage(): void {
    try {
      if (typeof window !== 'undefined' && window.localStorage && this.hasDecodingBasis) {
        const serialized = this.serializeBasis();
        if (serialized) {
          window.localStorage.setItem(`pca_basis_${this.rows}x${this.cols}_k${this.components!.length}`, serialized);
        }
      }
    } catch { /* ignore storage quota */ }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Encode — matrix -> z in [0,1]^K
  // ─────────────────────────────────────────────────────────────────────────

  public encode(x: Float32Array): number[] {
    if (!this.hasDecodingBasis || !this.mean || !this.components || this.components.length === 0) {
      throw new Error('Fit or import a PCA basis before encoding matrices.');
    }
    if (x.length !== this.inputDim) {
      throw new Error(`Cannot encode ${x.length} cells with a ${this.rows}×${this.cols} (${this.inputDim}-cell) basis.`);
    }
    const D = this.inputDim;
    const K = this.components!.length;
    const z = new Array<number>(K);

    for (let k = 0; k < K; k++) {
      let proj = 0;
      const comp = this.components![k];
      for (let d = 0; d < D; d++) {
        if (!Number.isFinite(x[d])) throw new Error(`Matrix contains a non-finite value at flattened cell ${d}.`);
        proj += (x[d] - this.mean![d]) * comp[d];
      }

      const minVal = this.zMin?.[k] ?? -1.0;
      const maxVal = this.zMax?.[k] ?? 1.0;
      const range = maxVal - minVal;

      z[k] = range > 1e-10
        ? Math.max(0, Math.min(1, (proj - minVal) / range))
        : 0.5;
    }
    return z;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Decode — z in [0,1]^K -> continuous reconstruction in original space
  // ─────────────────────────────────────────────────────────────────────────

  public decode(z: number[]): Float32Array {
    if (!this.hasDecodingBasis || !this.mean || !this.components || this.components.length === 0) {
      throw new Error('Import a trained PCA basis before decoding latent vectors.');
    }
    const D = this.inputDim;
    const K = this.components.length;
    if (z.length !== K) {
      throw new Error(`Latent vector has K=${z.length}; this basis requires exactly K=${K}.`);
    }
    const x = new Float32Array(D);

    for (let d = 0; d < D; d++) x[d] = this.mean[d];

    for (let k = 0; k < K; k++) {
      if (!Number.isFinite(z[k]) || z[k] < 0 || z[k] > 1) {
        throw new Error(`Latent coordinate z${k + 1} must be finite and inside [0,1].`);
      }
      const zk = z[k];
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
    exactMatchRate: number;
    balancedAccuracy: number;
    precision: number;
    recall: number;
    f1: number;
  } {
    const D = this.inputDim;
    const perMatrix: number[] = [];
    let exactMatches = 0;
    let tp = 0, tn = 0, fp = 0, fn = 0;
    for (const m of matrices) {
      const z = this.encode(m);
      const recon = this.decodeBinary(z, threshold);
      let correct = 0;
      for (let d = 0; d < D; d++) {
        const orig = m[d] > 0.5 ? 1 : 0;
        const pred = recon[d] > 0.5 ? 1 : 0;
        if (orig === pred) correct++;
        if (orig === 1 && pred === 1) tp++;
        else if (orig === 0 && pred === 0) tn++;
        else if (orig === 0 && pred === 1) fp++;
        else fn++;
      }
      if (correct === D) exactMatches++;
      perMatrix.push(correct / D);
    }
    const meanAccuracy = perMatrix.length > 0 
      ? perMatrix.reduce((a, b) => a + b, 0) / perMatrix.length 
      : 1.0;
    const truePositiveRate = tp + fn > 0 ? tp / (tp + fn) : 1;
    const trueNegativeRate = tn + fp > 0 ? tn / (tn + fp) : 1;
    const precision = tp + fp > 0 ? tp / (tp + fp) : 1;
    const recall = truePositiveRate;
    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
    return {
      meanAccuracy,
      perMatrix,
      exactMatchRate: matrices.length > 0 ? exactMatches / matrices.length : 1,
      balancedAccuracy: (truePositiveRate + trueNegativeRate) / 2,
      precision,
      recall,
      f1,
    };
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Basis serialisation / deserialisation
  // ─────────────────────────────────────────────────────────────────────────

  public serializeBasis(): string {
    if (!this.hasDecodingBasis || !this.mean || !this.components || this.components.length === 0) {
      throw new Error('There is no trained PCA basis to export. Fit at least one matrix first.');
    }
    this.modelId = this.computeModelId();

    const basis: PCABasis = {
      schemaVersion: PCA_BASIS_SCHEMA_VERSION,
      algorithm: PCA_BASIS_ALGORITHM,
      modelId: this.modelId,
      valueType: this.valueType,
      rows: this.rows,
      cols: this.cols,
      k: this.components!.length,
      mean: Array.from(this.mean!),
      components: this.components!.map(c => Array.from(c)),
      zMin: this.zMin || new Array(this.components!.length).fill(0),
      zMax: this.zMax || new Array(this.components!.length).fill(1),
      explainedVarianceRatios: this.explainedVarianceRatios || [],
      totalVariance: this.totalVariance || 0,
      fittedCount: this.fittedCount || 0,
      syntheticAnchorUsed: this.usedSyntheticAnchor,
    };
    return JSON.stringify(basis, null, 2);
  }

  public loadBasis(json: string): boolean {
    this.lastError = undefined;
    try {
      const b: PCABasis = JSON.parse(json);
      const fail = (message: string): false => {
        this.lastError = message;
        return false;
      };

      if (!b || typeof b !== 'object') return fail('Basis JSON must contain an object.');
      if (b.schemaVersion !== PCA_BASIS_SCHEMA_VERSION) {
        return fail('This basis uses an old or unsupported schema. Re-encode the matrices and export a new matching basis JSON.');
      }
      if (b.algorithm !== PCA_BASIS_ALGORITHM) {
        return fail(`Unsupported basis algorithm "${b.algorithm}".`);
      }
      if (!b.modelId) return fail('Basis modelId is missing. Re-export the basis from the encoder.');
      if (b.valueType !== 'binary_01' && b.valueType !== 'continuous_numeric') {
        return fail('Basis valueType must be binary_01 or continuous_numeric.');
      }
      if (!Number.isInteger(b.rows) || b.rows < 1 || !Number.isInteger(b.cols) || b.cols < 1) {
        return fail('Basis rows and columns must be positive integers.');
      }
      if (!Number.isInteger(b.k) || b.k < 1) return fail('Basis K must be a positive integer.');

      const D = b.rows * b.cols;
      if (!Array.isArray(b.mean) || b.mean.length !== D || b.mean.some(v => !Number.isFinite(v))) {
        return fail(`Basis mean must contain exactly ${D} finite values.`);
      }
      if (!Array.isArray(b.components) || b.components.length !== b.k) {
        return fail(`Basis must contain exactly K=${b.k} components.`);
      }
      for (let k = 0; k < b.components.length; k++) {
        const component = b.components[k];
        if (!Array.isArray(component) || component.length !== D || component.some(v => !Number.isFinite(v))) {
          return fail(`Basis component ${k + 1} must contain exactly ${D} finite values.`);
        }
        let normSq = 0;
        for (const value of component) normSq += value * value;
        if (Math.abs(Math.sqrt(normSq) - 1) > 1e-4) {
          return fail(`Basis component ${k + 1} is not normalized.`);
        }
      }
      if (!Array.isArray(b.zMin) || !Array.isArray(b.zMax) || b.zMin.length !== b.k || b.zMax.length !== b.k) {
        return fail(`Basis projection bounds must contain exactly K=${b.k} values.`);
      }
      for (let k = 0; k < b.k; k++) {
        if (!Number.isFinite(b.zMin[k]) || !Number.isFinite(b.zMax[k]) || b.zMax[k] - b.zMin[k] <= 1e-10) {
          return fail(`Basis projection bounds for z${k + 1} are invalid.`);
        }
      }

      const computedModelId = modelFingerprint({
        schemaVersion: PCA_BASIS_SCHEMA_VERSION,
        algorithm: PCA_BASIS_ALGORITHM,
        valueType: b.valueType,
        rows: b.rows,
        cols: b.cols,
        k: b.k,
        mean: b.mean,
        components: b.components,
        zMin: b.zMin,
        zMax: b.zMax,
      });
      if (b.modelId !== computedModelId) {
        return fail('Basis modelId does not match its numerical contents. The file may be damaged or edited.');
      }

      this.rows = b.rows;
      this.cols = b.cols;
      this.k = b.k;
      this.mean = new Float64Array(b.mean);
      this.components = b.components.map((c: number[]) => new Float64Array(c));
      this.zMin = [...b.zMin];
      this.zMax = [...b.zMax];
      this.explainedVarianceRatios = Array.isArray(b.explainedVarianceRatios)
        ? b.explainedVarianceRatios.filter(Number.isFinite)
        : [];
      this.totalVariance = Number.isFinite(b.totalVariance) ? b.totalVariance : 0;
      this.fittedCount = Number.isInteger(b.fittedCount) && b.fittedCount >= 0 ? b.fittedCount : 0;
      this.usedSyntheticAnchor = b.syntheticAnchorUsed === true;
      this.isFitted = true;
      this.modelSource = 'imported';
      this.modelId = computedModelId;
      this.valueType = b.valueType;

      const cumulative: number[] = [];
      let cum = 0;
      for (const r of (b.explainedVarianceRatios || [])) {
        cum += r;
        cumulative.push(Math.min(1, cum));
      }
      this.cumulativeVarianceRatios = cumulative;

      return true;
    } catch (error) {
      this.lastError = error instanceof Error ? error.message : 'Invalid basis JSON.';
      return false;
    }
  }

  public tryLoadFromStorage(): boolean {
    try {
      if (typeof window === 'undefined' || !window.localStorage) return false;
      const key = `pca_basis_${this.rows}x${this.cols}_k${this.k}`;
      const raw = window.localStorage.getItem(key);
      if (!raw) return false;
      const expectedRows = this.rows;
      const expectedCols = this.cols;
      const expectedK = this.k;
      const loaded = this.loadBasis(raw);
      if (!loaded || this.rows !== expectedRows || this.cols !== expectedCols || this.k !== expectedK) {
        this.rows = expectedRows;
        this.cols = expectedCols;
        this.k = expectedK;
        this.initCanonicalBasis();
        return false;
      }
      return true;
    } catch {
      return false;
    }
  }

  public clearStorage(): void {
    try {
      if (typeof window !== 'undefined' && window.localStorage) {
        window.localStorage.removeItem(`pca_basis_${this.rows}x${this.cols}_k${this.k}`);
      }
    } catch { /* ignore */ }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Python Script Generator for Bayesian Optimization & Scikit-learn
  // ─────────────────────────────────────────────────────────────────────────

  public generatePythonScript(threshold = 0.5): string {
    return `"""
Matrix PCA basis adapter for Bayesian Optimization
---------------------------------------------------
This script uses the exact basis JSON exported by Matrix PCA Studio.
It deliberately does not refit PCA or invent fallback components, so Python
and the web decoder always implement the same numerical model.

Expected app configuration when generated:
- Matrix dimension: ${this.rows} x ${this.cols}
- Latent dimension: K = ${this.k}
- Binary rule: reconstructed value > ${threshold} becomes 1; otherwise 0
"""

import csv
import json
from pathlib import Path

import numpy as np


SUPPORTED_SCHEMA_VERSION = 2
SUPPORTED_ALGORITHM = "pca-linear-v2"
DEFAULT_THRESHOLD = ${threshold}


class MatrixPCABOPipeline:
    """Encode and decode with one immutable Matrix PCA Studio basis."""

    def __init__(self, basis):
        if basis.get("schemaVersion") != SUPPORTED_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported or legacy basis schema. Re-export the basis from Matrix PCA Studio."
            )
        if basis.get("algorithm") != SUPPORTED_ALGORITHM:
            raise ValueError(f"Unsupported basis algorithm: {basis.get('algorithm')!r}")
        if not basis.get("modelId"):
            raise ValueError("Basis modelId is missing.")

        self.rows = int(basis["rows"])
        self.cols = int(basis["cols"])
        self.d = self.rows * self.cols
        self.k = int(basis["k"])
        self.model_id = str(basis["modelId"])
        self.value_type = basis.get("valueType")
        self.mean = np.asarray(basis["mean"], dtype=np.float64)
        self.components = np.asarray(basis["components"], dtype=np.float64)
        self.z_min = np.asarray(basis["zMin"], dtype=np.float64)
        self.z_max = np.asarray(basis["zMax"], dtype=np.float64)

        if self.rows < 1 or self.cols < 1 or self.k < 1:
            raise ValueError("Basis dimensions and K must be positive.")
        if self.value_type not in ("binary_01", "continuous_numeric"):
            raise ValueError("Basis valueType is missing or unsupported.")
        if self.mean.shape != (self.d,):
            raise ValueError(f"Basis mean must have shape ({self.d},).")
        if self.components.shape != (self.k, self.d):
            raise ValueError(f"Basis components must have shape ({self.k}, {self.d}).")
        if self.z_min.shape != (self.k,) or self.z_max.shape != (self.k,):
            raise ValueError(f"Basis bounds must each have shape ({self.k},).")
        if not all(
            np.all(np.isfinite(values))
            for values in (self.mean, self.components, self.z_min, self.z_max)
        ):
            raise ValueError("Basis contains non-finite values.")
        if np.any(self.z_max - self.z_min <= 1e-10):
            raise ValueError("Every latent component must have a positive projection range.")
        if not np.allclose(np.linalg.norm(self.components, axis=1), 1.0, atol=1e-4):
            raise ValueError("Basis components are not normalized.")

    @classmethod
    def from_basis(cls, filepath="pca_basis.json"):
        basis = json.loads(Path(filepath).read_text(encoding="utf-8"))
        return cls(basis)

    def _matrix_batch(self, matrices):
        values = np.asarray(matrices, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        elif values.ndim == 2 and values.shape == (self.rows, self.cols):
            values = values.reshape(1, -1)
        elif values.ndim == 3 and values.shape[1:] == (self.rows, self.cols):
            values = values.reshape(values.shape[0], -1)
        elif values.ndim != 2:
            raise ValueError("Matrices must be flattened or supplied as rows x columns arrays.")

        if values.shape[1] != self.d:
            raise ValueError(f"Each matrix must contain exactly {self.d} cells.")
        if not np.all(np.isfinite(values)):
            raise ValueError("Matrices contain non-finite values.")
        return values

    def encode(self, matrices):
        """Return normalized BO parameters in [0, 1]^K."""
        values = self._matrix_batch(matrices)
        projections = (values - self.mean) @ self.components.T
        normalized = (projections - self.z_min) / (self.z_max - self.z_min)
        return np.clip(normalized, 0.0, 1.0)

    def decode_continuous(self, z_normalized):
        """Decode one vector or a batch; invalid BO coordinates are rejected."""
        z = np.asarray(z_normalized, dtype=np.float64)
        if z.ndim == 1:
            z = z.reshape(1, -1)
        if z.ndim != 2 or z.shape[1] != self.k:
            raise ValueError(f"Each latent vector must contain exactly K={self.k} values.")
        if not np.all(np.isfinite(z)) or np.any(z < 0.0) or np.any(z > 1.0):
            raise ValueError("Every latent coordinate must be finite and inside [0, 1].")

        raw_z = z * (self.z_max - self.z_min) + self.z_min
        reconstructed = self.mean + raw_z @ self.components
        return reconstructed.reshape((-1, self.rows, self.cols))

    def decode_binary(self, z_normalized, threshold=DEFAULT_THRESHOLD):
        """Apply the app's strict binary rule: value > threshold."""
        return (self.decode_continuous(z_normalized) > threshold).astype(np.uint8)

    def read_latent_csv(self, filepath):
        """Read Matrix PCA Studio or BO CSV output by z-column name."""
        lines = [
            line
            for line in Path(filepath).read_text(encoding="utf-8-sig").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not lines:
            raise ValueError("Latent CSV has no data.")

        reader = csv.DictReader(lines)
        fieldnames = reader.fieldnames or []
        z_columns = sorted(
            (name for name in fieldnames if name.lower().startswith("z") and name[1:].isdigit()),
            key=lambda name: int(name[1:]),
        )
        if len(z_columns) != self.k:
            raise ValueError(
                f"Latent CSV has {len(z_columns)} z columns; basis requires K={self.k}."
            )

        ids = []
        rows = []
        for index, row in enumerate(reader, start=1):
            ids.append(row.get("matrix_id") or row.get("id") or f"candidate_{index}")
            rows.append([float(row[name]) for name in z_columns])
        z = np.asarray(rows, dtype=np.float64)
        self.decode_continuous(z)  # Validate shape, finiteness, and [0, 1] bounds.
        return ids, z


if __name__ == "__main__":
    pipeline = MatrixPCABOPipeline.from_basis("pca_basis.json")
    candidate_ids, candidate_z = pipeline.read_latent_csv("latent_vectors.csv")
    decoded = pipeline.decode_binary(candidate_z)
    print(
        f"Decoded {len(decoded)} matrices with model {pipeline.model_id} "
        f"({pipeline.rows}x{pipeline.cols}, K={pipeline.k})."
    )
`;
  }

  public generateKerasScript(): string {
    return this.generatePythonScript(0.5);
  }
}

// Shared singletons
export const defaultPCA = new MatrixPCA(45, 45, 16);
export { MatrixPCA as MatrixAutoencoder, defaultPCA as defaultAutoencoder };
