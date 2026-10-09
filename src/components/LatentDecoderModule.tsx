/**
 * Module 2: Latent Decoder & Reconstruction Module (Bright Theme)
 * Load a CSV file with multiple rows (each row = 1 matrix with K latent dimensions),
 * reconstructs the collection of matrices via the PCA basis, and allows downloading individual files
 * for each row (all of them, or a user-defined selection).
 *
 * Supports:
 * 1. Binary Matrices ({0, 1} with strictly '>' probability threshold)
 * 2. Generic Continuous Numerical Values (any real numbers)
 * 3. Standalone PCA Basis import (upload pca_basis.json)
 */
import React, { useState, useMemo, useEffect } from 'react';
import { 
  LatentRow, 
  ReconstructedMatrix, 
  OutputFormatMode, 
  MatrixValueType,
  MatrixItem
} from '../lib/types';
import { MatrixPCA, PCA_BASIS_SCHEMA_VERSION } from '../lib/autoencoder';
import { 
  parseLatentCsv,
  formatSparseCooCsv,
  formatCombinedDenseCsv,
  formatSingleDenseCsv,
  formatFastaSequences,
  formatFlattenedCsv,
  downloadBlob
} from '../lib/matrixFormats';
import { 
  SAMPLE_45X45_LATENT_CSV,
  SAMPLE_LATENT_CSV, 
  SAMPLE_CONTINUOUS_LATENT_CSV 
} from '../lib/samples';
import { MatrixHeatmap } from './MatrixHeatmap';
import { DesignValidationPanel } from './DesignValidationPanel';
import JSZip from 'jszip';
import { 
  Upload, 
  Download, 
  Sparkles, 
  Sliders, 
  FileCode, 
  FileText,
  Info,
  Grid,
  CheckSquare,
  Square,
  Search,
  ChevronLeft,
  ChevronRight,
  Archive,
  ArrowDownToLine,
  Binary,
  Hash,
  Scale,
  Check,
  CheckCircle2,
  AlertTriangle,
  FolderDown,
  ShieldCheck
} from 'lucide-react';

interface LatentDecoderModuleProps {
  autoencoder: MatrixPCA;
  initialLatentCsv?: string;
  groundTruthMatrices?: MatrixItem[];
  initialDimensions?: { rows: number; cols: number };
}

export const LatentDecoderModule: React.FC<LatentDecoderModuleProps> = ({
  autoencoder,
  initialLatentCsv,
  groundTruthMatrices,
  initialDimensions,
}) => {
  const [latentRows, setLatentRows] = useState<LatentRow[]>([]);
  const [activeModel, setActiveModel] = useState<MatrixPCA>(autoencoder);
  const [decoderError, setDecoderError] = useState<string | null>(null);
  
  // Matrix Value Mode: Binary {0, 1} vs Continuous numerical values
  const [matrixValueType, setMatrixValueType] = useState<MatrixValueType>('binary_01');
  
  // Binary mode controls: Strictly '>' threshold
  const [threshold, setThreshold] = useState<number>(0.50);
  const [decimalPrecision, setDecimalPrecision] = useState<number>(4);

  // File output options
  const [outputFormat, setOutputFormat] = useState<OutputFormatMode>('dense_zip');
  const [selectedMatrixId, setSelectedMatrixId] = useState<string | null>(null);
  const [pasteText, setPasteText] = useState<string>('');
  const [isPasteOpen, setIsPasteOpen] = useState<boolean>(false);
  const [showContinuousView, setShowContinuousView] = useState<boolean>(false);
  const [sourceLatentCsv, setSourceLatentCsv] = useState<string>(initialLatentCsv || '');
  const [isValidationVisible, setIsValidationVisible] = useState<boolean>(false);

  // Target reconstruction dimensions
  const [targetRows, setTargetRows] = useState<number>(initialDimensions?.rows || autoencoder.rows || 45);
  const [targetCols, setTargetCols] = useState<number>(initialDimensions?.cols || autoencoder.cols || 45);
  const [detectedDimensionNotice, setDetectedDimensionNotice] = useState<string | null>(null);
  const [basisStatusNotice, setBasisStatusNotice] = useState<string | null>(null);

  // Basis file state
  const [basisFileName, setBasisFileName] = useState<string | null>(null);

  // Selected row IDs for batch download selection
  const [selectedRowIds, setSelectedRowIds] = useState<Set<string>>(new Set());

  // Search & Pagination
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [currentPage, setCurrentPage] = useState<number>(1);
  const pageSize = 25;

  // Sync dimensions when initialDimensions changes
  useEffect(() => {
    if (initialDimensions) {
      setTargetRows(initialDimensions.rows);
      setTargetCols(initialDimensions.cols);
    }
  }, [initialDimensions]);

  // Apply dimension update
  const handleUpdateDimensions = (r: number, c: number) => {
    const validR = Math.max(1, Math.min(2000, r));
    const validC = Math.max(1, Math.min(2000, c));
    setTargetRows(validR);
    setTargetCols(validC);
  };

  const validateLatentRows = (parsed: LatentRow[], model: MatrixPCA): string | null => {
    if (parsed.length === 0) return 'No valid latent rows were found in the CSV.';
    const expectedK = parsed[0].z.length;
    for (const row of parsed) {
      if (row.z.length !== expectedK) return `Latent row "${row.id}" has a different K value.`;
      const invalidIndex = row.z.findIndex(value => !Number.isFinite(value) || value < 0 || value > 1);
      if (invalidIndex >= 0) return `Row "${row.id}" has an invalid z${invalidIndex + 1}; every coordinate must be in [0,1].`;
    }

    if (!model.hasDecodingBasis) return 'Latent CSV loaded. Upload its matching trained basis JSON to decode it.';
    const modelK = model.components?.length ?? model.k;
    if (expectedK !== modelK) return `Latent CSV uses K=${expectedK}, but the loaded basis requires K=${modelK}.`;

    const metadata = parsed[0];
    if (metadata.detectedK && metadata.detectedK !== expectedK) {
      return `Latent CSV declares K=${metadata.detectedK}, but contains ${expectedK} z columns.`;
    }
    if (metadata.detectedSchemaVersion && metadata.detectedSchemaVersion !== PCA_BASIS_SCHEMA_VERSION) {
      return `Latent CSV uses unsupported schema version ${metadata.detectedSchemaVersion}.`;
    }
    if (metadata.detectedRows && metadata.detectedCols &&
        (metadata.detectedRows !== model.rows || metadata.detectedCols !== model.cols)) {
      return `Latent CSV is ${metadata.detectedRows}×${metadata.detectedCols}, but the loaded basis is ${model.rows}×${model.cols}.`;
    }
    if (metadata.detectedModelId && metadata.detectedModelId !== model.modelId) {
      return `Model mismatch: CSV requires ${metadata.detectedModelId}, but the loaded basis is ${model.modelId}.`;
    }
    return null;
  };

  // Helper to load parsed rows and check for embedded metadata
  const applyLoadedLatentRows = (parsed: LatentRow[], sourceText: string) => {
    if (parsed.length === 0) {
      setDecoderError('No valid latent rows were found in the CSV.');
      return;
    }
    setLatentRows(parsed);
    setSourceLatentCsv(sourceText);
    setIsValidationVisible(false);
    setSelectedMatrixId(parsed[0].id);

    // Default select all rows
    setSelectedRowIds(new Set(parsed.map(p => p.id)));

    const firstRow = parsed[0];

    // If file specifies value type metadata
    if (firstRow.detectedValueType) {
      setMatrixValueType(firstRow.detectedValueType);
    }

    // If file has detected dimensions
    if (firstRow.detectedRows && firstRow.detectedCols) {
      setTargetRows(firstRow.detectedRows);
      setTargetCols(firstRow.detectedCols);
      setDetectedDimensionNotice(
        `Detected from CSV: ${firstRow.detectedRows} × ${firstRow.detectedCols} (K=${firstRow.detectedK || firstRow.z.length}, model ${firstRow.detectedModelId || 'legacy/unidentified'}).`
      );
      setTimeout(() => setDetectedDimensionNotice(null), 6000);
    }
    setDecoderError(validateLatentRows(parsed, activeModel));
  };

  // If initial CSV passed from Module 1 transfer
  useEffect(() => {
    if (initialLatentCsv) {
      const parsed = parseLatentCsv(initialLatentCsv);
      applyLoadedLatentRows(parsed, initialLatentCsv);
    }
  }, [initialLatentCsv]);

  // When latent CSV files are uploaded
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const text = await file.text();
    const parsed = parseLatentCsv(text);
    applyLoadedLatentRows(parsed, text);
    e.target.value = '';
  };

  // When PCA Basis JSON is uploaded
  const loadBasisText = (text: string, fileName: string) => {
    const candidate = new MatrixPCA(1, 1, 1);
    const success = candidate.loadBasis(text);
    if (!success) {
      setDecoderError(candidate.lastError || 'Invalid PCA basis JSON format.');
      return false;
    }

    setActiveModel(candidate);
    setBasisFileName(fileName);
    setTargetRows(candidate.rows);
    setTargetCols(candidate.cols);
    setMatrixValueType(candidate.valueType);
    setDecoderError(latentRows.length > 0 ? validateLatentRows(latentRows, candidate) : null);
    setBasisStatusNotice(
      `Loaded PCA basis ${candidate.modelId}: ${candidate.rows}×${candidate.cols}, K=${candidate.components?.length || candidate.k}.`
    );
    setTimeout(() => setBasisStatusNotice(null), 6000);
    return true;
  };

  const handleBasisFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      const text = await file.text();
      loadBasisText(text, file.name);
    } catch (error) {
      setDecoderError(error instanceof Error ? error.message : 'Could not read PCA basis JSON file.');
    }
    e.target.value = '';
  };

  // Drag and drop handler for latent CSV
  const handleDrop = async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (!file) return;

    if (file.name.endsWith('.json')) {
      const text = await file.text();
      loadBasisText(text, file.name);
      return;
    }

    const text = await file.text();
    const parsed = parseLatentCsv(text);
    applyLoadedLatentRows(parsed, text);
  };

  // Apply pasted latent CSV
  const handleApplyPaste = () => {
    if (!pasteText.trim()) return;
    const parsed = parseLatentCsv(pasteText);
    applyLoadedLatentRows(parsed, pasteText);
    setPasteText('');
    setIsPasteOpen(false);
  };

  // Load sample 45x45 BO candidates
  const handleLoad45x45Sample = () => {
    setMatrixValueType('binary_01');
    handleUpdateDimensions(45, 45);
    const parsed = parseLatentCsv(SAMPLE_45X45_LATENT_CSV);
    applyLoadedLatentRows(parsed, SAMPLE_45X45_LATENT_CSV);
  };

  // Load sample binary latent CSV (200x21)
  const handleLoadBinarySample = () => {
    setMatrixValueType('binary_01');
    handleUpdateDimensions(200, 21);
    const parsed = parseLatentCsv(SAMPLE_LATENT_CSV);
    applyLoadedLatentRows(parsed, SAMPLE_LATENT_CSV);
  };

  // Load sample continuous numerical latent CSV (10x10)
  const handleLoadContinuousSample = () => {
    setMatrixValueType('continuous_numeric');
    handleUpdateDimensions(10, 10);
    const parsed = parseLatentCsv(SAMPLE_CONTINUOUS_LATENT_CSV);
    applyLoadedLatentRows(parsed, SAMPLE_CONTINUOUS_LATENT_CSV);
  };

  // Ground truth map
  const gtMap = useMemo(() => {
    const map = new Map<string, Float32Array>();
    if (groundTruthMatrices) {
      groundTruthMatrices.forEach(m => {
        map.set(m.id, m.data);
      });
    }
    return map;
  }, [groundTruthMatrices]);

  const compatibilityError = latentRows.length > 0
    ? validateLatentRows(latentRows, activeModel) || (
      activeModel.hasDecodingBasis &&
      (targetRows !== activeModel.rows || targetCols !== activeModel.cols)
        ? `Output dimensions must match the loaded basis (${activeModel.rows}×${activeModel.cols}).`
        : null
    )
    : null;

  // Reconstruct all matrices based on options and target dimensions
  const reconstructedMatrices: ReconstructedMatrix[] = useMemo(() => {
    if (compatibilityError || !activeModel.hasDecodingBasis) return [];

    return latentRows.map(row => {
      const gt = gtMap.get(row.id);
      return activeModel.reconstruct(
        row.id, 
        row.z, 
        { threshold, valueType: matrixValueType },
        matrixValueType,
        gt
      );
    });
  }, [
    latentRows, 
    matrixValueType, 
    threshold, 
    targetRows, 
    targetCols, 
    gtMap,
    activeModel,
    compatibilityError,
  ]);

  // Toggle selection for a single row
  const toggleRowSelection = (id: string) => {
    setSelectedRowIds(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  // Select all or none
  const toggleSelectAll = () => {
    if (selectedRowIds.size === reconstructedMatrices.length) {
      setSelectedRowIds(new Set());
    } else {
      setSelectedRowIds(new Set(reconstructedMatrices.map(m => m.id)));
    }
  };

  // Filtered & Paginated reconstructed matrices
  const filteredMatrices = useMemo(() => {
    if (!searchQuery.trim()) return reconstructedMatrices;
    const q = searchQuery.toLowerCase();
    return reconstructedMatrices.filter(m => m.id.toLowerCase().includes(q));
  }, [reconstructedMatrices, searchQuery]);

  const totalPages = Math.ceil(filteredMatrices.length / pageSize) || 1;
  const paginatedMatrices = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return filteredMatrices.slice(start, start + pageSize);
  }, [filteredMatrices, currentPage, pageSize]);

  // Helper to format a single matrix file content
  const getSingleMatrixFileContent = (m: ReconstructedMatrix): { content: string; filename: string; mime: string } => {
    if (outputFormat === 'sparse_coo_csv') {
      const csv = formatSparseCooCsv([m], decimalPrecision);
      return { content: csv, filename: `${m.id}_sparse_coo.csv`, mime: 'text/csv' };
    }
    if (outputFormat === 'fasta_sequence' && m.cols === 21) {
      const fasta = formatFastaSequences([m]);
      return { content: fasta, filename: `${m.id}.fasta`, mime: 'text/plain' };
    }
    if (outputFormat === 'flattened_csv') {
      const flatCsv = formatFlattenedCsv([m], decimalPrecision);
      return { content: flatCsv, filename: `${m.id}_flattened.csv`, mime: 'text/csv' };
    }
    if (outputFormat === 'json') {
      const json = JSON.stringify({
        id: m.id,
        rows: m.rows,
        cols: m.cols,
        valueType: m.valueType,
        minVal: m.minVal,
        maxVal: m.maxVal,
        activeCount: m.nonZeroCount,
        sparsityPercent: m.sparsityPercent,
        data: Array.from(m.data)
      }, null, 2);
      return { content: json, filename: `${m.id}.json`, mime: 'application/json' };
    }
    // Default: Dense 2D CSV table
    const denseCsv = formatSingleDenseCsv(m, decimalPrecision);
    return { content: denseCsv, filename: `${m.id}_${m.rows}x${m.cols}.csv`, mime: 'text/csv' };
  };

  // Download a single specific matrix file directly
  const handleDownloadSingleFile = (m: ReconstructedMatrix) => {
    const { content, filename, mime } = getSingleMatrixFileContent(m);
    const blob = new Blob([content], { type: `${mime};charset=utf-8;` });
    downloadBlob(blob, filename);
  };

  // Download a collection of files as a ZIP (either ALL or SELECTED)
  const handleDownloadZipCollection = async (onlySelected: boolean = false) => {
    const targetMatrices = onlySelected 
      ? reconstructedMatrices.filter(m => selectedRowIds.has(m.id))
      : reconstructedMatrices;

    if (targetMatrices.length === 0) return;

    const zip = new JSZip();
    for (const m of targetMatrices) {
      const { content, filename } = getSingleMatrixFileContent(m);
      zip.file(filename, content);
    }

    const zipBlob = await zip.generateAsync({ type: 'blob' });
    const label = onlySelected ? `selected_${targetMatrices.length}` : `all_${targetMatrices.length}`;
    const modeTag = matrixValueType === 'binary_01' ? 'binary' : 'numeric';
    const filename = `reconstructed_${modeTag}_matrices_${label}_files_${targetRows}x${targetCols}.zip`;
    downloadBlob(zipBlob, filename);
  };

  // Combined single CSV download option
  const handleDownloadCombinedCsv = () => {
    const targetMatrices = reconstructedMatrices.filter(m => selectedRowIds.has(m.id));
    const list = targetMatrices.length > 0 ? targetMatrices : reconstructedMatrices;
    const content = formatCombinedDenseCsv(list, decimalPrecision);
    const blob = new Blob([content], { type: 'text/csv;charset=utf-8;' });
    const modeTag = matrixValueType === 'binary_01' ? 'binary' : 'numeric';
    const filename = `combined_${modeTag}_matrices_${list.length}_files_${targetRows}x${targetCols}.csv`;
    downloadBlob(blob, filename);
  };

  const selectedMatrix = reconstructedMatrices.find(m => m.id === selectedMatrixId) || reconstructedMatrices[0];
  const visibleDecoderError = compatibilityError || decoderError;

  return (
    <div className="flex flex-col gap-6">
      {/* Banner & Upload for Module 2 */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-xs">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-5 border-b border-slate-200">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold text-emerald-700 uppercase tracking-wider">Module 2</span>
              <span className="text-slate-300">·</span>
              <h2 className="text-lg font-bold text-slate-900">
                Latent Vector Decoder &amp; Matrix File Generator
              </h2>
            </div>
            <p className="text-xs text-slate-600 mt-1">
              Upload candidate latent coordinates <strong>z ∈ [0, 1]^K</strong> (e.g. proposed by Bayesian Optimizer). Reconstructs full matrices using strictly <strong>&apos;&gt;&apos; threshold</strong> and outputs individual or collective files.
            </p>
          </div>

          {/* Preset Sample Buttons */}
          <div className="flex flex-wrap items-center gap-2">
            <button
              onClick={handleLoad45x45Sample}
              className="text-xs px-3 py-1.5 rounded-lg bg-emerald-50 hover:bg-emerald-100 text-emerald-900 font-semibold transition-colors border border-emerald-300 flex items-center gap-1.5 shadow-2xs"
              title="Test with 45x45 Binary Bayesian Optimization candidates"
            >
              <Sparkles className="w-3.5 h-3.5 text-emerald-700" />
              Load 45×45 BO Candidates
            </button>
            <button
              onClick={handleLoadContinuousSample}
              className="text-xs px-3 py-1.5 rounded-lg bg-slate-50 hover:bg-slate-100 text-slate-700 font-medium transition-colors border border-slate-200 flex items-center gap-1.5 shadow-2xs"
              title="Test with continuous numerical matrices (10x10 floats)"
            >
              <Hash className="w-3.5 h-3.5 text-slate-600" />
              10×10 Floats
            </button>
            <button
              onClick={handleLoadBinarySample}
              className="text-xs px-3 py-1.5 rounded-lg bg-slate-50 hover:bg-slate-100 text-slate-700 font-medium transition-colors border border-slate-200 flex items-center gap-1.5 shadow-2xs"
              title="Test with binary 0/1 matrices (200x21)"
            >
              <Binary className="w-3.5 h-3.5 text-slate-600" />
              200×21 Protein
            </button>
          </div>
        </div>

        {/* Basis Status & Upload Bar */}
        <div className="pt-4 flex flex-col md:flex-row md:items-center justify-between gap-3 text-xs">
          <div className="flex items-center gap-2">
            <FileCode className="w-4 h-4 text-emerald-700" />
            <span className="font-semibold text-slate-800">PCA Basis Status:</span>
            {activeModel.hasDecodingBasis ? (
              <span className="px-2 py-0.5 bg-emerald-50 border border-emerald-300 rounded font-medium text-emerald-900 flex items-center gap-1">
                <Check className="w-3 h-3 text-emerald-600" />
                Active ({activeModel.rows}×{activeModel.cols}, K={activeModel.components?.length || activeModel.k}, model {activeModel.modelId})
                {basisFileName && <span className="text-slate-500 font-mono">[{basisFileName}]</span>}
              </span>
            ) : (
              <span className="px-2 py-0.5 bg-amber-50 border border-amber-300 rounded font-medium text-amber-900 flex items-center gap-1">
                <AlertTriangle className="w-3 h-3 text-amber-600" />
                No basis loaded (Module 1 active or upload basis JSON)
              </span>
            )}
          </div>

          <div className="flex items-center gap-2">
            <label className="cursor-pointer px-3 py-1.5 bg-slate-50 hover:bg-slate-100 border border-slate-200 text-slate-700 rounded-lg text-xs font-medium flex items-center gap-1.5 transition-colors shadow-2xs">
              <Upload className="w-3.5 h-3.5 text-slate-500" />
              <span>Upload Basis JSON</span>
              <input
                type="file"
                accept=".json"
                onChange={handleBasisFileUpload}
                className="hidden"
              />
            </label>

            <button
              onClick={() => setIsPasteOpen(!isPasteOpen)}
              className="px-3 py-1.5 bg-slate-50 hover:bg-slate-100 border border-slate-200 text-slate-700 rounded-lg text-xs font-medium flex items-center gap-1.5 transition-colors shadow-2xs"
            >
              <FileText className="w-3.5 h-3.5 text-slate-500" />
              <span>{isPasteOpen ? 'Hide Paste' : 'Paste Latent CSV'}</span>
            </button>
          </div>
        </div>

        {basisStatusNotice && (
          <div className="mt-3 p-2.5 bg-emerald-50 border border-emerald-200 rounded-lg text-xs text-emerald-800 flex items-center gap-2">
            <Check className="w-4 h-4 text-emerald-600 shrink-0" />
            <span>{basisStatusNotice}</span>
          </div>
        )}

        {visibleDecoderError && (
          <div className="mt-3 p-3 bg-amber-50 border border-amber-300 rounded-lg text-xs text-amber-950 flex items-start gap-2">
            <AlertTriangle className="w-4 h-4 text-amber-700 shrink-0 mt-0.5" />
            <span>{visibleDecoderError}</span>
          </div>
        )}

        {isPasteOpen && (
          <div className="mt-4 p-4 bg-slate-50 border border-slate-200 rounded-lg flex flex-col gap-2">
            <textarea
              value={pasteText}
              onChange={e => setPasteText(e.target.value)}
              placeholder="Paste latent CSV lines (e.g. matrix_1, 0.45, 0.12, 0.88, ...)..."
              rows={4}
              className="w-full bg-white border border-slate-300 rounded p-2 text-xs font-mono text-slate-800 focus:outline-hidden focus:border-emerald-600"
            />
            <div className="flex justify-end gap-2">
              <button
                onClick={() => setIsPasteOpen(false)}
                className="px-3 py-1 text-xs text-slate-500 hover:text-slate-800"
              >
                Cancel
              </button>
              <button
                onClick={handleApplyPaste}
                className="px-3 py-1 bg-emerald-700 hover:bg-emerald-800 text-white rounded text-xs font-medium shadow-2xs"
              >
                Decode Vectors
              </button>
            </div>
          </div>
        )}

        {/* Upload Drop Zone if empty */}
        {latentRows.length === 0 && (
          <div
            onDragOver={e => e.preventDefault()}
            onDrop={handleDrop}
            className="mt-4 border-2 border-dashed border-slate-300 hover:border-emerald-600 bg-slate-50/70 hover:bg-emerald-50/20 rounded-xl p-8 text-center transition-all cursor-pointer relative group"
          >
            <input
              type="file"
              accept=".csv,.tsv,.txt,.json"
              onChange={handleFileUpload}
              className="absolute inset-0 opacity-0 cursor-pointer w-full h-full"
            />
            <div className="flex flex-col items-center gap-2.5">
              <div className="w-11 h-11 rounded-full bg-white border border-slate-200 flex items-center justify-center text-emerald-700 group-hover:scale-105 shadow-xs transition-transform">
                <Upload className="w-5 h-5" />
              </div>
              <div className="text-sm font-semibold text-slate-800">
                Drop your latent CSV file here, or <span className="text-emerald-700 underline">browse file</span>
              </div>
              <p className="text-xs text-slate-500 max-w-md">
                Expects rows with latent coordinates <code>[z₁ .. z_K]</code> in range <code>[0.0, 1.0]</code>.
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Latent Vectors Loaded -> Controls & Reconstruction */}
      {latentRows.length > 0 && !compatibilityError && activeModel.hasDecodingBasis && (
        <div className="flex flex-col gap-6">
          {/* Target Dimensions & Mode Settings Bar */}
          <div className="p-4 bg-white border border-slate-200 rounded-xl flex flex-wrap items-center justify-between gap-4 text-xs shadow-xs">
            {/* Value Mode Toggle */}
            <div className="flex items-center gap-3">
              <span className="font-semibold text-slate-800">Mode:</span>
              <div className="inline-flex rounded-lg border border-slate-200 bg-slate-100 p-0.5">
                <button
                  onClick={() => setMatrixValueType('binary_01')}
                  className={`px-3 py-1 rounded-md font-medium text-xs transition-all ${
                    matrixValueType === 'binary_01'
                      ? 'bg-white text-emerald-800 shadow-2xs font-bold'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  Binary {`{0, 1}`}
                </button>
                <button
                  onClick={() => {
                    setMatrixValueType('continuous_numeric');
                    setIsValidationVisible(false);
                  }}
                  className={`px-3 py-1 rounded-md font-medium text-xs transition-all ${
                    matrixValueType === 'continuous_numeric'
                      ? 'bg-white text-cyan-800 shadow-2xs font-bold'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  Continuous Numeric
                </button>
              </div>
            </div>

            {/* Strict Threshold Slider (Binary Mode) */}
            {matrixValueType === 'binary_01' && (
              <div className="flex items-center gap-3 bg-slate-50 border border-slate-200 rounded-lg px-3 py-1.5 shadow-2xs">
                <span className="font-semibold text-slate-800">
                  Binary Threshold:
                </span>
                <input
                  type="range"
                  min="0.0"
                  max="1.0"
                  step="0.01"
                  value={threshold}
                  onChange={e => setThreshold(parseFloat(e.target.value))}
                  className="w-28 accent-emerald-600 cursor-pointer"
                />
                <span className="font-mono font-bold text-emerald-800 bg-white px-2 py-0.5 rounded border border-slate-200 text-xs">
                  θ = {threshold.toFixed(2)}
                </span>
                <span className="text-[11px] text-slate-500 font-mono">
                  (strictly &gt; θ)
                </span>
              </div>
            )}

            {/* Target Dimensions */}
            <div className="flex items-center gap-2">
              <Grid className="w-4 h-4 text-slate-400" />
              <span className="text-slate-600">Reconstructing:</span>
              <span className="font-mono font-bold text-slate-900">
                {targetRows} × {targetCols} ({targetRows * targetCols} cells)
              </span>
            </div>
          </div>

          {detectedDimensionNotice && (
            <div className="p-2.5 bg-emerald-50 border border-emerald-200 rounded-lg text-xs text-emerald-800 flex items-center gap-2">
              <Check className="w-4 h-4 text-emerald-600 shrink-0" />
              <span>{detectedDimensionNotice}</span>
            </div>
          )}

          {/* Main Inspection Area: Matrix List & Selected Heatmap */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Matrix List Column */}
            <div className="lg:col-span-5 bg-white border border-slate-200 rounded-xl p-5 shadow-xs flex flex-col">
              <div className="flex items-center justify-between pb-3 border-b border-slate-200 mb-3">
                <div className="flex items-center gap-2">
                  <button
                    onClick={toggleSelectAll}
                    className="text-slate-500 hover:text-slate-800"
                    title={selectedRowIds.size === reconstructedMatrices.length ? 'Deselect All' : 'Select All'}
                  >
                    {selectedRowIds.size === reconstructedMatrices.length ? (
                      <CheckSquare className="w-4 h-4 text-emerald-600" />
                    ) : (
                      <Square className="w-4 h-4 text-slate-400" />
                    )}
                  </button>
                  <h4 className="text-xs font-bold uppercase tracking-wider text-slate-700">
                    Reconstructed ({selectedRowIds.size}/{reconstructedMatrices.length})
                  </h4>
                </div>

                <div className="relative w-36">
                  <input
                    type="text"
                    placeholder="Search..."
                    value={searchQuery}
                    onChange={e => {
                      setSearchQuery(e.target.value);
                      setCurrentPage(1);
                    }}
                    className="w-full bg-slate-50 border border-slate-200 rounded-md pl-6 pr-2 py-1 text-xs text-slate-800 placeholder-slate-400"
                  />
                  <Search className="w-3.5 h-3.5 text-slate-400 absolute left-1.5 top-2" />
                </div>
              </div>

              {/* Rows List */}
              <div className="space-y-1.5 max-h-96 overflow-y-auto pr-1">
                {paginatedMatrices.map(m => {
                  const isSelected = m.id === selectedMatrixId;
                  const isChecked = selectedRowIds.has(m.id);

                  return (
                    <div
                      key={m.id}
                      onClick={() => setSelectedMatrixId(m.id)}
                      className={`p-2.5 rounded-lg border text-xs cursor-pointer transition-all flex items-center justify-between ${
                        isSelected
                          ? 'bg-emerald-50 border-emerald-400 shadow-2xs font-semibold text-emerald-950'
                          : 'bg-white border-slate-200 hover:border-slate-300 text-slate-700'
                      }`}
                    >
                      <div className="flex items-center gap-2 truncate">
                        <button
                          type="button"
                          onClick={e => {
                            e.stopPropagation();
                            toggleRowSelection(m.id);
                          }}
                          className="shrink-0"
                        >
                          {isChecked ? (
                            <CheckSquare className="w-3.5 h-3.5 text-emerald-600" />
                          ) : (
                            <Square className="w-3.5 h-3.5 text-slate-300" />
                          )}
                        </button>
                        <span className="truncate">{m.id}</span>
                      </div>

                      <div className="flex items-center gap-2 text-[11px] font-mono shrink-0">
                        {m.metrics?.hammingAccuracy !== undefined ? (
                          <span className="text-emerald-700 font-bold">
                            {(m.metrics.hammingAccuracy * 100).toFixed(1)}% acc
                          </span>
                        ) : (
                          <span className="text-slate-400">
                            {m.nonZeroCount} active
                          </span>
                        )}
                        <button
                          type="button"
                          onClick={e => {
                            e.stopPropagation();
                            handleDownloadSingleFile(m);
                          }}
                          className="p-1 text-slate-400 hover:text-emerald-700 rounded hover:bg-slate-100"
                          title="Download this single matrix file"
                        >
                          <Download className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Pagination */}
              {totalPages > 1 && (
                <div className="flex items-center justify-between pt-3 border-t border-slate-200 mt-3 text-xs text-slate-500">
                  <span>Page {currentPage} of {totalPages}</span>
                  <div className="flex items-center gap-1">
                    <button
                      disabled={currentPage <= 1}
                      onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                      className="p-1 rounded bg-slate-100 hover:bg-slate-200 disabled:opacity-40"
                    >
                      <ChevronLeft className="w-3.5 h-3.5" />
                    </button>
                    <button
                      disabled={currentPage >= totalPages}
                      onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                      className="p-1 rounded bg-slate-100 hover:bg-slate-200 disabled:opacity-40"
                    >
                      <ChevronRight className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* Selected Matrix Heatmap & Inspection */}
            <div className="lg:col-span-7 bg-white border border-slate-200 rounded-xl p-5 shadow-xs flex flex-col justify-between">
              {selectedMatrix && (
                <div>
                  <div className="flex items-center justify-between pb-3 border-b border-slate-200 mb-3">
                    <div className="flex items-center gap-2">
                      <h4 className="text-xs font-bold uppercase tracking-wider text-slate-700">
                        Reconstructed: <span className="text-emerald-900 font-mono">{selectedMatrix.id}</span>
                      </h4>
                    </div>

                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => setShowContinuousView(!showContinuousView)}
                        className={`px-2 py-0.5 rounded text-[11px] font-medium border transition-colors ${
                          showContinuousView 
                            ? 'bg-cyan-50 border-cyan-300 text-cyan-800' 
                            : 'bg-slate-50 border-slate-200 text-slate-600'
                        }`}
                        title="Show continuous pre-threshold inverse projection values"
                      >
                        {showContinuousView ? 'Showing Pre-Threshold' : 'Show Pre-Threshold'}
                      </button>
                      <button
                        onClick={() => handleDownloadSingleFile(selectedMatrix)}
                        className="px-2 py-0.5 rounded bg-emerald-700 hover:bg-emerald-800 text-white text-[11px] font-semibold flex items-center gap-1 shadow-2xs"
                      >
                        <Download className="w-3 h-3" />
                        Download File
                      </button>
                    </div>
                  </div>

                  {/* Heatmap */}
                  <div className="flex justify-center bg-slate-50 p-2 rounded-lg border border-slate-200 mb-4">
                    <MatrixHeatmap
                      rows={selectedMatrix.rows}
                      cols={selectedMatrix.cols}
                      data={showContinuousView ? selectedMatrix.continuous : selectedMatrix.data}
                      valueType={showContinuousView ? 'continuous_numeric' : selectedMatrix.valueType}
                      threshold={threshold}
                      groundTruth={selectedMatrix.groundTruth}
                    />
                  </div>

                  {/* Evaluation Metrics Card (if ground truth available) */}
                  {selectedMatrix.metrics?.hammingAccuracy !== undefined && (
                    <div className="p-3 bg-emerald-50 border border-emerald-200 rounded-lg text-xs grid grid-cols-2 sm:grid-cols-4 gap-2 text-center font-mono">
                      <div>
                        <div className="text-[10px] text-slate-500 font-sans">Accuracy</div>
                        <div className="font-bold text-emerald-800 text-sm">
                          {(selectedMatrix.metrics.hammingAccuracy * 100).toFixed(2)}%
                        </div>
                      </div>
                      <div>
                        <div className="text-[10px] text-slate-500 font-sans">Differing Cells</div>
                        <div className="font-bold text-slate-800 text-sm">
                          {selectedMatrix.metrics.hammingDistance} / {selectedMatrix.rows * selectedMatrix.cols}
                        </div>
                      </div>
                      <div>
                        <div className="text-[10px] text-slate-500 font-sans">Precision</div>
                        <div className="font-bold text-slate-800 text-sm">
                          {((selectedMatrix.metrics.precision || 0) * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div>
                        <div className="text-[10px] text-slate-500 font-sans">F1 Score</div>
                        <div className="font-bold text-slate-800 text-sm">
                          {((selectedMatrix.metrics.f1 || 0) * 100).toFixed(1)}%
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Binary-only design validation */}
          {matrixValueType === 'binary_01' && (
            <>
              <div className="flex flex-col gap-4 rounded-xl border border-violet-200 bg-gradient-to-r from-violet-50 via-white to-fuchsia-50 p-5 shadow-xs sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <h3 className="flex items-center gap-2 text-sm font-bold text-slate-900">
                    <ShieldCheck className="h-4 w-4 text-violet-700" />
                    Design Validation
                  </h3>
                  <p className="mt-1 max-w-3xl text-xs leading-5 text-slate-600">
                    Check whether every 1-cell is connected and whether at least two distinct corners contain an all-one 2×2 block. Validation is available only for binary matrices.
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => setIsValidationVisible(visible => !visible)}
                  className="inline-flex shrink-0 items-center justify-center gap-1.5 rounded-lg bg-violet-700 px-4 py-2 text-xs font-semibold text-white shadow-xs transition-colors hover:bg-violet-800"
                >
                  <ShieldCheck className="h-4 w-4" />
                  {isValidationVisible ? 'Hide Validation' : 'Validate'}
                </button>
              </div>

              {isValidationVisible && (
                <DesignValidationPanel
                  matrices={reconstructedMatrices}
                  latentRows={latentRows}
                  sourceLatentCsv={sourceLatentCsv}
                  threshold={threshold}
                  decimalPrecision={decimalPrecision}
                />
              )}
            </>
          )}

          {/* Batch Download Bar */}
          <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-xs flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
                <FolderDown className="w-4 h-4 text-emerald-700" />
                Batch Export Reconstructed Matrix Files
              </h3>
              <p className="text-xs text-slate-500 mt-0.5">
                Download collection of <strong>{selectedRowIds.size} selected</strong> (out of {reconstructedMatrices.length}) matrices as individual files or combined CSV.
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-2.5">
              <select
                value={outputFormat}
                onChange={e => setOutputFormat(e.target.value as OutputFormatMode)}
                className="bg-slate-50 border border-slate-300 rounded-lg px-2.5 py-1.5 text-xs text-slate-800 font-medium"
              >
                <option value="dense_zip">ZIP of Dense 2D CSV Tables</option>
                <option value="dense_combined_csv">Single Combined CSV</option>
                <option value="sparse_coo_csv">Sparse COO CSV (row,col,val)</option>
                <option value="flattened_csv">Flattened CSV (1 row per matrix)</option>
                <option value="json">JSON Array of Matrices</option>
              </select>

              {outputFormat === 'dense_combined_csv' ? (
                <button
                  onClick={handleDownloadCombinedCsv}
                  className="px-4 py-2 bg-emerald-700 hover:bg-emerald-800 text-white rounded-lg text-xs font-semibold shadow-xs flex items-center gap-1.5 transition-colors"
                >
                  <Download className="w-4 h-4" />
                  Download Combined CSV
                </button>
              ) : (
                <button
                  onClick={() => handleDownloadZipCollection(selectedRowIds.size < reconstructedMatrices.length && selectedRowIds.size > 0)}
                  className="px-4 py-2 bg-emerald-700 hover:bg-emerald-800 text-white rounded-lg text-xs font-semibold shadow-xs flex items-center gap-1.5 transition-colors"
                >
                  <Archive className="w-4 h-4" />
                  <span>
                    Download {selectedRowIds.size < reconstructedMatrices.length && selectedRowIds.size > 0 ? `Selected (${selectedRowIds.size})` : `All (${reconstructedMatrices.length})`} as ZIP
                  </span>
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
