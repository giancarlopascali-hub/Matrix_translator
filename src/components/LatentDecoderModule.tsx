/**
 * Module 2: Latent Decoder & Reconstruction Module (Bright Theme)
 * Load a CSV file with multiple rows (each row = 1 matrix with 8 latent dimensions),
 * reconstructs the collection of matrices, and allows downloading individual files
 * for each row (all of them, or a user-defined selection).
 *
 * Supports BOTH:
 * 1. Generic Numerical Values (Any real continuous numbers: floats, integers, positive/negative)
 * 2. Binary Matrices ({0, 1} with user-defined probability threshold)
 */
import React, { useState, useMemo, useEffect } from 'react';
import { 
  LatentRow, 
  ReconstructedMatrix, 
  OutputFormatMode, 
  DiscretizationMode,
  MatrixValueType,
  DecoderOptions,
  MatrixItem
} from '../lib/types';
import { MatrixAutoencoder } from '../lib/autoencoder';
import { 
  parseLatentCsv,
  formatSparseCooCsv,
  formatCombinedDenseCsv,
  formatSingleDenseCsv,
  formatFastaSequences,
  formatFlattenedCsv
} from '../lib/matrixFormats';
import { SAMPLE_LATENT_CSV, SAMPLE_CONTINUOUS_LATENT_CSV } from '../lib/samples';
import { MatrixHeatmap } from './MatrixHeatmap';
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
  CheckCircle2
} from 'lucide-react';

interface LatentDecoderModuleProps {
  autoencoder: MatrixAutoencoder;
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
  
  // Matrix Value Mode: Generic continuous numerical values vs Binary {0, 1}
  const [matrixValueType, setMatrixValueType] = useState<MatrixValueType>('continuous_numeric');
  
  // Binary mode controls
  const [threshold, setThreshold] = useState<number>(0.50);
  const [discretizationMode, setDiscretizationMode] = useState<DiscretizationMode>('threshold');
  
  // Continuous Generic Numerical Values controls
  const [continuousRangeMode, setContinuousRangeMode] = useState<'raw' | 'custom_range' | 'positive_only'>('raw');
  const [customMin, setCustomMin] = useState<number>(0.0);
  const [customMax, setCustomMax] = useState<number>(10.0);
  const [sparsityCutoff, setSparsityCutoff] = useState<number>(0.0); // Zero-out threshold for continuous values
  const [decimalPrecision, setDecimalPrecision] = useState<number>(4);

  // File output options
  const [outputFormat, setOutputFormat] = useState<OutputFormatMode>('dense_zip');
  const [selectedMatrixId, setSelectedMatrixId] = useState<string | null>(null);
  const [pasteText, setPasteText] = useState<string>('');
  const [isPasteOpen, setIsPasteOpen] = useState<boolean>(false);
  const [showProbabilitiesView, setShowProbabilitiesView] = useState<boolean>(false);

  // Target reconstruction dimensions
  const [targetRows, setTargetRows] = useState<number>(initialDimensions?.rows || autoencoder.config.inputRows);
  const [targetCols, setTargetCols] = useState<number>(initialDimensions?.cols || autoencoder.config.inputCols);
  const [detectedDimensionNotice, setDetectedDimensionNotice] = useState<string | null>(null);
  const [calibVersion, setCalibVersion] = useState<number>(0);

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
      autoencoder.reconfigure(initialDimensions.rows, initialDimensions.cols);
    }
  }, [initialDimensions, autoencoder]);

  // Apply dimension update
  const handleUpdateDimensions = (r: number, c: number) => {
    const validR = Math.max(1, Math.min(2000, r));
    const validC = Math.max(1, Math.min(2000, c));
    setTargetRows(validR);
    setTargetCols(validC);
    autoencoder.reconfigure(validR, validC);
  };

  // Helper to load parsed rows and check for embedded metadata
  const applyLoadedLatentRows = (parsed: LatentRow[]) => {
    if (parsed.length === 0) return;
    setLatentRows(parsed);
    setSelectedMatrixId(parsed[0].id);

    // Default select all rows
    setSelectedRowIds(new Set(parsed.map(p => p.id)));

    const firstRow = parsed[0];

    // If file specifies value type metadata
    if (firstRow.detectedValueType) {
      setMatrixValueType(firstRow.detectedValueType);
    }

    // If file has detected range metadata
    if (firstRow.detectedMinVal !== undefined && firstRow.detectedMaxVal !== undefined) {
      setCustomMin(firstRow.detectedMinVal);
      setCustomMax(firstRow.detectedMaxVal);
      if (firstRow.detectedMinVal !== 0 || firstRow.detectedMaxVal !== 1) {
        setContinuousRangeMode('custom_range');
      }
    }

    // If file has detected dimensions
    if (firstRow.detectedRows && firstRow.detectedCols) {
      setTargetRows(firstRow.detectedRows);
      setTargetCols(firstRow.detectedCols);
      autoencoder.reconfigure(firstRow.detectedRows, firstRow.detectedCols);
      setDetectedDimensionNotice(
        `Detected from CSV: ${firstRow.detectedRows} × ${firstRow.detectedCols} (${firstRow.detectedValueType === 'binary_01' ? 'Binary 0/1' : 'Numerical values'}). Autoencoder reconfigured.`
      );
      setTimeout(() => setDetectedDimensionNotice(null), 6000);
    }
  };

  // If initial CSV passed from Module 1 transfer
  useEffect(() => {
    if (initialLatentCsv) {
      const parsed = parseLatentCsv(initialLatentCsv);
      applyLoadedLatentRows(parsed);
    }
  }, [initialLatentCsv]);

  // When files are uploaded
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const text = await file.text();
    const parsed = parseLatentCsv(text);
    applyLoadedLatentRows(parsed);
    e.target.value = '';
  };

  // Drag and drop handler
  const handleDrop = async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (!file) return;

    const text = await file.text();
    const parsed = parseLatentCsv(text);
    applyLoadedLatentRows(parsed);
  };

  // Apply pasted latent CSV
  const handleApplyPaste = () => {
    if (!pasteText.trim()) return;
    const parsed = parseLatentCsv(pasteText);
    applyLoadedLatentRows(parsed);
    setPasteText('');
    setIsPasteOpen(false);
  };

  // Load sample binary latent CSV
  const handleLoadBinarySample = () => {
    setMatrixValueType('binary_01');
    handleUpdateDimensions(200, 21);
    const parsed = parseLatentCsv(SAMPLE_LATENT_CSV);
    applyLoadedLatentRows(parsed);
  };

  // Load sample continuous numerical latent CSV
  const handleLoadContinuousSample = () => {
    setMatrixValueType('continuous_numeric');
    setContinuousRangeMode('custom_range');
    setCustomMin(-4.50);
    setCustomMax(9.10);
    handleUpdateDimensions(10, 10);
    const parsed = parseLatentCsv(SAMPLE_CONTINUOUS_LATENT_CSV);
    applyLoadedLatentRows(parsed);
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

  // Reconstruct all matrices based on options and target dimensions
  const reconstructedMatrices: ReconstructedMatrix[] = useMemo(() => {
    if (autoencoder.config.inputRows !== targetRows || autoencoder.config.inputCols !== targetCols) {
      autoencoder.reconfigure(targetRows, targetCols);
    }

    const options: DecoderOptions = {
      matrixValueType,
      threshold,
      discretizationMode,
      continuousRangeMode,
      customMin,
      customMax,
      sparsityCutoff,
      decimalPrecision,
    };

    return latentRows.map(row => {
      const gt = gtMap.get(row.id);
      return autoencoder.reconstruct(row.id, row.z, options, gt);
    });
  }, [
    latentRows, 
    matrixValueType, 
    threshold, 
    discretizationMode, 
    continuousRangeMode, 
    customMin, 
    customMax, 
    sparsityCutoff, 
    decimalPrecision, 
    autoencoder, 
    targetRows, 
    targetCols, 
    gtMap,
    calibVersion
  ]);

  const selectedMatrix = useMemo(() => {
    return reconstructedMatrices.find(m => m.id === selectedMatrixId) || reconstructedMatrices[0];
  }, [reconstructedMatrices, selectedMatrixId]);

  // Toggle selection for a single row
  const toggleRowSelection = (id: string) => {
    setSelectedRowIds(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  // Select all / Deselect all
  const selectAllRows = () => {
    setSelectedRowIds(new Set(latentRows.map(r => r.id)));
  };

  const deselectAllRows = () => {
    setSelectedRowIds(new Set());
  };

  // Filtered & Paginated rows
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
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
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
    const url = URL.createObjectURL(zipBlob);
    const link = document.createElement('a');
    link.href = url;
    const label = onlySelected ? `selected_${targetMatrices.length}` : `all_${targetMatrices.length}`;
    const modeTag = matrixValueType === 'binary_01' ? 'binary' : 'numeric';
    link.download = `reconstructed_${modeTag}_matrices_${label}_files_${targetRows}x${targetCols}.zip`;
    link.click();
    URL.revokeObjectURL(url);
  };

  // Combined single CSV download option
  const handleDownloadCombinedCsv = () => {
    const targetMatrices = reconstructedMatrices.filter(m => selectedRowIds.has(m.id));
    const list = targetMatrices.length > 0 ? targetMatrices : reconstructedMatrices;
    const content = formatCombinedDenseCsv(list, decimalPrecision);
    const blob = new Blob([content], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    const modeTag = matrixValueType === 'binary_01' ? 'binary' : 'numeric';
    link.download = `combined_${modeTag}_matrices_${list.length}_files_${targetRows}x${targetCols}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex flex-col gap-6">
      {/* Banner & Upload for Module 2 (Bright Theme) */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-xs">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-5 border-b border-slate-200">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold text-emerald-700 uppercase tracking-wider">Module 2</span>
              <span className="text-slate-300">·</span>
              <h2 className="text-lg font-bold text-slate-900">
                Latent Vector Decoder &amp; Matrix File Collection Generator
              </h2>
            </div>
            <p className="text-xs text-slate-600 mt-1">
              Upload a CSV file where <strong>each row has 8 latent dimensions representing a matrix</strong>. The decoder reconstructs matrices with either <strong>Generic Numerical Values (any real numbers)</strong> or <strong>Binary 0/1 (with probability threshold)</strong>, and generates a collection of files ready for download.
            </p>
          </div>

          {/* Preset Sample Buttons */}
          <div className="flex flex-wrap items-center gap-2">
            <button
              onClick={handleLoadContinuousSample}
              className="text-xs px-3 py-1.5 rounded-lg bg-emerald-50 hover:bg-emerald-100 text-emerald-900 font-medium transition-colors border border-emerald-200 flex items-center gap-1.5 shadow-2xs"
              title="Test with continuous generic numerical matrices (10x10 floats [-4.5 to 9.1])"
            >
              <Hash className="w-3.5 h-3.5 text-emerald-700" />
              Load Continuous Numerical Sample (10×10)
            </button>
            <button
              onClick={handleLoadBinarySample}
              className="text-xs px-3 py-1.5 rounded-lg bg-slate-50 hover:bg-slate-100 text-slate-800 font-medium transition-colors border border-slate-200 flex items-center gap-1.5 shadow-2xs"
              title="Test with binary 0/1 matrices (200x21 one-hot sequences)"
            >
              <Binary className="w-3.5 h-3.5 text-slate-600" />
              Load Binary Sample (200×21)
            </button>
          </div>
        </div>

        {/* Load CSV controls */}
        <div className="pt-4 flex flex-col gap-3">
          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-700 font-semibold">Input: 8 Latent Dimensions CSV File (1 row per matrix)</span>
            <button
              onClick={() => setIsPasteOpen(!isPasteOpen)}
              className="text-xs text-slate-600 hover:text-slate-900 underline underline-offset-2 flex items-center gap-1"
            >
              <FileText className="w-3.5 h-3.5" />
              {isPasteOpen ? 'Hide Paste Box' : 'Or Paste CSV Rows Directly'}
            </button>
          </div>

          {isPasteOpen && (
            <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg flex flex-col gap-2">
              <textarea
                value={pasteText}
                onChange={e => setPasteText(e.target.value)}
                placeholder="Paste CSV lines (e.g. matrix_1, z1, z2, z3, z4, z5, z6, z7, z8)..."
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
                  Load Vectors
                </button>
              </div>
            </div>
          )}

          {/* Upload Drop Zone */}
          {latentRows.length === 0 && (
            <div
              onDragOver={e => e.preventDefault()}
              onDrop={handleDrop}
              className="border-2 border-dashed border-slate-300 hover:border-emerald-600 bg-slate-50/70 hover:bg-emerald-50/20 rounded-xl p-8 text-center transition-all cursor-pointer relative group"
            >
              <input
                type="file"
                accept=".csv,.tsv,.txt"
                onChange={handleFileUpload}
                className="absolute inset-0 opacity-0 cursor-pointer w-full h-full"
              />
              <div className="flex flex-col items-center gap-2.5">
                <div className="w-11 h-11 rounded-full bg-white border border-slate-200 flex items-center justify-center text-emerald-700 group-hover:scale-105 shadow-xs transition-transform">
                  <Upload className="w-5 h-5" />
                </div>
                <div className="text-sm font-semibold text-slate-800">
                  Drop your 8D latent CSV file here, or <span className="text-emerald-700 underline">browse file</span>
                </div>
                <p className="text-xs text-slate-500 max-w-md">
                  Expects multiple rows, each row containing 8 latent values $[z_1 \dots z_8]$ representing one matrix.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Latent Vectors Loaded -> Controls & Reconstruction */}
      {latentRows.length > 0 && (
        <div className="flex flex-col gap-6">
          {/* Target Dimensions Bar (Bright Theme) */}
          <div className="py-3 px-4 bg-white border border-slate-200 rounded-xl flex flex-wrap items-center justify-between gap-3 text-xs shadow-xs">
            <div className="flex items-center gap-2">
              <Grid className="w-4 h-4 text-emerald-700" />
              <span className="font-semibold text-slate-800">Target Matrix Reconstruction Shape:</span>
              <span className="px-2 py-0.5 bg-emerald-50 border border-emerald-200 rounded font-mono font-bold text-emerald-800 shadow-2xs">
                {targetRows} × {targetCols} ({targetRows * targetCols} cells)
              </span>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <div className="flex items-center gap-1">
                <span className="text-slate-500 text-[11px]">Presets:</span>
                {[
                  { r: 200, c: 21, label: '200×21' },
                  { r: 50, c: 50, label: '50×50' },
                  { r: 28, c: 28, label: '28×28' },
                  { r: 10, c: 10, label: '10×10' },
                ].map(p => (
                  <button
                    key={p.label}
                    onClick={() => handleUpdateDimensions(p.r, p.c)}
                    className={`px-2 py-0.5 rounded text-[11px] transition-colors ${
                      targetRows === p.r && targetCols === p.c
                        ? 'bg-emerald-700 text-white font-bold shadow-2xs'
                        : 'bg-slate-50 text-slate-600 hover:text-slate-900 border border-slate-200'
                    }`}
                  >
                    {p.label}
                  </button>
                ))}
              </div>

              {/* Custom Rows x Cols input */}
              <div className="flex items-center gap-1 bg-slate-50 border border-slate-200 rounded px-1.5 py-0.5 shadow-2xs">
                <span className="text-[11px] text-slate-500">Custom:</span>
                <input
                  type="number"
                  min="1"
                  max="2000"
                  value={targetRows}
                  onChange={e => {
                    const val = parseInt(e.target.value, 10);
                    if (!isNaN(val) && val > 0) handleUpdateDimensions(val, targetCols);
                  }}
                  className="w-12 px-1 py-0.5 bg-white border border-slate-300 rounded text-center text-[11px] font-mono text-emerald-800 font-semibold"
                  title="Target Rows"
                />
                <span className="text-slate-400 font-bold">×</span>
                <input
                  type="number"
                  min="1"
                  max="2000"
                  value={targetCols}
                  onChange={e => {
                    const val = parseInt(e.target.value, 10);
                    if (!isNaN(val) && val > 0) handleUpdateDimensions(targetRows, val);
                  }}
                  className="w-12 px-1 py-0.5 bg-white border border-slate-300 rounded text-center text-[11px] font-mono text-emerald-800 font-semibold"
                  title="Target Columns"
                />
              </div>
            </div>
          </div>

          {detectedDimensionNotice && (
            <div className="p-2.5 bg-emerald-50 border border-emerald-200 rounded-lg text-xs text-emerald-800 flex items-center gap-2">
              <Check className="w-4 h-4 text-emerald-600 shrink-0" />
              <span>{detectedDimensionNotice}</span>
            </div>
          )}

          {/* Model Calibration Status Notice */}
          {autoencoder.isCalibrated ? (
            <div className="p-3 bg-emerald-50 border border-emerald-300 rounded-xl text-xs text-emerald-900 flex items-center justify-between gap-3 shadow-xs">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
                <span>
                  <strong>8D Latent Subspace Calibrated:</strong> Active model is mathematically fitted to your matrices. Real numerical values will decode with exact fidelity down to 8+ decimal figures.
                </span>
              </div>
              <div className="flex items-center gap-2 shrink-0">
                <button
                  type="button"
                  onClick={() => {
                    autoencoder.resetCalibration();
                    setCalibVersion(v => v + 1);
                  }}
                  className="text-[11px] text-slate-500 hover:text-red-600 underline"
                  title="Switch back to default neural network weights"
                >
                  Reset Basis
                </button>
                <span className="text-[11px] font-mono font-bold bg-white text-emerald-800 px-2 py-0.5 rounded border border-emerald-200">
                  Lossless Mode Active
                </span>
              </div>
            </div>
          ) : (
            <div className="p-3 bg-amber-50 border border-amber-200 rounded-xl text-xs text-amber-900 flex items-center justify-between gap-2 shadow-xs">
              <div className="flex items-center gap-2">
                <Info className="w-4 h-4 text-amber-600 shrink-0" />
                <span>
                  <strong>Notice for custom/random numerical matrices:</strong> Compressing {targetRows * targetCols} independent numbers into 8 dimensions without fitting produces random projections. In Module 1, use <strong>"Auto-Fit Basis on Upload"</strong> or <strong>"Fit 8D Basis"</strong> to calibrate the 8D subspace for exact reconstruction.
                </span>
              </div>
              {typeof window !== 'undefined' && Boolean(window.localStorage?.getItem(`matrix_calib_${targetRows}x${targetCols}`) || window.localStorage?.getItem('matrix_calib_latest')) && (
                <button
                  type="button"
                  onClick={() => {
                    const raw = window.localStorage.getItem(`matrix_calib_${targetRows}x${targetCols}`) || window.localStorage.getItem('matrix_calib_latest');
                    if (raw && autoencoder.deserializeCalibration(raw)) {
                      setCalibVersion(v => v + 1);
                    }
                  }}
                  className="px-2.5 py-1 bg-amber-700 hover:bg-amber-800 text-white rounded text-xs font-semibold shrink-0 shadow-2xs"
                >
                  Restore Saved Basis
                </button>
              )}
            </div>
          )}

          {/* PRIMARY SELECTOR: Matrix Value Type Mode (Generic Numerical vs Binary 0/1) */}
          <div className="bg-white border-2 border-emerald-600/30 rounded-xl p-5 shadow-sm flex flex-col gap-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-200">
              <div>
                <span className="text-xs font-bold uppercase tracking-wider text-emerald-800">
                  Decoder Output Mode
                </span>
                <h3 className="text-sm font-bold text-slate-900 mt-0.5">
                  Select Matrix Numerical Representation
                </h3>
              </div>
              <span className="text-xs text-slate-500">
                Controls whether reconstructed matrices have continuous real numbers or discrete 0/1 values
              </span>
            </div>

            {/* Mode Toggle Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {/* Option 1: Generic Numerical Values (Any Real Numbers) */}
              <button
                type="button"
                onClick={() => setMatrixValueType('continuous_numeric')}
                className={`p-4 rounded-xl text-left border-2 transition-all flex flex-col gap-2 ${
                  matrixValueType === 'continuous_numeric'
                    ? 'border-emerald-600 bg-emerald-50/50 shadow-xs'
                    : 'border-slate-200 bg-slate-50/50 hover:bg-slate-100/70 text-slate-700'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className={`p-1.5 rounded-lg ${matrixValueType === 'continuous_numeric' ? 'bg-emerald-600 text-white' : 'bg-slate-200 text-slate-600'}`}>
                      <Hash className="w-4 h-4" />
                    </div>
                    <span className="font-bold text-sm text-slate-900">
                      Generic Numerical Values
                    </span>
                  </div>
                  {matrixValueType === 'continuous_numeric' && (
                    <span className="px-2 py-0.5 bg-emerald-600 text-white text-[10px] font-bold rounded-full">
                      ACTIVE
                    </span>
                  )}
                </div>
                <p className="text-xs text-slate-600 leading-relaxed">
                  Reconstructs matrices composed of <strong>any numerical values</strong> (continuous real numbers, floats, integers, negative &amp; positive). The probability threshold is <strong>disabled</strong> so numerical magnitudes are preserved.
                </p>
                <div className="text-[11px] font-mono text-emerald-800 bg-white/80 p-1.5 rounded border border-emerald-200/60 mt-1">
                  Output: real continuous floats e.g. [2.45, -0.80, 0.00, 7.15, ...]
                </div>
              </button>

              {/* Option 2: Binary Matrices ({0, 1} with Threshold) */}
              <button
                type="button"
                onClick={() => setMatrixValueType('binary_01')}
                className={`p-4 rounded-xl text-left border-2 transition-all flex flex-col gap-2 ${
                  matrixValueType === 'binary_01'
                    ? 'border-cyan-600 bg-cyan-50/50 shadow-xs'
                    : 'border-slate-200 bg-slate-50/50 hover:bg-slate-100/70 text-slate-700'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className={`p-1.5 rounded-lg ${matrixValueType === 'binary_01' ? 'bg-cyan-600 text-white' : 'bg-slate-200 text-slate-600'}`}>
                      <Binary className="w-4 h-4" />
                    </div>
                    <span className="font-bold text-sm text-slate-900">
                      Binary Matrices ({'{0, 1}'})
                    </span>
                  </div>
                  {matrixValueType === 'binary_01' && (
                    <span className="px-2 py-0.5 bg-cyan-600 text-white text-[10px] font-bold rounded-full">
                      ACTIVE
                    </span>
                  )}
                </div>
                <p className="text-xs text-slate-600 leading-relaxed">
                  Reconstructs matrices strictly composed of <strong>0 and 1</strong>. Applies a user-defined <strong>probability threshold ($\theta$)</strong> on sigmoid activations to discretize each cell into active (1) or inactive (0).
                </p>
                <div className="text-[11px] font-mono text-cyan-800 bg-white/80 p-1.5 rounded border border-cyan-200/60 mt-1">
                  Output: binary discrete bits e.g. [0, 1, 0, 0, 1, 0, ...]
                </div>
              </button>
            </div>
          </div>

          {/* Dedicated Configuration Bar depending on the selected Mode */}
          <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs flex flex-col gap-5">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              
              {/* LEFT CONTROL: Mode-Specific Parameterization */}
              {matrixValueType === 'continuous_numeric' ? (
                /* Continuous Generic Numerical Values Controls */
                <div className="flex flex-col gap-3 p-4 bg-emerald-50/30 border border-emerald-200 rounded-xl">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Scale className="w-4 h-4 text-emerald-700" />
                      <label className="text-xs font-bold text-slate-800 uppercase tracking-wider">
                        Numerical Scaling &amp; Value Range
                      </label>
                    </div>
                    <span className="text-[11px] text-emerald-700 font-semibold">Continuous Real Numbers</span>
                  </div>

                  {/* Range Mode Selection */}
                  <div className="grid grid-cols-3 gap-1 bg-white p-1 rounded-lg border border-slate-200 text-[11px] shadow-2xs">
                    <button
                      onClick={() => setContinuousRangeMode('raw')}
                      className={`py-1 rounded text-center transition-colors font-medium ${
                        continuousRangeMode === 'raw' ? 'bg-slate-800 text-white' : 'text-slate-600 hover:text-slate-900'
                      }`}
                      title="Direct unconstrained linear model outputs"
                    >
                      Raw Linear
                    </button>
                    <button
                      onClick={() => setContinuousRangeMode('custom_range')}
                      className={`py-1 rounded text-center transition-colors font-medium ${
                        continuousRangeMode === 'custom_range' ? 'bg-slate-800 text-white' : 'text-slate-600 hover:text-slate-900'
                      }`}
                      title="Normalize and scale to custom [min, max]"
                    >
                      Rescale [min, max]
                    </button>
                    <button
                      onClick={() => setContinuousRangeMode('positive_only')}
                      className={`py-1 rounded text-center transition-colors font-medium ${
                        continuousRangeMode === 'positive_only' ? 'bg-slate-800 text-white' : 'text-slate-600 hover:text-slate-900'
                      }`}
                      title="Non-negative values >= 0 (Softplus)"
                    >
                      Positive (&ge; 0)
                    </button>
                  </div>

                  {/* If custom_range is selected, show Min and Max inputs */}
                  {continuousRangeMode === 'custom_range' && (
                    <div className="p-2.5 bg-white border border-slate-200 rounded-lg flex flex-col gap-2 shadow-2xs">
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-slate-600">Scale to Range:</span>
                        <div className="flex items-center gap-1.5 font-mono text-xs">
                          <span className="text-slate-400">Min:</span>
                          <input
                            type="number"
                            step="0.5"
                            value={customMin}
                            onChange={e => setCustomMin(parseFloat(e.target.value) || 0)}
                            className="w-16 px-1.5 py-0.5 bg-slate-50 border border-slate-300 rounded text-center text-xs font-mono font-bold text-emerald-800"
                          />
                          <span className="text-slate-400">Max:</span>
                          <input
                            type="number"
                            step="0.5"
                            value={customMax}
                            onChange={e => setCustomMax(parseFloat(e.target.value) || 1)}
                            className="w-16 px-1.5 py-0.5 bg-slate-50 border border-slate-300 rounded text-center text-xs font-mono font-bold text-emerald-800"
                          />
                        </div>
                      </div>
                      <div className="flex items-center gap-1 text-[11px] text-slate-500">
                        <span>Range Presets:</span>
                        {[
                          { min: 0, max: 10, label: '[0, 10]' },
                          { min: -5, max: 5, label: '[-5, 5]' },
                          { min: 0, max: 100, label: '[0, 100]' },
                          { min: -1, max: 1, label: '[-1, 1]' },
                        ].map(pr => (
                          <button
                            key={pr.label}
                            onClick={() => {
                              setCustomMin(pr.min);
                              setCustomMax(pr.max);
                            }}
                            className="px-1.5 py-0.5 bg-slate-100 hover:bg-slate-200 rounded text-[10px] text-slate-700"
                          >
                            {pr.label}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Sparsity Cutoff Slider */}
                  <div className="flex flex-col gap-1.5 pt-1">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-slate-700 font-medium">Sparsity Zero-Cutoff ($\tau$):</span>
                      <span className="font-mono font-bold text-emerald-800">
                        {sparsityCutoff > 0 ? `|v| < ${sparsityCutoff.toFixed(2)} → 0.0` : 'None (0.00)'}
                      </span>
                    </div>
                    <div className="flex items-center gap-3">
                      <span className="text-[10px] font-mono text-slate-400">0.0</span>
                      <input
                        type="range"
                        min="0.0"
                        max="2.0"
                        step="0.05"
                        value={sparsityCutoff}
                        onChange={e => setSparsityCutoff(parseFloat(e.target.value))}
                        className="flex-1 accent-emerald-600 cursor-pointer h-2 bg-slate-200 rounded-lg"
                      />
                      <span className="text-[10px] font-mono text-slate-400">2.0</span>
                    </div>
                    <p className="text-[11px] text-slate-500">
                      Zeroes out values below cutoff threshold. Keep at 0.00 to preserve all continuous outputs, or raise it to filter out near-zero noise into true zeros.
                    </p>
                  </div>

                  {/* Decimal Precision */}
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pt-1 border-t border-slate-200 text-xs">
                    <span className="text-slate-600">Decimal Precision:</span>
                    <div className="flex flex-wrap items-center gap-1">
                      {[
                        { p: 0, label: 'Int (0)' },
                        { p: 2, label: '2 dec' },
                        { p: 4, label: '4 dec' },
                        { p: 6, label: '6 dec' },
                        { p: 8, label: '8 dec' },
                        { p: 10, label: '10 dec' },
                        { p: -1, label: 'Full Float' },
                      ].map(dp => (
                        <button
                          key={dp.p}
                          onClick={() => setDecimalPrecision(dp.p)}
                          className={`px-2 py-0.5 rounded text-[11px] font-mono transition-colors ${
                            decimalPrecision === dp.p
                              ? 'bg-emerald-700 text-white font-bold shadow-2xs'
                              : 'bg-white border border-slate-200 text-slate-600 hover:text-slate-900'
                          }`}
                        >
                          {dp.label}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              ) : (
                /* Binary {0, 1} Controls with Probability Threshold */
                <div className="flex flex-col gap-3 p-4 bg-cyan-50/30 border border-cyan-200 rounded-xl">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Sliders className="w-4 h-4 text-cyan-700" />
                      <label className="text-xs font-bold text-slate-800 uppercase tracking-wider">
                        Probability Threshold ($\theta$)
                      </label>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <span className="text-xs text-slate-500">Value:</span>
                      <input
                        type="number"
                        step="0.01"
                        min="0.01"
                        max="0.99"
                        value={threshold}
                        onChange={e => {
                          const val = parseFloat(e.target.value);
                          if (!isNaN(val)) setThreshold(Math.max(0.01, Math.min(0.99, val)));
                        }}
                        className="w-16 px-1.5 py-0.5 bg-white border border-slate-300 rounded text-center text-xs font-mono font-bold text-cyan-800 focus:outline-hidden focus:border-cyan-600 shadow-2xs"
                      />
                    </div>
                  </div>

                  {/* Range Slider */}
                  <div className="flex items-center gap-3">
                    <span className="text-[10px] font-mono text-slate-400">0.01</span>
                    <input
                      type="range"
                      min="0.01"
                      max="0.99"
                      step="0.01"
                      value={threshold}
                      onChange={e => setThreshold(parseFloat(e.target.value))}
                      className="flex-1 accent-cyan-600 cursor-pointer h-2 bg-slate-200 rounded-lg"
                    />
                    <span className="text-[10px] font-mono text-slate-400">0.99</span>
                  </div>

                  {/* Quick Threshold Presets */}
                  <div className="flex items-center justify-between pt-1">
                    <span className="text-[11px] text-slate-500">Presets:</span>
                    <div className="flex items-center gap-1.5">
                      {[0.20, 0.35, 0.50, 0.65, 0.80].map(p => (
                        <button
                          key={p}
                          onClick={() => setThreshold(p)}
                          className={`text-[11px] px-2 py-0.5 rounded transition-colors ${
                            Math.abs(threshold - p) < 0.001
                              ? 'bg-cyan-700 text-white font-bold shadow-2xs'
                              : 'bg-white text-slate-600 hover:text-slate-900 border border-slate-200'
                          }`}
                        >
                          {p.toFixed(2)}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Discretization mode: Threshold vs Argmax */}
                  <div className="flex items-center justify-between pt-1 border-t border-slate-200 text-xs">
                    <span className="text-slate-600">Discretization:</span>
                    <div className="flex items-center gap-1">
                      <button
                        onClick={() => setDiscretizationMode('threshold')}
                        className={`px-2.5 py-0.5 rounded text-[11px] transition-colors font-medium ${
                          discretizationMode === 'threshold'
                            ? 'bg-cyan-700 text-white font-bold'
                            : 'bg-white border border-slate-200 text-slate-600 hover:text-slate-900'
                        }`}
                      >
                        Binary Threshold (p &gt; {threshold.toFixed(2)})
                      </button>
                      <button
                        onClick={() => setDiscretizationMode('argmax')}
                        className={`px-2.5 py-0.5 rounded text-[11px] transition-colors font-medium ${
                          discretizationMode === 'argmax'
                            ? 'bg-cyan-700 text-white font-bold'
                            : 'bg-white border border-slate-200 text-slate-600 hover:text-slate-900'
                        }`}
                        title="Pick 1-hot max column per row"
                      >
                        Argmax (1-hot)
                      </button>
                    </div>
                  </div>

                  {/* Rule note */}
                  <div className="text-[11px] text-slate-600 bg-white p-2.5 rounded-lg border border-slate-200 flex items-start gap-1.5 shadow-2xs">
                    <Info className="w-3.5 h-3.5 text-cyan-600 shrink-0 mt-0.5" />
                    <span>
                      Rule: <code className="text-cyan-800 font-semibold">cell = probability &gt; {threshold.toFixed(2)} ? 1 : 0</code>
                      <br />
                      Adjusting threshold recalculates all binary reconstructions immediately.
                    </span>
                  </div>
                </div>
              )}

              {/* RIGHT CONTROL: Format for Generated Matrix Files */}
              <div className="flex flex-col gap-3 p-4 bg-slate-50 border border-slate-200 rounded-xl">
                <div className="flex items-center justify-between">
                  <label className="text-xs font-bold text-slate-800 uppercase tracking-wider flex items-center gap-2">
                    <FileCode className="w-4 h-4 text-slate-700" />
                    Format of Generated Matrix Files
                  </label>
                  <span className="text-[11px] text-slate-500">1 File per Matrix</span>
                </div>

                <div>
                  <label className="text-[11px] text-slate-500 block mb-1">Generated File Format:</label>
                  <select
                    value={outputFormat}
                    onChange={e => setOutputFormat(e.target.value as OutputFormatMode)}
                    className="w-full bg-white border border-slate-300 rounded-lg px-3 py-1.5 text-slate-800 text-xs focus:outline-hidden focus:border-emerald-600 font-medium shadow-2xs"
                  >
                    <option value="dense_zip">Dense 2D CSV Tables (e.g. matrix_01.csv, matrix_02.csv)</option>
                    <option value="sparse_coo_csv">Sparse Coordinate CSV (COO: row_index, col_index, value)</option>
                    <option value="flattened_csv">Flattened CSV (1 row per matrix)</option>
                    {targetCols === 21 && (
                      <option value="fasta_sequence">FASTA / Decoded Protein Sequences (.fasta)</option>
                    )}
                    <option value="json">Structured JSON (.json files)</option>
                  </select>
                </div>

                <div className="p-3 bg-white border border-slate-200 rounded-lg text-xs space-y-1.5 shadow-2xs text-slate-600">
                  <div className="font-semibold text-slate-800 flex items-center justify-between">
                    <span>Generated File Content:</span>
                    <span className="text-emerald-700 font-mono">
                      {matrixValueType === 'continuous_numeric' ? 'Real Numbers' : 'Binary {0, 1}'}
                    </span>
                  </div>
                  <p className="text-[11px] leading-relaxed">
                    {outputFormat === 'dense_zip' && 'Each file is a full 2D CSV grid containing the reconstructed matrix rows and columns.'}
                    {outputFormat === 'sparse_coo_csv' && 'Each file lists only non-zero coordinates: row_index, col_index, value.'}
                    {outputFormat === 'flattened_csv' && 'Single-row vector representation with all flattened cells.'}
                    {outputFormat === 'json' && 'JSON object with metadata, dimensions, and data array.'}
                    {outputFormat === 'fasta_sequence' && 'Translated biological sequence header and residues.'}
                  </p>
                </div>

                <div className="text-[11px] text-slate-600 pt-1">
                  Collection size: <span className="font-bold text-slate-800">{reconstructedMatrices.length} files</span> ready for generation
                </div>
              </div>
            </div>

            {/* Collection Download Toolbar (Download All or Selection) */}
            <div className="flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-slate-200">
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-xs text-slate-600 font-semibold mr-1">
                  Download Collection:
                </span>
                
                {/* Download ALL as ZIP */}
                <button
                  onClick={() => handleDownloadZipCollection(false)}
                  className="flex items-center gap-2 px-4 py-2 bg-emerald-700 hover:bg-emerald-800 text-white rounded-lg text-xs font-semibold shadow-xs transition-colors"
                >
                  <Archive className="w-4 h-4" />
                  Download ALL Files as ZIP ({reconstructedMatrices.length} files)
                </button>

                {/* Download SELECTED as ZIP */}
                <button
                  onClick={() => handleDownloadZipCollection(true)}
                  disabled={selectedRowIds.size === 0}
                  className="flex items-center gap-2 px-4 py-2 bg-cyan-700 hover:bg-cyan-800 disabled:opacity-40 disabled:pointer-events-none text-white rounded-lg text-xs font-semibold shadow-xs transition-colors"
                >
                  <ArrowDownToLine className="w-4 h-4" />
                  Download Selected Files ({selectedRowIds.size} files)
                </button>

                {/* Combined single CSV button */}
                <button
                  onClick={handleDownloadCombinedCsv}
                  className="flex items-center gap-1.5 px-3 py-2 bg-slate-50 hover:bg-slate-100 text-slate-700 rounded-lg text-xs font-medium border border-slate-200 transition-colors shadow-2xs"
                  title="Export all matrices concatenated into a single large CSV table"
                >
                  <Download className="w-3.5 h-3.5" />
                  Single Combined CSV
                </button>
              </div>

              {/* Selection buttons */}
              <div className="flex items-center gap-2 text-xs">
                <span className="text-slate-500 text-[11px]">
                  Selected: <strong className="text-slate-800">{selectedRowIds.size}</strong> of {reconstructedMatrices.length}
                </span>
                <button
                  onClick={selectAllRows}
                  className="px-2 py-1 rounded bg-slate-100 hover:bg-slate-200 text-slate-700 text-[11px] font-medium"
                >
                  Select All
                </button>
                <button
                  onClick={deselectAllRows}
                  className="px-2 py-1 rounded bg-slate-100 hover:bg-slate-200 text-slate-700 text-[11px] font-medium"
                >
                  Deselect All
                </button>
              </div>
            </div>
          </div>

          {/* Matrix Collection List & Heatmap Inspector */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left Column: Collection of Matrix Files List (5 cols) */}
            <div className="lg:col-span-5 flex flex-col gap-4">
              <div className="bg-white border border-slate-200 rounded-xl p-5 flex flex-col gap-3 shadow-xs">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-2 border-b border-slate-100">
                  <div className="flex items-center gap-2">
                    <h3 className="text-sm font-bold text-slate-900">
                      Generated File Collection ({reconstructedMatrices.length})
                    </h3>
                  </div>

                  <div className="flex items-center gap-2">
                    {/* Search box */}
                    <div className="relative">
                      <Search className="w-3.5 h-3.5 text-slate-400 absolute left-2 top-1/2 -translate-y-1/2" />
                      <input
                        type="text"
                        placeholder="Search file..."
                        value={searchQuery}
                        onChange={e => {
                          setSearchQuery(e.target.value);
                          setCurrentPage(1);
                        }}
                        className="pl-7 pr-2 py-0.5 text-xs bg-slate-50 border border-slate-200 rounded text-slate-800 placeholder-slate-400 w-32 focus:outline-hidden"
                      />
                    </div>
                  </div>
                </div>

                {/* Collection of Files with Direct Single Download & Checkboxes */}
                <div className="divide-y divide-slate-100 max-h-[520px] overflow-y-auto">
                  {paginatedMatrices.map((m, idx) => {
                    const isSelected = m.id === selectedMatrix?.id;
                    const isChecked = selectedRowIds.has(m.id);
                    const latentRow = latentRows.find(r => r.id === m.id);

                    return (
                      <div
                        key={m.id || idx}
                        onClick={() => setSelectedMatrixId(m.id)}
                        className={`p-3 rounded-lg cursor-pointer transition-colors flex items-center justify-between gap-3 ${
                          isSelected ? 'bg-emerald-50/70 border border-emerald-300' : 'hover:bg-slate-50'
                        }`}
                      >
                        {/* Checkbox for batch selection */}
                        <div 
                          onClick={(e) => {
                            e.stopPropagation();
                            toggleRowSelection(m.id);
                          }}
                          className="p-1 text-slate-400 hover:text-emerald-700"
                          title={isChecked ? 'Deselect this matrix' : 'Select this matrix'}
                        >
                          {isChecked ? (
                            <CheckSquare className="w-4 h-4 text-emerald-700" />
                          ) : (
                            <Square className="w-4 h-4 text-slate-300" />
                          )}
                        </div>

                        {/* File Details */}
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <span className="font-mono font-bold text-slate-900 truncate">
                              {m.id}
                            </span>
                            <span className="text-slate-500 text-[11px] shrink-0">
                              {m.rows} × {m.cols}
                            </span>
                            {m.valueType === 'continuous_numeric' ? (
                              <span className="px-1.5 py-0.2 bg-emerald-100/70 text-emerald-800 text-[10px] rounded font-mono font-medium">
                                [{m.minVal.toFixed(1)}, {m.maxVal.toFixed(1)}]
                              </span>
                            ) : (
                              <span className="px-1.5 py-0.2 bg-cyan-100/70 text-cyan-800 text-[10px] rounded font-mono font-medium">
                                {m.nonZeroCount} active
                              </span>
                            )}
                          </div>

                          {latentRow && (
                            <div className="text-[10px] font-mono text-slate-400 truncate">
                              z: [{latentRow.z.map(v => v.toFixed(2)).join(', ')}]
                            </div>
                          )}

                          {m.metrics && (
                            <div className="flex items-center gap-2 text-[10px] font-mono text-slate-500 pt-0.5">
                              {m.metrics.f1 !== undefined && (
                                <span className="text-emerald-700 font-semibold">
                                  F1: {(m.metrics.f1 * 100).toFixed(1)}%
                                </span>
                              )}
                              {m.metrics.mae !== undefined && (
                                <span className="text-emerald-700 font-semibold">
                                  MAE: {m.metrics.mae.toFixed(3)}
                                </span>
                              )}
                              {m.metrics.rmse !== undefined && (
                                <span>RMSE: {m.metrics.rmse.toFixed(3)}</span>
                              )}
                              <span>·</span>
                              <span>{m.sparsityPercent.toFixed(1)}% zero/sparse</span>
                            </div>
                          )}
                        </div>

                        {/* Action: Single File Download Icon Button */}
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDownloadSingleFile(m);
                          }}
                          className="p-1.5 rounded-md hover:bg-white text-slate-600 hover:text-emerald-700 border border-transparent hover:border-slate-200 transition-colors shadow-2xs"
                          title={`Download individual file: ${m.id}.csv`}
                        >
                          <Download className="w-4 h-4" />
                        </button>
                      </div>
                    );
                  })}
                </div>

                {/* Pagination */}
                {totalPages > 1 && (
                  <div className="flex items-center justify-between pt-2 border-t border-slate-100 text-xs text-slate-500">
                    <span>
                      {(currentPage - 1) * pageSize + 1}–{Math.min(filteredMatrices.length, currentPage * pageSize)} of {filteredMatrices.length}
                    </span>
                    <div className="flex items-center gap-1.5">
                      <button
                        onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                        disabled={currentPage === 1}
                        className="p-1 rounded border border-slate-200 disabled:opacity-40 hover:bg-slate-50"
                      >
                        <ChevronLeft className="w-4 h-4" />
                      </button>
                      <span className="font-medium text-slate-700 px-1">
                        {currentPage} / {totalPages}
                      </span>
                      <button
                        onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                        disabled={currentPage === totalPages}
                        className="p-1 rounded border border-slate-200 disabled:opacity-40 hover:bg-slate-50"
                      >
                        <ChevronRight className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Right Column: Reconstructed Matrix Heatmap & Single File Details (7 cols) */}
            <div className="lg:col-span-7 flex flex-col gap-4">
              {selectedMatrix ? (
                <div>
                  <div className="flex items-center justify-between pb-2">
                    <div className="flex items-center gap-2 text-xs">
                      <span className="font-bold text-slate-800">
                        Inspecting Matrix: <span className="text-emerald-700 font-mono">{selectedMatrix.id}</span>
                      </span>
                      <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-slate-100 text-slate-700">
                        {selectedMatrix.valueType === 'continuous_numeric' ? 'Continuous Numerical' : 'Binary 0/1'}
                      </span>
                    </div>

                    <div className="flex items-center gap-2">
                      {selectedMatrix.valueType === 'binary_01' && (
                        <button
                          onClick={() => setShowProbabilitiesView(!showProbabilitiesView)}
                          className={`text-xs px-2.5 py-1 rounded-lg transition-colors font-medium border ${
                            showProbabilitiesView
                              ? 'bg-cyan-700 text-white border-cyan-700'
                              : 'bg-white text-slate-700 hover:text-slate-900 border-slate-200 shadow-2xs'
                          }`}
                        >
                          {showProbabilitiesView ? 'Show Binarized Grid' : 'View Continuous Heatmap'}
                        </button>
                      )}

                      <button
                        onClick={() => handleDownloadSingleFile(selectedMatrix)}
                        className="flex items-center gap-1 text-xs px-2.5 py-1 bg-emerald-700 hover:bg-emerald-800 text-white rounded-lg font-medium shadow-2xs transition-colors"
                      >
                        <Download className="w-3.5 h-3.5" />
                        Download This File
                      </button>
                    </div>
                  </div>

                  <MatrixHeatmap
                    rows={selectedMatrix.rows}
                    cols={selectedMatrix.cols}
                    data={selectedMatrix.data}
                    probabilities={selectedMatrix.probabilities}
                    threshold={threshold}
                    groundTruth={selectedMatrix.groundTruth}
                    showProbabilities={showProbabilitiesView}
                    valueType={selectedMatrix.valueType}
                    minVal={selectedMatrix.minVal}
                    maxVal={selectedMatrix.maxVal}
                    title={`Reconstructed Matrix: ${selectedMatrix.id} (${selectedMatrix.rows} × ${selectedMatrix.cols}, ${selectedMatrix.valueType === 'continuous_numeric' ? `Range: [${selectedMatrix.minVal.toFixed(2)}, ${selectedMatrix.maxVal.toFixed(2)}]` : `Threshold: ${threshold.toFixed(2)}`})`}
                  />

                  {/* Summary Stats for Continuous Matrix */}
                  {selectedMatrix.valueType === 'continuous_numeric' && (
                    <div className="mt-4 p-4 bg-white border border-slate-200 rounded-xl shadow-xs flex flex-col gap-3">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-bold text-slate-800 uppercase tracking-wider">
                          Numerical Values Distribution
                        </span>
                        <span className="text-xs text-slate-500 font-mono">
                          {selectedMatrix.rows * selectedMatrix.cols} total values
                        </span>
                      </div>
                      <div className="grid grid-cols-4 gap-3 text-center">
                        <div className="p-2.5 bg-slate-50 border border-slate-200 rounded-lg">
                          <div className="text-[10px] text-slate-500 uppercase font-medium">Min Value</div>
                          <div className="text-sm font-extrabold text-slate-800 font-mono mt-0.5">
                            {selectedMatrix.minVal.toFixed(3)}
                          </div>
                        </div>
                        <div className="p-2.5 bg-slate-50 border border-slate-200 rounded-lg">
                          <div className="text-[10px] text-slate-500 uppercase font-medium">Max Value</div>
                          <div className="text-sm font-extrabold text-emerald-700 font-mono mt-0.5">
                            {selectedMatrix.maxVal.toFixed(3)}
                          </div>
                        </div>
                        <div className="p-2.5 bg-slate-50 border border-slate-200 rounded-lg">
                          <div className="text-[10px] text-slate-500 uppercase font-medium">Non-Zero Count</div>
                          <div className="text-sm font-extrabold text-cyan-700 font-mono mt-0.5">
                            {selectedMatrix.nonZeroCount}
                          </div>
                        </div>
                        <div className="p-2.5 bg-slate-50 border border-slate-200 rounded-lg">
                          <div className="text-[10px] text-slate-500 uppercase font-medium">Sparsity</div>
                          <div className="text-sm font-extrabold text-amber-700 font-mono mt-0.5">
                            {selectedMatrix.sparsityPercent.toFixed(1)}%
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Decoded Sequence if 21 columns and binary/categorical */}
                  {selectedMatrix.cols === 21 && (
                    <div className="mt-4 p-4 bg-white border border-slate-200 rounded-xl shadow-xs flex flex-col gap-2">
                      <div className="flex items-center justify-between text-xs">
                        <span className="font-bold text-slate-800">
                          Decoded Amino Acid Sequence ({selectedMatrix.rows} residues)
                        </span>
                        <span className="text-slate-500 text-[11px]">
                          Translated from 200×21 one-hot columns
                        </span>
                      </div>
                      <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg text-xs font-mono text-emerald-800 break-all select-all font-semibold">
                        {formatFastaSequences([selectedMatrix]).split('\n')[1] || '---'}
                      </div>
                    </div>
                  )}

                  {/* Ground Truth Metric Card if available */}
                  {selectedMatrix.metrics && (
                    <div className="mt-4 p-4 bg-white border border-emerald-300 rounded-xl shadow-xs flex flex-col gap-3">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-bold text-emerald-800 uppercase tracking-wider">
                          Reconstruction Fidelity vs Ground Truth
                        </span>
                        <span className="text-xs text-slate-500">
                          (Original from Module 1)
                        </span>
                      </div>

                      {selectedMatrix.valueType === 'binary_01' ? (
                        <div className="grid grid-cols-4 gap-3 text-center">
                          <div className="p-2.5 bg-slate-50 border border-slate-200 rounded-lg">
                            <div className="text-[10px] text-slate-500 uppercase font-medium">F1 Score</div>
                            <div className="text-base font-extrabold text-emerald-700 font-mono mt-0.5">
                              {selectedMatrix.metrics.f1 !== undefined ? `${(selectedMatrix.metrics.f1 * 100).toFixed(1)}%` : 'N/A'}
                            </div>
                          </div>
                          <div className="p-2.5 bg-slate-50 border border-slate-200 rounded-lg">
                            <div className="text-[10px] text-slate-500 uppercase font-medium">Precision</div>
                            <div className="text-base font-extrabold text-cyan-700 font-mono mt-0.5">
                              {selectedMatrix.metrics.precision !== undefined ? `${(selectedMatrix.metrics.precision * 100).toFixed(1)}%` : 'N/A'}
                            </div>
                          </div>
                          <div className="p-2.5 bg-slate-50 border border-slate-200 rounded-lg">
                            <div className="text-[10px] text-slate-500 uppercase font-medium">Recall</div>
                            <div className="text-base font-extrabold text-amber-700 font-mono mt-0.5">
                              {selectedMatrix.metrics.recall !== undefined ? `${(selectedMatrix.metrics.recall * 100).toFixed(1)}%` : 'N/A'}
                            </div>
                          </div>
                          <div className="p-2.5 bg-slate-50 border border-slate-200 rounded-lg">
                            <div className="text-[10px] text-slate-500 uppercase font-medium">Accuracy</div>
                            <div className="text-base font-extrabold text-slate-800 font-mono mt-0.5">
                              {selectedMatrix.metrics.accuracy !== undefined ? `${(selectedMatrix.metrics.accuracy * 100).toFixed(2)}%` : 'N/A'}
                            </div>
                          </div>
                        </div>
                      ) : (
                        <div className="grid grid-cols-2 gap-3 text-center">
                          <div className="p-2.5 bg-slate-50 border border-slate-200 rounded-lg">
                            <div className="text-[10px] text-slate-500 uppercase font-medium">Mean Absolute Error (MAE)</div>
                            <div className="text-base font-extrabold text-emerald-700 font-mono mt-0.5">
                              {selectedMatrix.metrics.mae !== undefined ? selectedMatrix.metrics.mae.toFixed(4) : 'N/A'}
                            </div>
                          </div>
                          <div className="p-2.5 bg-slate-50 border border-slate-200 rounded-lg">
                            <div className="text-[10px] text-slate-500 uppercase font-medium">Root Mean Squared Error (RMSE)</div>
                            <div className="text-base font-extrabold text-cyan-700 font-mono mt-0.5">
                              {selectedMatrix.metrics.rmse !== undefined ? selectedMatrix.metrics.rmse.toFixed(4) : 'N/A'}
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ) : (
                <div className="p-8 border border-slate-200 rounded-xl bg-white text-center text-xs text-slate-500 shadow-xs">
                  Select a reconstructed matrix to inspect
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
