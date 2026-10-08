/**
 * Module 1: Matrix Encoder Module (Bright Theme)
 * Upload any number of matrix files (in different formats and sizes),
 * extracts compact latent dimensions z ∈ [0, 1]^K via Principal Component Analysis (PCA),
 * and prepares collective CSV downloads and basis exports for Bayesian Optimization (BO).
 */
import React, { useState, useMemo, useEffect } from 'react';
import { MatrixItem, InputFormatMode, FitResult } from '../lib/types';
import { MatrixPCA } from '../lib/autoencoder';
import { 
  parseMatrixPayload, 
  formatLatentCsv,
} from '../lib/matrixFormats';
import { 
  SAMPLE_45X45_BINARY_COO_CSV,
  SAMPLE_FASTA_SEQUENCES, 
  SAMPLE_SPARSE_COO_CSV,
  SAMPLE_50X50_COO_CSV,
  SAMPLE_10X10_DENSE_CSV,
  SAMPLE_CONTINUOUS_FLOAT_CSV
} from '../lib/samples';
import { MatrixHeatmap } from './MatrixHeatmap';
import { 
  Upload, 
  FileText, 
  Download, 
  Copy, 
  Check, 
  ArrowRight, 
  Trash2, 
  Eye, 
  Sparkles,
  Grid,
  Search,
  ChevronLeft,
  ChevronRight,
  CheckCircle2,
  Layers,
  Zap,
  RefreshCw,
  Sliders,
  SlidersHorizontal,
  FileCode
} from 'lucide-react';

interface MatrixEncoderModuleProps {
  autoencoder: MatrixPCA;
  onTransferToDecoder?: (
    latentCsv: string, 
    groundTruthMatrices?: MatrixItem[], 
    dimensions?: { rows: number; cols: number }
  ) => void;
}

export const MatrixEncoderModule: React.FC<MatrixEncoderModuleProps> = ({
  autoencoder,
  onTransferToDecoder,
}) => {
  const [matrices, setMatrices] = useState<MatrixItem[]>([]);
  const [selectedFormat, setSelectedFormat] = useState<InputFormatMode>('auto');
  const [includeIdHeader, setIncludeIdHeader] = useState<boolean>(true);
  const [selectedMatrixId, setSelectedMatrixId] = useState<string | null>(null);
  const [copied, setCopied] = useState<boolean>(false);
  const [copiedBasis, setCopiedBasis] = useState<boolean>(false);
  const [pasteText, setPasteText] = useState<string>('');
  const [isPasteOpen, setIsPasteOpen] = useState<boolean>(false);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [progressStatus, setProgressStatus] = useState<string | null>(null);

  // Model calibration / fitting state
  const [targetK, setTargetK] = useState<number>(autoencoder.k || 16);
  const [calibrationInfo, setCalibrationInfo] = useState<{ 
    count: number; 
    varianceExplained: number; 
    accuracy: number;
    k: number;
  } | null>(null);

  // Dynamic matrix dimensions configuration
  const [targetRows, setTargetRows] = useState<number>(autoencoder.rows || 45);
  const [targetCols, setTargetCols] = useState<number>(autoencoder.cols || 45);
  const [autoAdaptDimensions, setAutoAdaptDimensions] = useState<boolean>(true);
  const [dimensionNotice, setDimensionNotice] = useState<string | null>(null);

  // Search & Pagination for large numbers of matrices
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [currentPage, setCurrentPage] = useState<number>(1);
  const pageSize = 25;

  // Fit PCA basis on current matrices
  const fitPcaOnItems = (items: MatrixItem[], currentK = targetK) => {
    if (items.length < 2) {
      // Just encode with whatever basis or identity
      const encoded = items.map(m => ({
        ...m,
        latent: autoencoder.encode(m.data)
      }));
      setMatrices(encoded);
      setCalibrationInfo(null);
      return;
    }

    autoencoder.reconfigure(targetRows, targetCols, currentK);
    const fitRes = autoencoder.fit(items.map(m => m.data));
    const roundTrip = autoencoder.roundTripMetrics(items.map(m => m.data), 0.5);

    const cumVar = fitRes.cumulativeVarianceRatios.slice(-1)[0] ?? 1.0;
    setCalibrationInfo({
      count: items.length,
      varianceExplained: cumVar,
      accuracy: roundTrip.meanAccuracy,
      k: fitRes.numComponents,
    });

    const encoded = items.map(m => ({
      ...m,
      latent: autoencoder.encode(m.data)
    }));

    setMatrices(encoded);
    if (!selectedMatrixId && encoded.length > 0) {
      setSelectedMatrixId(encoded[0].id);
    }
  };

  // Calibrate button
  const handleCalibrateModel = () => {
    if (matrices.length === 0) return;
    fitPcaOnItems(matrices, targetK);
  };

  // Reset to default weights
  const handleResetCalibration = () => {
    autoencoder.reset();
    setCalibrationInfo(null);
    setMatrices(prev => prev.map(m => ({
      ...m,
      latent: autoencoder.encode(m.data)
    })));
  };

  // Apply dimensions change
  const handleUpdateDimensions = (r: number, c: number, kVal = targetK) => {
    const validR = Math.max(1, Math.min(2000, r));
    const validC = Math.max(1, Math.min(2000, c));
    const validK = Math.max(2, Math.min(64, kVal));
    setTargetRows(validR);
    setTargetCols(validC);
    setTargetK(validK);
    autoencoder.reconfigure(validR, validC, validK);
    setCalibrationInfo(null);

    if (matrices.length > 0) {
      fitPcaOnItems(matrices, validK);
    }
    setDimensionNotice(`PCA reconfigured to ${validR} × ${validC} (${validR * validC} cells) → K=${validK} latent dims`);
    setTimeout(() => setDimensionNotice(null), 4000);
  };

  // Helper to process and encode a parsed list of MatrixItems
  const processAndSetMatrices = async (parsed: MatrixItem[]) => {
    if (parsed.length === 0) return;

    // Check if uploaded matrices have different dimensions from current configuration
    const sampleItem = parsed[0];
    let activeR = targetRows;
    let activeC = targetCols;

    if (autoAdaptDimensions && (sampleItem.rows !== targetRows || sampleItem.cols !== targetCols)) {
      activeR = sampleItem.rows;
      activeC = sampleItem.cols;
      setTargetRows(activeR);
      setTargetCols(activeC);
      autoencoder.reconfigure(activeR, activeC, targetK);
      setCalibrationInfo(null);
      setDimensionNotice(
        `Auto-detected dimensions: ${activeR} × ${activeC} (${activeR * activeC} features). PCA configured.`
      );
      setTimeout(() => setDimensionNotice(null), 5000);
    }

    const combinedAll = [...matrices, ...parsed];

    if (combinedAll.length >= 2) {
      setProgressStatus(`Fitting PCA basis to ${combinedAll.length} matrices...`);
      await new Promise(resolve => setTimeout(resolve, 10));
      fitPcaOnItems(combinedAll, targetK);
    } else {
      const itemsWithLatent = combinedAll.map(m => ({
        ...m,
        latent: autoencoder.encode(m.data)
      }));
      setMatrices(itemsWithLatent);
      if (!selectedMatrixId && itemsWithLatent.length > 0) {
        setSelectedMatrixId(itemsWithLatent[0].id);
      }
    }

    setProgressStatus(null);
  };

  // When files are selected
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    setIsProcessing(true);
    const totalFiles = files.length;
    const parsedList: MatrixItem[] = [];

    for (let i = 0; i < totalFiles; i++) {
      if (totalFiles > 10 && i % 10 === 0) {
        setProgressStatus(`Reading file ${i + 1} of ${totalFiles}...`);
        await new Promise(resolve => setTimeout(resolve, 0));
      }

      const file = files[i];
      const text = await file.text();
      const parsed = parseMatrixPayload(
        text, 
        file.name.replace(/\.[^/.]+$/, ''), 
        targetRows, 
        targetCols,
        selectedFormat === 'auto' ? undefined : selectedFormat
      );
      parsedList.push(...parsed);
    }

    await processAndSetMatrices(parsedList);
    setIsProcessing(false);
    setProgressStatus(null);
    e.target.value = '';
  };

  // Drag and drop handler
  const handleDrop = async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (!e.dataTransfer.files || e.dataTransfer.files.length === 0) return;

    setIsProcessing(true);
    const files = e.dataTransfer.files;
    const totalFiles = files.length;
    const parsedList: MatrixItem[] = [];

    for (let i = 0; i < totalFiles; i++) {
      if (totalFiles > 10 && i % 10 === 0) {
        setProgressStatus(`Reading file ${i + 1} of ${totalFiles}...`);
        await new Promise(resolve => setTimeout(resolve, 0));
      }

      const file = files[i];
      const text = await file.text();
      const parsed = parseMatrixPayload(
        text, 
        file.name.replace(/\.[^/.]+$/, ''), 
        targetRows, 
        targetCols,
        selectedFormat === 'auto' ? undefined : selectedFormat
      );
      parsedList.push(...parsed);
    }

    await processAndSetMatrices(parsedList);
    setIsProcessing(false);
    setProgressStatus(null);
  };

  // Paste handler
  const handleApplyPastedText = async () => {
    if (!pasteText.trim()) return;
    setIsProcessing(true);
    const parsed = parseMatrixPayload(
      pasteText,
      `pasted_matrix_${Date.now()}`,
      targetRows,
      targetCols,
      selectedFormat === 'auto' ? undefined : selectedFormat
    );

    await processAndSetMatrices(parsed);
    setPasteText('');
    setIsPasteOpen(false);
    setIsProcessing(false);
  };

  // Load sample dataset
  const handleLoadSample = (sampleType: '45x45_binary' | 'fasta' | 'sparse_coo' | '50x50' | '10x10' | 'continuous_floats') => {
    let text = '';
    let sampleName = '';
    let sRows = targetRows;
    let sCols = targetCols;

    if (sampleType === '45x45_binary') {
      text = SAMPLE_45X45_BINARY_COO_CSV;
      sampleName = 'binary_45x45';
      sRows = 45;
      sCols = 45;
    } else if (sampleType === 'fasta') {
      text = SAMPLE_FASTA_SEQUENCES;
      sampleName = 'protein_seq';
      sRows = 200;
      sCols = 21;
    } else if (sampleType === 'sparse_coo') {
      text = SAMPLE_SPARSE_COO_CSV;
      sampleName = 'sparse_coo';
      sRows = 200;
      sCols = 21;
    } else if (sampleType === '50x50') {
      text = SAMPLE_50X50_COO_CSV;
      sampleName = 'network_50x50';
      sRows = 50;
      sCols = 50;
    } else if (sampleType === '10x10') {
      text = SAMPLE_10X10_DENSE_CSV;
      sampleName = 'grid_10x10';
      sRows = 10;
      sCols = 10;
    } else if (sampleType === 'continuous_floats') {
      text = SAMPLE_CONTINUOUS_FLOAT_CSV;
      sampleName = 'floats_10x10';
      sRows = 10;
      sCols = 10;
    }

    setTargetRows(sRows);
    setTargetCols(sCols);
    autoencoder.reconfigure(sRows, sCols, targetK);

    const parsed = parseMatrixPayload(text, sampleName, sRows, sCols);
    if (parsed.length >= 2) {
      fitPcaOnItems(parsed, targetK);
    } else {
      const withLatent = parsed.map(item => ({
        ...item,
        latent: autoencoder.encode(item.data)
      }));
      setMatrices(withLatent);
      if (withLatent.length > 0) setSelectedMatrixId(withLatent[0].id);
    }
  };

  // Filtered & Paginated matrices
  const filteredMatrices = useMemo(() => {
    if (!searchQuery.trim()) return matrices;
    const q = searchQuery.toLowerCase();
    return matrices.filter(m => m.id.toLowerCase().includes(q) || m.name.toLowerCase().includes(q));
  }, [matrices, searchQuery]);

  const totalPages = Math.ceil(filteredMatrices.length / pageSize) || 1;
  const paginatedMatrices = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return filteredMatrices.slice(start, start + pageSize);
  }, [filteredMatrices, currentPage, pageSize]);

  // Generate collective download CSV
  const latentCsvText = useMemo(() => {
    if (matrices.length === 0) return '';
    const items = matrices.map(m => ({ 
      id: m.id, 
      latent: m.latent || new Array(targetK).fill(0.5) 
    }));
    const isAllBinary = matrices.every(m => m.isBinary);
    return formatLatentCsv(items, includeIdHeader, { 
      rows: targetRows, 
      cols: targetCols,
      k: targetK,
      valueType: isAllBinary ? 'binary_01' : 'continuous_numeric'
    });
  }, [matrices, includeIdHeader, targetRows, targetCols, targetK]);

  // Collective download action
  const handleDownloadCollectiveCsv = () => {
    if (!latentCsvText) return;
    const blob = new Blob([latentCsvText], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `latent_vectors_${matrices.length}matrices_${targetRows}x${targetCols}_k${targetK}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };

  // Export PCA Basis (JSON) action
  const handleDownloadBasisJson = () => {
    const serialized = autoencoder.serializeBasis();
    if (!serialized) return;
    const blob = new Blob([serialized], { type: 'application/json;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `pca_basis_${targetRows}x${targetCols}_k${targetK}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  // Copy collective CSV to clipboard
  const handleCopyCsv = () => {
    if (!latentCsvText) return;
    navigator.clipboard.writeText(latentCsvText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Transfer directly to Module 2
  const handleTransfer = () => {
    if (onTransferToDecoder && latentCsvText) {
      onTransferToDecoder(latentCsvText, matrices, { rows: targetRows, cols: targetCols });
    }
  };

  const selectedMatrix = matrices.find(m => m.id === selectedMatrixId) || matrices[0];

  return (
    <div className="flex flex-col gap-6">
      {/* Module Banner / Instructions */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-xs">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-5 border-b border-slate-200">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold text-cyan-700 uppercase tracking-wider">Module 1</span>
              <span className="text-slate-300">·</span>
              <h2 className="text-lg font-bold text-slate-900">
                Matrix Encoder &amp; PCA Latent Vector Extraction
              </h2>
            </div>
            <p className="text-xs text-slate-600 mt-1">
              Upload <strong>any number of matrices</strong> in extended format. The PCA engine extracts <strong>[0, 1]^K</strong> latent vectors ready for <strong>Bayesian Optimization (BO)</strong>.
            </p>
          </div>

          {/* Preset sample buttons */}
          <div className="flex flex-wrap items-center gap-1.5 text-xs">
            <span className="text-slate-500 text-[11px] mr-1">Instant Samples:</span>
            <button
              onClick={() => handleLoadSample('45x45_binary')}
              className="text-[11px] px-2.5 py-1.5 rounded-lg bg-cyan-50 hover:bg-cyan-100 text-cyan-900 font-semibold transition-colors border border-cyan-300 flex items-center gap-1 shadow-2xs"
              title="45x45 Binary Matrix (Primary BO format)"
            >
              <Sparkles className="w-3.5 h-3.5 text-cyan-600" />
              45×45 Binary (BO)
            </button>
            <button
              onClick={() => handleLoadSample('fasta')}
              className="text-[11px] px-2.5 py-1.5 rounded-lg bg-slate-50 hover:bg-slate-100 text-slate-700 font-medium transition-colors border border-slate-200 flex items-center gap-1 shadow-2xs"
            >
              200×21 Protein
            </button>
            <button
              onClick={() => handleLoadSample('sparse_coo')}
              className="text-[11px] px-2.5 py-1.5 rounded-lg bg-slate-50 hover:bg-slate-100 text-slate-700 font-medium transition-colors border border-slate-200 flex items-center gap-1 shadow-2xs"
            >
              200×21 COO
            </button>
            <button
              onClick={() => handleLoadSample('50x50')}
              className="text-[11px] px-2.5 py-1.5 rounded-lg bg-slate-50 hover:bg-slate-100 text-slate-700 font-medium transition-colors border border-slate-200 flex items-center gap-1 shadow-2xs"
            >
              50×50 Network
            </button>
            <button
              onClick={() => handleLoadSample('10x10')}
              className="text-[11px] px-2.5 py-1.5 rounded-lg bg-slate-50 hover:bg-slate-100 text-slate-700 font-medium transition-colors border border-slate-200 flex items-center gap-1 shadow-2xs"
            >
              10×10 Grid
            </button>
            <button
              onClick={() => handleLoadSample('continuous_floats')}
              className="text-[11px] px-2.5 py-1.5 rounded-lg bg-emerald-50 hover:bg-emerald-100 text-emerald-900 font-medium transition-colors border border-emerald-200 flex items-center gap-1 shadow-2xs"
              title="Matrix with generic continuous numerical values"
            >
              10×10 Floats
            </button>
          </div>
        </div>

        {/* Matrix Dimensions & K Control Bar */}
        <div className="py-3 px-4 mt-4 bg-slate-50 border border-slate-200 rounded-lg flex flex-wrap items-center justify-between gap-3 text-xs">
          <div className="flex items-center gap-2">
            <Grid className="w-4 h-4 text-cyan-700" />
            <span className="font-semibold text-slate-800">Dimensions:</span>
            <span className="px-2 py-0.5 bg-white border border-slate-300 rounded font-mono font-bold text-cyan-800 shadow-2xs">
              {targetRows} × {targetCols} ({targetRows * targetCols} cells)
            </span>
            <span className="text-slate-400">→</span>
            <div className="flex items-center gap-1">
              <span className="font-semibold text-slate-800">Latent K:</span>
              <span className="px-2 py-0.5 bg-white border border-slate-300 rounded font-mono font-bold text-emerald-800 shadow-2xs">
                K = {targetK} dims
              </span>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            {/* Quick Size Presets */}
            <div className="flex items-center gap-1">
              <span className="text-slate-500 text-[11px]">Size:</span>
              {[
                { r: 45, c: 45, label: '45×45' },
                { r: 50, c: 50, label: '50×50' },
                { r: 200, c: 21, label: '200×21' },
                { r: 10, c: 10, label: '10×10' },
              ].map(p => (
                <button
                  key={p.label}
                  onClick={() => handleUpdateDimensions(p.r, p.c, targetK)}
                  className={`px-2 py-0.5 rounded text-[11px] transition-colors ${
                    targetRows === p.r && targetCols === p.c
                      ? 'bg-cyan-700 text-white font-bold shadow-2xs'
                      : 'bg-white text-slate-600 hover:text-slate-900 border border-slate-200'
                  }`}
                >
                  {p.label}
                </button>
              ))}
            </div>

            {/* K Latent Dimensions Selector */}
            <div className="flex items-center gap-1.5 bg-white border border-slate-200 rounded px-2 py-0.5 shadow-2xs">
              <SlidersHorizontal className="w-3.5 h-3.5 text-slate-400" />
              <span className="text-[11px] text-slate-500">Latent K:</span>
              {[8, 16, 24, 32].map(kVal => (
                <button
                  key={kVal}
                  onClick={() => handleUpdateDimensions(targetRows, targetCols, kVal)}
                  className={`px-1.5 py-0.5 rounded text-[10px] font-mono transition-colors ${
                    targetK === kVal
                      ? 'bg-emerald-700 text-white font-bold'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  {kVal}
                </button>
              ))}
            </div>

            {/* Custom Rows x Cols input */}
            <div className="flex items-center gap-1 bg-white border border-slate-200 rounded px-1.5 py-0.5 shadow-2xs">
              <span className="text-[11px] text-slate-500">Custom:</span>
              <input
                type="number"
                min="1"
                max="2000"
                value={targetRows}
                onChange={e => {
                  const val = parseInt(e.target.value, 10);
                  if (!isNaN(val) && val > 0) handleUpdateDimensions(val, targetCols, targetK);
                }}
                className="w-12 px-1 py-0.5 bg-slate-50 border border-slate-300 rounded text-center text-[11px] font-mono text-cyan-800 font-semibold"
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
                  if (!isNaN(val) && val > 0) handleUpdateDimensions(targetRows, val, targetK);
                }}
                className="w-12 px-1 py-0.5 bg-slate-50 border border-slate-300 rounded text-center text-[11px] font-mono text-cyan-800 font-semibold"
                title="Target Columns"
              />
            </div>
          </div>
        </div>

        {/* Dimension auto-adapt notification */}
        {dimensionNotice && (
          <div className="mt-3 p-2.5 bg-cyan-50 border border-cyan-200 rounded-lg text-xs text-cyan-900 flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-cyan-600 shrink-0" />
            <span>{dimensionNotice}</span>
          </div>
        )}

        {/* PCA Subspace Fit Status Banner */}
        {calibrationInfo && (
          <div className="mt-3 p-3 bg-emerald-50 border border-emerald-200 rounded-lg text-xs text-emerald-900 flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <Zap className="w-4 h-4 text-emerald-600 shrink-0" />
              <div>
                <span className="font-bold">PCA Subspace Fitted:</span>{' '}
                <span>Fitted on {calibrationInfo.count} matrices with K = {calibrationInfo.k} components.</span>
                <span className="ml-2 px-1.5 py-0.5 bg-white border border-emerald-300 rounded font-semibold text-emerald-800">
                  {(calibrationInfo.varianceExplained * 100).toFixed(1)}% Variance Explained
                </span>
                <span className="ml-2 px-1.5 py-0.5 bg-white border border-emerald-300 rounded font-semibold text-cyan-800">
                  {(calibrationInfo.accuracy * 100).toFixed(2)}% Round-Trip Hamming Accuracy
                </span>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={handleDownloadBasisJson}
                className="px-2.5 py-1 bg-white hover:bg-emerald-100 border border-emerald-300 text-emerald-900 rounded font-medium flex items-center gap-1 shadow-2xs"
                title="Download PCA basis JSON for offline/future decoding in Module 2"
              >
                <FileCode className="w-3.5 h-3.5 text-emerald-700" />
                Export Basis JSON
              </button>
              <button
                onClick={handleCalibrateModel}
                className="px-2 py-1 bg-white hover:bg-emerald-100 border border-emerald-300 text-emerald-900 rounded font-medium flex items-center gap-1 shadow-2xs"
                title="Re-fit basis"
              >
                <RefreshCw className="w-3 h-3 text-emerald-700" />
                Re-fit
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Upload Zone & Formats */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Drag & Drop Upload Card */}
        <div 
          onDragOver={e => e.preventDefault()}
          onDrop={handleDrop}
          className="lg:col-span-2 border-2 border-dashed border-slate-300 hover:border-cyan-500 bg-white rounded-xl p-8 flex flex-col items-center justify-center text-center transition-all cursor-pointer shadow-xs group"
          onClick={() => document.getElementById('matrix-file-input')?.click()}
        >
          <input
            id="matrix-file-input"
            type="file"
            multiple
            accept=".csv,.txt,.fasta,.fa,.tsv,.json"
            onChange={handleFileUpload}
            className="hidden"
          />
          <div className="w-14 h-14 rounded-2xl bg-cyan-50 text-cyan-700 flex items-center justify-center mb-4 group-hover:scale-105 transition-transform shadow-xs">
            <Upload className="w-7 h-7" />
          </div>
          <h3 className="text-base font-bold text-slate-800 mb-1">
            Drag &amp; Drop Matrix Files Here (No File Limit)
          </h3>
          <p className="text-xs text-slate-500 max-w-md mb-4">
            Upload multiple files at once or single batch files containing many matrices. Supports <strong>CSV, TSV, FASTA, COO Coordinates, JSON</strong>.
          </p>
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={e => {
                e.stopPropagation();
                document.getElementById('matrix-file-input')?.click();
              }}
              className="px-4 py-2 bg-cyan-700 hover:bg-cyan-800 text-white rounded-lg text-xs font-semibold shadow-xs flex items-center gap-1.5 transition-colors"
            >
              <Upload className="w-3.5 h-3.5" />
              Browse Files
            </button>
            <button
              type="button"
              onClick={e => {
                e.stopPropagation();
                setIsPasteOpen(true);
              }}
              className="px-4 py-2 bg-slate-50 hover:bg-slate-100 border border-slate-200 text-slate-700 rounded-lg text-xs font-semibold shadow-2xs flex items-center gap-1.5 transition-colors"
            >
              <FileText className="w-3.5 h-3.5 text-slate-500" />
              Paste Raw Text
            </button>
          </div>
        </div>

        {/* Input format selector & options */}
        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs flex flex-col justify-between">
          <div>
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500 mb-3 flex items-center gap-1.5">
              <Sliders className="w-3.5 h-3.5 text-cyan-700" />
              Input Parsing Options
            </h4>
            <div className="space-y-3">
              <div>
                <label className="text-xs font-semibold text-slate-700 block mb-1">
                  Format Mode:
                </label>
                <select
                  value={selectedFormat}
                  onChange={e => setSelectedFormat(e.target.value as InputFormatMode)}
                  className="w-full bg-slate-50 border border-slate-300 rounded-lg px-2.5 py-1.5 text-xs text-slate-800 font-medium focus:ring-1 focus:ring-cyan-500"
                >
                  <option value="auto">Auto-Detect Format (Recommended)</option>
                  <option value="dense_csv">Dense 2D CSV Table</option>
                  <option value="sparse_coo">Sparse Coordinate (COO: row,col,val)</option>
                  <option value="flattened_csv">Flattened CSV (1 row per matrix)</option>
                  <option value="fasta_sequence">FASTA Sequence (One-Hot)</option>
                  <option value="json">JSON Array of Matrices</option>
                </select>
              </div>

              <div className="pt-2 border-t border-slate-100">
                <label className="flex items-center gap-2 cursor-pointer text-xs text-slate-700">
                  <input
                    type="checkbox"
                    checked={includeIdHeader}
                    onChange={e => setIncludeIdHeader(e.target.checked)}
                    className="rounded border-slate-300 text-cyan-600 focus:ring-cyan-500"
                  />
                  <span>Include <code>matrix_id</code> header in Latent CSV</span>
                </label>
              </div>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-slate-200 text-[11px] text-slate-500">
            Matrices loaded: <strong className="text-slate-800">{matrices.length}</strong>
            {matrices.length > 0 && (
              <button
                onClick={() => {
                  setMatrices([]);
                  setSelectedMatrixId(null);
                  setCalibrationInfo(null);
                }}
                className="ml-3 text-red-600 hover:text-red-700 font-semibold"
              >
                Clear All
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Progress status notification */}
      {progressStatus && (
        <div className="p-3 bg-cyan-50 border border-cyan-200 rounded-lg text-xs text-cyan-900 flex items-center gap-2 animate-pulse">
          <RefreshCw className="w-4 h-4 animate-spin text-cyan-600" />
          <span>{progressStatus}</span>
        </div>
      )}

      {/* Main Content Area: Matrix List & Latent CSV Inspector */}
      {matrices.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Matrix Explorer Column */}
          <div className="lg:col-span-5 bg-white border border-slate-200 rounded-xl p-5 shadow-xs flex flex-col">
            <div className="flex items-center justify-between pb-3 border-b border-slate-200 mb-3">
              <div className="flex items-center gap-2">
                <Layers className="w-4 h-4 text-cyan-700" />
                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-700">
                  Uploaded Matrices ({matrices.length})
                </h4>
              </div>
              {/* Search input */}
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

            {/* List */}
            <div className="space-y-1.5 max-h-96 overflow-y-auto pr-1">
              {paginatedMatrices.map(m => {
                const isSelected = m.id === selectedMatrixId;
                return (
                  <div
                    key={m.id}
                    onClick={() => setSelectedMatrixId(m.id)}
                    className={`p-2.5 rounded-lg border text-xs cursor-pointer transition-all flex items-center justify-between ${
                      isSelected
                        ? 'bg-cyan-50 border-cyan-400 shadow-2xs font-semibold text-cyan-950'
                        : 'bg-white border-slate-200 hover:border-slate-300 text-slate-700'
                    }`}
                  >
                    <div className="flex items-center gap-2 truncate">
                      <div className={`w-2 h-2 rounded-full ${m.isBinary ? 'bg-cyan-500' : 'bg-emerald-500'}`} />
                      <span className="truncate">{m.id}</span>
                    </div>
                    <div className="flex items-center gap-2 text-[11px] text-slate-400 shrink-0 font-mono">
                      <span>{m.rows}×{m.cols}</span>
                      <span className="text-slate-300">·</span>
                      <span className="text-cyan-700 font-bold">{m.nonZeroCount} active</span>
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

          {/* Selected Matrix Heatmap & Latent Vector Preview */}
          <div className="lg:col-span-7 bg-white border border-slate-200 rounded-xl p-5 shadow-xs flex flex-col justify-between">
            {selectedMatrix && (
              <div>
                <div className="flex items-center justify-between pb-3 border-b border-slate-200 mb-3">
                  <div className="flex items-center gap-2">
                    <Eye className="w-4 h-4 text-cyan-700" />
                    <h4 className="text-xs font-bold uppercase tracking-wider text-slate-700">
                      Matrix Preview: <span className="text-cyan-900 font-mono">{selectedMatrix.id}</span>
                    </h4>
                  </div>
                  <span className="text-[11px] px-2 py-0.5 rounded bg-slate-100 text-slate-600 font-medium">
                    {selectedMatrix.isBinary ? 'Binary {0, 1}' : 'Continuous Float'}
                  </span>
                </div>

                {/* Heatmap */}
                <div className="flex justify-center bg-slate-50 p-2 rounded-lg border border-slate-200 mb-4">
                  <MatrixHeatmap
                    rows={selectedMatrix.rows}
                    cols={selectedMatrix.cols}
                    data={selectedMatrix.data}
                    valueType={selectedMatrix.isBinary ? 'binary_01' : 'continuous_numeric'}
                  />
                </div>

                {/* Latent Coordinate Badges [0, 1]^K */}
                {selectedMatrix.latent && (
                  <div>
                    <h5 className="text-[11px] font-bold text-slate-600 mb-1.5 flex items-center justify-between">
                      <span>Normalized Latent Coordinates (z ∈ [0, 1]^{targetK}):</span>
                      <span className="text-cyan-700 font-mono">BO Search Hypercube</span>
                    </h5>
                    <div className="grid grid-cols-4 sm:grid-cols-8 gap-1.5">
                      {selectedMatrix.latent.map((val, idx) => (
                        <div 
                          key={idx}
                          className="bg-slate-50 border border-slate-200 rounded p-1 text-center font-mono text-[10px]"
                        >
                          <div className="text-slate-400 text-[9px]">z{idx + 1}</div>
                          <div className="font-bold text-cyan-800">{val.toFixed(3)}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Latent CSV Collective Download & Module 2 Bridge Bar */}
      {matrices.length > 0 && (
        <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-xs flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
              <Download className="w-4 h-4 text-cyan-700" />
              Collective Latent Vectors CSV (Ready for Bayesian Optimization)
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Contains <strong>{matrices.length} rows</strong> (1 row per matrix) with <strong>{targetK} normalized parameters [z₁ .. z_{targetK}]</strong> in range <code>[0.0, 1.0]</code>.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2.5">
            <button
              onClick={handleCopyCsv}
              className="px-3.5 py-2 bg-slate-50 hover:bg-slate-100 border border-slate-200 text-slate-700 rounded-lg text-xs font-semibold shadow-2xs flex items-center gap-1.5 transition-colors"
            >
              {copied ? <Check className="w-4 h-4 text-emerald-600" /> : <Copy className="w-4 h-4 text-slate-500" />}
              {copied ? 'Copied CSV' : 'Copy CSV'}
            </button>

            <button
              onClick={handleDownloadBasisJson}
              className="px-3.5 py-2 bg-slate-50 hover:bg-slate-100 border border-slate-200 text-slate-700 rounded-lg text-xs font-semibold shadow-2xs flex items-center gap-1.5 transition-colors"
              title="Download PCA basis JSON for offline or standalone Module 2 decoding"
            >
              <FileCode className="w-4 h-4 text-emerald-600" />
              Download Basis JSON
            </button>

            <button
              onClick={handleDownloadCollectiveCsv}
              className="px-4 py-2 bg-cyan-700 hover:bg-cyan-800 text-white rounded-lg text-xs font-semibold shadow-xs flex items-center gap-1.5 transition-colors"
            >
              <Download className="w-4 h-4" />
              Download Latent CSV
            </button>

            {onTransferToDecoder && (
              <button
                onClick={handleTransfer}
                className="px-4 py-2 bg-emerald-700 hover:bg-emerald-800 text-white rounded-lg text-xs font-semibold shadow-xs flex items-center gap-1.5 transition-colors"
              >
                <span>Transfer to Decoder</span>
                <ArrowRight className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>
      )}

      {/* Paste Modal */}
      {isPasteOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-xs">
          <div className="bg-white border border-slate-200 rounded-xl p-6 max-w-xl w-full shadow-2xl space-y-4">
            <h3 className="text-sm font-bold text-slate-900">Paste Raw Matrix Data</h3>
            <p className="text-xs text-slate-500">
              Paste dense 2D rows, sparse COO lines (matrix_id, row, col, val), or flattened vectors.
            </p>
            <textarea
              rows={8}
              value={pasteText}
              onChange={e => setPasteText(e.target.value)}
              placeholder="Paste raw matrix CSV or text here..."
              className="w-full bg-slate-50 border border-slate-200 rounded-lg p-3 text-xs font-mono text-slate-800 focus:ring-1 focus:ring-cyan-500"
            />
            <div className="flex justify-end gap-2 text-xs">
              <button
                onClick={() => setIsPasteOpen(false)}
                className="px-3.5 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-lg font-medium"
              >
                Cancel
              </button>
              <button
                onClick={handleApplyPastedText}
                className="px-4 py-1.5 bg-cyan-700 hover:bg-cyan-800 text-white rounded-lg font-semibold shadow-xs"
              >
                Parse &amp; Encode
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
