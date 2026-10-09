import React, { useMemo, useState } from 'react';
import {
  CheckCircle2,
  Download,
  FileSpreadsheet,
  ShieldCheck,
  XCircle,
} from 'lucide-react';
import { validateBinaryDesign, DesignCorner } from '../lib/designValidation';
import { LatentRow, ReconstructedMatrix } from '../lib/types';
import {
  downloadBlob,
  formatSingleDenseCsv,
  formatValidatedLatentCsv,
} from '../lib/matrixFormats';

interface DesignValidationPanelProps {
  matrices: ReconstructedMatrix[];
  latentRows: LatentRow[];
  sourceLatentCsv: string;
  threshold: number;
  decimalPrecision: number;
}

const CORNER_LABELS: Record<DesignCorner, string> = {
  top_left: 'top-left',
  top_right: 'top-right',
  bottom_left: 'bottom-left',
  bottom_right: 'bottom-right',
};

export const DesignValidationPanel: React.FC<DesignValidationPanelProps> = ({
  matrices,
  latentRows,
  sourceLatentCsv,
  threshold,
  decimalPrecision,
}) => {
  const [exportError, setExportError] = useState<string | null>(null);
  const results = useMemo(
    () => matrices.map(matrix => validateBinaryDesign(matrix)),
    [matrices],
  );
  const passedCount = results.reduce((count, result) => count + result.validation, 0);

  const handleDownloadMatrix = (matrix: ReconstructedMatrix) => {
    const content = formatSingleDenseCsv(matrix, decimalPrecision);
    const blob = new Blob([content], { type: 'text/csv;charset=utf-8;' });
    downloadBlob(blob, `${matrix.id}_${matrix.rows}x${matrix.cols}.csv`);
  };

  const handleDownloadValidatedLatents = () => {
    try {
      const content = formatValidatedLatentCsv(
        sourceLatentCsv,
        latentRows,
        results.map(result => result.validation),
        threshold,
      );
      const blob = new Blob([content], { type: 'text/csv;charset=utf-8;' });
      downloadBlob(blob, `latent_candidates_validated_${matrices.length}.csv`);
      setExportError(null);
    } catch (error) {
      setExportError(error instanceof Error ? error.message : 'Could not export the validated latent CSV.');
    }
  };

  return (
    <section className="rounded-xl border border-violet-200 bg-white p-5 shadow-xs">
      <div className="flex flex-col gap-4 border-b border-slate-200 pb-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <div className="flex items-center gap-2">
            <ShieldCheck className="h-5 w-5 text-violet-700" />
            <h3 className="text-sm font-bold text-slate-900">Validation Results</h3>
            <span className="rounded-full border border-violet-200 bg-violet-50 px-2 py-0.5 font-mono text-[11px] font-bold text-violet-800">
              {passedCount}/{results.length} passed
            </span>
          </div>
          <p className="mt-1 max-w-3xl text-xs leading-5 text-slate-500">
            A design passes when no more than one connected region is larger than 2×2 and at least two distinct matrix corners contain an all-one 2×2 block. Smaller disconnected islands are ignored.
          </p>
        </div>

        <button
          type="button"
          onClick={handleDownloadValidatedLatents}
          className="inline-flex shrink-0 items-center justify-center gap-1.5 rounded-lg bg-violet-700 px-4 py-2 text-xs font-semibold text-white shadow-xs transition-colors hover:bg-violet-800"
        >
          <FileSpreadsheet className="h-4 w-4" />
          Download Latent CSV + Validation
        </button>
      </div>

      {exportError && (
        <div className="mt-3 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-800">
          {exportError}
        </div>
      )}

      <div className="mt-4 max-h-[30rem] space-y-2 overflow-y-auto pr-1">
        {matrices.map((matrix, index) => {
          const result = results[index];
          const cornerText = result.qualifyingCorners.length > 0
            ? result.qualifyingCorners.map(corner => CORNER_LABELS[corner]).join(', ')
            : 'no qualifying corners';
          const connectivityText = result.connectivityPassed
            ? result.significantComponentCount === 1
              ? '1 larger region retained'
              : 'no larger region retained'
            : `${result.significantComponentCount} disconnected regions larger than 2×2`;
          const ignoredText = `${result.ignoredSmallComponentCount} small island${result.ignoredSmallComponentCount === 1 ? '' : 's'} ignored`;

          return (
            <div
              key={`${matrix.id}-${index}`}
              className={`flex flex-col gap-3 rounded-lg border px-3 py-3 sm:flex-row sm:items-center sm:justify-between ${
                result.validation === 1
                  ? 'border-emerald-200 bg-emerald-50/60'
                  : 'border-red-200 bg-red-50/50'
              }`}
            >
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  {result.validation === 1 ? (
                    <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-700" />
                  ) : (
                    <XCircle className="h-4 w-4 shrink-0 text-red-700" />
                  )}
                  <span className="truncate font-mono text-xs font-bold text-slate-900">{matrix.id}</span>
                  <span className={`rounded px-2 py-0.5 font-mono text-xs font-black ${
                    result.validation === 1
                      ? 'bg-emerald-700 text-white'
                      : 'bg-red-700 text-white'
                  }`}>
                    validation = {result.validation}
                  </span>
                </div>
                <p className="mt-1 pl-6 text-[11px] leading-4 text-slate-600">
                  {result.binaryValuesOnly
                    ? `${connectivityText} · ${ignoredText} · ${result.qualifyingCorners.length} corner blocks (${cornerText}) · ${result.activeCellCount} active cells`
                    : 'Validation failed because the reconstructed matrix is not strictly binary.'}
                </p>
              </div>

              <button
                type="button"
                onClick={() => handleDownloadMatrix(matrix)}
                className="inline-flex shrink-0 items-center justify-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs font-semibold text-slate-700 shadow-2xs transition-colors hover:border-violet-300 hover:text-violet-800"
                title="Download this complete dense matrix CSV"
              >
                <Download className="h-3.5 w-3.5" />
                Matrix CSV
              </button>
            </div>
          );
        })}
      </div>
    </section>
  );
};
