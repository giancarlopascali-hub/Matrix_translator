/**
 * Model Architecture Modal & Python BO Pipeline Generator (Bright Theme)
 */
import React, { useState } from 'react';
import { MatrixPCA } from '../lib/autoencoder';
import { downloadBlob } from '../lib/matrixFormats';
import { X, Copy, Check, Download, Sparkles, Sliders, Cpu, AlertTriangle } from 'lucide-react';

interface ModelArchitectureModalProps {
  isOpen: boolean;
  onClose: () => void;
  autoencoder: MatrixPCA;
}

export const ModelArchitectureModal: React.FC<ModelArchitectureModalProps> = ({
  isOpen,
  onClose,
  autoencoder,
}) => {
  const [copied, setCopied] = useState(false);
  const pythonScript = autoencoder.generatePythonScript(0.5);

  if (!isOpen) return null;

  const handleCopy = () => {
    navigator.clipboard.writeText(pythonScript);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const blob = new Blob([pythonScript], { type: 'text/x-python;charset=utf-8;' });
    const filename = `matrix_pca_${autoencoder.rows}x${autoencoder.cols}_k${autoencoder.k}_bo_pipeline.py`;
    downloadBlob(blob, filename);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-xs">
      <div className="relative w-full max-w-4xl max-h-[90vh] flex flex-col bg-white border border-slate-200 rounded-xl shadow-2xl text-slate-800 overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200 bg-slate-50">
          <div className="flex items-center gap-2">
            <Cpu className="w-5 h-5 text-cyan-700" />
            <div>
              <h2 className="text-base font-bold text-slate-900">
                Matrix PCA Architecture &amp; Bayesian Optimization Pipeline
              </h2>
              <p className="text-xs text-slate-500">
                Linear Subspace Compression into [0, 1]^K Bounds for EDOS-Family Bayesian Optimizers
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content body */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* Architecture diagram */}
          <div>
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500 mb-3">
              Mathematical Pipeline Flow
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-5 gap-3 items-center text-center font-mono text-xs">
              <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg shadow-2xs">
                <div className="text-slate-500 text-[10px] font-sans">Input Matrix</div>
                <div className="font-bold text-cyan-800 text-sm mt-1">{autoencoder.inputDim} cells</div>
                <div className="text-[10px] text-slate-400 mt-0.5">{autoencoder.rows} × {autoencoder.cols} matrix</div>
              </div>

              <div className="text-slate-400 font-bold hidden md:block">→</div>

              {/* PCA Subspace */}
              <div className="p-3 bg-cyan-50 border-2 border-cyan-600 rounded-lg shadow-xs">
                <div className="text-cyan-800 font-bold text-[10px] font-sans">Latent Space (BO)</div>
                <div className="font-extrabold text-cyan-900 text-base mt-1">K = {autoencoder.k}</div>
                <div className="text-[10px] text-cyan-700 font-semibold mt-0.5">[z₁ .. z_K] ∈ [0, 1]^K</div>
              </div>

              <div className="text-slate-400 font-bold hidden md:block">→</div>

              {/* Reconstruction */}
              <div className="p-3 bg-emerald-50 border border-emerald-500 rounded-lg shadow-2xs">
                <div className="text-emerald-800 text-[10px] font-sans">Strict Threshold Decoder</div>
                <div className="font-bold text-emerald-800 text-sm mt-1">{autoencoder.inputDim} cells</div>
                <div className="text-[10px] text-emerald-600 mt-0.5">value &gt; threshold ? 1 : 0</div>
              </div>
            </div>
          </div>

          {/* Details & BO explanation */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-slate-700">
            <div className="bg-slate-50 p-4 border border-slate-200 rounded-lg space-y-2">
              <h4 className="font-bold text-slate-900 flex items-center gap-1.5">
                <Sparkles className="w-4 h-4 text-cyan-700" />
                Why PCA for Bayesian Optimization?
              </h4>
              <p className="leading-relaxed">
                Rather than treating all <strong>{autoencoder.inputDim}</strong> matrix entries independently, the model uses data-supported PCA directions and deterministic orthogonal completion to keep every vector exactly <strong>K = {autoencoder.k}</strong> values long.
              </p>
              <p className="leading-relaxed">
                Each projection is normalized to the unit hypercube <strong>[0, 1]^K</strong>, which matches the standard parameter bounds expected by <strong>EDOS</strong> and Gaussian Process Bayesian Optimizers.
              </p>
              <p className="leading-relaxed">
                Exact binary reconstruction is measured after fitting. It is guaranteed for the fitted samples only when the retained components span their affine variation; otherwise the encoder reports the representation as lossy.
              </p>
            </div>

            <div className="bg-slate-50 p-4 border border-slate-200 rounded-lg space-y-2">
              <h4 className="font-bold text-slate-900 flex items-center gap-1.5">
                <Sliders className="w-4 h-4 text-emerald-700" />
                Reconstruction &amp; Strictly &apos;&gt;&apos; Threshold
              </h4>
              <p className="leading-relaxed">
                The continuous reconstruction from coordinates <code>z</code> is obtained via the inverse PCA transform:
              </p>
              <code className="block bg-white p-2 rounded border border-slate-200 font-mono text-[11px] text-slate-800">
                x_recon = mean + sum(z_raw_k * component_k)
              </code>
              <p className="leading-relaxed text-slate-600">
                For binary matrices, cells are mapped using the strict inequality: <code>value &gt; threshold ? 1 : 0</code> (default θ = 0.50).
              </p>
            </div>
          </div>

          <div className="flex items-start gap-2 rounded-lg border border-amber-300 bg-amber-50 p-3 text-xs text-amber-950">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-700" />
            <span>
              Latent CSV values and basis JSON are one matched model. Keep them together and do not decode a CSV with a different model ID. The Python helper below loads that same exported basis; it does not refit or approximate it.
            </span>
          </div>

          {/* Python code section */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500">
                Python / NumPy Basis Adapter for BO Integration
              </h3>
              <div className="flex items-center gap-2">
                <button
                  onClick={handleCopy}
                  className="flex items-center gap-1.5 px-3 py-1 bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs rounded transition-colors"
                >
                  {copied ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
                  {copied ? 'Copied' : 'Copy Code'}
                </button>
                <button
                  onClick={handleDownload}
                  className="flex items-center gap-1.5 px-3 py-1 bg-cyan-700 hover:bg-cyan-800 text-white text-xs rounded transition-colors shadow-2xs"
                >
                  <Download className="w-3.5 h-3.5" />
                  Download .py
                </button>
              </div>
            </div>
            <pre className="p-4 bg-slate-900 border border-slate-800 rounded-lg text-xs font-mono text-slate-200 overflow-x-auto max-h-72">
              <code>{pythonScript}</code>
            </pre>
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-slate-200 bg-slate-50 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-1.5 text-xs font-medium text-slate-700 hover:text-slate-900 bg-white hover:bg-slate-100 border border-slate-200 rounded-lg transition-colors shadow-2xs"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
