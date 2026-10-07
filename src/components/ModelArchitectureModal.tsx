/**
 * Model Architecture Modal & Python/Keras Script Generator (Bright Theme)
 */
import React, { useState } from 'react';
import { MatrixAutoencoder } from '../lib/autoencoder';
import { X, Copy, Check, Download, Layers } from 'lucide-react';

interface ModelArchitectureModalProps {
  isOpen: boolean;
  onClose: () => void;
  autoencoder: MatrixAutoencoder;
}

export const ModelArchitectureModal: React.FC<ModelArchitectureModalProps> = ({
  isOpen,
  onClose,
  autoencoder,
}) => {
  const [copied, setCopied] = useState(false);
  const kerasScript = autoencoder.generateKerasScript();

  if (!isOpen) return null;

  const handleCopy = () => {
    navigator.clipboard.writeText(kerasScript);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const blob = new Blob([kerasScript], { type: 'text/x-python;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'matrix_autoencoder_8d.py';
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-xs">
      <div className="relative w-full max-w-4xl max-h-[90vh] flex flex-col bg-white border border-slate-200 rounded-xl shadow-2xl text-slate-800 overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200 bg-slate-50">
          <div className="flex items-center gap-2">
            <Layers className="w-5 h-5 text-cyan-700" />
            <h2 className="text-base font-bold text-slate-900">
              8-Latent Autoencoder Architecture &amp; Python Keras Model
            </h2>
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
              Neural Topology Flow
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-7 gap-2 items-center text-center font-mono text-xs">
              <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
                <div className="text-slate-500 text-[10px]">Input Layer</div>
                <div className="font-bold text-cyan-800 text-sm mt-1">{autoencoder.config.inputDim}</div>
                <div className="text-[10px] text-slate-400 mt-0.5">{autoencoder.config.inputRows} × {autoencoder.config.inputCols} matrix</div>
              </div>

              <div className="text-slate-400 font-bold hidden md:block">→</div>

              <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
                <div className="text-slate-500 text-[10px]">Enc Hidden 1</div>
                <div className="font-bold text-slate-700 text-sm mt-1">{autoencoder.config.hidden1}</div>
                <div className="text-[10px] text-slate-400 mt-0.5">ReLU</div>
              </div>

              <div className="text-slate-400 font-bold hidden md:block">→</div>

              <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
                <div className="text-slate-500 text-[10px]">Enc Hidden 2</div>
                <div className="font-bold text-slate-700 text-sm mt-1">{autoencoder.config.hidden2}</div>
                <div className="text-[10px] text-slate-400 mt-0.5">ReLU</div>
              </div>

              <div className="text-slate-400 font-bold hidden md:block">→</div>

              {/* Bottleneck 8 */}
              <div className="p-3 bg-cyan-50 border-2 border-cyan-600 rounded-lg shadow-xs">
                <div className="text-cyan-800 font-bold text-[10px]">Bottleneck</div>
                <div className="font-extrabold text-cyan-900 text-base mt-1">8 Latent</div>
                <div className="text-[10px] text-cyan-700 font-semibold mt-0.5">[z1 .. z8]</div>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-7 gap-2 items-center text-center font-mono text-xs mt-3">
              <div className="p-3 bg-cyan-50 border-2 border-cyan-600 rounded-lg">
                <div className="text-cyan-800 font-bold text-[10px]">Bottleneck</div>
                <div className="font-extrabold text-cyan-900 text-base mt-1">8 Latent</div>
                <div className="text-[10px] text-cyan-700 font-semibold mt-0.5">Decoder Input</div>
              </div>

              <div className="text-slate-400 font-bold hidden md:block">→</div>

              <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
                <div className="text-slate-500 text-[10px]">Dec Hidden 1</div>
                <div className="font-bold text-slate-700 text-sm mt-1">{autoencoder.config.hidden2}</div>
                <div className="text-[10px] text-slate-400 mt-0.5">ReLU</div>
              </div>

              <div className="text-slate-400 font-bold hidden md:block">→</div>

              <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
                <div className="text-slate-500 text-[10px]">Dec Hidden 2</div>
                <div className="font-bold text-slate-700 text-sm mt-1">{autoencoder.config.hidden1}</div>
                <div className="text-[10px] text-slate-400 mt-0.5">ReLU</div>
              </div>

              <div className="text-slate-400 font-bold hidden md:block">→</div>

              <div className="p-3 bg-emerald-50 border border-emerald-500 rounded-lg">
                <div className="text-emerald-800 text-[10px]">Sigmoid Output</div>
                <div className="font-bold text-emerald-800 text-sm mt-1">{autoencoder.config.inputDim}</div>
                <div className="text-[10px] text-emerald-600 mt-0.5">Reconstruct &amp; Threshold</div>
              </div>
            </div>
          </div>

          {/* Details & Discretization notes */}
          <div className="bg-slate-50 p-4 border border-slate-200 rounded-lg text-xs space-y-2 text-slate-700 leading-relaxed">
            <h4 className="font-bold text-slate-900">Reconstruction with Probability Thresholding</h4>
            <p>
              The decoder output layer uses a <code className="text-cyan-800 font-semibold bg-white px-1.5 py-0.5 rounded border border-slate-200">sigmoid</code> activation producing continuous values in range <code className="text-cyan-800 font-semibold bg-white px-1.5 py-0.5 rounded border border-slate-200">[0.0, 1.0]</code>. These represent probabilities that an entry is active (or amino acid present).
            </p>
            <p>
              In Module 2, the user-defined threshold $\theta$ converts these probabilities back to discrete matrix entries:
              <br />
              <code className="text-amber-800 bg-white px-2 py-0.5 rounded border border-slate-200 font-mono font-semibold">
                reconstructed_value = probability &gt;= threshold ? 1 : 0
              </code>
            </p>
          </div>

          {/* Python Keras code section */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500">
                Executable Python / TensorFlow Keras Code
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
              <code>{kerasScript}</code>
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
