/**
 * Matrix Latent Autoencoder Studio - Main Application (Bright Theme)
 */
import React, { useState } from 'react';
import { defaultAutoencoder } from './lib/autoencoder';
import { MatrixItem } from './lib/types';
import { Header } from './components/Header';
import { MatrixEncoderModule } from './components/MatrixEncoderModule';
import { LatentDecoderModule } from './components/LatentDecoderModule';
import { ModelArchitectureModal } from './components/ModelArchitectureModal';

export default function App() {
  const [activeTab, setActiveTab] = useState<'module1' | 'module2' | 'split'>('module1');
  const [isModelModalOpen, setIsModelModalOpen] = useState(false);
  const [transferredLatentCsv, setTransferredLatentCsv] = useState<string>('');
  const [groundTruthMatrices, setGroundTruthMatrices] = useState<MatrixItem[]>([]);
  const [transferredDimensions, setTransferredDimensions] = useState<{ rows: number; cols: number } | undefined>();

  // Bridge callback from Module 1 -> Module 2
  const handleTransferToDecoder = (
    latentCsv: string, 
    gt?: MatrixItem[], 
    dimensions?: { rows: number; cols: number }
  ) => {
    setTransferredLatentCsv(latentCsv);
    if (gt) setGroundTruthMatrices(gt);
    if (dimensions) setTransferredDimensions(dimensions);
    setActiveTab('module2');
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col font-sans selection:bg-cyan-600/20 selection:text-cyan-900">
      {/* Top Navigation */}
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        onOpenModelModal={() => setIsModelModalOpen(true)}
      />

      {/* Main Body */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-6">
        {activeTab === 'module1' && (
          <MatrixEncoderModule
            autoencoder={defaultAutoencoder}
            onTransferToDecoder={handleTransferToDecoder}
          />
        )}

        {activeTab === 'module2' && (
          <LatentDecoderModule
            autoencoder={defaultAutoencoder}
            initialLatentCsv={transferredLatentCsv}
            groundTruthMatrices={groundTruthMatrices}
            initialDimensions={transferredDimensions}
          />
        )}

        {activeTab === 'split' && (
          <div className="flex flex-col gap-10">
            <div className="border-b border-slate-200 pb-8">
              <MatrixEncoderModule
                autoencoder={defaultAutoencoder}
                onTransferToDecoder={handleTransferToDecoder}
              />
            </div>
            <div>
              <LatentDecoderModule
                autoencoder={defaultAutoencoder}
                initialLatentCsv={transferredLatentCsv}
                groundTruthMatrices={groundTruthMatrices}
                initialDimensions={transferredDimensions}
              />
            </div>
          </div>
        )}
      </main>

      {/* Model Architecture & Python Script Modal */}
      <ModelArchitectureModal
        isOpen={isModelModalOpen}
        onClose={() => setIsModelModalOpen(false)}
        autoencoder={defaultAutoencoder}
      />

      {/* Footer (Bright Theme) */}
      <footer className="border-t border-slate-200 bg-white py-4 px-6 text-center text-xs text-slate-500 shadow-2xs">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>Matrix Latent Autoencoder Studio · 8-Dimensional Bottleneck Pipeline for Any Matrix Size</span>
          <span className="text-slate-600 font-mono">
            Active: {defaultAutoencoder.config.inputRows} × {defaultAutoencoder.config.inputCols} ({defaultAutoencoder.config.inputDim} features) → 8 Latent Dimensions
          </span>
        </div>
      </footer>
    </div>
  );
}
