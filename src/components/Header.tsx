/**
 * Header and Navigation Component (Bright Theme)
 */
import React from 'react';
import { Cpu, ArrowLeftRight, Binary, BookOpen } from 'lucide-react';

export type AppTab = 'guide' | 'module1' | 'module2' | 'split';

interface HeaderProps {
  activeTab: AppTab;
  setActiveTab: (tab: AppTab) => void;
  onOpenModelModal: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  setActiveTab,
  onOpenModelModal,
}) => {
  return (
    <header className="border-b border-slate-200 bg-white/95 backdrop-blur-md sticky top-0 z-40 shadow-xs">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-3.5 flex flex-col md:flex-row md:items-center justify-between gap-4">
        {/* Brand & Title */}
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-gradient-to-tr from-cyan-600 to-emerald-600 flex items-center justify-center text-white font-black shadow-xs">
            <Binary className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-base font-bold text-slate-900 flex items-center gap-2">
              Matrix PCA Studio
            </h1>
            <p className="text-xs text-slate-500">
              Compact Subspace Representation for Bayesian Optimization &amp; Strict Threshold Reconstruction
            </p>
          </div>
        </div>

        {/* Navigation Tabs (Interactive Segmented Control in Bright Theme) */}
        <div className="flex w-full min-w-0 items-center gap-2.5 md:w-auto">
          <div className="flex min-w-0 flex-1 items-center overflow-x-auto bg-slate-100 border border-slate-200 rounded-lg p-1 text-xs md:flex-none">
            <button
              onClick={() => setActiveTab('guide')}
              className={`flex shrink-0 items-center gap-1.5 px-3 py-1.5 rounded-md font-medium transition-all ${
                activeTab === 'guide'
                  ? 'bg-white text-violet-800 shadow-xs font-semibold'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <BookOpen className="w-3.5 h-3.5" />
              Quick Guide
            </button>
            <button
              onClick={() => setActiveTab('module1')}
              className={`shrink-0 px-3 py-1.5 rounded-md font-medium transition-all ${
                activeTab === 'module1'
                  ? 'bg-white text-cyan-800 shadow-xs font-semibold'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <span className="sm:hidden">Encoder</span>
              <span className="hidden sm:inline">Module 1: Matrix Encoder</span>
            </button>
            <button
              onClick={() => setActiveTab('module2')}
              className={`shrink-0 px-3 py-1.5 rounded-md font-medium transition-all ${
                activeTab === 'module2'
                  ? 'bg-white text-emerald-800 shadow-xs font-semibold'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <span className="sm:hidden">Decoder</span>
              <span className="hidden sm:inline">Module 2: Latent Decoder</span>
            </button>
            <button
              onClick={() => setActiveTab('split')}
              className={`px-3 py-1.5 rounded-md font-medium transition-all hidden sm:flex items-center gap-1.5 ${
                activeTab === 'split'
                  ? 'bg-white text-slate-900 shadow-xs font-semibold'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <ArrowLeftRight className="w-3.5 h-3.5" />
              Side-by-Side
            </button>
          </div>

          {/* Model Architecture Button */}
          <button
            onClick={onOpenModelModal}
            className="flex shrink-0 items-center gap-1.5 px-3 py-1.5 bg-slate-50 hover:bg-slate-100 border border-slate-200 text-slate-700 hover:text-slate-900 rounded-lg text-xs font-medium transition-colors shadow-2xs"
          >
            <Cpu className="w-3.5 h-3.5 text-cyan-700" />
            <span className="hidden sm:inline">PCA &amp; BO</span> Pipeline
          </button>
        </div>
      </div>
    </header>
  );
};
