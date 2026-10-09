/**
 * A visual, non-technical walkthrough of the matrix -> BO -> matrix workflow.
 */
import React from 'react';
import {
  ArrowDown,
  ArrowRight,
  Binary,
  BookOpen,
  CheckCircle2,
  FileJson,
  FileSpreadsheet,
  KeyRound,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Upload,
} from 'lucide-react';

interface QuickGuideProps {
  onOpenEncoder: () => void;
  onOpenDecoder: () => void;
}

const sourceCells = [1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0];
const resultCells = [1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0];

const MiniMatrix: React.FC<{ cells: number[]; tone?: 'cyan' | 'emerald' }> = ({
  cells,
  tone = 'cyan',
}) => (
  <div className="grid grid-cols-4 gap-1" aria-hidden="true">
    {cells.map((cell, index) => (
      <div
        key={index}
        className={`h-4 w-4 rounded-[3px] border ${
          cell
            ? tone === 'cyan'
              ? 'border-cyan-600 bg-cyan-500'
              : 'border-emerald-600 bg-emerald-500'
            : 'border-slate-200 bg-white'
        }`}
      />
    ))}
  </div>
);

const FlowArrow: React.FC = () => (
  <div className="flex items-center justify-center text-slate-300" aria-hidden="true">
    <ArrowRight className="hidden h-6 w-6 lg:block" />
    <ArrowDown className="h-6 w-6 lg:hidden" />
  </div>
);

const StepBadge: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <span className="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-slate-900 text-xs font-black text-white">
    {children}
  </span>
);

export const QuickGuide: React.FC<QuickGuideProps> = ({ onOpenEncoder, onOpenDecoder }) => {
  return (
    <div className="space-y-6">
      <section className="overflow-hidden rounded-2xl border border-cyan-200 bg-gradient-to-br from-cyan-50 via-white to-emerald-50 shadow-sm">
        <div className="grid gap-6 px-6 py-7 md:grid-cols-[1fr_auto] md:items-center md:px-8">
          <div>
            <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-cyan-200 bg-white px-3 py-1 text-xs font-bold uppercase tracking-wider text-cyan-800">
              <BookOpen className="h-3.5 w-3.5" />
              Quick Guide
            </div>
            <h2 className="text-2xl font-black tracking-tight text-slate-950 sm:text-3xl">
              Two separate tasks, joined by a matching basis
            </h2>
            <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-600">
              Use the Encoder when creating numerical representations. Use the Decoder later when turning optimizer
              candidates back into matrices. The <strong>basis JSON</strong> is the key that connects the two tasks.
            </p>
          </div>
          <div className="flex flex-wrap gap-2 md:max-w-xs md:justify-end">
            {['1+ source matrices', 'K values per row', 'CSV + basis JSON'].map(label => (
              <span key={label} className="rounded-lg border border-white bg-white/90 px-3 py-2 text-xs font-semibold text-slate-700 shadow-xs">
                {label}
              </span>
            ))}
          </div>
        </div>
      </section>

      <section className="rounded-2xl border border-cyan-200 bg-white p-5 shadow-sm sm:p-7">
        <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <div className="flex items-center gap-2">
              <span className="rounded-md bg-cyan-100 px-2 py-1 text-[10px] font-black uppercase tracking-wider text-cyan-800">Task A</span>
              <p className="text-xs font-bold uppercase tracking-wider text-cyan-700">Encoding workflow</p>
            </div>
            <h3 className="mt-2 text-lg font-bold text-slate-900">Turn source matrices into fixed-K parameters</h3>
            <p className="mt-1 text-xs leading-5 text-slate-500">Do this when preparing a training library for Bayesian optimization.</p>
          </div>
          <button type="button" onClick={onOpenEncoder} className="inline-flex items-center justify-center gap-2 rounded-lg bg-cyan-700 px-4 py-2 text-xs font-bold text-white hover:bg-cyan-600">
            Open Encoder <ArrowRight className="h-4 w-4" />
          </button>
        </div>

        <div className="grid items-stretch gap-3 lg:grid-cols-[1fr_auto_1fr_auto_1fr_auto_1.15fr]">
          <div className="flex min-h-40 flex-col items-center justify-center rounded-xl border border-cyan-200 bg-cyan-50 p-4 text-center">
            <StepBadge>1</StepBadge>
            <div className="mt-3"><MiniMatrix cells={sourceCells} /></div>
            <div className="mt-3 text-sm font-bold text-slate-900">Load matrices</div>
            <div className="mt-1 text-[11px] leading-4 text-slate-500">One or more matrices<br />with the same shape</div>
          </div>
          <FlowArrow />
          <div className="flex min-h-40 flex-col items-center justify-center rounded-xl border border-cyan-200 bg-white p-4 text-center">
            <StepBadge>2</StepBadge>
            <SlidersHorizontal className="mt-3 h-7 w-7 text-cyan-700" />
            <div className="mt-3 text-sm font-bold text-slate-900">Choose K and fit</div>
            <div className="mt-1 text-[11px] leading-4 text-slate-500">PCA + orthogonal completion<br />always returns exactly K values</div>
          </div>
          <FlowArrow />
          <div className="flex min-h-40 flex-col items-center justify-center rounded-xl border border-cyan-200 bg-white p-4 text-center">
            <StepBadge>3</StepBadge>
            <CheckCircle2 className="mt-3 h-7 w-7 text-cyan-700" />
            <div className="mt-3 text-sm font-bold text-slate-900">Check the result</div>
            <div className="mt-1 text-[11px] leading-4 text-slate-500">Review round-trip accuracy<br />and exact-match rate</div>
          </div>
          <FlowArrow />
          <div className="flex min-h-40 flex-col justify-center rounded-xl border border-violet-200 bg-violet-50 p-4">
            <div className="mb-2 flex items-center gap-2"><StepBadge>4</StepBadge><span className="text-sm font-bold text-slate-900">Download both files</span></div>
            <div className="flex items-center gap-2 rounded-lg border border-violet-200 bg-white px-3 py-2">
              <FileSpreadsheet className="h-5 w-5 shrink-0 text-violet-700" />
              <div><div className="text-xs font-bold text-slate-900">Latent CSV</div><div className="font-mono text-[10px] text-slate-500">K columns + dimension roles</div></div>
            </div>
            <div className="mt-2 flex items-center gap-2 rounded-lg border border-amber-200 bg-white px-3 py-2">
              <FileJson className="h-5 w-5 shrink-0 text-amber-700" />
              <div><div className="text-xs font-bold text-slate-900">Basis JSON</div><div className="text-[10px] text-slate-500">Save unchanged for decoding</div></div>
            </div>
          </div>
        </div>

        <div className="mt-4 flex items-start gap-3 rounded-xl border border-cyan-100 bg-cyan-50/60 px-4 py-3 text-xs leading-5 text-cyan-950">
          <Upload className="mt-0.5 h-4 w-4 shrink-0 text-cyan-700" />
          <span><strong>Single-matrix encoding is supported.</strong> A hidden anchor supplies the first learned direction, is never exported, and the output row still contains exactly K values.</span>
        </div>
      </section>

      <section className="rounded-2xl border border-fuchsia-200 bg-gradient-to-r from-fuchsia-50 via-white to-violet-50 px-5 py-5 shadow-sm sm:px-7">
        <div className="grid gap-5 md:grid-cols-[1fr_auto_1fr] md:items-center">
          <div className="flex items-center gap-3 rounded-xl border border-violet-200 bg-white p-4">
            <FileSpreadsheet className="h-7 w-7 shrink-0 text-violet-700" />
            <div><div className="text-sm font-bold text-slate-900">Encoder’s latent CSV</div><div className="text-xs text-slate-500">Use z1…zK as [0,1] parameters</div></div>
          </div>
          <div className="flex flex-col items-center text-fuchsia-700">
            <ArrowRight className="hidden h-6 w-6 md:block" />
            <ArrowDown className="h-6 w-6 md:hidden" />
          </div>
          <div className="flex items-center gap-3 rounded-xl border border-fuchsia-200 bg-white p-4">
            <Sparkles className="h-7 w-7 shrink-0 text-fuchsia-700" />
            <div><div className="text-sm font-bold text-slate-900">Optimizer proposes candidate CSV</div><div className="text-xs text-slate-500">Keep the same z1…zK column names</div></div>
          </div>
        </div>
        <p className="mt-3 text-center text-[11px] leading-5 text-slate-500">
          This happens outside Matrix PCA Studio. Objective columns may be added; the decoder ignores non-z columns.
        </p>
      </section>

      <section className="rounded-2xl border border-emerald-200 bg-white p-5 shadow-sm sm:p-7">
        <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <div className="flex items-center gap-2">
              <span className="rounded-md bg-emerald-100 px-2 py-1 text-[10px] font-black uppercase tracking-wider text-emerald-800">Task B</span>
              <p className="text-xs font-bold uppercase tracking-wider text-emerald-700">Decoding workflow</p>
            </div>
            <h3 className="mt-2 text-lg font-bold text-slate-900">Turn optimizer candidates into new matrices</h3>
            <p className="mt-1 text-xs leading-5 text-slate-500">Do this after your optimizer has proposed new z1…zK values.</p>
          </div>
          <button type="button" onClick={onOpenDecoder} className="inline-flex items-center justify-center gap-2 rounded-lg bg-emerald-700 px-4 py-2 text-xs font-bold text-white hover:bg-emerald-600">
            Open Decoder <ArrowRight className="h-4 w-4" />
          </button>
        </div>

        <div className="grid items-stretch gap-3 lg:grid-cols-[1.15fr_auto_1fr_auto_1fr_auto_1fr]">
          <div className="flex min-h-40 flex-col justify-center gap-2 rounded-xl border border-emerald-200 bg-emerald-50 p-4">
            <div className="mb-1 flex items-center gap-2"><StepBadge>1</StepBadge><span className="text-sm font-bold text-slate-900">Load the two inputs</span></div>
            <div className="flex items-center gap-2 rounded-lg border border-fuchsia-200 bg-white px-3 py-2"><FileSpreadsheet className="h-5 w-5 text-fuchsia-700" /><span className="text-xs font-bold">Optimizer candidate CSV</span></div>
            <div className="flex items-center gap-2 rounded-lg border border-amber-200 bg-white px-3 py-2"><FileJson className="h-5 w-5 text-amber-700" /><span className="text-xs font-bold">Matching basis JSON</span></div>
          </div>
          <FlowArrow />
          <div className="flex min-h-40 flex-col items-center justify-center rounded-xl border border-emerald-200 bg-white p-4 text-center">
            <StepBadge>2</StepBadge>
            <KeyRound className="mt-3 h-7 w-7 text-emerald-700" />
            <div className="mt-3 text-sm font-bold text-slate-900">Validate the pair</div>
            <div className="mt-1 text-[11px] leading-4 text-slate-500">Model ID, dimensions, K,<br />and coordinate bounds must match</div>
          </div>
          <FlowArrow />
          <div className="flex min-h-40 flex-col items-center justify-center rounded-xl border border-emerald-200 bg-white p-4 text-center">
            <StepBadge>3</StepBadge>
            <Binary className="mt-3 h-7 w-7 text-emerald-700" />
            <div className="mt-3 text-sm font-bold text-slate-900">Reconstruct</div>
            <div className="mt-1 text-[11px] leading-4 text-slate-500">For binary output:<br />value &gt; 0.5 becomes 1</div>
          </div>
          <FlowArrow />
          <div className="flex min-h-40 flex-col items-center justify-center rounded-xl border border-emerald-200 bg-emerald-50 p-4 text-center">
            <StepBadge>4</StepBadge>
            <div className="mt-3"><MiniMatrix cells={resultCells} tone="emerald" /></div>
            <div className="mt-3 text-sm font-bold text-slate-900">Download matrices</div>
            <div className="mt-1 text-[11px] leading-4 text-slate-500">Choose the required<br />extended output format</div>
          </div>
        </div>

        <div className="mt-4 flex items-start gap-3 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3">
          <KeyRound className="mt-0.5 h-5 w-5 shrink-0 text-amber-700" />
          <p className="text-xs leading-5 text-amber-950">
            <strong>Important:</strong> use the exact basis JSON downloaded during the encoding task. It bypasses the optimizer and must not be edited or replaced by a basis from another run.
          </p>
        </div>

        <div className="mt-4 flex flex-col gap-3 rounded-xl border border-violet-200 bg-violet-50 px-4 py-3 sm:flex-row sm:items-center">
          <ShieldCheck className="h-6 w-6 shrink-0 text-violet-700" />
          <div>
            <div className="text-xs font-bold text-violet-950">Optional binary design validation</div>
            <p className="mt-0.5 text-[11px] leading-5 text-violet-900/80">
              In binary mode, ignore disconnected islands fitting within 2×2, reject multiple larger regions, check the corner 2×2 blocks, and download the latent CSV with an added <code>validation</code> column.
            </p>
          </div>
        </div>
      </section>

      <section className="rounded-2xl border border-slate-200 bg-slate-900 px-5 py-5 text-white shadow-sm sm:px-6">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h3 className="font-bold">Ready to try it?</h3>
            <p className="mt-1 text-xs leading-5 text-slate-300">
              A single matrix works too: the hidden bootstrap anchor is never exported, and the vector still has exactly K values.
            </p>
          </div>
          <div className="flex shrink-0 flex-wrap gap-2">
            <button
              type="button"
              onClick={onOpenEncoder}
              className="inline-flex items-center gap-2 rounded-lg bg-cyan-500 px-4 py-2 text-xs font-bold text-slate-950 transition-colors hover:bg-cyan-400"
            >
              Open Encoder <ArrowRight className="h-4 w-4" />
            </button>
            <button
              type="button"
              onClick={onOpenDecoder}
              className="inline-flex items-center gap-2 rounded-lg border border-slate-600 px-4 py-2 text-xs font-bold text-white transition-colors hover:bg-slate-800"
            >
              Open Decoder
            </button>
          </div>
        </div>
      </section>
    </div>
  );
};
