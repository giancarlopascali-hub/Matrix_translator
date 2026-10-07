/**
 * High-performance Canvas Matrix Heatmap with hover inspect and zoom (Bright Theme)
 */
import React, { useEffect, useRef, useState, useMemo } from 'react';
import { AMINO_ACIDS } from '../lib/matrixFormats';

interface MatrixHeatmapProps {
  rows: number;
  cols: number;
  data: Float32Array; // values (any real numbers or 0/1)
  probabilities?: Float32Array; // raw probabilities if available
  threshold?: number;
  groundTruth?: Float32Array;
  title?: string;
  showProbabilities?: boolean;
  valueType?: 'continuous_numeric' | 'binary_01';
  minVal?: number;
  maxVal?: number;
}

export const MatrixHeatmap: React.FC<MatrixHeatmapProps> = ({
  rows,
  cols,
  data,
  probabilities,
  threshold = 0.5,
  groundTruth,
  title,
  showProbabilities = false,
  valueType = 'binary_01',
  minVal,
  maxVal,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [zoom, setZoom] = useState<'fit' | '1x' | '2x'>('fit');
  const [hoverInfo, setHoverInfo] = useState<{
    r: number;
    c: number;
    val: number;
    prob?: number;
    label?: string;
    gt?: number;
    x: number;
    y: number;
  } | null>(null);

  // Compute actual min/max if not provided
  const { computedMin, computedMax } = useMemo(() => {
    let mMin = minVal !== undefined ? minVal : Infinity;
    let mMax = maxVal !== undefined ? maxVal : -Infinity;
    if (minVal === undefined || maxVal === undefined) {
      for (let i = 0; i < data.length; i++) {
        const v = data[i];
        if (v < mMin) mMin = v;
        if (v > mMax) mMax = v;
      }
      if (mMin === Infinity) { mMin = 0; mMax = 1; }
    }
    return { computedMin: mMin, computedMax: mMax };
  }, [data, minVal, maxVal]);

  // Layout calculations
  const cellWidth = zoom === 'fit' ? Math.max(8, Math.min(24, Math.floor(640 / cols))) : (zoom === '1x' ? 14 : 24);
  const cellHeight = zoom === 'fit' ? Math.max(2, Math.min(8, Math.floor(320 / rows))) : (zoom === '1x' ? 4 : 8);

  const canvasWidth = cols * cellWidth;
  const canvasHeight = rows * cellHeight;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = canvasWidth;
    canvas.height = canvasHeight;

    // Bright Background
    ctx.fillStyle = '#f8fafc'; // slate-50
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);

    const span = Math.max(1e-6, computedMax - computedMin);

    // Render cells
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const idx = r * cols + c;
        const val = data[idx] || 0;
        const prob = probabilities ? probabilities[idx] : val;
        const gt = groundTruth ? groundTruth[idx] : undefined;

        let fillStyle = '#f8fafc';

        if (groundTruth && !showProbabilities) {
          // Difference mode
          if (valueType === 'binary_01') {
            const gtBin = gt && gt > 0 ? 1 : 0;
            const predBin = val > 0 ? 1 : 0;
            if (gtBin === 1 && predBin === 1) fillStyle = '#059669';
            else if (gtBin === 0 && predBin === 1) fillStyle = '#dc2626';
            else if (gtBin === 1 && predBin === 0) fillStyle = '#d97706';
            else fillStyle = (r % 2 === 0) ? '#f8fafc' : '#f1f5f9';
          } else {
            // Continuous error heatmap (green = exact match, orange/red = error)
            const err = Math.abs(val - (gt || 0));
            if (err < 0.05) fillStyle = '#059669';
            else if (err < 0.5) fillStyle = '#d97706';
            else fillStyle = '#dc2626';
          }
        } else if (showProbabilities && probabilities) {
          // Probability gradient in bright mode
          if (prob > 0.005) {
            const intensity = Math.min(1.0, Math.max(0.0, prob));
            const rCol = Math.round(240 - intensity * 215);
            const gCol = Math.round(245 - intensity * 155);
            const bCol = Math.round(255 - intensity * 50);
            fillStyle = `rgb(${rCol}, ${gCol}, ${bCol})`;
          } else {
            fillStyle = (r % 2 === 0) ? '#f8fafc' : '#f1f5f9';
          }
        } else if (valueType === 'continuous_numeric') {
          // Continuous generic real numbers gradient
          if (val !== 0) {
            if (val > 0) {
              // Positive numbers: Cyan/Azure to Deep Sapphire
              const intensity = Math.min(1.0, Math.max(0.1, val / (Math.max(1e-6, computedMax))));
              const rCol = Math.round(224 - intensity * 190);
              const gCol = Math.round(242 - intensity * 140);
              const bCol = Math.round(254 - intensity * 50);
              fillStyle = `rgb(${rCol}, ${gCol}, ${bCol})`;
            } else {
              // Negative numbers: Soft Rose / Coral
              const intensity = Math.min(1.0, Math.max(0.1, Math.abs(val) / (Math.max(1e-6, Math.abs(computedMin)))));
              const rCol = Math.round(255 - intensity * 30);
              const gCol = Math.round(230 - intensity * 140);
              const bCol = Math.round(230 - intensity * 140);
              fillStyle = `rgb(${rCol}, ${gCol}, ${bCol})`;
            }
          } else {
            fillStyle = (r % 2 === 0) ? '#ffffff' : '#f8fafc';
          }
        } else {
          // Binary active/inactive
          if (val > 0) {
            fillStyle = '#0284c7'; // sky-600 vibrant active
          } else {
            fillStyle = (r % 2 === 0) ? '#ffffff' : '#f8fafc';
          }
        }

        ctx.fillStyle = fillStyle;
        ctx.fillRect(c * cellWidth, r * cellHeight, cellWidth, cellHeight);

        // Grid border
        if (cellWidth >= 12 && cellHeight >= 6) {
          ctx.strokeStyle = 'rgba(148, 163, 184, 0.25)'; // slate-400
          ctx.strokeRect(c * cellWidth, r * cellHeight, cellWidth, cellHeight);
        }
      }
    }
  }, [rows, cols, data, probabilities, groundTruth, showProbabilities, valueType, computedMin, computedMax, cellWidth, cellHeight, canvasWidth, canvasHeight, threshold]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const c = Math.floor(x / cellWidth);
    const r = Math.floor(y / cellHeight);

    if (r >= 0 && r < rows && c >= 0 && c < cols) {
      const idx = r * cols + c;
      const val = data[idx] || 0;
      const prob = probabilities ? probabilities[idx] : undefined;
      const gt = groundTruth ? groundTruth[idx] : undefined;
      const label = cols === 21 ? AMINO_ACIDS[c] : `Col ${c}`;
      setHoverInfo({ r, c, val, prob, label, gt, x: e.clientX, y: e.clientY });
    } else {
      setHoverInfo(null);
    }
  };

  const handleMouseLeave = () => {
    setHoverInfo(null);
  };

  // Active counts
  const nonZero = useMemo(() => {
    let count = 0;
    for (let i = 0; i < data.length; i++) if (data[i] > 0) count++;
    return count;
  }, [data]);

  return (
    <div className="flex flex-col gap-2.5 bg-white p-4 border border-slate-200 rounded-xl shadow-xs text-slate-800">
      <div className="flex flex-wrap items-center justify-between gap-3 text-xs">
        <div className="flex items-center gap-2">
          {title && <span className="font-semibold text-slate-900">{title}</span>}
          <span className="text-slate-500">
            {rows} × {cols} ({rows * cols} cells)
          </span>
          <span className="text-slate-300">·</span>
          <span className="text-emerald-700 font-medium">
            Active: {nonZero} ({(100 - (nonZero / (rows * cols)) * 100).toFixed(1)}% sparse)
          </span>
        </div>

        <div className="flex items-center gap-2">
          {groundTruth && (
            <div className="flex items-center gap-2 mr-2 text-[11px]">
              <span className="inline-block w-2.5 h-2.5 bg-emerald-600 rounded-xs"></span>
              <span className="text-slate-600">Match</span>
              <span className="inline-block w-2.5 h-2.5 bg-red-600 rounded-xs"></span>
              <span className="text-slate-600">False Pos</span>
              <span className="inline-block w-2.5 h-2.5 bg-amber-600 rounded-xs"></span>
              <span className="text-slate-600">Missed</span>
            </div>
          )}

          <div className="flex items-center bg-slate-100 border border-slate-200 rounded p-0.5">
            <button
              onClick={() => setZoom('fit')}
              className={`px-2 py-0.5 rounded transition-colors text-[11px] font-medium ${zoom === 'fit' ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-500 hover:text-slate-800'}`}
            >
              Fit
            </button>
            <button
              onClick={() => setZoom('1x')}
              className={`px-2 py-0.5 rounded transition-colors text-[11px] font-medium ${zoom === '1x' ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-500 hover:text-slate-800'}`}
            >
              1×
            </button>
            <button
              onClick={() => setZoom('2x')}
              className={`px-2 py-0.5 rounded transition-colors text-[11px] font-medium ${zoom === '2x' ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-500 hover:text-slate-800'}`}
            >
              2×
            </button>
          </div>
        </div>
      </div>

      {/* Column labels if cols === 21 */}
      {cols === 21 && (
        <div className="overflow-x-auto pb-1">
          <div 
            className="flex text-[10px] font-mono text-slate-500 border-b border-slate-200 pb-1"
            style={{ width: `${canvasWidth}px` }}
          >
            {AMINO_ACIDS.map((aa) => (
              <div 
                key={aa} 
                style={{ width: `${cellWidth}px` }} 
                className="text-center font-bold text-slate-700"
              >
                {aa}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Canvas container with scroll */}
      <div className="max-h-[380px] overflow-auto border border-slate-200 rounded-lg bg-slate-50 relative shadow-inner">
        <canvas
          ref={canvasRef}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          className="cursor-crosshair block"
        />

        {hoverInfo && (
          <div 
            className="fixed pointer-events-none z-50 bg-slate-900/95 border border-slate-700 text-slate-100 text-xs px-3 py-2 rounded-lg shadow-xl backdrop-blur-xs font-mono"
            style={{ 
              left: `${hoverInfo.x + 12}px`, 
              top: `${hoverInfo.y + 12}px` 
            }}
          >
            <div className="text-cyan-300 font-semibold">
              Position: Row {hoverInfo.r}, {hoverInfo.label} (Col {hoverInfo.c})
            </div>
            <div>Value: <span className="font-bold text-white">{hoverInfo.val > 0 ? '1 (Active)' : '0'}</span></div>
            {hoverInfo.prob !== undefined && (
              <div>Probability: <span className="text-amber-300">{(hoverInfo.prob * 100).toFixed(2)}%</span></div>
            )}
            {hoverInfo.gt !== undefined && (
              <div>Ground Truth: <span className={hoverInfo.gt > 0 ? 'text-emerald-400' : 'text-slate-400'}>{hoverInfo.gt > 0 ? '1' : '0'}</span></div>
            )}
          </div>
        )}
      </div>

      <div className="flex items-center justify-between text-[11px] text-slate-500 pt-0.5">
        <span>Rows: Positions 0 to {rows - 1} · Columns: Categories / Features 0 to {cols - 1}</span>
        <span>Hover cells for coordinate &amp; probability values</span>
      </div>
    </div>
  );
};
