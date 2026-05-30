import type { PredictionResult } from "@/lib/api";
import type { PredictionSource } from "@/lib/predictionHistory";
import { ShieldCheck, AlertTriangle, AudioLines } from "lucide-react";

interface ResultCardProps {
  result: PredictionResult;
  audioUrl: string;
  audioName: string;
  source: PredictionSource;
  isLatest?: boolean;
}

export default function ResultCard({
  result,
  audioUrl,
  audioName,
  source,
  isLatest,
}: ResultCardProps) {
  const isHealthy = result.label === "healthy";
  const confidencePercent = Math.round(result.confidence * 100);

  const accentColor = isHealthy ? "var(--green)" : "var(--amber)";
  const badgeText = isHealthy ? "LOW RISK" : "REVIEW RECOMMENDED";
  const Icon = isHealthy ? ShieldCheck : AlertTriangle;

  return (
    <div
      className={`relative overflow-hidden rounded-2xl border bg-white transition-shadow duration-500 ${
        isLatest
          ? "border-[var(--brand-red)]/30 shadow-[0_0_0_1px_rgba(214,40,40,0.08),0_20px_50px_-16px_rgba(214,40,40,0.18)]"
          : "border-slate-200"
      }`}
    >
      <div className="p-6 sm:p-8">
        <div className="mb-6 flex items-start justify-between gap-4">
          <div className="flex items-center gap-3">
            {isLatest && (
              <span className="absolute -top-px left-6 rounded-b-lg bg-[var(--brand-red)] px-3 py-1 text-[0.6rem] font-bold uppercase tracking-widest text-white">
                Latest
              </span>
            )}
            <span
              className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full"
              style={{ backgroundColor: isHealthy ? "#ecfdf5" : "#fffbeb", color: accentColor }}
            >
              <Icon className="h-5 w-5" />
            </span>
            <div>
              <p className="text-sm font-medium text-slate-500">
                Screening result
              </p>
              <p className="mt-0.5 text-2xl font-semibold tracking-tight text-slate-950">
                {result.label === "healthy" ? "Healthy pattern" : "Closer look"}
              </p>
            </div>
          </div>
          <span
            className="shrink-0 rounded-full px-3 py-1.5 text-[0.68rem] font-bold uppercase tracking-wider text-white"
            style={{ backgroundColor: accentColor }}
          >
            {badgeText}
          </span>
        </div>

        <div className="rounded-xl border border-slate-100 bg-slate-50/50 p-5">
          <div className="mb-3 flex items-center justify-between gap-4">
            <p className="text-sm font-medium text-slate-600">Confidence</p>
            <p className="font-[family-name:var(--font-mono)] text-2xl font-semibold tabular-nums text-slate-950">
              {confidencePercent}%
            </p>
          </div>
          <div className="h-2.5 overflow-hidden rounded-full bg-slate-200">
            <div
              className="h-full rounded-full transition-[width] duration-700 ease-out"
              style={{
                width: `${confidencePercent}%`,
                backgroundColor: accentColor,
              }}
            />
          </div>
        </div>

        <div className="mt-4 rounded-xl border border-slate-100 p-5">
          <div className="mb-3 flex items-center gap-2 text-sm font-medium text-slate-700">
            <AudioLines className="h-4 w-4 text-slate-400" />
            <span>{source === "recorded" ? "Recorded sample" : "Uploaded sample"}</span>
          </div>
          <p className="mb-3 truncate text-xs text-slate-400">{audioName}</p>
          <audio controls src={audioUrl} className="w-full" />
        </div>

        {result.ensemble_size && result.ensemble_size > 1 && (
          <p className="mt-4 text-center text-xs text-slate-400">
            Averaged across {result.ensemble_size} models
          </p>
        )}
      </div>

      {(result.spectrogram || result.gradcam) && (
        <div className="border-t border-slate-200 bg-slate-950 p-4 space-y-3">
          {result.spectrogram && (
            <div>
              <p className="mb-2 text-xs font-medium text-slate-400">Spectrogram</p>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={result.spectrogram}
                alt="Mel spectrogram of analyzed cough"
                className="w-full rounded-lg"
              />
            </div>
          )}
          {result.gradcam && (
            <div>
              <p className="mb-2 text-xs font-medium text-slate-400">Model focus areas</p>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={result.gradcam}
                alt="Grad-CAM heatmap showing regions that influenced the prediction"
                className="w-full rounded-lg"
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
