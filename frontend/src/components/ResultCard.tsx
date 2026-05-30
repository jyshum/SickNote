import type { PredictionResult } from "@/lib/api";
import { ShieldCheck, AlertTriangle } from "lucide-react";

interface ResultCardProps {
  result: PredictionResult;
}

export default function ResultCard({ result }: ResultCardProps) {
  const isHealthy = result.label === "healthy";

  const accentColor = isHealthy ? "var(--green)" : "var(--amber)";
  const badgeText = isHealthy ? "LOW RISK" : "REVIEW RECOMMENDED";
  const Icon = isHealthy ? ShieldCheck : AlertTriangle;

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-6">
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Icon className="h-6 w-6" style={{ color: accentColor }} />
          <div>
            <p className="text-xl font-bold" style={{ color: accentColor }}>
              {result.label.toUpperCase()}
            </p>
            <p className="text-sm text-slate-500">
              {(result.confidence * 100).toFixed(0)}% confidence
            </p>
          </div>
        </div>
        <span
          className="rounded-full px-3 py-1 text-xs font-semibold text-white"
          style={{ backgroundColor: accentColor }}
        >
          {badgeText}
        </span>
      </div>

      {result.spectrogram && (
        <div className="overflow-hidden rounded-lg bg-slate-900">
          <img
            src={result.spectrogram}
            alt="Mel spectrogram of analyzed cough"
            className="w-full"
          />
        </div>
      )}
    </div>
  );
}
