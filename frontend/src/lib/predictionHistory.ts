export type PredictionSource = "recorded" | "uploaded";

export interface PredictionHistoryItem {
  id: string;
  label: "healthy" | "abnormal";
  confidence: number;
  source: PredictionSource;
  createdAt: string;
  audioUrl: string;
}

interface NewPredictionHistoryItem {
  label: "healthy" | "abnormal";
  confidence: number;
  source: PredictionSource;
  audioUrl: string;
  createdAt?: string;
}

export function addPredictionToHistory(
  current: PredictionHistoryItem[],
  next: NewPredictionHistoryItem,
): PredictionHistoryItem[] {
  const createdAt = next.createdAt ?? new Date().toISOString();
  const id = `${next.source}-${createdAt}-${Math.random().toString(36).slice(2, 8)}`;

  return [
    {
      id,
      label: next.label,
      confidence: next.confidence,
      source: next.source,
      createdAt,
      audioUrl: next.audioUrl,
    },
    ...current,
  ].slice(0, 5);
}
