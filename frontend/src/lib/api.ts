const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface PredictionResult {
  label: "healthy" | "abnormal";
  confidence: number;
  spectrogram: string; // base64 data URI
  gradcam?: string; // base64 data URI — Grad-CAM heatmap overlay
  ensemble_size?: number;
}

export async function predictCough(file: File): Promise<PredictionResult> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_URL}/api/predict`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Prediction failed (${res.status}): ${detail}`);
  }

  return res.json();
}
