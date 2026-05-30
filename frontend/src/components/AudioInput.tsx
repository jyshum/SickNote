"use client";

import { useRef } from "react";
import { Upload } from "lucide-react";
import RecordButton from "./RecordButton";
import type { PredictionSource } from "@/lib/predictionHistory";

interface AudioInputProps {
  onAudioReady: (file: File, source: PredictionSource) => void;
  disabled?: boolean;
}

export default function AudioInput({ onAudioReady, disabled }: AudioInputProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) {
      onAudioReady(file, "uploaded");
      e.target.value = "";
    }
  }

  return (
    <div className="space-y-4">
      {/* Primary action: Record */}
      <RecordButton
        onRecorded={(file) => onAudioReady(file, "recorded")}
        disabled={disabled}
      />

      {/* Secondary action: Upload */}
      <div className="flex items-center gap-4">
        <div className="h-px flex-1 bg-slate-200" />
        <span className="text-xs font-medium text-slate-400">or</span>
        <div className="h-px flex-1 bg-slate-200" />
      </div>

      <button
        onClick={() => fileInputRef.current?.click()}
        disabled={disabled}
        className="flex w-full items-center justify-center gap-2.5 rounded-full border border-slate-200 bg-white px-6 py-3.5 text-sm font-semibold text-slate-700 transition duration-300 hover:-translate-y-0.5 hover:border-slate-300 hover:bg-slate-50 active:translate-y-0 disabled:pointer-events-none disabled:opacity-40"
      >
        <Upload className="h-4 w-4" />
        Upload audio file
      </button>

      <p className="text-center text-xs text-slate-400">
        Accepts .webm, .wav, .ogg, or .mp3
      </p>

      <input
        ref={fileInputRef}
        type="file"
        accept=".webm,.wav,.ogg,.mp3"
        onChange={handleFileChange}
        className="hidden"
      />
    </div>
  );
}
