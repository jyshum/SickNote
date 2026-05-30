"use client";

import { useRef } from "react";
import { Upload } from "lucide-react";
import RecordButton from "./RecordButton";

interface AudioInputProps {
  onAudioReady: (file: File) => void;
  disabled?: boolean;
}

export default function AudioInput({ onAudioReady, disabled }: AudioInputProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) {
      onAudioReady(file);
      e.target.value = "";
    }
  }

  return (
    <div className="grid grid-cols-2 gap-4">
      <RecordButton onRecorded={onAudioReady} disabled={disabled} />

      <div>
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={disabled}
          className="flex w-full flex-col items-center justify-center rounded-xl border border-slate-200 bg-white p-8 text-center transition-colors hover:border-slate-300 hover:bg-slate-50 disabled:opacity-40 disabled:pointer-events-none"
        >
          <Upload className="mb-2 h-8 w-8 text-slate-700" />
          <p className="text-sm font-semibold text-slate-700">Upload</p>
          <p className="mt-1 text-xs text-slate-400">.webm, .wav, .ogg, .mp3</p>
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept=".webm,.wav,.ogg,.mp3"
          onChange={handleFileChange}
          className="hidden"
        />
      </div>
    </div>
  );
}
