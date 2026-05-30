"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { Mic, Square } from "lucide-react";

interface RecordButtonProps {
  onRecorded: (file: File) => void;
  disabled?: boolean;
}

export default function RecordButton({ onRecorded, disabled }: RecordButtonProps) {
  const [state, setState] = useState<"idle" | "recording" | "unsupported">(() =>
    typeof navigator !== "undefined" && !navigator.mediaDevices
      ? "unsupported"
      : "idle"
  );
  const [elapsed, setElapsed] = useState(0);
  const mediaRecorder = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunks = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
      chunks.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.current.push(e.data);
      };

      recorder.onstop = () => {
        const blob = new Blob(chunks.current, { type: "audio/webm" });
        const file = new File([blob], "recording.webm", { type: "audio/webm" });
        stream.getTracks().forEach((t) => t.stop());
        onRecorded(file);
        setElapsed(0);
      };

      mediaRecorder.current = recorder;
      recorder.start();
      setState("recording");
      setElapsed(0);
      timerRef.current = setInterval(() => setElapsed((e) => e + 1), 1000);
    } catch {
      setState("unsupported");
    }
  }, [onRecorded]);

  const stopRecording = useCallback(() => {
    mediaRecorder.current?.stop();
    if (timerRef.current) clearInterval(timerRef.current);
    setState("idle");
  }, []);

  if (state === "unsupported") {
    return (
      <div className="flex flex-col items-center justify-center rounded-xl border border-slate-200 bg-slate-50 p-8 text-center opacity-60">
        <Mic className="mb-2 h-8 w-8 text-slate-400" />
        <p className="text-sm text-slate-500">Microphone not available</p>
      </div>
    );
  }

  if (state === "recording") {
    return (
      <button
        onClick={stopRecording}
        className="flex flex-col items-center justify-center rounded-xl border-2 border-red-300 bg-red-50 p-8 text-center transition-colors hover:bg-red-100"
      >
        <div className="animate-pulse-recording mb-2 h-4 w-4 rounded-full bg-red-500" />
        <Square className="mb-1 h-6 w-6 text-red-600" />
        <p className="text-sm font-semibold text-red-700">Stop</p>
        <p className="mt-1 text-xs tabular-nums text-red-500">
          {Math.floor(elapsed / 60)}:{(elapsed % 60).toString().padStart(2, "0")}
        </p>
      </button>
    );
  }

  return (
    <button
      onClick={startRecording}
      disabled={disabled}
      className="flex flex-col items-center justify-center rounded-xl border border-slate-200 bg-white p-8 text-center transition-colors hover:border-slate-300 hover:bg-slate-50 disabled:opacity-40 disabled:pointer-events-none"
    >
      <Mic className="mb-2 h-8 w-8 text-slate-700" />
      <p className="text-sm font-semibold text-slate-700">Record</p>
      <p className="mt-1 text-xs text-slate-400">Click to start</p>
    </button>
  );
}
