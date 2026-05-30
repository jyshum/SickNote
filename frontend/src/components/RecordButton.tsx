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
      <div className="flex items-center justify-center gap-3 rounded-full border border-slate-200 bg-slate-50 px-6 py-4 text-sm text-slate-500">
        <Mic className="h-4 w-4 text-slate-400" />
        Microphone unavailable — upload an audio file instead.
      </div>
    );
  }

  if (state === "recording") {
    return (
      <button
        onClick={stopRecording}
        className="flex w-full items-center justify-center gap-4 rounded-full border border-red-200 bg-[var(--brand-red-light)] px-6 py-4 transition duration-300 hover:bg-red-100 active:scale-[0.99]"
      >
        <span className="animate-pulse-recording h-3 w-3 rounded-full bg-[var(--brand-red)]" />
        <span className="font-[family-name:var(--font-mono)] text-sm font-semibold tabular-nums text-[var(--brand-red)]">
          {Math.floor(elapsed / 60)}:
          {(elapsed % 60).toString().padStart(2, "0")}
        </span>
        <span className="flex h-8 w-8 items-center justify-center rounded-full bg-[var(--brand-red)] text-white">
          <Square className="h-3.5 w-3.5" />
        </span>
        <span className="text-sm font-semibold text-red-800">
          Stop recording
        </span>
      </button>
    );
  }

  return (
    <button
      onClick={startRecording}
      disabled={disabled}
      className="group flex w-full items-center justify-center gap-3 rounded-full bg-[var(--brand-red)] px-8 py-4 text-base font-semibold text-white shadow-[0_16px_48px_-12px_rgba(214,40,40,0.4)] transition duration-300 hover:-translate-y-0.5 hover:bg-[var(--brand-red-hover)] active:translate-y-0 disabled:pointer-events-none disabled:opacity-40"
    >
      <Mic className="h-5 w-5" />
      Record cough
    </button>
  );
}
