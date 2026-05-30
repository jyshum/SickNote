"use client";

import { useState } from "react";
import { predictCough, type PredictionResult } from "@/lib/api";
import AudioInput from "@/components/AudioInput";
import ResultCard from "@/components/ResultCard";
import { Loader2 } from "lucide-react";

type PageState = "idle" | "loading" | "result" | "error";

export default function Home() {
  const [state, setState] = useState<PageState>("idle");
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [errorMsg, setErrorMsg] = useState("");
  const [lastFile, setLastFile] = useState<File | null>(null);

  async function handleAudioReady(file: File) {
    setState("loading");
    setResult(null);
    setErrorMsg("");
    setLastFile(file);

    try {
      const prediction = await predictCough(file);
      setResult(prediction);
      setState("result");
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Something went wrong.");
      setState("error");
    }
  }

  function handleReset() {
    setState("idle");
    setResult(null);
    setErrorMsg("");
    setLastFile(null);
  }

  async function handleRetry() {
    if (lastFile) {
      await handleAudioReady(lastFile);
    }
  }

  return (
    <div className="mx-auto max-w-2xl px-6 py-10">
      {/* Hero */}
      <div className="mb-8 rounded-2xl bg-gradient-to-br from-[var(--navy)] to-[var(--navy-light)] px-8 py-10 text-center">
        <h1 className="text-2xl font-bold text-white">SickNote</h1>
        <p className="mt-2 text-sm text-slate-300">
          AI-powered cough screening
        </p>
      </div>

      {/* Audio Input */}
      <div className="mb-6">
        <AudioInput
          onAudioReady={handleAudioReady}
          disabled={state === "loading"}
        />
      </div>

      {/* Loading */}
      {state === "loading" && (
        <div className="flex flex-col items-center rounded-xl border border-slate-200 bg-white p-8">
          <Loader2 className="mb-3 h-8 w-8 animate-spin text-slate-400" />
          <p className="text-sm text-slate-500">Analyzing your cough...</p>
        </div>
      )}

      {/* Result */}
      {state === "result" && result && (
        <div>
          <ResultCard result={result} />
          <button
            onClick={handleReset}
            className="mt-4 w-full rounded-lg border border-slate-200 py-2.5 text-sm font-medium text-slate-600 transition-colors hover:bg-slate-50"
          >
            Analyze another
          </button>
        </div>
      )}

      {/* Error */}
      {state === "error" && (
        <div className="rounded-xl border border-red-200 bg-red-50 p-6 text-center">
          <p className="text-sm text-red-700">{errorMsg}</p>
          <button
            onClick={handleRetry}
            className="mt-3 rounded-lg bg-red-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-red-700"
          >
            Try again
          </button>
        </div>
      )}

      {/* Disclaimer */}
      <p className="mt-8 text-center text-xs text-slate-400">
        Screening tool only. Not a medical diagnosis. See a doctor if concerned.
      </p>
    </div>
  );
}
