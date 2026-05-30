"use client";

import { useState } from "react";
import { predictCough, type PredictionResult } from "@/lib/api";
import AudioInput from "@/components/AudioInput";
import ResultCard from "@/components/ResultCard";
import type { PredictionSource } from "@/lib/predictionHistory";
import { AlertCircle, ArrowDown, ChevronLeft, ChevronRight, Loader2 } from "lucide-react";

interface RecordingEntry {
  id: string;
  result: PredictionResult;
  source: PredictionSource;
  audioUrl: string;
  fileName: string;
  createdAt: string;
}

export default function Home() {
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [recordings, setRecordings] = useState<RecordingEntry[]>([]);
  const [activeIndex, setActiveIndex] = useState(0);

  const hasResults = recordings.length > 0 || loading || errorMsg;

  async function handleAudioReady(file: File, source: PredictionSource) {
    setLoading(true);
    setErrorMsg("");

    const audioUrl = URL.createObjectURL(file);

    try {
      const prediction = await predictCough(file);
      const entry: RecordingEntry = {
        id: `${source}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        result: prediction,
        source,
        audioUrl,
        fileName: file.name,
        createdAt: new Date().toISOString(),
      };
      setRecordings((prev) => [entry, ...prev].slice(0, 10));
      setActiveIndex(0);
      setLoading(false);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Something went wrong.");
      setLoading(false);
    }
  }

  const activeEntry = recordings[activeIndex];
  const total = recordings.length;

  function goPrev() {
    setActiveIndex((i) => Math.max(0, i - 1));
  }

  function goNext() {
    setActiveIndex((i) => Math.min(recordings.length - 1, i + 1));
  }

  return (
    <main className="min-h-[100dvh] w-full max-w-full overflow-x-hidden">
      {/* ── Hero ── */}
      <section className="relative flex min-h-[70vh] flex-col items-center justify-center px-6 py-20 md:py-28">
        <div className="pointer-events-none absolute inset-0 -z-10 bg-[radial-gradient(ellipse_80%_50%_at_50%_-20%,rgba(214,40,40,0.08),transparent)]" />

        <h1 className="max-w-5xl text-center text-[clamp(2.6rem,6vw,5.5rem)] font-semibold leading-[1.05] tracking-tight text-slate-950">
          Know when your cough
          <br />
          needs a <span className="text-[var(--brand-red)]">closer look</span>
        </h1>

        <p className="mt-8 max-w-lg text-center text-lg leading-relaxed text-slate-500 font-light">
          Record a short sample. Our classifier returns a screening result
          with confidence scoring and spectrogram analysis.
        </p>

        <a
          href="#analyze"
          className="mt-12 inline-flex items-center gap-3 rounded-full bg-[var(--brand-red)] px-8 py-4 text-base font-semibold text-white shadow-[0_16px_48px_-12px_rgba(214,40,40,0.4)] transition duration-300 hover:bg-[var(--brand-red-hover)] hover:-translate-y-0.5 active:translate-y-0"
        >
          Record your cough
          <ArrowDown className="h-4 w-4" />
        </a>

        <p className="mt-12 max-w-sm text-center text-xs leading-5 text-slate-400">
          Screening support only. This does not replace medical
          diagnosis or clinical care.
        </p>
      </section>

      {/* ── Analyze Section ── */}
      <section
        id="analyze"
        className="border-t border-slate-200/60 bg-white px-6 py-16 md:py-24"
      >
        <div className="mx-auto max-w-6xl">
          <div
            className={`grid transition-all duration-700 ease-out ${
              hasResults
                ? "lg:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)] gap-10 lg:gap-14"
                : "grid-cols-1"
            }`}
          >
            {/* ── Left: Input ── */}
            <div
              className={`transition-all duration-700 ease-out ${
                hasResults ? "" : "mx-auto max-w-2xl w-full"
              }`}
            >
              <div className="lg:sticky lg:top-28">
                <h2 className="text-[clamp(1.6rem,3vw,2.4rem)] font-semibold leading-tight tracking-tight text-slate-950">
                  Submit a sample
                </h2>
                <p className="mt-3 text-base text-slate-500">
                  Use your microphone or upload an existing audio file.
                </p>

                <div className="mt-10">
                  <AudioInput
                    onAudioReady={handleAudioReady}
                    disabled={loading}
                  />
                </div>
              </div>
            </div>

            {/* ── Right: Paginated Results ── */}
            {hasResults && (
              <div>
                {/* Header with navigation */}
                <div className="mb-5 flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-slate-950">
                    Results
                  </h3>

                  {total > 1 && (
                    <div className="flex items-center gap-1">
                      <button
                        onClick={goPrev}
                        disabled={activeIndex === 0}
                        className="flex h-8 w-8 items-center justify-center rounded-full border border-slate-200 bg-white text-slate-600 transition duration-200 hover:bg-slate-50 disabled:opacity-30 disabled:pointer-events-none"
                      >
                        <ChevronLeft className="h-4 w-4" />
                      </button>

                      {/* Dot indicators */}
                      <div className="flex items-center gap-1.5 px-2">
                        {recordings.map((_, i) => (
                          <button
                            key={recordings[i].id}
                            onClick={() => setActiveIndex(i)}
                            className={`rounded-full transition-all duration-300 ${
                              i === activeIndex
                                ? i === 0
                                  ? "h-2.5 w-2.5 bg-[var(--brand-red)]"
                                  : "h-2.5 w-2.5 bg-slate-950"
                                : i === 0
                                  ? "h-1.5 w-1.5 bg-[var(--brand-red)]/40 hover:bg-[var(--brand-red)]/70"
                                  : "h-1.5 w-1.5 bg-slate-300 hover:bg-slate-400"
                            }`}
                          />
                        ))}
                      </div>

                      <button
                        onClick={goNext}
                        disabled={activeIndex === total - 1}
                        className="flex h-8 w-8 items-center justify-center rounded-full border border-slate-200 bg-white text-slate-600 transition duration-200 hover:bg-slate-50 disabled:opacity-30 disabled:pointer-events-none"
                      >
                        <ChevronRight className="h-4 w-4" />
                      </button>
                    </div>
                  )}
                </div>

                {/* Page counter */}
                {total > 0 && (
                  <p className="mb-4 font-[family-name:var(--font-mono)] text-xs tabular-nums text-slate-400">
                    {activeIndex + 1} / {total}
                  </p>
                )}

                {/* Loading skeleton */}
                {loading && (
                  <div className="rounded-2xl border border-slate-200 bg-white p-6">
                    <div className="flex items-center gap-4">
                      <span className="flex h-11 w-11 items-center justify-center rounded-full bg-slate-100 text-slate-700">
                        <Loader2 className="h-5 w-5 animate-spin" />
                      </span>
                      <div>
                        <p className="text-base font-semibold text-slate-950">
                          Analyzing your cough
                        </p>
                        <p className="mt-1 text-sm text-slate-500">
                          Preparing audio, generating spectrogram, running classifier.
                        </p>
                      </div>
                    </div>
                    <div className="mt-6 space-y-3">
                      <div className="h-3 w-full animate-pulse rounded-full bg-slate-100" />
                      <div className="h-3 w-4/5 animate-pulse rounded-full bg-slate-100" />
                      <div className="h-16 animate-pulse rounded-xl bg-slate-100" />
                    </div>
                  </div>
                )}

                {/* Error */}
                {errorMsg && !loading && (
                  <div className="rounded-2xl border border-amber-200 bg-amber-50 p-5">
                    <div className="flex gap-3">
                      <AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-amber-600" />
                      <div>
                        <p className="font-semibold text-amber-900">
                          Analysis did not complete
                        </p>
                        <p className="mt-1 text-sm leading-6 text-amber-800">
                          {errorMsg}
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                {/* Active result card */}
                {!loading && activeEntry && (
                  <ResultCard
                    key={activeEntry.id}
                    result={activeEntry.result}
                    audioUrl={activeEntry.audioUrl}
                    audioName={activeEntry.fileName}
                    source={activeEntry.source}
                    isLatest={activeIndex === 0}
                  />
                )}
              </div>
            )}
          </div>
        </div>
      </section>
    </main>
  );
}
