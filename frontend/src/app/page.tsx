"use client";

import { useEffect, useRef, useState } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { predictCough, type PredictionResult } from "@/lib/api";
import AudioInput from "@/components/AudioInput";
import ResultCard from "@/components/ResultCard";
import type { PredictionSource } from "@/lib/predictionHistory";
import {
  AlertCircle,
  ArrowDown,
  ChevronLeft,
  ChevronRight,
  Loader2,
  Stethoscope,
} from "lucide-react";

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

  const pageRef = useRef<HTMLElement>(null);
  const analyzerRef = useRef<HTMLElement>(null);

  const hasResults = recordings.length > 0 || loading || errorMsg;

  useEffect(() => {
    gsap.registerPlugin(ScrollTrigger);

    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      return;
    }

    const context = gsap.context(() => {
      gsap.from("[data-hero-reveal]", {
        y: 28,
        opacity: 0,
        duration: 0.9,
        ease: "power3.out",
        stagger: 0.1,
      });

      gsap.from("[data-analyzer-reveal]", {
        y: 34,
        opacity: 0,
        duration: 0.8,
        ease: "power3.out",
        stagger: 0.08,
        scrollTrigger: {
          trigger: analyzerRef.current,
          start: "top 74%",
        },
      });
    }, pageRef);

    return () => context.revert();
  }, []);

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
    <main
      ref={pageRef}
      className="relative min-h-[100dvh] w-full max-w-full overflow-x-hidden bg-[var(--background)] text-slate-950"
    >
      <div className="noise-overlay" />

      <section className="relative isolate flex min-h-[92dvh] items-center px-5 pb-20 pt-32 sm:px-8 md:pt-36 lg:px-10">
        <div className="clinic-grid pointer-events-none absolute inset-0 -z-20" />
        <div className="pointer-events-none absolute inset-0 -z-10 bg-[radial-gradient(circle_at_50%_18%,rgba(214,40,40,0.10),transparent_30%),linear-gradient(180deg,rgba(255,255,255,0.7),rgba(251,250,247,0.96)_72%)]" />

        <div className="mx-auto flex max-w-6xl flex-col items-center text-center">
          <p
            data-hero-reveal
            className="mb-6 max-w-xl font-[family-name:var(--font-mono)] text-xs uppercase leading-6 tracking-[0.28em] text-[var(--brand-red)]"
          >
            Cough screening support, clearly explained
          </p>

          <h1
            data-hero-reveal
            className="max-w-5xl text-[clamp(2.75rem,5.8vw,5.8rem)] font-black leading-[0.98] tracking-normal text-slate-950"
          >
            Know when your cough needs a closer look.
          </h1>

          <p
            data-hero-reveal
            className="mt-7 max-w-2xl text-lg leading-8 text-slate-600 sm:text-xl"
          >
            Record a short sample or upload audio. SickNote returns a screening
            result with confidence scoring and spectrogram context.
          </p>

          <div data-hero-reveal className="mt-10 flex flex-col gap-3 sm:flex-row">
            <a
              href="#analyze"
              className="inline-flex items-center justify-center gap-3 rounded-full bg-[var(--brand-red)] px-7 py-4 text-base font-bold text-white transition duration-300 hover:-translate-y-0.5 hover:bg-[var(--brand-red-hover)] focus:outline-none focus:ring-2 focus:ring-[var(--brand-red)] focus:ring-offset-2 focus:ring-offset-[var(--background)] active:translate-y-0"
            >
              Start screening
              <ArrowDown className="h-4 w-4" />
            </a>
            <a
              href="/technical"
              className="inline-flex items-center justify-center rounded-full border border-slate-200 bg-white px-7 py-4 text-base font-bold text-slate-800 transition duration-300 hover:-translate-y-0.5 hover:border-slate-300 hover:bg-slate-50 focus:outline-none focus:ring-2 focus:ring-slate-300 focus:ring-offset-2 focus:ring-offset-[var(--background)] active:translate-y-0"
            >
              How it works
            </a>
          </div>

        </div>
      </section>

      <section
        ref={analyzerRef}
        id="analyze"
        className="border-t border-slate-200/80 bg-white px-5 py-20 sm:px-8 md:py-28 lg:px-10"
      >
        <div className="mx-auto max-w-6xl">
          <div data-analyzer-reveal className="mb-10 max-w-3xl">
            <div className="mb-4 flex items-center gap-3">
              <span className="flex h-10 w-10 items-center justify-center rounded-full bg-[var(--brand-red-light)] text-[var(--brand-red)]">
                <Stethoscope className="h-5 w-5" />
              </span>
              <p className="font-[family-name:var(--font-mono)] text-xs uppercase tracking-[0.24em] text-slate-400">
                Analyzer
              </p>
            </div>
            <h2 className="text-[clamp(2rem,4vw,3.4rem)] font-black leading-tight tracking-normal text-slate-950">
              Submit a sample and review the result in one place.
            </h2>
            <p className="mt-4 max-w-2xl text-base leading-7 text-slate-600">
              This is screening support only. It does not replace medical diagnosis
              or clinical care.
            </p>
          </div>

          <div
            className={`grid gap-6 transition-all duration-700 ease-out ${
              hasResults ? "lg:grid-cols-[0.82fr_1.18fr]" : "lg:grid-cols-[minmax(0,0.82fr)_minmax(0,1.18fr)]"
            }`}
          >
            <article
              data-analyzer-reveal
              className="rounded-[1.75rem] border border-slate-200 bg-[#fbfaf7] p-5 shadow-[0_24px_80px_-54px_rgba(15,23,42,0.36)] sm:p-7"
            >
              <AudioInput onAudioReady={handleAudioReady} disabled={loading} />
            </article>

            <article
              data-analyzer-reveal
              className="min-h-[300px] rounded-[1.75rem] border border-slate-200 bg-white p-5 shadow-[0_24px_80px_-54px_rgba(15,23,42,0.28)] sm:p-7"
            >
              <div className="mb-5 flex items-center justify-between gap-4">
                <div>
                  <h3 className="text-lg font-bold tracking-normal text-slate-950">
                    Results
                  </h3>
                  <p className="mt-1 text-sm text-slate-500">
                    Your latest analysis appears here.
                  </p>
                </div>

                {total > 1 && (
                  <div className="flex items-center gap-2">
                    <button
                      onClick={goPrev}
                      disabled={activeIndex === 0}
                      className="flex h-9 w-9 items-center justify-center rounded-full border border-slate-200 bg-white text-slate-700 transition duration-200 hover:bg-slate-50 disabled:pointer-events-none disabled:opacity-30"
                      aria-label="Previous result"
                    >
                      <ChevronLeft className="h-4 w-4" />
                    </button>
                    <button
                      onClick={goNext}
                      disabled={activeIndex === total - 1}
                      className="flex h-9 w-9 items-center justify-center rounded-full border border-slate-200 bg-white text-slate-700 transition duration-200 hover:bg-slate-50 disabled:pointer-events-none disabled:opacity-30"
                      aria-label="Next result"
                    >
                      <ChevronRight className="h-4 w-4" />
                    </button>
                  </div>
                )}
              </div>

              {total > 0 && (
                <p className="mb-4 font-[family-name:var(--font-mono)] text-xs tabular-nums text-slate-400">
                  {activeIndex + 1} / {total}
                </p>
              )}

              {!hasResults && (
                <div className="flex min-h-[210px] items-center justify-center rounded-2xl border border-dashed border-slate-200 bg-slate-50/70 p-8 text-center">
                  <p className="max-w-sm text-sm leading-6 text-slate-500">
                    Record or upload a cough sample to see confidence scoring,
                    result status, and audio playback.
                  </p>
                </div>
              )}

              {loading && (
                <div className="rounded-2xl border border-slate-200 bg-slate-50 p-6">
                  <div className="flex items-center gap-4">
                    <span className="flex h-12 w-12 items-center justify-center rounded-full bg-white text-[var(--brand-red)]">
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
                    <div className="h-3 w-full animate-pulse rounded-full bg-slate-200" />
                    <div className="h-3 w-4/5 animate-pulse rounded-full bg-slate-200" />
                    <div className="h-20 animate-pulse rounded-2xl bg-slate-200" />
                  </div>
                </div>
              )}

              {errorMsg && !loading && (
                <div className="rounded-2xl border border-amber-200 bg-amber-50 p-5">
                  <div className="flex gap-3">
                    <AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-amber-600" />
                    <div>
                      <p className="font-semibold text-amber-950">
                        Analysis did not complete
                      </p>
                      <p className="mt-1 text-sm leading-6 text-amber-800">
                        {errorMsg}
                      </p>
                    </div>
                  </div>
                </div>
              )}

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
            </article>
          </div>
        </div>
      </section>
    </main>
  );
}
