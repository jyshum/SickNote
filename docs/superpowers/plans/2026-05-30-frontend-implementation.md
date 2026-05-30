# SickNote Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-page Next.js frontend — product page (audio upload/record → ML prediction) and technical page (static ML pipeline explanation).

**Architecture:** Next.js App Router with four client components (`Navbar`, `AudioInput`, `RecordButton`, `ResultCard`). Product page orchestrates state (idle → recording → loading → result). Technical page is static markup. API connector (`src/lib/api.ts`) already exists.

**Tech Stack:** Next.js 16, React 19, Tailwind CSS 4, TypeScript, lucide-react (icons)

**Spec:** `docs/superpowers/specs/2026-05-30-frontend-design.md`

---

## File Structure

```
frontend/src/
├── app/
│   ├── layout.tsx              ← MODIFY: add Navbar, update metadata
│   ├── page.tsx                ← REWRITE: product page with hero + cards
│   ├── technical/
│   │   └── page.tsx            ← CREATE: static technical page
│   └── globals.css             ← MODIFY: add pulse animation keyframe
├── components/
│   ├── Navbar.tsx              ← CREATE: shared top nav
│   ├── AudioInput.tsx          ← CREATE: record + upload cards
│   ├── RecordButton.tsx        ← CREATE: MediaRecorder wrapper
│   └── ResultCard.tsx          ← CREATE: prediction result display
└── lib/
    └── api.ts                  ← EXISTS: no changes needed
```

---

### Task 1: Install lucide-react and set up globals

**Files:**
- Modify: `frontend/package.json` (via npm install)
- Modify: `frontend/src/app/globals.css`

- [ ] **Step 1: Install lucide-react**

Run from `frontend/`:
```bash
npm install lucide-react
```
Expected: package.json updated, lock file updated, no errors.

- [ ] **Step 2: Update globals.css**

Replace the contents of `frontend/src/app/globals.css` with:

```css
@import "tailwindcss";

:root {
  --background: #ffffff;
  --foreground: #0f172a;
  --green: #059669;
  --amber: #d97706;
  --navy: #0f172a;
  --navy-light: #1e3a5f;
}

@theme inline {
  --color-background: var(--background);
  --color-foreground: var(--foreground);
  --font-sans: var(--font-geist-sans);
  --font-mono: var(--font-geist-mono);
}

body {
  background: var(--background);
  color: var(--foreground);
}

@keyframes pulse-recording {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}

.animate-pulse-recording {
  animation: pulse-recording 1.2s ease-in-out infinite;
}
```

- [ ] **Step 3: Verify the dev server starts**

Run from `frontend/`:
```bash
npm run dev
```
Expected: compiles without errors, page loads at `http://localhost:3000`.

- [ ] **Step 4: Commit**

```bash
git add frontend/package.json frontend/package-lock.json frontend/src/app/globals.css
git commit -m "feat(frontend): install lucide-react, update global styles"
```

---

### Task 2: Create Navbar component

**Files:**
- Create: `frontend/src/components/Navbar.tsx`
- Modify: `frontend/src/app/layout.tsx`

- [ ] **Step 1: Create Navbar.tsx**

Create `frontend/src/components/Navbar.tsx`:

```tsx
"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity } from "lucide-react";

export default function Navbar() {
  const pathname = usePathname();

  return (
    <nav className="w-full border-b border-slate-200 bg-white">
      <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-4">
        <Link href="/" className="flex items-center gap-2">
          <Activity className="h-5 w-5 text-[var(--green)]" />
          <span className="text-lg font-bold text-slate-900">SickNote</span>
        </Link>
        <div className="flex gap-6 text-sm font-medium">
          <Link
            href="/"
            className={pathname === "/" ? "text-slate-900" : "text-slate-500 hover:text-slate-700"}
          >
            Analyze
          </Link>
          <Link
            href="/technical"
            className={pathname === "/technical" ? "text-slate-900" : "text-slate-500 hover:text-slate-700"}
          >
            How it Works
          </Link>
        </div>
      </div>
    </nav>
  );
}
```

- [ ] **Step 2: Update layout.tsx**

Replace the contents of `frontend/src/app/layout.tsx` with:

```tsx
import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import Navbar from "@/components/Navbar";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "SickNote — Cough Screening Tool",
  description: "AI-powered cough screening. Not a medical diagnosis.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col bg-white text-slate-900">
        <Navbar />
        <main className="flex-1">{children}</main>
      </body>
    </html>
  );
}
```

- [ ] **Step 3: Verify nav renders on dev server**

Run from `frontend/`:
```bash
npm run dev
```
Expected: page loads at `http://localhost:3000` with nav bar showing "SickNote" logo, "Analyze" link (active), and "How it Works" link.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/Navbar.tsx frontend/src/app/layout.tsx
git commit -m "feat(frontend): add Navbar component with navigation"
```

---

### Task 3: Create RecordButton component

**Files:**
- Create: `frontend/src/components/RecordButton.tsx`

- [ ] **Step 1: Create RecordButton.tsx**

Create `frontend/src/components/RecordButton.tsx`:

```tsx
"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { Mic, Square } from "lucide-react";

interface RecordButtonProps {
  onRecorded: (file: File) => void;
  disabled?: boolean;
}

export default function RecordButton({ onRecorded, disabled }: RecordButtonProps) {
  const [state, setState] = useState<"idle" | "recording" | "unsupported">("idle");
  const [elapsed, setElapsed] = useState(0);
  const mediaRecorder = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (typeof navigator !== "undefined" && !navigator.mediaDevices) {
      setState("unsupported");
    }
  }, []);

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
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
```

- [ ] **Step 2: Verify it compiles**

Run from `frontend/`:
```bash
npx tsc --noEmit
```
Expected: no type errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/RecordButton.tsx
git commit -m "feat(frontend): add RecordButton with MediaRecorder API"
```

---

### Task 4: Create AudioInput component

**Files:**
- Create: `frontend/src/components/AudioInput.tsx`

- [ ] **Step 1: Create AudioInput.tsx**

Create `frontend/src/components/AudioInput.tsx`:

```tsx
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

      <button
        onClick={() => fileInputRef.current?.click()}
        disabled={disabled}
        className="flex flex-col items-center justify-center rounded-xl border border-slate-200 bg-white p-8 text-center transition-colors hover:border-slate-300 hover:bg-slate-50 disabled:opacity-40 disabled:pointer-events-none"
      >
        <Upload className="mb-2 h-8 w-8 text-slate-700" />
        <p className="text-sm font-semibold text-slate-700">Upload</p>
        <p className="mt-1 text-xs text-slate-400">.webm, .wav, .ogg, .mp3</p>
        <input
          ref={fileInputRef}
          type="file"
          accept=".webm,.wav,.ogg,.mp3"
          onChange={handleFileChange}
          className="hidden"
        />
      </button>
    </div>
  );
}
```

- [ ] **Step 2: Verify it compiles**

Run from `frontend/`:
```bash
npx tsc --noEmit
```
Expected: no type errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/AudioInput.tsx
git commit -m "feat(frontend): add AudioInput with record + upload cards"
```

---

### Task 5: Create ResultCard component

**Files:**
- Create: `frontend/src/components/ResultCard.tsx`

- [ ] **Step 1: Create ResultCard.tsx**

Create `frontend/src/components/ResultCard.tsx`:

```tsx
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
```

- [ ] **Step 2: Verify it compiles**

Run from `frontend/`:
```bash
npx tsc --noEmit
```
Expected: no type errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/ResultCard.tsx
git commit -m "feat(frontend): add ResultCard with label, confidence, spectrogram"
```

---

### Task 6: Build the product page

**Files:**
- Rewrite: `frontend/src/app/page.tsx`

- [ ] **Step 1: Rewrite page.tsx**

Replace the contents of `frontend/src/app/page.tsx` with:

```tsx
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
```

- [ ] **Step 2: Start mock API and test end-to-end**

Terminal 1 (from project root):
```bash
source .venv/bin/activate && uvicorn api.main:app --port 8000
```

Terminal 2 (from `frontend/`):
```bash
npm run dev
```

Open `http://localhost:3000`. Verify:
- Hero header renders with dark gradient
- Record and Upload cards render side by side
- Clicking Upload and selecting any audio file triggers loading state
- Result card appears with label, confidence, spectrogram
- "Analyze another" resets to idle
- Disclaimer footer visible

- [ ] **Step 3: Verify build passes**

Run from `frontend/`:
```bash
npm run build
```
Expected: no build errors.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/app/page.tsx
git commit -m "feat(frontend): build product page with hero, audio input, results"
```

---

### Task 7: Build the technical page

**Files:**
- Create: `frontend/src/app/technical/page.tsx`

- [ ] **Step 1: Create the technical directory and page**

Create `frontend/src/app/technical/page.tsx`:

```tsx
import { Database, Filter, Tag, AudioWaveform, BarChart3, AlertCircle, Cpu, FlaskConical } from "lucide-react";

export default function TechnicalPage() {
  return (
    <div className="mx-auto max-w-3xl px-6 py-10">
      {/* Hero */}
      <div className="mb-12 rounded-2xl bg-gradient-to-br from-[var(--navy)] to-[var(--navy-light)] px-8 py-10 text-center">
        <h1 className="text-2xl font-bold text-white">How SickNote Works</h1>
        <p className="mt-2 text-sm text-slate-300">
          The machine learning behind cough screening
        </p>
      </div>

      {/* Overview */}
      <section className="mb-12">
        <h2 className="mb-3 text-xl font-bold text-slate-900">Overview</h2>
        <p className="leading-7 text-slate-600">
          SickNote is a binary cough classifier that distinguishes healthy coughs
          from abnormal ones. It analyzes audio recordings by converting them into
          mel spectrograms — visual representations of sound frequencies — and
          feeding them through a convolutional neural network (CNN). The model is
          trained on the COUGHVID dataset, a collection of crowdsourced cough
          recordings with expert physician annotations.
        </p>
      </section>

      {/* Dataset */}
      <section className="mb-12">
        <div className="mb-3 flex items-center gap-2">
          <Database className="h-5 w-5 text-[var(--green)]" />
          <h2 className="text-xl font-bold text-slate-900">Dataset</h2>
        </div>
        <p className="mb-4 leading-7 text-slate-600">
          The COUGHVID dataset contains ~34,400 crowdsourced cough recordings.
          Of these, 2,841 were reviewed by four expert physicians who provided
          diagnostic labels.
        </p>
        <div className="grid grid-cols-3 gap-4">
          <div className="rounded-lg border border-slate-200 p-4 text-center">
            <p className="text-2xl font-bold text-slate-900">34,400</p>
            <p className="text-xs text-slate-500">Total recordings</p>
          </div>
          <div className="rounded-lg border border-slate-200 p-4 text-center">
            <p className="text-2xl font-bold text-slate-900">2,841</p>
            <p className="text-xs text-slate-500">Expert-labeled</p>
          </div>
          <div className="rounded-lg border border-slate-200 p-4 text-center">
            <p className="text-2xl font-bold text-slate-900">~2,300</p>
            <p className="text-xs text-slate-500">After filtering</p>
          </div>
        </div>
        <div className="mt-4 rounded-lg border border-slate-200 p-4">
          <p className="mb-2 text-sm font-semibold text-slate-700">Filtering Criteria</p>
          <ul className="space-y-1 text-sm text-slate-600">
            <li>Cough detection confidence &gt; 0.8</li>
            <li>At least one expert diagnosis present</li>
            <li>Quality rated acceptable by majority of experts</li>
          </ul>
          <div className="mt-4">
            <p className="mb-1 text-sm font-semibold text-slate-700">Class Distribution</p>
            <div className="flex h-6 overflow-hidden rounded-full">
              <div className="flex items-center justify-center bg-amber-500" style={{ width: "78%" }}>
                <span className="text-xs font-medium text-white">78% abnormal</span>
              </div>
              <div className="flex items-center justify-center bg-emerald-500" style={{ width: "22%" }}>
                <span className="text-xs font-medium text-white">22%</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Pipeline */}
      <section className="mb-12">
        <div className="mb-3 flex items-center gap-2">
          <FlaskConical className="h-5 w-5 text-[var(--green)]" />
          <h2 className="text-xl font-bold text-slate-900">Processing Pipeline</h2>
        </div>
        <div className="flex flex-wrap items-center justify-center gap-2 rounded-lg border border-slate-200 p-6">
          {[
            { icon: AudioWaveform, label: "Raw Audio" },
            { icon: Filter, label: "Filter" },
            { icon: Tag, label: "Label" },
            { icon: BarChart3, label: "Spectrogram" },
            { icon: Cpu, label: "Normalize" },
            { icon: Database, label: "Split" },
          ].map((step, i) => (
            <div key={step.label} className="flex items-center gap-2">
              <div className="flex flex-col items-center rounded-lg bg-slate-50 px-4 py-3">
                <step.icon className="mb-1 h-5 w-5 text-slate-600" />
                <span className="text-xs font-medium text-slate-700">{step.label}</span>
              </div>
              {i < 5 && <span className="text-slate-300">→</span>}
            </div>
          ))}
        </div>
      </section>

      {/* Spectrogram Explanation */}
      <section className="mb-12">
        <div className="mb-3 flex items-center gap-2">
          <BarChart3 className="h-5 w-5 text-[var(--green)]" />
          <h2 className="text-xl font-bold text-slate-900">Spectrograms</h2>
        </div>
        <p className="mb-4 leading-7 text-slate-600">
          A mel spectrogram converts audio into a visual representation where the
          x-axis is time, the y-axis is frequency (on the mel scale, which mimics
          human hearing), and color intensity represents energy. This transforms
          the audio classification problem into an image classification problem,
          allowing standard computer vision techniques to apply.
        </p>
        <div className="grid grid-cols-2 gap-4">
          <div className="rounded-lg border border-slate-200 p-4 text-center">
            <div className="mb-2 flex h-24 items-center justify-center rounded bg-slate-900">
              <span className="text-xs text-slate-500">Healthy spectrogram placeholder</span>
            </div>
            <p className="text-sm font-medium text-emerald-600">Healthy Cough</p>
          </div>
          <div className="rounded-lg border border-slate-200 p-4 text-center">
            <div className="mb-2 flex h-24 items-center justify-center rounded bg-slate-900">
              <span className="text-xs text-slate-500">Abnormal spectrogram placeholder</span>
            </div>
            <p className="text-sm font-medium text-amber-600">Abnormal Cough</p>
          </div>
        </div>
        <p className="mt-2 text-center text-xs text-slate-400">
          Real spectrogram images will be added after model training.
        </p>
      </section>

      {/* Model */}
      <section className="mb-12">
        <div className="mb-3 flex items-center gap-2">
          <Cpu className="h-5 w-5 text-[var(--green)]" />
          <h2 className="text-xl font-bold text-slate-900">Model Architecture</h2>
        </div>
        <p className="mb-4 leading-7 text-slate-600">
          A small convolutional neural network (CNN) trained from scratch on the
          filtered dataset. The model processes mel spectrograms as single-channel
          images through three convolutional blocks, then classifies via two fully
          connected layers.
        </p>
        <div className="rounded-lg border border-slate-200 p-4">
          <div className="space-y-2 font-mono text-sm text-slate-700">
            <p>Input: (1, 64, T) mel spectrogram</p>
            <p>&nbsp;&nbsp;→ Conv2d(16) + BatchNorm + ReLU + MaxPool</p>
            <p>&nbsp;&nbsp;→ Conv2d(32) + BatchNorm + ReLU + MaxPool</p>
            <p>&nbsp;&nbsp;→ Conv2d(64) + BatchNorm + ReLU + MaxPool</p>
            <p>&nbsp;&nbsp;→ Flatten</p>
            <p>&nbsp;&nbsp;→ Linear(128) + ReLU + Dropout(0.3)</p>
            <p>&nbsp;&nbsp;→ Linear(1) → logit</p>
          </div>
        </div>
        <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
          <div className="rounded-lg border border-slate-200 p-3">
            <p className="font-semibold text-slate-700">Loss Function</p>
            <p className="text-slate-500">BCEWithLogitsLoss + pos_weight</p>
          </div>
          <div className="rounded-lg border border-slate-200 p-3">
            <p className="font-semibold text-slate-700">Optimizer</p>
            <p className="text-slate-500">Adam (lr=1e-3, wd=1e-4)</p>
          </div>
          <div className="rounded-lg border border-slate-200 p-3">
            <p className="font-semibold text-slate-700">Checkpoint</p>
            <p className="text-slate-500">Best val_AUC + early stopping</p>
          </div>
          <div className="rounded-lg border border-slate-200 p-3">
            <p className="font-semibold text-slate-700">Hardware</p>
            <p className="text-slate-500">Apple M2 MPS backend</p>
          </div>
        </div>
      </section>

      {/* Results */}
      <section className="mb-12">
        <div className="mb-3 flex items-center gap-2">
          <BarChart3 className="h-5 w-5 text-[var(--green)]" />
          <h2 className="text-xl font-bold text-slate-900">Evaluation Results</h2>
        </div>
        <table className="w-full rounded-lg border border-slate-200 text-sm">
          <thead>
            <tr className="border-b border-slate-200 bg-slate-50">
              <th className="px-4 py-3 text-left font-semibold text-slate-700">Metric</th>
              <th className="px-4 py-3 text-left font-semibold text-slate-700">Minimum</th>
              <th className="px-4 py-3 text-left font-semibold text-slate-700">Target</th>
              <th className="px-4 py-3 text-left font-semibold text-slate-700">Actual</th>
            </tr>
          </thead>
          <tbody>
            {[
              { metric: "AUC-ROC", min: "0.75", target: "> 0.82", actual: "—" },
              { metric: "Accuracy", min: "70%", target: "> 78%", actual: "—" },
              { metric: "Sensitivity", min: "0.65", target: "> 0.75", actual: "—" },
              { metric: "Specificity", min: "0.60", target: "> 0.70", actual: "—" },
            ].map((row) => (
              <tr key={row.metric} className="border-b border-slate-100">
                <td className="px-4 py-3 font-medium text-slate-900">{row.metric}</td>
                <td className="px-4 py-3 text-slate-500">{row.min}</td>
                <td className="px-4 py-3 text-slate-500">{row.target}</td>
                <td className="px-4 py-3 text-slate-400">{row.actual}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="mt-2 text-center text-xs text-slate-400">
          Actual results will be filled in after model training and evaluation.
        </p>
      </section>

      {/* Limitations */}
      <section className="mb-12">
        <div className="mb-3 flex items-center gap-2">
          <AlertCircle className="h-5 w-5 text-amber-500" />
          <h2 className="text-xl font-bold text-slate-900">Known Limitations</h2>
        </div>
        <ul className="space-y-2 text-sm leading-6 text-slate-600">
          <li>All COUGHVID recordings are voluntary intentional coughs</li>
          <li>Trained on ~2,300 expert-labeled samples — small by production ML standards</li>
          <li>Class imbalance: ~78% abnormal / ~22% healthy after expert filtering</li>
          <li>No external validation dataset — generalization to new devices unknown</li>
          <li>Binary only — does not distinguish COVID vs URTI vs LRTI vs other</li>
          <li>COUGHVID collected during COVID pandemic — label distribution reflects that context</li>
          <li>Screening tool only — not a diagnostic</li>
        </ul>
      </section>
    </div>
  );
}
```

- [ ] **Step 2: Verify the page renders**

Run from `frontend/`:
```bash
npm run dev
```

Navigate to `http://localhost:3000/technical`. Verify:
- Dark hero header with "How SickNote Works"
- All 7 sections render in order
- Pipeline flow diagram shows arrows between steps
- Metrics table shows placeholder dashes in "Actual" column
- Nav link "How it Works" is active (bold)
- Clicking "Analyze" nav link returns to product page

- [ ] **Step 3: Verify build passes**

Run from `frontend/`:
```bash
npm run build
```
Expected: both `/` and `/technical` routes built successfully.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/app/technical/page.tsx
git commit -m "feat(frontend): add technical page with ML pipeline explanation"
```

---

### Task 8: Final verification and cleanup

**Files:**
- Modify: `frontend/.gitignore` (if needed)

- [ ] **Step 1: Remove boilerplate assets**

Delete the default Next.js SVGs that are no longer used:

```bash
rm frontend/public/file.svg frontend/public/globe.svg frontend/public/next.svg frontend/public/vercel.svg frontend/public/window.svg
rm frontend/README.md 2>/dev/null
```

- [ ] **Step 2: Full build check**

Run from `frontend/`:
```bash
npm run build
```
Expected: clean build, both routes render.

- [ ] **Step 3: End-to-end smoke test**

Terminal 1 (from project root):
```bash
source .venv/bin/activate && uvicorn api.main:app --port 8000
```

Terminal 2 (from `frontend/`):
```bash
npm run dev
```

Test checklist:
- `http://localhost:3000` — product page loads, hero visible
- Upload an audio file → loading spinner → result card with label + confidence + spectrogram
- Click "Analyze another" → resets to idle
- Navigate to `/technical` → all sections render
- Navigate back to `/` → product page loads
- Microphone record button appears (or shows "not available" if no mic)

- [ ] **Step 4: Commit and push**

```bash
git add -A
git commit -m "feat(frontend): cleanup boilerplate, final build verification"
git push origin main
```

---

## Execution Summary

| Task | What it produces | Files |
|---|---|---|
| 1 | Icons + global styles | package.json, globals.css |
| 2 | Shared navigation | Navbar.tsx, layout.tsx |
| 3 | Microphone recording | RecordButton.tsx |
| 4 | Audio input cards | AudioInput.tsx |
| 5 | Result display | ResultCard.tsx |
| 6 | Product page | page.tsx |
| 7 | Technical page | technical/page.tsx |
| 8 | Cleanup + verification | boilerplate removal |
