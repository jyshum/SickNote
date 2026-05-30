# SickNote Frontend Design Spec

## Overview

Next.js + Tailwind CSS frontend for the SickNote cough screening tool. Two pages: a product page for end-user interaction and a technical page explaining the ML pipeline. Clinical/professional visual style.

## Pages

### Product Page (`/`)

**Layout:** Hero + Cards (option C from brainstorming). Dark branded header, side-by-side action cards for record/upload, unified result card with embedded spectrogram.

**States:**

| State | UI Behavior |
|---|---|
| Idle | Hero header visible. Record and Upload cards active. No result area. |
| Recording | Record card shows pulsing indicator + stop button. Upload card dimmed. |
| Loading | Both cards dimmed. Result area shows skeleton/spinner. "Analyzing your cough..." |
| Result | Label (HEALTHY/ABNORMAL) + confidence percentage + spectrogram image + "Analyze another" reset button. |

**Result color coding:**
- Healthy: green accent `#059669`, badge text "LOW RISK"
- Abnormal: amber accent `#d97706`, badge text "REVIEW RECOMMENDED" (not red — too alarming for screening)

**Error handling:** Inline message replaces result area. "Something went wrong. Try again." with retry button. No modals or toasts.

**Footer:** Medical disclaimer — "Screening tool only. Not a medical diagnosis. See a doctor if concerned."

### Technical Page (`/technical`)

**Layout:** Scrolling single page. Same dark hero header with subtitle "How it works."

**Sections (in order):**

1. **Overview** — One paragraph: what SickNote does, binary classification, COUGHVID dataset.
2. **Dataset** — Stats: 34,400 total recordings, 2,841 expert-labeled, filtering criteria (cough_detected > 0.8, quality filter, majority vote), class distribution (~78% abnormal / ~22% healthy).
3. **Pipeline** — Horizontal flow diagram with icons: Raw Audio → Filter → Label → Spectrogram → Normalize → Split.
4. **Spectrogram Explanation** — Side-by-side healthy vs abnormal spectrogram (placeholder images until P1 provides real ones). Brief text on what mel-spectrograms are and why they work.
5. **Model** — CNN architecture: 3 conv-batchnorm-relu-maxpool blocks → flatten → 2 FC layers. BCEWithLogitsLoss, early stopping on val_AUC, MPS backend.
6. **Results** — Metrics table: AUC, accuracy, sensitivity, specificity. Placeholder values until P1 delivers.
7. **Limitations** — Bullet list from ARCHITECTURE.md known limitations section.

All content is static. No API calls. P1 swaps in real spectrogram images and final metrics later.

## Component Architecture

```
frontend/src/
├── app/
│   ├── layout.tsx              shared nav bar
│   ├── page.tsx                product page
│   ├── technical/
│   │   └── page.tsx            technical page
│   └── globals.css
├── components/
│   ├── Navbar.tsx              top nav: logo + links to / and /technical
│   ├── AudioInput.tsx          record card + upload card side by side
│   ├── ResultCard.tsx          label + confidence badge + spectrogram
│   └── RecordButton.tsx        MediaRecorder API, pulsing indicator, stop
└── lib/
    └── api.ts                  predictCough(file) → PredictionResult (exists)
```

### Component Responsibilities

**Navbar** — renders on both pages via `layout.tsx`. Logo on the left, two navigation links on the right ("Analyze" → `/`, "How it Works" → `/technical`). White background, subtle bottom border.

**AudioInput** — two cards side by side. Left card: `RecordButton` (microphone). Right card: file input styled as a drop zone. Accepts `.webm`, `.wav`, `.ogg`, `.mp3`. Calls `onAudioReady(file: File)` when audio is captured or selected.

**RecordButton** — wraps the MediaRecorder API. Three internal states: idle (show mic icon + "Record" label), recording (pulsing red dot + "Stop" button + elapsed time counter), done (passes recorded blob up as `onRecorded(file: File)`). Outputs a `File` with type `audio/webm`. Browser compatibility: MediaRecorder is supported in all modern browsers. If `navigator.mediaDevices` is unavailable, the record card shows a disabled state with "Microphone not available" text.

**ResultCard** — receives `PredictionResult | null`. When null, renders nothing. When populated, renders the label with color-coded badge, confidence percentage, and spectrogram as an `<img>` from the base64 data URI.

### Data Flow (Product Page)

```
User clicks Record → RecordButton captures audio → File
User clicks Upload → file input → File
    ↓
AudioInput calls onAudioReady(file)
    ↓
page.tsx sets loading state, calls predictCough(file)
    ↓
API returns { label, confidence, spectrogram }
    ↓
page.tsx sets result state → ResultCard renders
    ↓
User clicks "Analyze another" → reset to idle
```

## API Contract

```
POST http://localhost:8000/api/predict
Content-Type: multipart/form-data
Body: file (audio)

Response:
{
    "label": "healthy" | "abnormal",
    "confidence": 0.87,
    "spectrogram": "data:image/png;base64,..."
}
```

The connector in `src/lib/api.ts` is already built and tested against the mock API.

## Visual Style

- **Palette:** White backgrounds, slate-700/900 text, green (`#059669`) for healthy, amber (`#d97706`) for abnormal. Dark navy (`#0f172a` → `#1e3a5f` gradient) for hero header.
- **Typography:** Geist Sans (already configured in layout.tsx). Clean, clinical.
- **Spacing:** Generous padding, rounded corners (12px cards, 8px inner elements).
- **No dark mode.** Light clinical feel only.
- **No emojis in production UI.** Use SVG icons or Lucide icons if needed.

## What Is NOT Included

- No authentication
- No analysis history
- No drag-and-drop upload
- No waveform visualizer
- No dark mode toggle
- No demo clip buttons (can be added later)
- No animations beyond the recording pulse indicator
