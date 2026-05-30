import { Database, Filter, Tag, AudioWaveform, BarChart3, AlertCircle, Cpu, FlaskConical } from "lucide-react";

export default function TechnicalPage() {
  return (
    <div className="bg-[#f7f9fb]">
    <div className="mx-auto max-w-5xl px-4 py-6 sm:px-6 lg:px-8 lg:py-10">
      {/* Hero */}
      <div className="mb-10 rounded-[2rem] bg-slate-950 px-6 py-8 text-left shadow-[0_24px_80px_-48px_rgba(15,23,42,0.85)] sm:px-8 sm:py-10">
        <h1 className="max-w-2xl text-3xl font-semibold leading-none tracking-tight text-white sm:text-4xl">
          How SickNote works
        </h1>
        <p className="mt-3 max-w-xl text-sm leading-6 text-slate-300">
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
        <div className="grid gap-3 sm:grid-cols-3">
          <div className="rounded-2xl border border-slate-200 bg-white p-4">
            <p className="text-2xl font-bold text-slate-900">34,400</p>
            <p className="text-xs text-slate-500">Total recordings</p>
          </div>
          <div className="rounded-2xl border border-slate-200 bg-white p-4">
            <p className="text-2xl font-bold text-slate-900">2,841</p>
            <p className="text-xs text-slate-500">Expert-labeled</p>
          </div>
          <div className="rounded-2xl border border-slate-200 bg-white p-4">
            <p className="text-2xl font-bold text-slate-900">~2,300</p>
            <p className="text-xs text-slate-500">After filtering</p>
          </div>
        </div>
        <div className="mt-4 rounded-2xl border border-slate-200 bg-white p-4">
          <p className="mb-2 text-sm font-semibold text-slate-700">Filtering Criteria</p>
          <ul className="space-y-1 text-sm text-slate-600">
            <li>Cough detection confidence &gt; 0.8</li>
            <li>At least one expert diagnosis present</li>
            <li>Quality rated acceptable by majority of experts</li>
          </ul>
          <div className="mt-4">
            <p className="mb-1 text-sm font-semibold text-slate-700">Class Distribution</p>
            <div className="flex h-7 overflow-hidden rounded-full">
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
        <div className="grid gap-2 rounded-2xl border border-slate-200 bg-white p-4 sm:grid-cols-[repeat(11,minmax(0,auto))] sm:items-center sm:justify-center sm:p-6">
          {[
            { icon: AudioWaveform, label: "Raw Audio" },
            { icon: Filter, label: "Filter" },
            { icon: Tag, label: "Label" },
            { icon: BarChart3, label: "Spectrogram" },
            { icon: Cpu, label: "Normalize" },
            { icon: Database, label: "Split" },
          ].map((step, i) => (
            <div key={step.label} className="contents">
              <div className="flex items-center gap-3 rounded-2xl bg-slate-50 px-4 py-3 sm:flex-col sm:gap-1">
                <step.icon className="mb-1 h-5 w-5 text-slate-600" />
                <span className="text-xs font-medium text-slate-700">{step.label}</span>
              </div>
              {i < 5 && <span className="hidden text-slate-300 sm:block">→</span>}
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
        <div className="grid gap-4 sm:grid-cols-2">
          <div className="rounded-2xl border border-slate-200 bg-white p-4 text-center">
            <div className="mb-2 flex h-24 items-center justify-center rounded bg-slate-900">
              <span className="text-xs text-slate-500">Healthy spectrogram placeholder</span>
            </div>
            <p className="text-sm font-medium text-emerald-600">Healthy Cough</p>
          </div>
          <div className="rounded-2xl border border-slate-200 bg-white p-4 text-center">
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
        <div className="overflow-x-auto rounded-2xl border border-slate-200 bg-white p-4">
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
        <div className="mt-4 grid gap-3 text-sm sm:grid-cols-2">
          <div className="rounded-2xl border border-slate-200 bg-white p-3">
            <p className="font-semibold text-slate-700">Loss Function</p>
            <p className="text-slate-500">BCEWithLogitsLoss + pos_weight</p>
          </div>
          <div className="rounded-2xl border border-slate-200 bg-white p-3">
            <p className="font-semibold text-slate-700">Optimizer</p>
            <p className="text-slate-500">Adam (lr=1e-3, wd=1e-4)</p>
          </div>
          <div className="rounded-2xl border border-slate-200 bg-white p-3">
            <p className="font-semibold text-slate-700">Checkpoint</p>
            <p className="text-slate-500">Best val_AUC + early stopping</p>
          </div>
          <div className="rounded-2xl border border-slate-200 bg-white p-3">
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
        <div className="overflow-x-auto rounded-2xl border border-slate-200 bg-white">
        <table className="w-full min-w-[34rem] text-sm">
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
        </div>
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
    </div>
  );
}
