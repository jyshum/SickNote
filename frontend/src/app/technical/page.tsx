"use client";

import { useEffect, useRef } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import {
  Database,
  Filter,
  Tag,
  AudioWaveform,
  BarChart3,
  AlertCircle,
  Cpu,
  FlaskConical,
  Layers,
  ScanEye,
  ArrowLeft,
} from "lucide-react";

export default function TechnicalPage() {
  const pageRef = useRef<HTMLDivElement>(null);
  const act2Ref = useRef<HTMLElement>(null);

  useEffect(() => {
    gsap.registerPlugin(ScrollTrigger);

    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

    const ctx = gsap.context(() => {
      gsap.from("[data-reveal]", {
        y: 24,
        opacity: 0,
        duration: 0.7,
        ease: "power3.out",
        stagger: 0.08,
        scrollTrigger: { trigger: "[data-reveal]", start: "top 82%" },
      });

      document
        .querySelectorAll<HTMLElement>("[data-section-reveal]")
        .forEach((section) => {
          gsap.from(section.querySelectorAll("[data-item]"), {
            y: 28,
            opacity: 0,
            duration: 0.7,
            ease: "power3.out",
            stagger: 0.08,
            scrollTrigger: { trigger: section, start: "top 78%" },
          });
        });

      if (act2Ref.current) {
        gsap.from(act2Ref.current.querySelectorAll("[data-decision]"), {
          y: 32,
          opacity: 0,
          duration: 0.8,
          ease: "power3.out",
          stagger: 0.12,
          scrollTrigger: { trigger: act2Ref.current, start: "top 72%" },
        });
      }
    }, pageRef);

    return () => ctx.revert();
  }, []);

  return (
    <div
      ref={pageRef}
      className="relative min-h-[100dvh] w-full max-w-full overflow-x-hidden"
    >
      <div className="noise-overlay" />

      {/* ── ACT 1: How It Works ── */}
      <div className="bg-[var(--background)]">
        <div className="clinic-grid pointer-events-none absolute inset-0 -z-10" />

        <div className="mx-auto max-w-5xl px-5 pb-20 pt-28 sm:px-8 md:pt-32 lg:px-10">
          {/* Back link */}
          <a
            href="/"
            data-reveal
            className="mb-10 inline-flex items-center gap-2 text-sm font-medium text-slate-500 transition hover:text-slate-800"
          >
            <ArrowLeft className="h-4 w-4" />
            Back to screening
          </a>

          {/* Hero */}
          <div
            data-reveal
            className="mb-16 rounded-[2rem] bg-slate-950 px-7 py-10 shadow-[0_32px_80px_-40px_rgba(15,23,42,0.7)] sm:px-10 sm:py-12"
          >
            <h1 className="max-w-3xl text-3xl font-black leading-tight tracking-normal text-white sm:text-4xl">
              How SickNote works
            </h1>
            <p className="mt-3 max-w-xl text-base leading-7 text-slate-400">
              From raw audio to screening result in under 3 seconds
            </p>
          </div>

          {/* Overview */}
          <section data-section-reveal className="mb-20">
            <h2 data-item className="mb-4 text-2xl font-bold text-slate-950">
              Overview
            </h2>
            <p data-item className="max-w-3xl text-base leading-7 text-slate-600">
              SickNote is a binary cough classifier that distinguishes healthy
              coughs from abnormal ones. It converts audio recordings into mel
              spectrograms — visual representations of sound frequencies — and
              feeds them through an ensemble of five convolutional neural
              networks. Each prediction includes a Grad-CAM heatmap showing
              which spectrogram regions drove the result.
            </p>
          </section>

          {/* Dataset */}
          <section data-section-reveal className="mb-20">
            <div data-item className="mb-4 flex items-center gap-3">
              <span className="flex h-9 w-9 items-center justify-center rounded-full bg-emerald-50 text-emerald-600">
                <Database className="h-4 w-4" />
              </span>
              <h2 className="text-2xl font-bold text-slate-950">Dataset</h2>
            </div>
            <p data-item className="mb-6 max-w-3xl text-base leading-7 text-slate-600">
              The COUGHVID dataset contains ~34,400 crowdsourced cough
              recordings. Of these, 2,841 were reviewed by four expert
              physicians who provided diagnostic labels.
            </p>
            <div data-item className="grid gap-4 sm:grid-cols-3">
              {[
                { value: "34,400", label: "Total recordings" },
                { value: "2,841", label: "Expert-labeled" },
                { value: "~2,300", label: "After filtering" },
              ].map((s) => (
                <div
                  key={s.label}
                  className="rounded-[1.75rem] border border-slate-200 bg-white p-5 shadow-[0_8px_30px_-12px_rgba(15,23,42,0.12)]"
                >
                  <p className="text-3xl font-black tabular-nums text-slate-950">
                    {s.value}
                  </p>
                  <p className="mt-1 text-sm text-slate-500">{s.label}</p>
                </div>
              ))}
            </div>

            <div
              data-item
              className="mt-5 rounded-[1.75rem] border border-slate-200 bg-white p-5 shadow-[0_8px_30px_-12px_rgba(15,23,42,0.12)]"
            >
              <p className="mb-3 text-sm font-bold text-slate-800">
                Filtering criteria
              </p>
              <ul className="space-y-1.5 text-sm leading-6 text-slate-600">
                <li>Cough detection confidence &gt; 0.8</li>
                <li>At least one expert diagnosis present</li>
                <li>Quality rated acceptable by majority of experts</li>
              </ul>
              <div className="mt-5">
                <p className="mb-2 text-sm font-bold text-slate-800">
                  Class distribution
                </p>
                <div className="flex h-8 overflow-hidden rounded-full">
                  <div
                    className="flex items-center justify-center bg-amber-500"
                    style={{ width: "78%" }}
                  >
                    <span className="text-xs font-bold text-white">
                      78% abnormal
                    </span>
                  </div>
                  <div
                    className="flex items-center justify-center bg-emerald-500"
                    style={{ width: "22%" }}
                  >
                    <span className="text-xs font-bold text-white">22%</span>
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* Pipeline */}
          <section data-section-reveal className="mb-20">
            <div data-item className="mb-4 flex items-center gap-3">
              <span className="flex h-9 w-9 items-center justify-center rounded-full bg-emerald-50 text-emerald-600">
                <FlaskConical className="h-4 w-4" />
              </span>
              <h2 className="text-2xl font-bold text-slate-950">
                Processing pipeline
              </h2>
            </div>
            <div
              data-item
              className="grid gap-2 rounded-[1.75rem] border border-slate-200 bg-white p-5 shadow-[0_8px_30px_-12px_rgba(15,23,42,0.12)] sm:grid-cols-[repeat(11,minmax(0,auto))] sm:items-center sm:justify-center sm:p-7"
            >
              {[
                { icon: AudioWaveform, label: "Raw Audio" },
                { icon: Filter, label: "Filter" },
                { icon: Tag, label: "Label" },
                { icon: BarChart3, label: "Spectrogram" },
                { icon: Cpu, label: "Normalize" },
                { icon: Database, label: "Split" },
              ].map((step, i) => (
                <div key={step.label} className="contents">
                  <div className="flex items-center gap-3 rounded-2xl bg-slate-50 px-4 py-3 sm:flex-col sm:gap-1.5">
                    <step.icon className="h-5 w-5 text-slate-600" />
                    <span className="text-xs font-semibold text-slate-700">
                      {step.label}
                    </span>
                  </div>
                  {i < 5 && (
                    <span className="hidden text-slate-300 sm:block">
                      &rarr;
                    </span>
                  )}
                </div>
              ))}
            </div>
          </section>

          {/* Model Architecture */}
          <section data-section-reveal className="mb-20">
            <div data-item className="mb-4 flex items-center gap-3">
              <span className="flex h-9 w-9 items-center justify-center rounded-full bg-emerald-50 text-emerald-600">
                <Cpu className="h-4 w-4" />
              </span>
              <h2 className="text-2xl font-bold text-slate-950">
                Model architecture
              </h2>
            </div>
            <p data-item className="mb-5 max-w-3xl text-base leading-7 text-slate-600">
              A small convolutional neural network trained from scratch. Five
              models are trained with different random seeds and their
              predictions are averaged for robustness. Each result includes a
              Grad-CAM heatmap showing which spectrogram regions influenced the
              classification.
            </p>
            <div
              data-item
              className="overflow-x-auto rounded-[1.75rem] border border-slate-200 bg-white p-5 shadow-[0_8px_30px_-12px_rgba(15,23,42,0.12)]"
            >
              <div className="space-y-1.5 font-[family-name:var(--font-mono)] text-sm leading-relaxed text-slate-700">
                <p>Input: (1, 64, T) mel spectrogram</p>
                <p className="text-slate-400">
                  &nbsp;&nbsp;&darr;
                </p>
                <p>
                  &nbsp;&nbsp;Conv2d(8) + BatchNorm + ReLU + MaxPool
                </p>
                <p>
                  &nbsp;&nbsp;Conv2d(16) + BatchNorm + ReLU + MaxPool
                </p>
                <p>
                  &nbsp;&nbsp;Conv2d(32) + BatchNorm + ReLU + MaxPool
                </p>
                <p className="text-slate-400">
                  &nbsp;&nbsp;&darr;
                </p>
                <p>&nbsp;&nbsp;Flatten</p>
                <p>
                  &nbsp;&nbsp;Linear(128) + ReLU + Dropout(0.5)
                </p>
                <p>&nbsp;&nbsp;Linear(1) &rarr; logit</p>
              </div>
            </div>

            <div data-item className="mt-5 grid gap-4 sm:grid-cols-2">
              {[
                {
                  icon: BarChart3,
                  title: "Loss function",
                  desc: "BCEWithLogitsLoss + pos_weight",
                },
                {
                  icon: Cpu,
                  title: "Optimizer",
                  desc: "Adam (lr=3e-4, wd=1e-4)",
                },
                {
                  icon: Layers,
                  title: "Ensemble",
                  desc: "5 models, different seeds, averaged probabilities",
                },
                {
                  icon: ScanEye,
                  title: "Explainability",
                  desc: "Grad-CAM heatmaps on last conv block",
                },
              ].map((c) => (
                <div
                  key={c.title}
                  className="flex gap-4 rounded-[1.75rem] border border-slate-200 bg-white p-5 shadow-[0_8px_30px_-12px_rgba(15,23,42,0.12)]"
                >
                  <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-slate-100 text-slate-600">
                    <c.icon className="h-4 w-4" />
                  </span>
                  <div>
                    <p className="text-sm font-bold text-slate-900">
                      {c.title}
                    </p>
                    <p className="mt-0.5 text-sm text-slate-500">{c.desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* Evaluation Metrics */}
          <section data-section-reveal className="mb-20">
            <div data-item className="mb-4 flex items-center gap-3">
              <span className="flex h-9 w-9 items-center justify-center rounded-full bg-emerald-50 text-emerald-600">
                <BarChart3 className="h-4 w-4" />
              </span>
              <h2 className="text-2xl font-bold text-slate-950">
                Evaluation metrics
              </h2>
            </div>
            <div
              data-item
              className="overflow-x-auto rounded-[1.75rem] border border-slate-200 bg-white shadow-[0_8px_30px_-12px_rgba(15,23,42,0.12)]"
            >
              <table className="w-full min-w-[30rem] text-sm">
                <thead>
                  <tr className="border-b border-slate-200 bg-slate-50/80">
                    <th className="px-5 py-3.5 text-left font-bold text-slate-800">
                      Metric
                    </th>
                    <th className="px-5 py-3.5 text-left font-bold text-slate-800">
                      Actual
                    </th>
                    <th className="px-5 py-3.5 text-left font-bold text-slate-800">
                      Target
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    { metric: "AUC-ROC", actual: "0.73", target: "> 0.82", note: "" },
                    { metric: "Accuracy", actual: "76%", target: "> 78%", note: "" },
                    { metric: "Sensitivity", actual: "0.68", target: "> 0.75", note: "" },
                    { metric: "Specificity", actual: "0.77", target: "> 0.70", note: "" },
                  ].map((row) => (
                    <tr
                      key={row.metric}
                      className="border-b border-slate-100 last:border-b-0"
                    >
                      <td className="px-5 py-3.5 font-semibold text-slate-900">
                        {row.metric}
                      </td>
                      <td className="px-5 py-3.5 font-[family-name:var(--font-mono)] tabular-nums text-slate-700">
                        {row.actual}
                        {row.note && (
                          <span className="ml-2 font-sans text-xs text-slate-400">
                            {row.note}
                          </span>
                        )}
                      </td>
                      <td className="px-5 py-3.5 text-slate-500">
                        {row.target}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p data-item className="mt-4 max-w-3xl text-sm leading-6 text-slate-500">
              Classification threshold optimized from 0.50 to 0.52 using
              Youden&apos;s J statistic to balance sensitivity and specificity.
              Metrics reported on the held-out test set (15% of data, never seen
              during training).
            </p>
          </section>

          {/* Limitations */}
          <section data-section-reveal className="mb-10">
            <div data-item className="mb-4 flex items-center gap-3">
              <span className="flex h-9 w-9 items-center justify-center rounded-full bg-amber-50 text-amber-600">
                <AlertCircle className="h-4 w-4" />
              </span>
              <h2 className="text-2xl font-bold text-slate-950">
                Known limitations
              </h2>
            </div>
            <ul
              data-item
              className="max-w-3xl space-y-2.5 text-sm leading-6 text-slate-600"
            >
              <li>
                All COUGHVID recordings are voluntary intentional coughs
              </li>
              <li>
                ~2,300 expert-labeled samples, split into ~1,600 for training —
                small by production ML standards
              </li>
              <li>
                Class imbalance: ~78% abnormal / ~22% healthy after expert
                filtering
              </li>
              <li>
                No external validation dataset — generalization to new devices
                unknown
              </li>
              <li>
                Binary only — does not distinguish COVID vs URTI vs LRTI vs
                other
              </li>
              <li>
                COUGHVID was collected during the COVID pandemic — label
                distribution reflects that context
              </li>
              <li>Screening tool only — not a diagnostic</li>
            </ul>
          </section>
        </div>
      </div>

      {/* ── ACT 2: Design Decisions ── */}
      <section
        ref={act2Ref}
        className="relative bg-slate-950 px-5 py-24 sm:px-8 md:py-32 lg:px-10"
      >
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_30%_20%,rgba(214,40,40,0.06),transparent_50%)]" />

        <div className="relative mx-auto max-w-5xl">
          <div className="mb-16">
            <h2 className="text-3xl font-black tracking-normal text-white sm:text-4xl">
              Design decisions
            </h2>
            <p className="mt-3 max-w-xl text-base leading-7 text-slate-400">
              The engineering choices behind SickNote
            </p>
          </div>

          <div className="space-y-5">
            {[
              {
                category: "Architecture",
                title: "Why binary classification",
                body: "Multi-class classification (COVID, URTI, LRTI, obstructive disease) dropped accuracy sharply because pathological coughs occupy overlapping feature space in the spectrogram domain. Binary classification — healthy vs. abnormal — is medically honest and performs significantly better with limited data. The output is \"something sounds off,\" not a specific diagnosis.",
              },
              {
                category: "Architecture",
                title: "Why we built a CNN from scratch",
                body: "With only ~2,300 expert-labeled samples, a small 3-layer CNN with aggressive dropout (0.5) and reduced channel widths [8, 16, 32] was the right fit. Larger architectures overfit before learning useful features. We tuned every hyperparameter against real dataset statistics from explore.py before writing a single training loop.",
              },
              {
                category: "Training",
                title: "Why we tried transfer learning and reverted",
                body: "We attempted to replace our CNN with a pretrained ResNet18 backbone to leverage features learned from millions of ImageNet images. The plan: freeze pretrained layers, train only a classifier head on our cough spectrograms. With ~1,600 training samples, the model's higher capacity worked against us — it overfit faster than our from-scratch CNN. The pretrained features were too general for the narrow spectrogram patterns that distinguish healthy from abnormal coughs. We reverted to the original architecture.",
              },
              {
                category: "Data",
                title: "Why data augmentation didn't help",
                body: "Standard audio augmentation (noise injection, time stretching) on a dataset this small amplified noise rather than adding signal. The augmented samples were too similar to the originals to provide new information, and synthetic noise patterns confused the model. We removed augmentation entirely and kept the raw expert-labeled data.",
              },
              {
                category: "Training",
                title: "Why ensemble + threshold tuning",
                body: "Instead of relying on a single model's opinion, we train five models with different random seeds and average their predictions. This smooths out the variance inherent to our small dataset. We then optimized the classification threshold from 0.50 to 0.52 using Youden's J statistic, which maximizes the balance between sensitivity (catching abnormal coughs) and specificity (correctly identifying healthy ones).",
              },
            ].map((decision) => (
              <div
                key={decision.title}
                data-decision
                className="rounded-[1.75rem] border border-slate-700/40 bg-slate-900/60 p-6 backdrop-blur-sm sm:p-8"
              >
                <p className="mb-2 font-[family-name:var(--font-mono)] text-xs uppercase tracking-[0.2em] text-slate-500">
                  {decision.category}
                </p>
                <h3 className="mb-3 text-lg font-bold text-white">
                  {decision.title}
                </h3>
                <p className="max-w-3xl text-sm leading-7 text-slate-400">
                  {decision.body}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
