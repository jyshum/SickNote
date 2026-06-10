# STATS_PROJECT.md — Phase 1: Scoping & Discovery

> **What this document is**: Context and instructions for Claude Code to scope the existing SickNote codebase and report back findings. Do NOT write implementation code yet. The goal of this phase is discovery only.

---

## PROJECT CONTEXT

SickNote is a binary cough audio classifier (healthy vs. abnormal) built on the COUGHVID dataset. We are adding a **controlled experiment** to study the relationship between training data size and model classification performance (AUC-ROC). This experiment is for an AP Statistics 12 final project.

**The experiment in one sentence**: Train 50 independent CNN models at 10 different training set sizes (5 random seeds each), record AUC-ROC for each, then perform a regression slope t-test to determine if more training data significantly predicts better performance.

### Experimental Design Summary

- **50 total runs**: 10 training sizes × 5 random seeds
- **Training sizes**: 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 1950 recordings
- **Random seeds**: 1, 2, 3, 4, 5 (controls subset selection + weight initialization)
- **Architecture**: Identical to existing SickNote CNN (single model, NOT the 5-model ensemble)
- **Hyperparameters**: All fixed — same as existing SickNote config (lr, optimizer, dropout, etc.)
- **Epochs**: Fixed at 30 per run, no early stopping
- **Test set**: Fixed 15% stratified holdout, same for every run (split with a constant seed=0)
- **Training subsets**: Stratified random samples from the training pool, preserving class ratio
- **Evaluation**: AUC-ROC, accuracy, sensitivity, specificity on the fixed test set
- **No data augmentation, no ensemble, no hyperparameter tuning between runs**
- **pos_weight**: Recalculated per run based on the training subset's class ratio

### What Needs to Be Built

A new `stats_experiment/` directory containing:
1. An experiment runner that trains 50 models and logs results to CSV
2. An analysis script that computes regression statistics and generates graphs

Both scripts must **import and reuse** existing SickNote components (model class, data loading, preprocessing). No duplication.

---

## DISCOVERY TASKS

Read the codebase and report back the following. Be specific — include exact file paths, class/function names, and constructor/function signatures.

### Task 1: Project Structure

Run `find . -type f -name "*.py"` and provide the full file tree. Also list any config files (`.yaml`, `.json`, `.env`, etc.) and data directories.

### Task 2: CNN Model Architecture

Find the CNN model class definition. Report:
- File path
- Class name
- Constructor signature (what arguments it takes)
- Forward method signature
- Layer structure (channels, kernel sizes, etc.) — confirm it matches: Conv2d(8) → Conv2d(16) → Conv2d(32) → Linear(128) → Linear(1)
- How the model is instantiated in training (what args are passed)

### Task 3: Dataset & Preprocessing

Find how COUGHVID data is loaded and preprocessed. Report:
- Where are the raw audio files stored? What format (.ogg, .wav, etc.)?
- Where/how are mel spectrograms generated? (torchaudio transforms? librosa? preprocessed and cached?)
- Is there a Dataset class? What's its name, file path, and `__getitem__` return signature?
- How are labels encoded? (0/1? which class is 0, which is 1?)
- How is the train/test split currently done? (random_state? stratified? what ratio?)
- What filtering is applied? (cough_detected > 0.8? expert labels present? quality filter?)
- What is the exact mel spectrogram config? (n_mels, sample_rate, n_fft, hop_length, duration/padding)

### Task 4: Training Loop

Find the training function/script. Report:
- File path and function name
- What arguments does it take? (model, dataloader, optimizer, epochs, device, etc.)
- How is the loss computed? (BCEWithLogitsLoss? pos_weight passed in or hardcoded?)
- How is the optimizer configured? (lr, weight_decay, etc.)
- What batch size is used?
- Is there early stopping? Checkpointing? Learning rate scheduling?
- How is the model evaluated after training? (separate function? inline?)

### Task 5: Evaluation Metrics

Find where AUC-ROC and other metrics are computed. Report:
- File path and function name(s)
- What inputs do the functions expect? (predictions, labels, probabilities, logits?)
- What metrics are returned? (AUC-ROC, accuracy, sensitivity, specificity, others?)
- How is the classification threshold determined? (0.5? 0.52 via Youden's J? Is this relevant to our experiment?)

### Task 6: Existing Configuration

Find any configuration values. Report:
- Learning rate
- Weight decay
- Batch size
- Number of epochs used in existing training
- Dropout rate
- pos_weight value or formula
- Any other hyperparameters

### Task 7: Data Inventory

Report:
- Total number of recordings available after filtering
- Class distribution (how many healthy, how many abnormal)
- Whether preprocessed spectrograms are cached to disk or generated on-the-fly
- Approximate size of the dataset on disk

### Task 8: Dependencies & Environment

Report:
- Key packages in requirements.txt or environment (torch, torchaudio, sklearn, scipy, etc.)
- Python version
- Whether the project currently runs on GPU or CPU

---

## WHAT TO REPORT BACK

After completing all 8 tasks, compile your findings into a structured summary. Use this exact format:

```
## CODEBASE DISCOVERY REPORT

### Project Structure
[file tree]

### CNN Model
- File: [path]
- Class: [name]
- Constructor: [signature]
- Forward: [signature]
- Architecture confirmed: [yes/no, with notes if different]

### Dataset & Preprocessing
- Raw data location: [path]
- Audio format: [format]
- Spectrogram generation: [method, config]
- Dataset class: [name, path, return signature]
- Label encoding: [scheme]
- Train/test split: [method, ratio, seed]
- Filtering: [criteria]

### Training
- File: [path]
- Function: [name, signature]
- Loss: [config]
- Optimizer: [config]
- Batch size: [value]
- Epochs: [value]
- Early stopping: [yes/no]
- LR scheduling: [yes/no]

### Evaluation
- File: [path]
- Functions: [names, signatures]
- Threshold: [value, method]
- Metrics returned: [list]

### Configuration
- lr: [value]
- weight_decay: [value]
- batch_size: [value]
- dropout: [value]
- pos_weight: [value or formula]
- n_mels: [value]
- sample_rate: [value]

### Data Inventory
- Total recordings: [count]
- Abnormal: [count] ([%])
- Healthy: [count] ([%])
- Cached spectrograms: [yes/no]
- Dataset size on disk: [approx]

### Dependencies
- Python: [version]
- Key packages: [list with versions]
- GPU available: [yes/no]

### Potential Issues / Notes
[Anything unexpected, missing, or that might affect the experiment design]
```

---

## WHAT NOT TO DO IN THIS PHASE

- Do NOT create any new files yet
- Do NOT modify any existing files
- Do NOT run any training
- Do NOT install any packages
- Do NOT write implementation code

Discovery only. Report back. Implementation spec comes next.
