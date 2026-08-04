# Source-Disjoint Audio Spectrogram Transformers with Consistency Regularization for Bioacoustic Sound Classification

Official implementation of the **Source-Disjoint Audio Spectrogram Transformer (AST)** framework for the Field Sound Classification 2022 (**FSC22**) bioacoustic benchmark dataset.

---

## 🌟 Overview

Bioacoustic sound classification is frequently affected by **data leakage** when multi-segment audio recordings or pitch-augmented clip variants derived from the exact same physical recording source span across training and evaluation partitions.

This repository introduces:
1. **Source-Disjoint Evaluation Protocol**: Uses nested `StratifiedGroupKFold` partitioning based on underlying FreeSound recording IDs (`Source File Name`), strictly isolating physical audio sources across train, validation, and test splits (zero source-group overlap).
2. **Consistency-Regularized AST Fine-Tuning**: Fine-tunes an Audio Spectrogram Transformer backbone with 8 unfrozen encoder layers, incorporating focal-smoothed classification loss and multi-task logit (Jensen-Shannon divergence) and embedding (cosine distance) consistency loss across stochastic views.
3. **Dual Performance Benchmarks**:
   - **Clean Unseen Test Set**: **87.41% Accuracy** / **87.54% Macro F1** (100% unseen acoustic recording sources).
   - **Transductive Source-Consistent TTA**: **98.44% Accuracy** / **98.42% Macro F1** (3-seed probability ensemble).

---

## 📊 Experimental Results

| Experimental Setup | Split Strategy | Accuracy (%) | Macro F1 (%) | Description |
| :--- | :--- | :---: | :---: | :--- |
| **FSC22 Base Paper CNN** *(Lin et al., 2022)* | Augmentation-before-split | 92.59% | — | Original baseline reference |
| **Baseline AST** *(Seed 42)* | Clip-level Random Split | 94.49% | 94.41% | Standard clip-level split |
| **Cross-Fitted Calibrated AST** | Clip-level Calibrated | 95.06% | 94.98% | Calibrated ensemble |
| **AST v2 Source-Consistent** *(Seeds 101, 202, 303)* | Transductive TTA | 98.27% | 98.25% | Multi-seed consistency fine-tuned |
| **Legacy 3-Seed AST** *(Seeds 42, 17, 73)* | Transductive TTA | **98.44%** | **98.42%** | Transductive augmentation-overlap result |
| **Clean Unseen AST Ensemble (Proposed)** | **Source-Disjoint Unseen Test** | **87.41%** | **87.54%** | **Strict Unseen-Recording Benchmark** |

---

## ⚙️ Installation & Setup

### 1. Environment Setup
Clone the repository and create the Conda environment:

```bash
git clone https://github.com/KanishkaM108/FSC22_Bioacoustic_AST.git
cd FSC22_Bioacoustic_AST

# Create and activate environment
conda create -n fsc22_research python=3.10 -y
conda activate fsc22_research

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage Guide

### 1. Prepare Source-Disjoint Protocol
Generate the leakage-free source-grouped manifest and lock the test split:

```bash
python src/prepare_clean_grouped_protocol.py
```

### 2. Train Clean AST Models
Train 3 independent seeds (101, 202, 303) on source-disjoint partitions:

```bash
python src/train_ast_v2_source_consistent.py --seed 101 --epochs 40 --unfrozen-blocks 8 --tag clean_ast_v1
python src/train_ast_v2_source_consistent.py --seed 202 --epochs 40 --unfrozen-blocks 8 --tag clean_ast_v1
python src/train_ast_v2_source_consistent.py --seed 303 --epochs 40 --unfrozen-blocks 8 --tag clean_ast_v1
```

Or run the full automated pipeline via batch script:

```bat
run_clean_unseen_training.bat
```

### 3. Evaluate Locked Test Split
Evaluate the 3-seed ensemble on the locked, source-disjoint unseen test set:

```bash
python src/evaluate_clean_ast_ensemble.py --tag clean_ast_v1
```

---

## 📁 Repository Structure

```text
FSC22_Bioacoustic_AST/
├── src/
│   ├── prepare_clean_grouped_protocol.py   # Source-disjoint StratifiedGroupKFold splitter
│   ├── train_ast_v2_source_consistent.py     # AST fine-tuning with focal & consistency loss
│   ├── evaluate_clean_ast_ensemble.py      # Locked unseen test set evaluator
│   ├── evaluate_source_consistent_ast_v2.py # Transductive source-consistent evaluator
│   └── train_baseline.py                   # Baseline CNN training script
├── outputs/                                # Manifests, metrics, and confusion matrix plots
├── run_clean_unseen_training.bat           # Automated training & evaluation pipeline
├── requirements.txt                        # Python dependencies list
└── README.md                               # Project documentation
```

---

## 📝 Citation & Research Paper

If you find this codebase or protocol design useful in your research, please cite our manuscript:

```bibtex
@article{kanishka2026source,
  title={Source-Disjoint Audio Spectrogram Transformers with Consistency Regularization for Bioacoustic Sound Classification},
  author={Kanishka et al.},
  journal={Bioacoustic Intelligence Benchmark},
  year={2026}
}
```
