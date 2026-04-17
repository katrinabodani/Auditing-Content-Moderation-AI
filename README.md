# Responsible & Explainable AI — Assignment 2
**FAST-NUCES | Auditing Content Moderation AI for Bias, Adversarial Robustness & Safety**

---

## Environment

| Item | Detail |
|---|---|
| Python | 3.10 |
| GPU | NVIDIA RTX 4050 (6 GB VRAM) |
| CUDA | 12.1 |
| OS | Windows 11 / WSL2 |

---

## Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd responsible-ai-assignment

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS / WSL
# venv\Scripts\activate         # Windows CMD

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Dataset

This assignment uses the **Jigsaw Unintended Bias in Toxicity Classification** dataset.

1. Create a free account at [kaggle.com](https://kaggle.com)
2. Accept the competition at: https://kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification
3. Download **only** these two files from the Data tab:
   - `jigsaw-unintended-bias-train.csv`
   - `validation.csv`
4. Place them in a `data/` folder at the root of this repo (this folder is gitignored)

---

## Reproducing Results

Run the notebooks **in order**. Each part depends on artifacts saved by the previous one.

| Notebook | Description |
|---|---|
| `part1.ipynb` | Baseline DistilBERT classifier |
| `part2.ipynb` | Bias audit across identity cohorts |
| `part3.ipynb` | Adversarial attacks (evasion + poisoning) |
| `part4.ipynb` | Bias mitigation techniques |
| `part5.ipynb` | Guardrail pipeline demonstration |

Saved model checkpoints are written to `./checkpoints/` (gitignored). You must run Part 1 before any subsequent part.

---

## Repository Structure

```
.
├── part1.ipynb
├── part2.ipynb
├── part3.ipynb
├── part4.ipynb
├── part5.ipynb
├── pipeline.py          # ModerationPipeline class (Part 5)
├── requirements.txt
├── README.md
├── .gitignore
└── data/                # NOT committed — place dataset files here
```

---

## .gitignore Note

Dataset files (`*.csv`), model checkpoints (`*.pt`, `*.bin`, `saved_model/`, `checkpoints/`), and the virtual environment (`venv/`) are excluded from version control via `.gitignore`.
