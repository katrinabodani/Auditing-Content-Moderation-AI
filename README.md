# Responsible & Explainable AI -- Assignment 2
**Auditing Content Moderation AI for Bias, Adversarial Robustness & Safety**

---

## Environment

| Item    | Detail                          |
|---------|---------------------------------|
| Python  | 3.10                            |
| GPU     | NVIDIA GeForce RTX 4050 Laptop  |
| VRAM    | 6 GB                            |
| CUDA    | 12.4                            |
| OS      | Windows 11                      |

---

## Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd responsible-ai-assignment

# 2. Create and activate a conda environment
conda create -n responsible-ai python=3.10 -y
conda activate responsible-ai

# 3. Install PyTorch with CUDA 12.4
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. Register the kernel for Jupyter / VSCode
python -m ipykernel install --user --name responsible-ai --display-name "Python (responsible-ai)"
```

---

## Dataset

This assignment uses the **Jigsaw Unintended Bias in Toxicity Classification** dataset.

1. Create a free account at [kaggle.com](https://kaggle.com)
2. Accept the competition at: https://www.kaggle.com/competitions/jigsaw-multilingual-toxic-comment-classification/data
3. Download only these two files from the Data tab:
   - `jigsaw-unintended-bias-train.csv`
   - `validation.csv`
4. Place them in a `data/` folder at the root of this repo (gitignored)

---

## Reproducing Results

Run the notebooks **in order**. Each part depends on artefacts saved by the previous one.
All artefacts are saved to `checkpoints/` (gitignored).

| Notebook     | Description                          | Runtime (approx.) |
|--------------|--------------------------------------|-------------------|
| `part1.ipynb` | Baseline DistilBERT classifier      | 35-50 min         |
| `part2.ipynb` | Bias audit across identity cohorts  | < 5 min           |
| `part3.ipynb` | Adversarial attacks                 | 35-50 min         |
| `part4.ipynb` | Bias mitigation techniques          | 90-120 min        |
| `part5.ipynb` | Guardrail pipeline demonstration    | 10-15 min         |

---

## Results Summary

### Part 1 -- Baseline Classifier

| Metric        | Value  |
|---------------|--------|
| Accuracy      | 0.9463 |
| F1 Macro      | 0.8114 |
| F1 Toxic      | 0.6520 |
| AUC-ROC       | 0.9419 |
| Threshold     | 0.4    |

### Part 2 -- Bias Audit

| Metric                      | High-black | Reference | Disparity |
|-----------------------------|------------|-----------|-----------|
| TPR                         | 0.6444     | 0.5660    | +0.0784   |
| FPR                         | 0.1345     | 0.1049    | +0.0296   |
| FNR                         | 0.3556     | 0.4340    | -0.0784   |
| Disparate Impact ratio      | --         | --        | 1.2822    |
| Statistical Parity Diff     | --         | --        | -0.0448   |
| Equal Opportunity Diff      | --         | --        | -0.0296   |

The model applies a lower effective decision threshold to the high-black cohort, flagging more of everything (both toxic and non-toxic content). The Disparate Impact ratio of 1.28 exceeds the 1.25 concern threshold.

### Part 3 -- Adversarial Attacks

**Attack 1 -- Character-level evasion**

| Metric                   | Value  |
|--------------------------|--------|
| Sample size              | 500    |
| Avg confidence before    | 0.9690 |
| Avg confidence after     | 0.0036 |
| Attack Success Rate      | 99.80% |

**Attack 2 -- Label-flipping poisoning (5% of training labels flipped)**

| Metric    | Baseline | Poisoned | Delta   |
|-----------|----------|----------|---------|
| Accuracy  | 0.9463   | 0.9433   | -0.0030 |
| F1 Macro  | 0.8114   | 0.8028   | -0.0086 |
| F1 Toxic  | 0.6520   | 0.6363   | -0.0157 |
| FNR       | 0.3702   | 0.3796   | +0.0094 |
| FPR       | 0.0262   | 0.0286   | +0.0024 |

### Part 4 -- Mitigation

| Technique                 | Overall F1 | High-black FPR | Reference FPR | Stat. Parity D | Equal Opp. D |
|---------------------------|------------|----------------|---------------|----------------|--------------|
| Baseline (Part 1)         | 0.8114     | 0.1345         | 0.1049        | -0.0448        | -0.0296      |
| Reweighing (pre-proc)     | 0.8090     | 0.1345         | 0.0909        | -0.0703        | -0.0435      |
| ThresholdOptimizer (post) | 0.8120     | 0.1261         | 0.1189        | -0.0132        | -0.0072      |
| Oversampling (data aug)   | 0.8111     | 0.2521         | 0.0979        | -0.1322        | -0.1542      |

Best mitigated model: **ThresholdOptimizer** (post-processing, no retraining required).

### Part 5 -- Guardrail Pipeline (1,000 comments)

| Layer                    | Comments | Percentage |
|--------------------------|----------|------------|
| Layer 1 (input filter)   | 0        | 0.0%       |
| Layer 2 (model)          | 981      | 98.1%      |
| Layer 3 (review queue)   | 19       | 1.9%       |

| Metric                        | Value  |
|-------------------------------|--------|
| Auto-action F1                | 0.7727 |
| Auto-action Precision         | 0.6957 |
| Auto-action Recall            | 0.4848 |
| Review queue toxic rate       | 57.89% |

Threshold sensitivity:

| Band               | Review (N) | Auto F1 | Auto Precision | Auto Recall |
|--------------------|------------|---------|----------------|-------------|
| Narrow (0.45-0.55) | 9          | 0.7580  | 0.6600         | 0.4648      |
| Default (0.40-0.60)| 19         | 0.7727  | 0.6957         | 0.4848      |
| Wide (0.30-0.70)   | 44         | 0.7873  | 0.8889         | 0.4444      |

---

## Repository Structure

```
.
+-- part1.ipynb       Baseline classifier
+-- part2.ipynb       Bias audit
+-- part3.ipynb       Adversarial attacks
+-- part4.ipynb       Bias mitigation
+-- part5.ipynb       Guardrail pipeline
+-- pipeline.py       ModerationPipeline class
+-- requirements.txt
+-- README.md
+-- .gitignore
+-- data/             NOT committed -- place dataset CSV files here
+-- checkpoints/      NOT committed -- model checkpoints and artefacts
```

---

## Notes

- All notebooks must be run with the `responsible-ai` conda environment selected as the kernel
- The `checkpoints/` directory is created automatically when Part 1 runs
- The `data/` directory must be created manually before running Part 1
- Part 5 imports `pipeline.py` directly -- both files must be in the same directory
