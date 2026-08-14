# Data Analysis — Individual Task 1: Part 1

Data Science role: **Graduate – Data Engineer** at Whitehaven Coal.
Course: Case Studies in Data Science, RMIT University.
Author: Bhavesh Kanyal (s4189426)

This repository contains the machine learning analysis referenced in
Part 1.3 (Data Analysis) of the report. Two algorithms — a **Decision
Tree Classifier** and a **Support Vector Machine (RBF kernel)** — are
applied separately to two datasets relevant to a mining data engineer:
equipment failure prediction and workplace safety-incident severity.

## Datasets

| # | Dataset | Source | Rows |
|---|---------|--------|------|
| 1 | AI4I 2020 Predictive Maintenance | [Kaggle](https://www.kaggle.com/datasets/geetanjalisikarwar/equipment-failure-prediction-dataset) / [UCI](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) | 10,000 |
| 2 | Industrial Safety and Health Analytics Database | [Kaggle](https://www.kaggle.com/datasets/ihmstefanini/industrial-safety-and-health-analytics-database) | 425 |

Neither dataset's raw CSV is committed to this repo (Kaggle's terms
discourage redistribution). Run `download_data.py` to fetch both from
public GitHub mirrors before running the analysis.

## Setup

**macOS / Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/download_data.py
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/download_data.py
```

If pandas/scikit-learn are already installed globally (e.g. via
Anaconda), the venv steps are optional — `pip install -r
requirements.txt` and `python src/download_data.py` alone are enough.

## Run the analysis

```bash
python src/ai4i_analysis.py
python src/safety_analysis.py
```

Each script prints accuracy, precision, recall, F1, and ROC-AUC for
both models, plus the top Decision Tree feature importances, and saves
a results JSON to `results/`.

## Methodology summary

- **Target 1 (AI4I):** binary `Machine failure` label, predicted from
  sensor telemetry (air/process temperature, rotational speed, torque,
  tool wear) and product quality type.
- **Target 2 (Industrial Safety):** binary `severe` label (Accident
  Level III–V vs I–II), predicted from incident attributes (industry
  sector, country, gender, employee/third-party status, critical risk
  category).
- Both targets are rare-event / imbalanced, so **precision, recall,
  F1, and ROC-AUC** are reported instead of accuracy alone — a model
  that always predicts the majority class would otherwise look
  artificially strong.
- Both models use `class_weight="balanced"` to counter that imbalance
  during training.

## Repository structure

```
.
├── data/                  # datasets (gitignored — run download_data.py)
├── results/                # output JSON metrics (gitignored)
├── src/
│   ├── download_data.py
│   ├── ai4i_analysis.py
│   └── safety_analysis.py
├── requirements.txt
└── README.md
```
