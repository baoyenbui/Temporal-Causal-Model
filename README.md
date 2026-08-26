# Temporal Alignment for Construct-Level Treatment-Effect Prediction

This repository contains the implementation and frozen experimental evidence for evaluating the predictive value of temporal alignment in construct-level treatment-effect prediction using the CausalEdu benchmark.

## Research Question

> Under leakage-safe leave-one-treatment-construct-out (LOTO) evaluation, does preserving the original within-student temporal alignment improve out-of-fold mean absolute error (MAE) relative to a matched circular-shift control?

The study compares a **Temporal** representation with a **Matched Circular-Shift (MCS)** representation under the same feature architecture, preprocessing, and Random Forest estimator. Three feature-free baselines are also evaluated.

## Experimental Protocol

- **Dataset:** CausalEdu
- **Evaluation units:** 88 treatment–question–year units
- **Evaluation:** 15 treatment-construct LOTO folds
- **Primary metric:** MAE
- **Secondary metrics:** RMSE, Spearman correlation, sign agreement
- **Cluster bootstrap:** 2,000 replicates, seed 42
- **Circular-shift seed:** 43
- **Random Forest:** 200 trees, seed 42

Outcome-related `Checkout` and `CheckoutRetry` interactions are excluded before feature construction to prevent outcome leakage.

## Frozen Experimental Evidence

The final experiment is frozen at commit:

```text
f4e0dc7a1ca3f8b6fbcbce922802eb2ccb865b4d
The frozen manifest records the experimental configuration, input counts and hashes, fold assignments, model settings, out-of-fold predictions, lookup coverage, and pairwise statistical results.

**Manifest:**  
https://github.com/baoyenbui/Temporal-Causal-Model/blob/f4e0dc7a1ca3f8b6fbcbce922802eb2ccb865b4d/final_manifest.json

## Data Availability

The study uses the CausalEdu benchmark released by Eedi and Microsoft Research.

**Official repository:**  
https://github.com/Eedi/CausalEdu

The dataset is not redistributed in this repository. Please follow the conditions specified by the dataset authors.

## Installation

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run the evaluation:

```bash
python run_option_a.py
```

Run tests:

```bash
pytest
```
