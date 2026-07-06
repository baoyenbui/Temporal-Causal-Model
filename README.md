# Temporal Causal Learning Model in Student Interaction Sequences

## Core Idea

We model student learning as a temporal sequence of dependent interactions and learn causal structure over time.

Layer 1 - Temporal Representation:
Convert raw logs into ordered learning windows with behavioral features and student clustering.

Layer 2 - Causal Structure Learning:
Construct system-level time series and learn lagged causal relations using PCMCI with stability selection and latent proxy variables.

Layer 3 - Counterfactual Reasoning:
Apply interventions on the learned causal graph to simulate temporal effects of changes in key learning variables.

---

## Project Structure

Repository is organized into modular components for data processing, causal discovery, baseline comparison, and counterfactual analysis.

```
project/
│── data/                     # raw datasets
│
│── src/                      # core implementation modules
│   ├── data_loader.py
│   ├── preprocessing.ipynb
│   ├── causal_discovery.py
│   ├── baseline.py
│   ├── cf.py
│   ├── evaluation.py
│   └── utils.py
│
│── main.py                   # run pipeline
│── run_baseline.py           # run baseline models
│── run_cf.py                 # run counterfactual analysis
│── requirements.txt
│── README.md
```

---

## Installation
---

### 1. Create environment

```
python -m venv venv
```

Activate environment:

```
# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate
```

---

### 2. Install dependencies

```
pip install -r requirements.txt
```

---

### 3. Optional (DYNOTEARS)

```
pip install git+https://github.com/sessions-lab/dynotears.git
```
