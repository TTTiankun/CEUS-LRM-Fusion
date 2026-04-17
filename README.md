# CEUS-LRM-Fusion

Public reference implementation for the study:

**AI-Assisted Differentiation of Hepatocellular Carcinoma From Non-HCC Malignancies in CEUS LI-RADS LR-M Lesions: Retrospective Development and Prospective Test Study**

This repository is organized as a paper companion codebase rather than a local experiment workspace. It retains the final methodological mainline only:

- `CEUS-GRU`: temporal CEUS sequence model
- `Clinical-LR`: logistic regression model using structured clinical variables
- `LRM-Fusion`: case-level fusion of CEUS and clinical probabilities

Patient-derived datasets, trained weights, cached figures, and historical experiment outputs are intentionally excluded from the public repository.

## Study task

The target task is binary classification within **CEUS LI-RADS LR-M lesions**:

- positive class: `HCC`
- negative class: `non-HCC malignancy`

In the manuscript, the LR-M category defines the inclusion scope. It is not the prediction target itself.

## Repository structure

```text
.
├── configs/
│   ├── ceus_gru.yaml
│   ├── clinical_lr.yaml
│   └── fusion.yaml
├── docs/
│   ├── data_format.md
│   └── manuscript/
├── scripts/
│   ├── train_ceus.py
│   ├── train_clinical.py
│   ├── train_fusion.py
│   ├── evaluate_ceus.py
│   ├── evaluate_clinical.py
│   ├── evaluate_fusion.py
│   ├── prepare_clinical_txt.py
│   └── prepare_clinical_folds.py
└── src/
    └── ceus_lrm_fusion/
        ├── ceus/
        ├── clinical/
        └── fusion/
```

## Installation

Create a Python environment and install the package in editable mode:

```bash
pip install -e .
```

If you prefer requirements-based installation:

```bash
pip install -r requirements.txt
```

## Data organization

The repository does not ship with clinical or CEUS data.

- CEUS temporal data: `data/ceus/{train,val,test}/`
- Clinical feature files: `data/clinical/{train,val,test}/{HCC,UNHCC}/`
- Fusion labels: `data/fusion/*.csv`


## Training workflow

### 1. Train the CEUS temporal branch

```bash
python scripts/train_ceus.py --config configs/ceus_gru.yaml
```

Outputs are saved under `runs/ceus_gru/` by default, including:

- `best.pt`
- `last.pt`
- `history.json`
- `summary.json`

### 2. Train the clinical branch

```bash
python scripts/train_clinical.py --config configs/clinical_lr.yaml
```

Outputs are saved under `runs/clinical_lr/`, including:

- `model_bundle.pkl`
- `coefficients.csv`
- `summary.json`

### 3. Train the fusion branch

Run the CEUS and clinical evaluation steps first to generate case-level probability CSV files, then update `configs/fusion.yaml` and train:

```bash
python scripts/train_fusion.py --config configs/fusion.yaml
```

## Evaluation workflow

### CEUS-GRU

```bash
python scripts/evaluate_ceus.py --checkpoint runs/ceus_gru/best.pt --output reports/ceus_gru
```

The evaluator exports:

- per-sample prediction tables
- attention weight tables
- ROC and PR curve points
- confusion matrix, ROC, and PR figures
- validation-set Youden threshold

### Clinical-LR

```bash
python scripts/evaluate_clinical.py --bundle runs/clinical_lr/model_bundle.pkl --output reports/clinical_lr
```

The evaluator exports:

- per-sample prediction tables
- coefficient table with bootstrap confidence intervals
- confusion matrix, ROC, PR, and coefficient figures

### LRM-Fusion

```bash
python scripts/evaluate_fusion.py --bundle runs/fusion/model_bundle.pkl --output reports/fusion
```

## Prediction workflow

### CEUS-GRU

```bash
python -m ceus_lrm_fusion.ceus.predict --checkpoint runs/ceus_gru/best.pt --input data/ceus/inference --output reports/ceus_predictions
```

### Clinical-LR

```bash
python -m ceus_lrm_fusion.clinical.predict --bundle runs/clinical_lr/model_bundle.pkl --input data/clinical/inference --output reports/clinical_predictions
```

### LRM-Fusion

```bash
python -m ceus_lrm_fusion.fusion.predict --bundle runs/fusion/model_bundle.pkl --ceus reports/ceus_predictions/predictions.csv --clinical reports/clinical_predictions/predictions.csv --output reports/fusion_predictions
```

## Reported metrics

The public code exports the core binary metrics used in the manuscript-oriented workflow:

- AUC
- average precision
- accuracy
- sensitivity
- specificity
- precision
- Brier score
- expected calibration error

Decision curve analysis is not precomputed automatically in the current public release, but the exported per-sample probability tables are sufficient for downstream DCA scripts.

## Interpretability outputs

The public repository preserves the interpretability directions emphasized in the manuscript:

- temporal attention weights from `CEUS-GRU`
- coefficient magnitude and bootstrap confidence intervals from `Clinical-LR`
- branch-level probability contributions for `LRM-Fusion`

## Clinical data preparation

Two helper scripts are included for local dataset preparation:

- `scripts/prepare_clinical_txt.py`: convert a spreadsheet into per-case text files
- `scripts/prepare_clinical_folds.py`: create fold directories from a split manifest

These scripts are intentionally generic and may require local adaptation to match institutional export formats.

## Privacy and data availability

No patient-level data, trained checkpoints, or private experimental artifacts are provided in this repository.

For manuscript reproduction, users should prepare an institutional dataset with the same file structure and variable definitions. Public redistribution should follow local ethics, privacy, and data-sharing requirements.

## Paper-to-code mapping

- Manuscript temporal CEUS model: `src/ceus_lrm_fusion/ceus/`
- Manuscript clinical logistic model: `src/ceus_lrm_fusion/clinical/`
- Manuscript fusion model: `src/ceus_lrm_fusion/fusion/`

## Citation

If you use this repository in academic work, please cite the associated manuscript.
