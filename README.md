# CEUS-LRM-Fusion

Reference codebase for the CEUS LI-RADS LR-M study:

**AI-Assisted Differentiation of Hepatocellular Carcinoma From Non-HCC Malignancies in CEUS LI-RADS LR-M Lesions**

## Task Definition

The prediction target is binary classification within **CEUS LI-RADS LR-M lesions**:

- positive class: `HCC`
- negative class: `non-HCC malignancy`

The LR-M category defines the inclusion scope, not the prediction target.

## Branches

- `CEUS-GRU`: temporal CEUS sequence model using depthwise-pointwise feature projection, trainable-scale multi-head self-attention, and residual bidirectional GRUs
- `Clinical-LR`: logistic regression over structured clinical variables
- `LRM-Fusion`: neural fusion branch using the same `AttentionGRUModel` family as the CEUS-GRU, but with a fusion-sequence input format instead of a temporal CEUS feature sequence

## Repository Structure

```text
.
├── configs/
│   ├── ceus_gru.yaml
│   ├── clinical_lr.yaml
│   └── fusion.yaml
├── scripts/
│   ├── train_ceus.py
│   ├── train_clinical.py
│   ├── train_fusion.py
│   ├── evaluate_ceus.py
│   ├── evaluate_clinical.py
│   ├── evaluate_fusion.py
│   ├── prepare_clinical_txt.py
│   ├── prepare_clinical_folds.py
│   └── report_model_stats.py
└── src/
    └── ceus_lrm_fusion/
        ├── ceus/
        ├── clinical/
        └── fusion/
```

## Installation

```bash
pip install -e .
```

or

```bash
pip install -r requirements.txt
```

## Data Layout

The repository does not include patient data or checkpoints.

### CEUS-GRU

- training split: `data/ceus/train`
- validation split: `data/ceus/val`
- test split: `data/ceus/test`
- supported file types: `.npz` feature sequences or `.txt` probability sequences

### Clinical-LR

- training split: `data/clinical/train/{HCC,UNHCC}`
- validation split: `data/clinical/val/{HCC,UNHCC}`
- test split: `data/clinical/test/{HCC,UNHCC}`
- each sample is a whitespace-separated feature text file

### LRM-Fusion

- training split: `data/fusion/train`
- validation split: `data/fusion/val`
- test split: `data/fusion/test`
- each sample is a temporal fusion sequence in `.txt` or `.npz` form

The reconciled fusion branch expects prepared multimodal sequences. It does **not** use the previous case-level CSV logistic fusion design.

## Training

### CEUS-GRU

```bash
python scripts/train_ceus.py --config configs/ceus_gru.yaml
```

### Clinical-LR

```bash
python scripts/train_clinical.py --config configs/clinical_lr.yaml
```

### LRM-Fusion

```bash
python scripts/train_fusion.py --config configs/fusion.yaml
```

Outputs are saved under the configured `save_dir` and include `best.pt`, `last.pt`, `history.json`, `summary.json`, and optional `best_swa.pt`.

## Evaluation

### CEUS-GRU

```bash
python scripts/evaluate_ceus.py --checkpoint runs/ceus_gru/best.pt --output reports/ceus_gru
```

### Clinical-LR

```bash
python scripts/evaluate_clinical.py --bundle runs/clinical_lr/model_bundle.pkl --output reports/clinical_lr
```

### LRM-Fusion

```bash
python scripts/evaluate_fusion.py --checkpoint runs/fusion/best.pt --output reports/fusion
```

The CEUS and fusion evaluators export:

- per-sample prediction tables
- attention-weight tables
- ROC and PR points
- confusion matrix, ROC, and PR figures
- validation-set Youden threshold

## Prediction

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
python -m ceus_lrm_fusion.fusion.predict --checkpoint runs/fusion/best.pt --input data/fusion/inference --output reports/fusion_predictions
```

## Parameter Count Utility

Use the formula-based reporter:

```bash
python scripts/report_model_stats.py
```

## Privacy

No patient data, trained weights, or private experimental artifacts are distributed here.

## Citation

If you use this repository in academic work, cite the associated manuscript.
