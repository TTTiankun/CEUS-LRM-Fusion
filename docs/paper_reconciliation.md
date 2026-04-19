# Paper Reconciliation Notes

## 1. What Should Be Treated As Ground Truth

- `CEUS-GRU` implementation details: follow the original `GRU_Code` behavior and the legacy config from `0717experience_result/result_02/output/config.yaml`
- `LRM-Fusion` implementation details: follow the original `GRU_Code` behavior and the legacy config from `20250814Muti/result03/output/config.yaml`
- `Clinical-LR`: follow the implementation in this public repository

## 2. Confirmed Mismatches Between Word Files and Code/Config

### CEUS-GRU

The supplementary Word text does not fully match the legacy config used as implementation truth.

Relevant differences:

- Word supplement says attention embedding dimension `96`; legacy config uses `attention_dim = 40`
- Word supplement says batch size `32`; legacy config uses `batch_size = 64`
- Word supplement says early-stopping patience `100`; legacy config uses `150`
- Word supplement says Gaussian noise `sigma = 0.05`; legacy config uses `0.15`
- Word supplement says feature masking `30%`; legacy config disables feature masking
- Legacy config enables `time_mask`, `temporal_jitter`, and `time_warp`, which is stronger than the current Word summary

### LRM-Fusion

The LRM-Fusion structural description in the supplementary Word file is not fully consistent with the legacy implementation path.

Relevant differences:

- Word text describes a `256 + 2 = 258` dimensional concatenated fusion representation
- legacy code/config uses the same `AttentionGRUModel_Pro` family with `attention_dim = 80`, `gru_dims = [64, 32]`, and prepared multimodal temporal fusion sequences
- legacy fusion config includes `confidence_suppression`, which is not reflected in the Word description

### Clinical-LR

The current public code and the manuscript wording are not fully identical.

Relevant differences:

- Word supplement says `L2`, `C = 10.0`, `max_iter = 1000`, and a final model built from the top 20 features
- current public repo config uses `L2`, `C = 1.0`, `max_iter = 5000`
- current public repo uses 10 raw structured variables with preprocessing and one-hot expansion, so the exact fitted coefficient count is data-dependent unless you explicitly freeze a top-20-feature pipeline

## 3. Parameter Counts

Using the reconciled legacy configs and the released architecture:

| Component | Assumption | Trainable Parameters |
| --- | --- | ---: |
| CEUS-GRU temporal model | `D_in=96`, `attention_dim=40`, `gru_dims=[2048,1024,512,256]` | `111,321,980` |
| LRM-Fusion module | `D_in=2`, `attention_dim=80`, `gru_dims=[64,32]` | `131,828` |
| Clinical-LR | 20 retained features + 1 bias | `21` |
| Full released system excluding image backbone | CEUS-GRU + LRM-Fusion + Clinical-LR | `111,453,829` |

Important limitation:

- the exact image feature extractor parameter count cannot be recovered unambiguously from the released repositories because the image-backbone training code and definitive ConvNeXtV2 variant identifier are not present

## 4. Short Tuning Description For The Paper

You can use this wording directly or with light editing:

> All model selection and hyperparameter tuning were performed exclusively on the retrospective internal training/validation split, and the independent prospective test cohort was kept fully isolated for one-time blinded evaluation. For CEUS-GRU, the released implementation used an attention-enhanced residual bidirectional GRU with attention dimension 40, GRU widths 2048/1024/512/256, 8 attention heads, dropout 0.2, AdamW optimization (learning rate 1e-4, weight decay 0.005), 10 warm-up epochs, cosine decay, label smoothing 0.05, gradient clipping 0.5, SWA, and early stopping with patience 150; the best checkpoint was selected on the validation set. For LRM-Fusion, the released implementation used the same model family with attention dimension 80 and GRU widths 64/32, together with AdamW, dropout 0.2, label smoothing 0.05, gradient clipping 0.5, 10 warm-up epochs, SWA, and validation-based checkpoint selection. Clinical-LR hyperparameters were fixed before prospective testing according to the released configuration and assessed on the internal validation set only. The prospective test cohort was not used for hyperparameter selection, early stopping, threshold selection, or model comparison tuning.

## 5. What Was Selected On Validation

- best CEUS-GRU checkpoint
- best LRM-Fusion checkpoint
- any reported validation-set Youden threshold used for threshold analysis
- backbone/model-family screening decisions described before final locked testing
- clinical regularization settings only if you want to describe them as tuned rather than prespecified

## 6. Recommendation For The Manuscript

If you want the paper to match the reconciled repository cleanly:

- update the CEUS-GRU hyperparameter paragraph to match the legacy config values
- revise the LRM-Fusion structure paragraph to remove or qualify the inconsistent `258`-dimensional statement unless you can point to a separate unreleased fusion-preprocessing implementation
- explicitly state whether Clinical-LR follows the current public repo configuration or the older top-20/C=10 version, because those are not the same pipeline
