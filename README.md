# PRISM-F1: Interpretable AI for F1 Pit Stop Prediction

PRISM-F1 is a small, fully interpretable neural network for predicting pit stop
decisions on the Kaggle Playground Series S6E5 ("F1 Pit Stop Prediction")
competition. It produces a decision (PIT / NO_PIT), a probability, and a
plain-English strategist-style explanation for every prediction.

**Public leaderboard ROC-AUC: 0.909** (joint model, λ_concept=20)

---

## What's in this repo

├── notebooks/  
│   ├── 01_data_exploration.ipynb        # Data inspection, feature engineering, concept GT  
│   ├── 02_blocks_and_stage1.ipynb       # Model blocks + Stage 1 training (Reader+Imagination)  
│   ├── 03_stage2_concepts.ipynb         # Stage 2: Concept Block training (frozen Reader)  
│   ├── 04_stage3_decision.ipynb         # Stage 3: Decision Block training (frozen above)  
│   ├── 05_inference_submission.ipynb    # Inference + Kaggle submission (staged model)  
│   ├── 06_joint_training_experiment.ipynb   # Reviewer's critique: joint vs staged  
│   ├── 07_joint_inference_submission.ipynb  # Joint model inference + submission  
│   └── 08_inference_with_reasoning.ipynb    # Per-row reasoning generation  
├── outputs/  
│   ├── submission.csv                   # Staged model (0.70 public AUC)  
│   ├── submission_joint_lambda20.csv    # Joint model λ=20 (0.909 public AUC)  
│   ├── audit_trail_demo.md              # 5 worked examples with full attribution  
│   ├── demo_predictions_v2.md           # 80 stratified predictions with plain-English reasoning  
│   └── lambda_ablation_results.csv      # Full Pareto-curve data  
├── docs/  
│   └── PRISM_F1_Architecture.md         # The architecture specification  
├── .gitignore  
└── README.md  

Data files (`data/*.csv`, `data/*.pkl`) and model checkpoints (`checkpoints/*.pt`)
are not committed due to size. See [Reproducing the results](#reproducing-the-results).

---

## The core idea

PRISM-F1 forces every pit decision to flow through six named, human-readable
concepts. A pit-wall engineer can read the output and immediately understand
*why* the model said what it said.
  
                  input features (13)
                          │  
                          ▼  
              ┌──────────────────────────┐  
              │  Concept Block (MLP)     │  ──► 6 interpretable concepts in [0, 1]  
              └──────────────────────────┘  
                          │  
                          ▼  
              ┌──────────────────────────┐  
              │  Decision Block (Linear) │  ──► pit probability  
              └──────────────────────────┘  
                                       

### The six concepts

| Concept | What it captures |
|---|---|
| `degradation_severity` | How worn the tyres are relative to the compound's typical cliff |
| `pace_decay_rate` | Whether lap-time is degrading lap to lap |
| `strategic_window` | Whether we're in the typical pit window for this compound |
| `track_position_risk` | Whether pitting risks a defendable on-track position |
| `undercut_pressure` | Whether competitors have pitted and threaten an undercut |
| `endgame_proximity` | Whether the race is too close to finished to recover the pit cost |

Because the Decision Block is linear, every prediction has an exact additive
decomposition:  

logit = w₁·c₁ + w₂·c₂ + ... + w₆·c₆ + bias

You can attribute the decision to specific concept contributions, in seconds,
with no SHAP or LIME wrapper required.

---

## Example: a real prediction from the test set

Driver: HAM, Monaco Grand Prix 2024, Lap 45 (RaceProgress 0.58)
Compound: MEDIUM, TyreLife: 51, Position: P5
Concepts (model output):
degradation_severity   0.99   ████████████████████
pace_decay_rate        0.01
strategic_window       0.02
track_position_risk    0.89   █████████████████
undercut_pressure      0.99   ████████████████████
endgame_proximity      0.04
Decision: PIT (probability 99.4%)
Reasoning:
"The tyres are deep into their wear phase, and another few laps at racing
pace risks a sharp drop-off and lost lap time — the call is to pit. On top
of that, a rival has pitted and is poised to undercut — failing to respond
this lap likely hands them the position once stops cycle through."

This is **not** a post-hoc explanation. It's a direct reading of the model's
own decomposition into named concepts plus their relative contributions to
the logit.

---

## The empirical story: staged vs joint training

This repo contains an unusually thorough ablation. The original PRISM-F1
design used **staged training** — Reader pretrained on a self-supervised
next-lap prediction task, then frozen; Concept Block trained against
analytical concept ground truth; Decision Block trained against pit labels.
That gave **0.70 public AUC**.

A reviewer suggested a simpler architecture: no Reader, no staged training,
just `x → ConceptBlock → DecisionBlock` trained end-to-end with auxiliary
concept supervision. To test the critique honestly, we ran a λ-ablation
across the auxiliary loss weight:

| Training mode | λ_concept | Val AUC | Public AUC | Mean concept ρ | Concepts ≥ 0.7 |
|---|---|---|---|---|---|
| Staged (original) | — | 0.740 | 0.700 | 0.834 | 5/6 |
| Joint | 0.0 | 0.904 | — | 0.174 | 0/6 |
| Joint | 0.1 | 0.905 | — | 0.253 | 0/6 |
| Joint | 0.5 | 0.905 | — | 0.540 | 3/6 |
| Joint | 2.0 | 0.898 | — | 0.733 | 4/6 |
| Joint | 5.0 | 0.903 | **0.922** | 0.759 | 4/6 |
| Joint | 20.0 | 0.889 | **0.909** | 0.893 | 5/6 |

**The reviewer was right.** For this task — 13 tabular features computed by
formula — joint training with auxiliary concept supervision dominates
staged training on both axes (prediction quality *and* concept fidelity).
The Reader and the staged pipeline did not earn their keep here.

The joint model at λ=20 was chosen as the final submission because it
**matches the staged model's interpretability (5/6 concepts at ρ≥0.7) while
beating it by 21 percentage points on public AUC**.

This is a real finding, not a setback. It tells us when the PRISM family
pattern earns its complexity (high-dimensional sensor streams) and when it
doesn't (small tabular tasks with handcrafted features).

---

## Reproducing the results

### Setup

```bash
git clone <this-repo-url>
cd prism-f1
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install torch numpy pandas matplotlib scikit-learn jupyter
```

### Data

Download `train.csv` and `test.csv` from the
[Kaggle Playground S6E5](https://www.kaggle.com/competitions/playground-series-s6e5)
competition and place them in `data/`. Also place the parent dataset
`f1_strategy_dataset_v4.csv` in `data/` if you want to retrain Stage 1
(otherwise skip notebook 02).

### Run

Open the notebooks in order:

1. **`01_data_exploration.ipynb`** — produces `data/x_train.pkl`,
   `data/x_test.pkl`, `data/parent_pairs.pkl`
2. **`02_blocks_and_stage1.ipynb`** through **`04_stage3_decision.ipynb`**
   — original staged PRISM-F1 pipeline
3. **`05_inference_submission.ipynb`** — staged submission (0.70 public)
4. **`06_joint_training_experiment.ipynb`** — runs the λ-ablation
5. **`07_joint_inference_submission.ipynb`** — joint λ=20 submission (0.909 public)
6. **`08_inference_with_reasoning.ipynb`** — generates plain-English reasoning

Total training time on CPU: roughly 30-40 minutes for the full pipeline,
plus ~1.5 hours for the joint λ-ablation in notebook 06.

---

## Model size and deployment notes

| Variant | Parameters | Use case |
|---|---|---|
| PRISM-F1 staged | 3,357 (runtime) + 1,637 (training-only) | Original design — useful as a family example, but joint variant is better for this task |
| Joint λ=20 (final) | **717** | The submission. Small enough to deploy on a microcontroller. |

The joint model fits comfortably in INT8 quantization on an STM32-class MCU.

---

## Honest acknowledgements

- The original staged design was over-engineered for this task. The repo
  preserves it as Notebooks 02-05 because the empirical comparison is
  itself the most interesting part of the work.
- The synthetic Kaggle test set contains some non-physical rows (e.g.,
  Lap 1 with RaceProgress = 1.0). Notebook 08's reasoning layer includes
  coherence guards to prevent confidently nonsense outputs on those rows.
- ROC-AUC was the competition metric. Calibration was not optimized for
  and was visibly miscalibrated (the model's mean predicted probability is
  ~2× the true base rate); this would matter for log-loss-evaluated
  competitions or production deployment.

---

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you reference this work:
