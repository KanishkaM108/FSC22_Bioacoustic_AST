# FSC22 metric-upgrade runbook

This package adds two experiments without overwriting your existing models or
results. Extract the ZIP directly into your existing `FSC22_Research` folder;
it adds two new Python files and two optional batch files. Then run all commands
from:

`C:\Users\Kanishka\Downloads\FSC22_Research`

Keep the `fsc22_research` Conda environment active.

## 1. Run this first: no retraining

```bat
python src\evaluate_source_consistent_ast_v2.py
```

This loads the existing seed 42, 17, and 73 AST checkpoints and evaluates all
6,075 original/pitch variants.  For each original recording, it averages model
probabilities over its three variants.  The rule does not read any label while
constructing a prediction.

Results are written to:

`outputs\source_consistent_ast_legacy_three_seed`

The terminal prints accuracy, macro precision, macro recall, and macro-F1, plus
whether both 97% targets were reached.  Zip and upload that output folder:

```bat
powershell -Command "Compress-Archive -Path outputs\source_consistent_ast_legacy_three_seed -DestinationPath source_consistent_v2_results.zip -Force"
```

Important: this is a transductive augmentation-overlap result.  It uses source
identity and training-partition variants during inference.  It can be used to
demonstrate why the paper-compatible protocol is optimistic, but it must not be
reported as clean unseen-recording performance.

## 2. If the first run is still below the target: train AST v2

Run one seed at a time.  Seed 101 gives the first new result; seeds 202 and 303
add ensemble diversity.

```bat
python src\train_ast_v2_source_consistent.py --seed 101
python src\train_ast_v2_source_consistent.py --seed 202
python src\train_ast_v2_source_consistent.py --seed 303
```

The v2 trainer is designed for the RTX 3050 Laptop GPU:

- batch size 4, gradient accumulation 4;
- six unfrozen AST blocks with differential learning rates;
- independent time/frequency/noise augmentation for same-recording views;
- focal-smoothed classification loss;
- logit and embedding consistency regularization;
- no test-derived class weights.

Each seed creates a new checkpoint such as:

`models\fsc22_ast_v2_seed101.pt`

It does not replace the old models.  If CUDA reports an out-of-memory error,
rerun that seed with four blocks:

```bat
python src\train_ast_v2_source_consistent.py --seed 101 --unfrozen-blocks 4 --force
```

After all three seeds finish, evaluate their source-consistent ensemble:

```bat
python src\evaluate_source_consistent_ast_v2.py --seeds 101 202 303 --checkpoint-template "models/fsc22_ast_v2_seed{seed}.pt" --tag ast_v2
```

The results will be in:

`outputs\source_consistent_ast_ast_v2`

Zip them for review:

```bat
powershell -Command "Compress-Archive -Path outputs\ast_v2,outputs\source_consistent_ast_ast_v2 -DestinationPath ast_v2_results.zip -Force"
```

## What can be claimed

Keep these results separate in the paper:

| Result | Accuracy | Macro-F1 | Correct interpretation |
|---|---:|---:|---|
| Clean original-recording AST | 90.12% | 89.90% | Unseen-recording estimate |
| Paper-protocol AST, seed 42 | 94.49% | 94.41% | Augmentation-overlap development result |
| Cross-fitted calibrated ensemble | 95.06% | 94.98% | Best current model-only overlap result |
| Source-overlap label audit | 98.93% | 98.93% | Audit only; not a model result |
| New source-consistent AST | Pending run | Pending run | Transductive overlap result; not clean performance |

Do not write 97% or 98% in the manuscript until the new output files show the
number.  A 97% score on 1,215 rows requires at least 1,179 correct predictions;
98% requires at least 1,191.
