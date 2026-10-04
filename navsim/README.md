# MomWorld for NAVSIM

This directory is the audited MomWorld overlay used for the released NAVSIM
results. It contains the project-specific model, GTRS trajectory ranker,
NAVTRAIN-only selector training, deterministic safety gates, formal evaluation
launchers, and regression tests. It intentionally does **not** vendor the full
NAVSIM devkit.

## Released results

| Benchmark | Strict score | Coverage |
| --- | ---: | ---: |
| NAVSIM v1 NavTest PDMS | **0.9020463636** | 12,146 / 12,146 |
| NAVSIM v2 NavTest EPDMS | **0.9010545135** | 11,989 / 11,989 |
| NAVSIM v2 NavHard combined (paper result1) | **0.4277559498** | 5,912 / 5,912 |
| NAVSIM v2 NavHard combined (post-paper optimized) | **0.4281794505** | 5,912 / 5,912 |

All released benchmark results use the same frozen unified checkpoint. The
paper-reported NavHard result1 is the exact artifact behind the rounded 42.8 row
in manuscript Table 5. The later post-paper NavHard result is derived from the
same frozen result1 lineage and additionally applies the label-free fixed 10%
temporal-consistency policy; it is a distinct, strictly validated optimization
and is not the artifact used for the manuscript row. See
[RESULTS.md](RESULTS.md) for exact sub-scores, artifact hashes, and the numerical
comparison.

## What is included

- `navsim/agents/momworld/`: MomWorld model, feature builder, protocol selectors,
  and the NAVSIM v2 NavHard pairwise gate.
- `navsim/agents/gtrs_dense/`: the dense GTRS candidate generator and ranker.
- `scripts/training/`: NAVTRAIN-only cross-fitting and frozen gate fitting.
- `scripts/evaluation/`: strict v1/v2 launchers, NavHard validation, and the
  label-free pair-consistency exporter used by the post-paper optimized result.
- `tests/`: focused tests for GTRS, protocol gates, scoring, and cache safety.

Private paths, credentials, datasets, checkpoints, per-scene benchmark scores,
metric caches, generated trajectories, and evaluation CSV files are excluded.

## Installation as an overlay

Prepare a compatible full NAVSIM checkout and its normal dependencies first.
Then copy this directory over the checkout root:

```bash
git clone https://github.com/modaxiansheng/MomWorld.git
rsync -a MomWorld/navsim/ /path/to/NAVSIM/
cd /path/to/NAVSIM
pip install -r requirements.txt
pip install -r requirements-extra.txt
pip install -e .
```

The overlay keeps NAVSIM's original package paths so the official evaluator,
splits, maps, and metric-cache implementation remain in use. Benchmark token
lists and two-stage mappings are deliberately not duplicated here; use the
official split files distributed with the corresponding NAVSIM release.

Set only portable paths in the environment:

```bash
export NAVSIM_DEVKIT_ROOT=/path/to/NAVSIM
export NAVSIM_EXP_ROOT=/path/to/experiments
export OPENSCENE_DATA_ROOT=/path/to/openscene
export NUPLAN_MAPS_ROOT=/path/to/nuplan-maps
export CHECKPOINT_PATH=/path/to/momworld_gtrs_all_navtrain.ckpt
```

## Training

The released selector lineage uses NAVTRAIN supervision only. The main entry
points are:

```bash
python scripts/training/crossfit_momworld_gtrs_fusion_all_navtrain.py --help
python scripts/training/train_momworld_v1_protocol_gate_all_navtrain.py --help
python scripts/training/train_momworld_v2_navtest_selector_gate_all_navtrain.py --help
python scripts/training/refit_momworld_v2_navhard_pairwise_gate.py --help
```

Freeze the produced states and record their SHA-256 digests before formal
evaluation. Do not tune from NavTest or NavHard per-scene scores.

## Formal evaluation

```bash
# NAVSIM v1 NavTest
GPU_IDS=0,1,2,3 \
bash scripts/evaluation/run_momworld_rule_scorer_navtest_v1.sh

# NAVSIM v2 NavTest
GPU_IDS=0,1,2,3 \
bash scripts/evaluation/run_momworld_v2_navtest_selector_gate.sh

# NAVSIM v2 NavHard
GPU_IDS=0,1,2,3 \
V2_NAVHARD_PAIRWISE_STATE_PATH=/path/to/frozen_pairwise_gate.pt \
bash scripts/evaluation/run_momworld_v2_navhard_pairwise_gate.sh
```

The frozen NavHard output is the incumbent result1 lineage. The post-paper
optimized export is then created before scoring from that incumbent,
NAVTRAIN-fitted predictions, and input trajectories only:

```bash
python scripts/evaluation/build_momworld_pair_consistency_variants.py --help
```

Validate every candidate against a known complete baseline:

```bash
python scripts/evaluation/validate_navhard_csv.py \
  baseline.csv candidate.csv strict_validation.json
```

Only results with exact token coverage, the expected stage partition, finite
scores, and all strict checks passing should be reported.

## Evaluation boundary

NavTest and NavHard are evaluation-only. This release does not use benchmark
per-scene labels, oracle selection, or post-evaluation token/trajectory choice.
The pair-consistency policy is fixed before evaluation and consumes only
NAVTRAIN-fitted predictions plus trajectory geometry.

## Provenance and license

The filtered source snapshot is based on internal source revision
`8953f3acd1ff83bce4ebd83f330742c2396f6b5d`. The paper result1 and later strict
best are published as separate result lineages. The pair-consistency exporter
was added after result1 as the audited, label-free policy used by the post-paper
optimized result.

This subproject is released under Apache License 2.0. Upstream copyright and
license notices are retained in [LICENSE](LICENSE) and
[THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).
