# Strict NAVSIM results

All counts and digests below come from complete formal outputs. Raw CSV files,
checkpoints, trajectories, datasets, caches, and per-scene benchmark values are
not distributed.

## Shared checkpoint

- Artifact: `momworld_gtrs_all_navtrain_98fb9bb.ckpt`
- SHA-256: `4201d129bfb3a5e00025a12d7921d3280d298c301a5b99f4bcfbd466521e8592`
- Training boundary: NAVTRAIN only

## NAVSIM v1 NavTest

| Metric | Value |
| --- | ---: |
| PDMS | **0.9020463636307717** |
| No at-fault collision | 0.9864976123826774 |
| Drivable-area compliance | 0.9817223777375268 |
| Ego progress | 0.8569272922145696 |
| Time to collision | 0.9387452659311708 |
| Comfort | 0.9996706734727482 |
| Scenarios | 12,146 |

Formal CSV SHA-256:
`52cad329bfef25b302f6527d0c01c3f6b6adac2d1f45928a1bb4c135361e7b12`

## NAVSIM v2 NavTest

| Metric | Value |
| --- | ---: |
| EPDMS | **0.9010545135192526** |
| No at-fault collision | 0.9810659771457169 |
| Drivable-area compliance | 0.9809825673534073 |
| Driving-direction compliance | 0.9954958712152807 |
| Traffic-light compliance | 0.9979147551922596 |
| Ego progress | 0.8885466050412962 |
| Time to collision | 0.9819834848611227 |
| Lane keeping | 0.9608808074067896 |
| History comfort | 0.9827341729919092 |
| Scenarios | 11,989 |

Formal CSV SHA-256:
`21d64c2d4d5fd6c02adf98c776d9f71bfdd23a551d9fffd87202125c94ad4254`

## NAVSIM v2 NavHard

Two different complete-split artifacts are published. They share the unified
NAVTRAIN checkpoint but are intentionally identified as separate lineages.

### Paper-reported result1 (Table 5)

This is the exact result used for the MomWorld manuscript. Its values round to
the Stage-1, Stage-2, and combined entries shown in Table 5.

| Metric | Stage 1 | Stage 2 |
| --- | ---: | ---: |
| Stage score | 0.7337023328387648 | 0.5679657176434654 |
| No at-fault collision | 0.9688888888888889 | 0.8644121320298176 |
| Drivable-area compliance | 0.9355555555555556 | 0.8820404082251109 |
| Driving-direction compliance | 0.9977777777777778 | 0.9418986317698694 |
| Traffic-light compliance | 0.9977777777777778 | 0.9826115322131169 |
| Ego progress | 0.8036405366925962 | 0.8166788825021591 |
| Time to collision | 0.9688888888888889 | 0.8438997008913939 |
| Lane keeping | 0.9644444444444444 | 0.5526428180480907 |
| History comfort | 0.9755555555555555 | 0.9699806785118246 |
| Two-frame extended comfort | 0.6000000000000000 | 0.5465962769690780 |

- Combined: **0.4277559497825215**
- Scenarios: 5,912
- Formal CSV SHA-256:
  `b4c0a4baff0b4d1055b4be5a85a93b797ce31b4ada8beb260c24175df29fae05`
- Manuscript display: Stage-2 TTC 84.4, LK 55.3, EC 54.7, combined 42.8

### Post-paper optimized result

This later result is strictly better than the paper artifact. Starting from the
frozen result1 incumbent, it additionally applies the label-free fixed 10%
temporal-consistency policy. It was not substituted retroactively into the
manuscript's Table 5 row.

| Metric | Value |
| --- | ---: |
| Combined | **0.4281794504595703** |
| Stage 1 | 0.7337023328387648 |
| Stage 2 | 0.5682663154988540 |
| Stage-2 ego progress | 0.8172219891722398 |
| Stage-2 time to collision | 0.8414525443774834 |
| Stage-2 lane keeping | 0.5550281708986450 |
| Stage-2 two-frame extended comfort | 0.5562940122260941 |
| Scenarios | 5,912 |

- Formal CSV SHA-256:
  `719e3ec2186066c07089e0083359390d204b5ffea26a469465e0af18bbe51627`
- Frozen pairwise-gate SHA-256:
  `a5038efd255da7698d847c1b7d96ea50778a055ec52c2e5310076ff6ddd1df1e`
- Frozen selected-trajectory SHA-256:
  `74077c26b512f4c3b8e194872b458133057c1920ee0211910f39376d1a5dfcfb`
- Policy: pair consistency, fixed 10% fraction
- Strict validation: 5,912 unique real tokens, 450 Stage-1 and 5,462
  Stage-2 scenes, three summary rows, finite scores, exact token set, all rows
  valid, all checks passed

### Comparison

The post-paper optimized result improves combined by **0.0004235006770488**,
equivalent to approximately **0.0423501 points** on the 0--100 display scale.
Both results round to 42.8, but their CSV hashes and Stage-2 sub-scores show that
they are different formal evaluation artifacts.

## Interpretation

These numbers document one shared frozen checkpoint lineage and two distinct
NavHard result artifacts. Hashes are provided for provenance, not as a
substitute for the benchmark data licenses. The release makes no claim that
benchmark outputs can be reconstructed without the official NAVSIM datasets,
maps, caches, and compatible model artifacts.
