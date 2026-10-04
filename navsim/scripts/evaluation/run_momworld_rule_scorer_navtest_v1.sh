#!/bin/bash
set -euo pipefail

: "${NAVSIM_DEVKIT_ROOT:?Set NAVSIM_DEVKIT_ROOT to the UniLAW checkout}"
: "${NAVSIM_EXP_ROOT:?Set NAVSIM_EXP_ROOT to a writable experiment directory}"
: "${OPENSCENE_DATA_ROOT:?Set OPENSCENE_DATA_ROOT to the NAVSIM data root}"
: "${NUPLAN_MAPS_ROOT:?Set NUPLAN_MAPS_ROOT to the nuplan maps}"
: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to the rule-scorer checkpoint}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDS="${GPU_IDS:-0}"
TRAINER_DEVICES="${TRAINER_DEVICES:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SCORE_WORKERS="${SCORE_WORKERS:-8}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-${NAVSIM_EXP_ROOT}/navtest_v1_exact_metric_cache}"
V1_PROTOCOL_GATE_STATE_PATH="${V1_PROTOCOL_GATE_STATE_PATH:-}"
V1_PROTOCOL_GATE_STATE_SHA256="${V1_PROTOCOL_GATE_STATE_SHA256:-}"
v1_protocol_gate_enabled=false
if [[ -n "${V1_PROTOCOL_GATE_STATE_PATH}${V1_PROTOCOL_GATE_STATE_SHA256}" ]]; then
  if [[ ! -f "${V1_PROTOCOL_GATE_STATE_PATH}" || ! "${V1_PROTOCOL_GATE_STATE_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "V1 protocol gate requires an existing state path and lowercase SHA256" >&2
    exit 2
  fi
  actual_gate_sha256="$(sha256sum -- "${V1_PROTOCOL_GATE_STATE_PATH}" | awk '{print $1}')"
  if [[ "${actual_gate_sha256}" != "${V1_PROTOCOL_GATE_STATE_SHA256}" ]]; then
    echo "V1 protocol gate state SHA256 mismatch" >&2
    exit 2
  fi
  v1_protocol_gate_enabled=true
fi
BASE_EXPERIMENT_NAME="momworld_rule_scorer_navtest_v1"
DEFAULT_EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME}"
if [[ -n "${RULE_PROTOCOL_PROXY_WEIGHT:-}${RULE_LEARNED_ZSCORE_WEIGHT:-}${RULE_SCORE_ZSCORE_EPSILON:-}${V1_PROTOCOL_GATE_STATE_PATH}" ]]; then
  RANK_FUSION_MODE="${RANK_FUSION_MODE:-proxy_only}"
  if [[ "${RANK_FUSION_MODE}" != "hybrid" && "${RANK_FUSION_MODE}" != "proxy_only" ]]; then
    echo "RANK_FUSION_MODE must be hybrid or proxy_only" >&2
    exit 2
  fi
  : "${RULE_LEARNED_SCORE_WEIGHT:?Set all eight NAVTRAIN rank-fusion parameters}"
  : "${RULE_COLLISION_WEIGHT:?Set all eight NAVTRAIN rank-fusion parameters}"
  : "${RULE_KINEMATIC_WEIGHT:?Set all eight NAVTRAIN rank-fusion parameters}"
  : "${RULE_MOMENTUM_WEIGHT:?Set all eight NAVTRAIN rank-fusion parameters}"
  : "${RULE_COLLISION_THRESHOLD:?Set all eight NAVTRAIN rank-fusion parameters}"
  : "${RULE_PROTOCOL_PROXY_WEIGHT:?Set both NAVTRAIN rank-fusion weights and epsilon}"
  : "${RULE_LEARNED_ZSCORE_WEIGHT:?Set both NAVTRAIN rank-fusion weights and epsilon}"
  : "${RULE_SCORE_ZSCORE_EPSILON:?Set both NAVTRAIN rank-fusion weights and epsilon}"
  zero_pattern='^[-+]?(0+([.]0*)?|[.]0+)([eE][-+]?0+)?$'
  if [[ \
    ! "${RULE_LEARNED_SCORE_WEIGHT}" =~ ${zero_pattern} \
    || ! "${RULE_COLLISION_WEIGHT}" =~ ${zero_pattern} \
    || ! "${RULE_KINEMATIC_WEIGHT}" =~ ${zero_pattern} \
    || ! "${RULE_MOMENTUM_WEIGHT}" =~ ${zero_pattern} \
    || "${RULE_COLLISION_THRESHOLD}" != "1.000001" \
  ]]; then
    echo "Rank-fusion candidate requires four zero legacy weights and threshold 1.000001" >&2
    exit 2
  fi
  if [[ \
    "${RANK_FUSION_MODE}" == "proxy_only" \
    && ! "${RULE_LEARNED_ZSCORE_WEIGHT}" =~ ${zero_pattern} \
  ]]; then
    echo "proxy_only requires RULE_LEARNED_ZSCORE_WEIGHT=0" >&2
    exit 2
  fi
  rank_fusion_source_status="$(git -C "${NAVSIM_DEVKIT_ROOT}" status --porcelain -- \
    navsim/agents/momworld/momworld_config.py \
    navsim/agents/momworld/momworld_model.py \
    scripts/training/train_momworld_v1_protocol_gate_all_navtrain.py \
    scripts/evaluation/prepare_momworld_rank_fusion_manifest.py \
    scripts/evaluation/run_momworld_rule_scorer_navtest_v1.sh)"
  if [[ -n "${rank_fusion_source_status}" ]]; then
    echo "Refusing rank fusion from dirty evaluation source files" >&2
    exit 2
  fi
  rank_fusion_source_commit="$(git -C "${NAVSIM_DEVKIT_ROOT}" rev-parse HEAD)"
  rank_fusion_checkpoint_sha256="$(sha256sum -- "${CHECKPOINT_PATH}" | awk '{print $1}')"
  if [[ ! "${rank_fusion_checkpoint_sha256}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "Unable to compute checkpoint SHA256: ${CHECKPOINT_PATH}" >&2
    exit 2
  fi
  rank_fusion_signature="$({
    printf '%s\n' \
      "${rank_fusion_source_commit}" \
      "${rank_fusion_checkpoint_sha256}" \
      "${RANK_FUSION_MODE}" \
      "${RULE_LEARNED_SCORE_WEIGHT:-}" \
      "${RULE_COLLISION_WEIGHT:-}" \
      "${RULE_KINEMATIC_WEIGHT:-}" \
      "${RULE_MOMENTUM_WEIGHT:-}" \
      "${RULE_COLLISION_THRESHOLD:-}" \
      "${RULE_PROTOCOL_PROXY_WEIGHT:-}" \
      "${RULE_LEARNED_ZSCORE_WEIGHT:-}" \
      "${RULE_SCORE_ZSCORE_EPSILON:-}"
  } | git hash-object --stdin)"
  rank_fusion_signature="${rank_fusion_signature:0:12}"
  if [[ "${v1_protocol_gate_enabled}" == true ]]; then
    v1_protocol_gate_signature="$({
      printf '%s\n' \
        "${rank_fusion_source_commit}" \
        "${rank_fusion_signature}" \
        "${V1_PROTOCOL_GATE_STATE_SHA256}"
    } | git hash-object --stdin)"
    v1_protocol_gate_signature="${v1_protocol_gate_signature:0:12}"
    DEFAULT_EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME}_protocol_gate_${v1_protocol_gate_signature}"
  else
    DEFAULT_EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME}_protocol_proxy_${rank_fusion_signature}"
  fi
fi
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${DEFAULT_EXPERIMENT_NAME}}"
if [[ "${EXPERIMENT_NAME}" == */* || "${EXPERIMENT_NAME}" == "." || "${EXPERIMENT_NAME}" == ".." ]]; then
  echo "EXPERIMENT_NAME must be one directory name without path traversal" >&2
  exit 2
fi
if [[ \
  -n "${RULE_PROTOCOL_PROXY_WEIGHT:-}${RULE_LEARNED_ZSCORE_WEIGHT:-}${RULE_SCORE_ZSCORE_EPSILON:-}" \
  && "${EXPERIMENT_NAME}" == "${BASE_EXPERIMENT_NAME}" \
]]; then
  echo "Refusing to reuse the first-round experiment name for rank fusion" >&2
  exit 2
fi
TRAJECTORY_PATH="${TRAJECTORY_PATH:-${NAVSIM_EXP_ROOT}/${EXPERIMENT_NAME}/trajectories.pkl}"
checkpoint_name=$(basename "${CHECKPOINT_PATH}" .ckpt)
checkpoint_signature=$(stat -c '%s-%Y' "${CHECKPOINT_PATH}")
if [[ -n "${rank_fusion_signature:-}" ]]; then
  checkpoint_signature="${checkpoint_signature}-${rank_fusion_checkpoint_sha256:0:12}-${rank_fusion_signature}"
fi
if [[ -n "${v1_protocol_gate_signature:-}" ]]; then
  checkpoint_signature="${checkpoint_signature}-${v1_protocol_gate_signature}"
fi
CHECKPOINT_CACHE_PATH="${NAVSIM_EXP_ROOT}/${EXPERIMENT_NAME}/checkpoint_cache/${checkpoint_name}-${checkpoint_signature}"
PREDICTION_CACHE_PATH="${PREDICTION_CACHE_PATH:-${CHECKPOINT_CACHE_PATH}/trajectories}"
SCORE_CACHE_PATH="${SCORE_CACHE_PATH:-${CHECKPOINT_CACHE_PATH}/scores}"
export TRAJECTORY_PATH
export PREDICTION_CACHE_PATH
export SCORE_CACHE_PATH
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}/.deps:${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}"

RULE_OVERRIDE_ARGS=()
if [[ -n "${RULE_SCORER_INPUT_MODE:-}" ]]; then
  if [[ "${RULE_SCORER_INPUT_MODE}" != "raw" && "${RULE_SCORER_INPUT_MODE}" != "raw_plus_scene_zscore" ]]; then
    echo "RULE_SCORER_INPUT_MODE must be raw or raw_plus_scene_zscore" >&2
    exit 2
  fi
  RULE_OVERRIDE_ARGS+=(
    "++agent.config.rule_scorer_input_mode=${RULE_SCORER_INPUT_MODE}"
  )
fi
if [[ -n "${RULE_LEARNED_SCORE_WEIGHT:-}${RULE_COLLISION_WEIGHT:-}${RULE_KINEMATIC_WEIGHT:-}${RULE_MOMENTUM_WEIGHT:-}${RULE_COLLISION_THRESHOLD:-}" ]]; then
  : "${RULE_LEARNED_SCORE_WEIGHT:?Set all five tuned rule parameters}"
  : "${RULE_COLLISION_WEIGHT:?Set all five tuned rule parameters}"
  : "${RULE_KINEMATIC_WEIGHT:?Set all five tuned rule parameters}"
  : "${RULE_MOMENTUM_WEIGHT:?Set all five tuned rule parameters}"
  : "${RULE_COLLISION_THRESHOLD:?Set all five tuned rule parameters}"
  RULE_OVERRIDE_ARGS+=(
    "++agent.config.rule_learned_score_weight=${RULE_LEARNED_SCORE_WEIGHT}"
    "++agent.config.rule_collision_weight=${RULE_COLLISION_WEIGHT}"
    "++agent.config.rule_kinematic_weight=${RULE_KINEMATIC_WEIGHT}"
    "++agent.config.rule_momentum_weight=${RULE_MOMENTUM_WEIGHT}"
    "++agent.config.rule_collision_filter_threshold=${RULE_COLLISION_THRESHOLD}"
  )
fi
if [[ -n "${RULE_PROTOCOL_PROXY_WEIGHT:-}${RULE_LEARNED_ZSCORE_WEIGHT:-}${RULE_SCORE_ZSCORE_EPSILON:-}" ]]; then
  RULE_OVERRIDE_ARGS+=(
    "++agent.config.rule_rank_fusion_mode=${RANK_FUSION_MODE}"
    "++agent.config.rule_protocol_proxy_weight=${RULE_PROTOCOL_PROXY_WEIGHT}"
    "++agent.config.rule_learned_zscore_weight=${RULE_LEARNED_ZSCORE_WEIGHT}"
    "++agent.config.rule_score_zscore_epsilon=${RULE_SCORE_ZSCORE_EPSILON}"
  )
  rank_fusion_experiment_dir="${NAVSIM_EXP_ROOT}/${EXPERIMENT_NAME}"
  rank_fusion_manifest_path="${rank_fusion_experiment_dir}/rank_fusion_run_manifest.json"
  "${PYTHON_BIN}" \
    "${NAVSIM_DEVKIT_ROOT}/scripts/evaluation/prepare_momworld_rank_fusion_manifest.py" \
    --manifest "${rank_fusion_manifest_path}" \
    --experiment-dir "${rank_fusion_experiment_dir}" \
    --launcher run_momworld_rule_scorer_navtest_v1.sh \
    --experiment-name "${EXPERIMENT_NAME}" \
    --train-test-split navtest \
    --protocol v1 \
    --source-commit "${rank_fusion_source_commit}" \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --checkpoint-sha256 "${rank_fusion_checkpoint_sha256}" \
    --rank-fusion-signature "${rank_fusion_signature}" \
    --rank-fusion-mode "${RANK_FUSION_MODE}" \
    --trajectory-path "${TRAJECTORY_PATH}" \
    --prediction-cache-path "${PREDICTION_CACHE_PATH}" \
    --score-cache-path "${SCORE_CACHE_PATH}" \
    --metric-cache-path "${METRIC_CACHE_PATH}" \
    --rule-learned-score-weight "${RULE_LEARNED_SCORE_WEIGHT}" \
    --rule-collision-weight "${RULE_COLLISION_WEIGHT}" \
    --rule-kinematic-weight "${RULE_KINEMATIC_WEIGHT}" \
    --rule-momentum-weight "${RULE_MOMENTUM_WEIGHT}" \
    --rule-collision-threshold "${RULE_COLLISION_THRESHOLD}" \
    --rule-protocol-proxy-weight "${RULE_PROTOCOL_PROXY_WEIGHT}" \
    --rule-learned-zscore-weight "${RULE_LEARNED_ZSCORE_WEIGHT}" \
    --rule-score-zscore-epsilon "${RULE_SCORE_ZSCORE_EPSILON}"
fi
if [[ "${v1_protocol_gate_enabled}" == true ]]; then
  RULE_OVERRIDE_ARGS+=(
    "++agent.config.rule_v1_protocol_gate_enabled=true"
    "++agent.config.rule_v1_protocol_gate_state_path='${V1_PROTOCOL_GATE_STATE_PATH}'"
    "++agent.config.rule_v1_protocol_gate_state_sha256=${V1_PROTOCOL_GATE_STATE_SHA256}"
  )
fi

CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON_BIN}" \
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_gpu_v1.py" \
  train_test_split=navtest \
  agent=momworld_rule_scorer_vov_eval \
  agent.config.rule_protocol=v1 \
  agent.checkpoint_path="'${CHECKPOINT_PATH}'" \
  experiment_name="${EXPERIMENT_NAME}" \
  metric_cache_path="${METRIC_CACHE_PATH}" \
  traffic_agents=non_reactive \
  scorer.config.human_penalty_filter=false \
  worker=single_machine_thread_pool \
  worker.max_workers="${SCORE_WORKERS}" \
  dataloader.params.batch_size="${BATCH_SIZE}" \
  dataloader.params.num_workers="${NUM_WORKERS}" \
  +trainer.params.devices="${TRAINER_DEVICES}" \
  trainer.params.precision=16-mixed \
  "${RULE_OVERRIDE_ARGS[@]}"
