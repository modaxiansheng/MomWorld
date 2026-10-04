#!/bin/bash
set -euo pipefail

: "${NAVSIM_DEVKIT_ROOT:?Set NAVSIM_DEVKIT_ROOT}"
: "${NAVSIM_EXP_ROOT:?Set NAVSIM_EXP_ROOT}"
: "${OPENSCENE_DATA_ROOT:?Set OPENSCENE_DATA_ROOT}"
: "${NUPLAN_MAPS_ROOT:?Set NUPLAN_MAPS_ROOT}"
: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH}"
: "${V2_NAVTEST_GATE_STATE_PATH:?Set V2_NAVTEST_GATE_STATE_PATH}"
: "${V2_NAVTEST_GATE_STATE_SHA256:?Set V2_NAVTEST_GATE_STATE_SHA256}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDS="${GPU_IDS:-0}"
TRAINER_DEVICES="${TRAINER_DEVICES:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SCORE_WORKERS="${SCORE_WORKERS:-8}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-${NAVSIM_EXP_ROOT}/navtest_two_stage_metric_cache}"

if [[ ! -f "${CHECKPOINT_PATH}" || ! -f "${V2_NAVTEST_GATE_STATE_PATH}" ]]; then
  echo "Checkpoint and V2 gate state must exist" >&2
  exit 2
fi
if [[ ! "${V2_NAVTEST_GATE_STATE_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "V2 gate state SHA256 must be lowercase hexadecimal" >&2
  exit 2
fi
actual_gate_sha256="$(sha256sum -- "${V2_NAVTEST_GATE_STATE_PATH}" | awk '{print $1}')"
if [[ "${actual_gate_sha256}" != "${V2_NAVTEST_GATE_STATE_SHA256}" ]]; then
  echo "V2 NavTest selector state SHA256 mismatch" >&2
  exit 2
fi

source_files=(
  navsim/agents/momworld/momworld_config.py
  navsim/agents/momworld/momworld_model.py
  scripts/training/train_momworld_v2_navtest_selector_gate_all_navtrain.py
  scripts/evaluation/run_momworld_v2_navtest_selector_gate.sh
)
source_status="$(git -C "${NAVSIM_DEVKIT_ROOT}" status --porcelain -- "${source_files[@]}")"
if [[ -n "${source_status}" ]]; then
  echo "Refusing formal evaluation with dirty selector sources" >&2
  printf '%s\n' "${source_status}" >&2
  exit 2
fi
source_commit="$(git -C "${NAVSIM_DEVKIT_ROOT}" rev-parse HEAD)"
checkpoint_sha256="$(sha256sum -- "${CHECKPOINT_PATH}" | awk '{print $1}')"
signature="$(printf '%s\n' \
  "${source_commit}" \
  "${checkpoint_sha256}" \
  "${V2_NAVTEST_GATE_STATE_SHA256}" \
  v2 navtest_two_stage hybrid 1.0 2.0 0.0 0.1 0.0 0.0 0.55 0.25 \
  | git hash-object --stdin)"
signature="${signature:0:12}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-momworld_v2_navtest_selector_gate_${signature}}"
if [[ "${EXPERIMENT_NAME}" == */* || "${EXPERIMENT_NAME}" == "." || "${EXPERIMENT_NAME}" == ".." ]]; then
  echo "EXPERIMENT_NAME must be one directory name" >&2
  exit 2
fi

experiment_dir="${NAVSIM_EXP_ROOT}/${EXPERIMENT_NAME}"
TRAJECTORY_PATH="${TRAJECTORY_PATH:-${experiment_dir}/trajectories.pkl}"
checkpoint_name="$(basename "${CHECKPOINT_PATH}" .ckpt)"
checkpoint_cache="${experiment_dir}/checkpoint_cache/${checkpoint_name}-${signature}"
PREDICTION_CACHE_PATH="${PREDICTION_CACHE_PATH:-${checkpoint_cache}/trajectories}"
SCORE_CACHE_PATH="${SCORE_CACHE_PATH:-${checkpoint_cache}/scores}"
mkdir -p "${experiment_dir}"
export TRAJECTORY_PATH PREDICTION_CACHE_PATH SCORE_CACHE_PATH
export SUBSCORE_PATH="${TRAJECTORY_PATH}"
export PROGRESS_MODE=eval
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}/.deps:${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}"

manifest="${experiment_dir}/selector_gate_run_manifest.txt"
if [[ ! -e "${manifest}" ]]; then
  umask 002
  {
    printf 'schema=momworld-v2-navtest-selector-gate-formal-v1\n'
    printf 'source_commit=%s\n' "${source_commit}"
    printf 'checkpoint_path=%s\n' "${CHECKPOINT_PATH}"
    printf 'checkpoint_sha256=%s\n' "${checkpoint_sha256}"
    printf 'gate_state_path=%s\n' "${V2_NAVTEST_GATE_STATE_PATH}"
    printf 'gate_state_sha256=%s\n' "${V2_NAVTEST_GATE_STATE_SHA256}"
    printf 'signature=%s\n' "${signature}"
    printf 'split=navtest_two_stage\nprotocol=v2\n'
  } > "${manifest}"
fi

CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON_BIN}" \
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_gpu_v2.py" \
  train_test_split=navtest_two_stage \
  agent=momworld_rule_scorer_vov_eval \
  agent.config.rule_protocol=v2 \
  agent.checkpoint_path="'${CHECKPOINT_PATH}'" \
  +combined_inference=false \
  +cache_path=null \
  experiment_name="${EXPERIMENT_NAME}" \
  metric_cache_path="${METRIC_CACHE_PATH}" \
  worker=single_machine_thread_pool \
  worker.use_process_pool=true \
  worker.max_workers="${SCORE_WORKERS}" \
  dataloader.params.batch_size="${BATCH_SIZE}" \
  dataloader.params.num_workers="${NUM_WORKERS}" \
  +trainer.params.devices="${TRAINER_DEVICES}" \
  trainer.params.precision=16-mixed \
  ++agent.config.rule_learned_score_weight=1.0 \
  ++agent.config.rule_collision_weight=2.0 \
  ++agent.config.rule_kinematic_weight=0.0 \
  ++agent.config.rule_momentum_weight=0.1 \
  ++agent.config.rule_collision_filter_threshold=1.000001 \
  ++agent.config.rule_rank_fusion_mode=hybrid \
  ++agent.config.rule_protocol_proxy_weight=0.0 \
  ++agent.config.rule_learned_zscore_weight=0.0 \
  ++agent.config.rule_score_zscore_epsilon=0.0001 \
  ++agent.config.rule_safety_filter_enabled=true \
  ++agent.config.rule_safety_ttc_min=0.55 \
  ++agent.config.rule_safety_lane_min=0.25 \
  ++agent.config.rule_relative_safety_filter_enabled=false \
  ++agent.config.rule_monotonic_residual_enabled=false \
  ++agent.config.rule_collision_calibrator_enabled=false \
  ++agent.config.context_ranker_enabled=false \
  ++agent.config.rule_v2_navtest_selector_gate_enabled=true \
  ++agent.config.rule_v2_navtest_selector_gate_state_path="'${V2_NAVTEST_GATE_STATE_PATH}'" \
  ++agent.config.rule_v2_navtest_selector_gate_state_sha256="${V2_NAVTEST_GATE_STATE_SHA256}"
