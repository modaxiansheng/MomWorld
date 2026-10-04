"""Multi-GPU trajectory inference and classic NAVSIM v1 PDMS evaluation."""

import logging
import os
import pickle
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_utils import worker_map
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SensorConfig, Trajectory
from navsim.common.dataloader import MetricCacheLoader, SceneFilter, SceneLoader
from navsim.evaluate.pdm_score_v1 import PDMScorerV1, pdm_score_v1
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.training.dataset import Dataset
from navsim.planning.script.momworld_prediction_shards import (
    load_prediction_file,
    shard_configuration_from_environment,
    shard_tokens,
    write_prediction_shard,
)

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score_gpu_v1"
SCORE_CACHE_VERSION = "navsim-v1.1-exact-3e8291b"


class TrajectoryPredictionModule(pl.LightningModule):
    """Prediction-only wrapper that keeps just one compact trajectory/token."""

    def __init__(
        self, agent: AbstractAgent, prediction_cache_path: Optional[Path] = None
    ):
        super().__init__()
        self.agent = agent
        self.prediction_cache_path = prediction_cache_path

    def predict_step(self, batch, batch_idx: int):
        del batch_idx
        features, _, tokens = batch
        predictions = self.agent.forward(features)
        poses = predictions["trajectory"].detach().cpu().numpy()
        result: Dict[str, Trajectory] = {}
        for token, pose in zip(tokens, poses):
            trajectory = Trajectory(pose, self.agent._trajectory_sampling)
            if self.prediction_cache_path is not None:
                _save_cached_trajectory(
                    self.prediction_cache_path, token, trajectory
                )
            else:
                result[token] = trajectory
        return result


def _prediction_cache_file(cache_path: Path, token: str) -> Path:
    """Return a traversal-safe per-token cache path."""
    if not token or Path(token).name != token or token in {".", ".."}:
        raise ValueError(f"Invalid prediction token: {token!r}")
    return cache_path / f"{token}.pkl"


def _atomic_pickle_dump(value, output_path: Path) -> None:
    """Persist a pickle atomically so interrupted writes are never reused."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.tmp"
    )
    try:
        with temporary_path.open("wb") as file:
            pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _save_cached_trajectory(
    cache_path: Path, token: str, trajectory: Trajectory
) -> None:
    _atomic_pickle_dump(trajectory, _prediction_cache_file(cache_path, token))


def _load_cached_trajectories(
    cache_path: Path, expected_tokens: Iterable[str]
) -> Dict[str, Trajectory]:
    """Load valid per-token trajectories, ignoring incomplete/corrupt files."""
    trajectories: Dict[str, Trajectory] = {}
    for token in expected_tokens:
        cache_file = _prediction_cache_file(cache_path, token)
        if not cache_file.is_file():
            continue
        try:
            with cache_file.open("rb") as file:
                trajectory = pickle.load(file)
            if not isinstance(trajectory, Trajectory):
                raise TypeError(
                    f"Expected Trajectory, got {type(trajectory).__name__}"
                )
            trajectories[token] = trajectory
        except Exception as error:
            logger.warning(
                "Ignoring invalid prediction cache %s: %s", cache_file, error
            )
    return trajectories


def _score_cache_file(cache_path: Path, token: str) -> Path:
    """Return a traversal-safe per-token score cache path."""
    if not token or Path(token).name != token or token in {".", ".."}:
        raise ValueError(f"Invalid score token: {token!r}")
    return cache_path / f"{token}.pkl"


def _save_cached_score(
    cache_path: Path, token: str, score_row: Dict[str, Union[str, bool, float]]
) -> None:
    payload = {"version": SCORE_CACHE_VERSION, "row": score_row}
    _atomic_pickle_dump(payload, _score_cache_file(cache_path, token))


def _load_cached_score(
    cache_path: Path, token: str
) -> Optional[Dict[str, Union[str, bool, float]]]:
    cache_file = _score_cache_file(cache_path, token)
    if not cache_file.is_file():
        return None
    try:
        with cache_file.open("rb") as file:
            payload = pickle.load(file)
        if not isinstance(payload, dict) or payload.get("version") != SCORE_CACHE_VERSION:
            raise ValueError("score cache version mismatch")
        score_row = payload.get("row")
        if not isinstance(score_row, dict) or score_row.get("token") != token:
            raise ValueError("invalid score cache row")
        if score_row.get("valid") is not True:
            raise ValueError("only successful score rows may be reused")
        return score_row
    except Exception as error:
        logger.warning("Ignoring invalid score cache %s: %s", cache_file, error)
        return None


def _score_precomputed_trajectories(
    args: List[Dict[str, Union[List[str], DictConfig, Dict[str, Trajectory]]]],
) -> List[Dict[str, Union[str, bool, float]]]:
    """Score a subset of already inferred trajectories on CPU."""
    log_names = [arg["log_file"] for arg in args]
    tokens = [token for arg in args for token in arg["tokens"]]
    cfg: DictConfig = args[0]["cfg"]
    trajectories: Dict[str, Trajectory] = args[0]["trajectories"]
    score_cache_env = os.environ.get("SCORE_CACHE_PATH")
    score_cache_path = Path(score_cache_env) if score_cache_env else None

    simulator: PDMSimulator = instantiate(cfg.simulator)
    configured_scorer: PDMScorer = instantiate(cfg.scorer)
    if simulator.proposal_sampling != configured_scorer.proposal_sampling:
        raise ValueError("Simulator and scorer sampling must match")
    scorer = PDMScorerV1(
        proposal_sampling=configured_scorer.proposal_sampling,
        config=configured_scorer._config,
    )

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        original_sensor_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )

    rows: List[Dict[str, Union[str, bool, float]]] = []
    tokens_to_score = sorted(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    for token in tokens_to_score:
        if score_cache_path is not None:
            cached_score = _load_cached_score(score_cache_path, token)
            if cached_score is not None:
                rows.append(cached_score)
                continue
        score_row: Dict[str, Union[str, bool, float]] = {"token": token, "valid": True}
        try:
            metric_cache = metric_cache_loader.get_from_token(token)
            score_row.update(
                pdm_score_v1(
                    metric_cache=metric_cache,
                    model_trajectory=trajectories[token],
                    future_sampling=simulator.proposal_sampling,
                    simulator=simulator,
                    scorer=scorer,
                )
            )
        except Exception:
            logger.warning("Agent or scorer failed for token %s", token)
            traceback.print_exc()
            score_row["valid"] = False
        if score_cache_path is not None and score_row["valid"] is True:
            _save_cached_score(score_cache_path, token, score_row)
        rows.append(score_row)
    return rows


def _merge_prediction_shards(shards) -> Dict[str, Trajectory]:
    merged: Dict[str, Trajectory] = {}
    for rank_predictions in shards:
        if rank_predictions is None:
            continue
        for batch_predictions in rank_predictions:
            overlap = set(merged) & set(batch_predictions)
            if overlap:
                raise RuntimeError(f"Duplicate prediction tokens: {sorted(overlap)[:3]}")
            merged.update(batch_predictions)
    return merged


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    build_logger(cfg)
    full_scene_filter = instantiate(cfg.train_test_split.scene_filter)
    full_scene_loader = SceneLoader(
        original_sensor_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=full_scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    full_tokens = set(full_scene_loader.tokens)
    shard_configuration = shard_configuration_from_environment()
    score_only = os.environ.get("MOMWORLD_SCORE_ONLY", "").lower() == "true"
    if score_only and shard_configuration is not None:
        raise ValueError("MOMWORLD_SCORE_ONLY and prediction sharding are mutually exclusive")
    all_tokens = (
        set(
            shard_tokens(
                full_tokens,
                int(shard_configuration["shard_id"]),
                int(shard_configuration["num_shards"]),
            )
        )
        if shard_configuration is not None
        else full_tokens
    )
    prediction_cache_env = os.environ.get("PREDICTION_CACHE_PATH")
    prediction_cache_path = (
        Path(prediction_cache_env) if prediction_cache_env else None
    )
    if score_only:
        merged_path_env = os.environ.get("MOMWORLD_MERGED_TRAJECTORY_PATH")
        if not merged_path_env:
            raise ValueError("MOMWORLD_SCORE_ONLY requires MOMWORLD_MERGED_TRAJECTORY_PATH")
        trajectories = load_prediction_file(Path(merged_path_env), "v1")
        if set(trajectories) != full_tokens:
            raise RuntimeError("Score-only NAVSIM v1 trajectory coverage is not full split")
    else:
        memory_fraction = os.environ.get("GPU_MEMORY_FRACTION")
        if memory_fraction:
            fraction = float(memory_fraction)
            if not 0.0 < fraction <= 1.0:
                raise ValueError("GPU_MEMORY_FRACTION must be in (0, 1]")
            torch.cuda.set_per_process_memory_fraction(fraction)
            logger.info("Capped this process to %.1f%% of one GPU", 100.0 * fraction)
        agent: AbstractAgent = instantiate(cfg.agent)
        agent.initialize()
        trajectories = (
            _load_cached_trajectories(prediction_cache_path, all_tokens)
            if prediction_cache_path is not None
            else {}
        )
    missing_prediction_tokens = sorted(all_tokens - set(trajectories))
    logger.info(
        "Prediction resume state: total=%d, cached=%d, missing=%d",
        len(all_tokens),
        len(trajectories),
        len(missing_prediction_tokens),
    )

    local_predictions = []
    if not score_only and missing_prediction_tokens:
        inference_scene_filter = instantiate(cfg.train_test_split.scene_filter)
        inference_scene_filter.tokens = missing_prediction_tokens
        inference_loader = SceneLoader(
            original_sensor_path=Path(cfg.original_sensor_path),
            data_path=Path(cfg.navsim_log_path),
            scene_filter=inference_scene_filter,
            sensor_config=agent.get_sensor_config(),
        )
        dataset = Dataset(
            scene_loader=inference_loader,
            feature_builders=agent.get_feature_builders(),
            target_builders=[],
            cache_path=None,
            force_cache_computation=False,
            append_token_to_batch=True,
            agent_input_only=True,
            is_training=False,
        )
        dataloader = DataLoader(dataset, **cfg.dataloader.params, shuffle=False)

        trainer = pl.Trainer(**cfg.trainer.params)
        local_predictions = trainer.predict(
            TrajectoryPredictionModule(agent, prediction_cache_path),
            dataloader,
            return_predictions=prediction_cache_path is None,
        )

    if not score_only and dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        if prediction_cache_path is not None:
            dist.barrier()
            gathered_predictions = []
        else:
            gathered_predictions = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered_predictions, local_predictions)
    elif not score_only:
        rank = 0
        gathered_predictions = [local_predictions]
    else:
        rank = 0
        gathered_predictions = []

    if rank != 0:
        return

    if score_only:
        pass
    elif prediction_cache_path is not None:
        trajectories = _load_cached_trajectories(
            prediction_cache_path, all_tokens
        )
    else:
        trajectories = _merge_prediction_shards(gathered_predictions)
    missing_prediction_tokens = all_tokens - set(trajectories)
    if missing_prediction_tokens:
        raise RuntimeError(
            f"Missing {len(missing_prediction_tokens)} trajectory predictions; "
            f"first tokens: {sorted(missing_prediction_tokens)[:10]}"
        )

    if shard_configuration is not None:
        write_prediction_shard(
            trajectories,
            full_tokens,
            "v1",
            str(shard_configuration["split"]),
            shard_configuration,
        )
        logger.info(
            "Completed prediction-only shard %d/%d with %d unique tokens",
            int(shard_configuration["shard_id"]),
            int(shard_configuration["num_shards"]),
            len(trajectories),
        )
        return

    trajectory_path = os.environ.get("TRAJECTORY_PATH")
    if trajectory_path and not score_only:
        trajectory_file = Path(trajectory_path)
        _atomic_pickle_dump({"predictions": [trajectories]}, trajectory_file)
        logger.info("Saved %d trajectories to %s", len(trajectories), trajectory_file)

    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    scene_loader = SceneLoader(
        original_sensor_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    expected_tokens = set(scene_loader.tokens) & set(metric_cache_loader.tokens)
    logger.info(
        "Starting classic NAVSIM v1 scoring for %d scenarios (%d predictions)",
        len(expected_tokens),
        len(trajectories),
    )
    score_cache_env = os.environ.get("SCORE_CACHE_PATH")
    if score_cache_env:
        score_cache_path = Path(score_cache_env)
        cached_score_count = sum(
            _load_cached_score(score_cache_path, token) is not None
            for token in expected_tokens
        )
        logger.info(
            "Score resume state: total=%d, cached=%d, missing=%d",
            len(expected_tokens),
            cached_score_count,
            len(expected_tokens) - cached_score_count,
        )

    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": token_list,
            "trajectories": trajectories,
        }
        for log_file, token_list in scene_loader.get_tokens_list_per_log().items()
    ]
    worker = build_worker(cfg)
    worker_rows = worker_map(worker, _score_precomputed_trajectories, data_points)
    rows = []
    for item in worker_rows:
        if isinstance(item, list):
            rows.extend(item)
        else:
            rows.append(item)
    score_df = pd.DataFrame(rows)
    if score_df.empty:
        raise RuntimeError("NAVSIM v1 evaluation produced no score rows")

    missing_rows = expected_tokens - set(score_df["token"])
    for token in sorted(missing_rows):
        score_df.loc[len(score_df)] = {"token": token, "valid": False}

    valid_df = score_df[score_df["valid"] == True]  # noqa: E712
    metric_columns = [
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "ego_progress",
        "time_to_collision_within_bound",
        "comfort",
        "score",
    ]
    average = valid_df[metric_columns].mean(skipna=True)
    average["token"] = "average"
    average["valid"] = len(valid_df) == len(expected_tokens)
    score_df = pd.concat([score_df, pd.DataFrame([average])], ignore_index=True)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{datetime.now().strftime('%Y.%m.%d.%H.%M.%S')}.csv"
    score_df.to_csv(result_path, index=False)

    num_failed = len(expected_tokens) - len(valid_df)
    logger.info(
        "Finished NAVSIM v1 evaluation: valid=%d, failed=%d, PDMS=%.6f, csv=%s",
        len(valid_df),
        num_failed,
        float(average["score"]),
        result_path,
    )
    if num_failed:
        failed_tokens = score_df[score_df["valid"] == False]["token"].tolist()  # noqa: E712
        raise RuntimeError(
            f"NAVSIM v1 evaluation failed for {num_failed} scenarios; "
            f"first tokens: {failed_tokens[:10]}"
        )


if __name__ == "__main__":
    main()
