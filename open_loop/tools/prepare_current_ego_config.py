"""Prepare a NEW ego-conditioned config and strictly migrated warm-start weights.

Does not train, use CUDA, modify data/evaluators, or resume an old optimizer.
Before a formal experiment, audit CAN coverage and timestamp alignment: the
legacy converter uses nearest messages and silently fills missing CAN with 0.
"""

import argparse
import copy
import hashlib
import inspect
import ntpath
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import posixpath
import sys


HEAD_TYPE = "MotionPlanningHead_MomAD_World_model_6s_V2"
SUPPORTED_HEAD_MODULE = (
    "projects.mmdet3d_plugin.models.motion."
    "motion_planning_head_MomAD_World_model_6s_v2_oracle_reg_backup_20260628"
)
UNSUPPORTED_HEAD_MODULE = (
    "projects.mmdet3d_plugin.models.motion.motion_planning_head_MomAD_World_model_6s_v2"
)


def _resolved_run_path(value, path_base=None, *, absolute=False):
    """Resolve config paths without interpreting a POSIX path as a Windows drive.

    Relative paths use the training invocation's cwd, or an explicit path_base
    for configurations intended for another OS. Native paths also resolve
    existing symlinks/junctions; foreign paths can only be checked lexically.
    """
    if not value:
        raise ValueError("An explicit archived/new work_dir is required")
    text = os.fspath(value)
    windows = PureWindowsPath(text)
    if windows.drive:
        if not windows.is_absolute():
            raise ValueError("Drive-relative paths are ambiguous")
        result = windows
    elif text.startswith("/"):
        if "\\" in text:
            raise ValueError("Do not mix Windows separators into POSIX paths")
        result = PurePosixPath(text)
    else:
        if text.startswith("\\"):
            raise ValueError("Windows rooted paths need an explicit drive")
        if absolute:
            raise ValueError("Use an absolute, non-traversing new work_dir")
        base = _resolved_run_path(path_base if path_base is not None else str(Path.cwd()), absolute=True)
        if isinstance(base, PurePosixPath) and "\\" in text:
            raise ValueError("Do not mix Windows separators into POSIX paths")
        result = base / text
    if absolute and ".." in result.parts:
        raise ValueError("Use an absolute, non-traversing new work_dir")
    is_windows = isinstance(result, PureWindowsPath)
    normalizer = ntpath.normpath if is_windows else posixpath.normpath
    result = type(result)(normalizer(str(result)))
    if is_windows == (os.name == "nt"):
        # Resolve aliases even when the final output does not yet exist.
        result = type(result)(str(Path(str(result)).resolve()))
    return result


def _protected_run(base, path_base=None):
    old = _resolved_run_path(base.get("work_dir"), path_base)
    leaf = old.name.casefold() if isinstance(old, PureWindowsPath) else old.name
    return old.parent if leaf == "training" else old


def _require_outside(target, protected):
    if type(target) is not type(protected):
        raise ValueError("Cannot compare work_dir namespaces; specify a matching path_base")
    if target == protected or protected in target.parents:
        raise ValueError("Do not write into the archived run or its descendants")


def _validate_head_imports(cfg):
    custom = cfg.get("custom_imports")
    if not isinstance(custom, dict) or custom.get("allow_failed_imports", False) is not False:
        raise ValueError("Require explicit, non-optional custom_imports for the supported head")
    imports = custom.get("imports")
    imports = [imports] if isinstance(imports, str) else imports
    if not isinstance(imports, (list, tuple)) or not all(isinstance(x, str) for x in imports):
        raise ValueError("custom_imports must name the supported head module")
    if imports.count(SUPPORTED_HEAD_MODULE) != 1 or UNSUPPORTED_HEAD_MODULE in imports:
        raise ValueError("Import only the supported oracle-backup V2 head, not its incompatible namesake")


def validate_registered_head(registry):
    """Check the implementation actually selected by MMCV, before construction."""
    head_class = registry.get(HEAD_TYPE)
    if head_class is None or head_class.__module__ != SUPPORTED_HEAD_MODULE:
        raise ValueError("The registered V2 head is not the supported ego-conditioned implementation")
    if "use_current_ego_status" not in inspect.signature(head_class.__init__).parameters:
        raise ValueError("The registered head does not support use_current_ego_status")


def prepare_config(base, work_dir, checkpoint, *, path_base=None):
    """Preserve the verified baseline; opt in and unfreeze the new encoder."""
    cfg = copy.deepcopy(base)
    head = cfg["model"]["head"]["motion_plan_head"]
    if head.get("type") != HEAD_TYPE:
        raise ValueError("This adapter requires the verified 6s V2 planning head")
    _validate_head_imports(cfg)
    if not work_dir or not checkpoint:
        raise ValueError("An independent work_dir and explicit checkpoint are required")
    protected = _protected_run(cfg, path_base)
    new_dir = _resolved_run_path(work_dir, absolute=True)
    _require_outside(new_dir, protected)
    work_dir = str(new_dir)

    def check_rotations(value):
        if isinstance(value, dict):
            if "rot3d_range" in value and any(float(v) != 0 for v in value["rot3d_range"]):
                raise ValueError("Nonzero lidar rotation needs explicit ego-frame handling")
            for child in value.values():
                check_rotations(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                check_rotations(child)

    # Rotation is usually a dataset aug_config, not a pipeline-step field.
    check_rotations(cfg)

    def check_pipeline(pipeline):
        if not isinstance(pipeline, list):
            raise ValueError("Missing model input pipeline")
        collectors = []
        for step in pipeline:
            if "rot3d_range" in step and any(float(v) != 0 for v in step["rot3d_range"]):
                raise ValueError("Nonzero lidar rotation needs explicit ego-frame handling")
            if step.get("type") == "Collect":
                collectors.append(step)
        if not collectors or any("ego_status" not in step.get("keys", []) for step in collectors):
            raise ValueError("Every model input Collect must include ego_status")

    def check_dataset(dataset):
        if isinstance(dataset, list):
            for child in dataset:
                check_dataset(child)
        elif "dataset" in dataset:
            check_dataset(dataset["dataset"])
        elif "datasets" in dataset:
            check_dataset(dataset["datasets"])
        else:
            check_pipeline(dataset.get("pipeline"))

    for split in ("train", "val", "test"):
        check_dataset(cfg["data"][split])

    def relocate(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "work_dir":
                    value[key] = str(work_dir)
                else:
                    relocate(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                relocate(child)

    relocate(cfg)
    head["use_current_ego_status"] = True
    matched = False
    for hook in cfg.get("custom_hooks", []):
        if hook.get("type") == "FreezeForLatentWorldModelMomAD6sHook":
            keywords = list(hook.get("train_keywords", (
                "motion_plan_head.latent_world_model", "motion_plan_head.final_fusion_bias",
            )))
            if "motion_plan_head.ego_state_encoder" not in keywords:
                keywords.append("motion_plan_head.ego_state_encoder")
            hook["train_keywords"] = tuple(keywords)
            matched = True
    if not matched:
        raise ValueError("Missing freeze hook: new encoder must explicitly be trainable")
    cfg.update(work_dir=str(work_dir), resume_from=None, load_from=str(checkpoint))
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--path-base", help=(
        "Absolute training cwd used to resolve a relative archived work_dir; "
        "defaults to this process's cwd. Set explicitly for cross-OS config preparation."))
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--output-config", required=True)
    args = parser.parse_args()
    output_checkpoint = Path(args.output_checkpoint)
    output_config = Path(args.output_config)
    source_checkpoint = Path(args.source_checkpoint)
    for target in (output_checkpoint, output_config):
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite {target}")
    if output_checkpoint.resolve() == output_config.resolve():
        raise ValueError("Config and checkpoint outputs must differ")
    # CPU-only migration; no launch or GPU allocation is performed here.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    from mmcv import Config
    from mmcv.utils import import_modules_from_strings
    from mmdet.models import HEADS, build_detector
    import projects.mmdet3d_plugin
    from projects.mmdet3d_plugin.models.motion.current_ego_state import load_current_ego_warmstart

    torch.manual_seed(0)
    base = Config.fromfile(args.base_config)
    prepared = prepare_config(base._cfg_dict, args.work_dir, str(output_checkpoint),
                              path_base=args.path_base)
    # Actual artifacts are local filesystem writes, even when preparing a
    # foreign-OS work_dir. Protect the archived config path as interpreted on
    # this host, and its target namespace when that namespace is native.
    old_dir = Path(base["work_dir"]).resolve()
    old_leaf = old_dir.name.casefold() if os.name == "nt" else old_dir.name
    protected = old_dir.parent if old_leaf == "training" else old_dir
    configured_protected = _protected_run(base._cfg_dict, args.path_base)
    for target in (output_checkpoint.resolve(), output_config.resolve()):
        if target == protected or protected in target.parents:
            raise ValueError("Do not place any new artifact inside the archived run")
        native_target = _resolved_run_path(str(target), absolute=True)
        if type(native_target) is type(configured_protected):
            _require_outside(native_target, configured_protected)
    import_modules_from_strings(**prepared["custom_imports"])
    validate_registered_head(HEADS)
    model = build_detector(prepared["model"], train_cfg=prepared.get("train_cfg"),
                           test_cfg=prepared.get("test_cfg"))
    # Every legacy tensor is replaced by the strict parent load below. The
    # new encoder initializes itself in its constructor; avoid fetching an
    # unrelated backbone initialization checkpoint during this migration.
    # Legacy MMCV checkpoints can contain pickled metadata. Only load trusted
    # checkpoints; make the choice explicit on newer PyTorch versions while
    # retaining compatibility with older versions without weights_only.
    load_options = dict(map_location="cpu")
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_options["weights_only"] = False
    with source_checkpoint.open("rb") as handle:
        before = os.fstat(handle.fileno())
        source_sha = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            source_sha.update(block)
        handle.seek(0)
        original = torch.load(handle, **load_options)
        after = os.fstat(handle.fileno())
        signature = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if signature(before) != signature(after) or signature(after) != signature(source_checkpoint.stat()):
            raise RuntimeError("Source checkpoint changed during preparation; no outputs written")
    receipt = load_current_ego_warmstart(model, original["state_dict"])
    receipt.update(parent_checkpoint=str(source_checkpoint), parent_sha256=source_sha.hexdigest(),
                   direct_current_ego_status=True, adapter_training_completed=False)
    for target in (output_checkpoint, output_config):
        target.parent.mkdir(parents=True, exist_ok=True)
    with output_checkpoint.open("xb") as handle:
        torch.save(dict(state_dict=model.state_dict(), meta=dict(current_ego_warmstart=receipt)), handle)
    with output_config.open("x", encoding="utf-8") as handle:
        handle.write(Config(prepared).pretty_text)
    print(receipt)
    print("Prepared only: new encoder requires training; no experiment was launched.")


if __name__ == "__main__":
    main()
