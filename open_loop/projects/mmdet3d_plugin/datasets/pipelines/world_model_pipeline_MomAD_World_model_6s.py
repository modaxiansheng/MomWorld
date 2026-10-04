"""World-state aware pipeline components for the independent 6s experiment.

The original MomAD pipeline is intentionally left untouched.  These wrappers
keep the new per-agent world-state labels aligned with every range/name filter,
apply the same geometric augmentation, and convert them to DataContainer.
"""

import numpy as np

from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import to_tensor

from .augment import BBoxRotation
from .transform import NuScenesSparse4DAdaptor


AGENT_WORLD_KEYS = (
    "gt_world_model_agent_states",
    "gt_world_model_agent_masks",
)


def _filter_agent_fields(input_dict, mask):
    for key in (
        "gt_agent_fut_trajs",
        "gt_agent_fut_masks",
        *AGENT_WORLD_KEYS,
    ):
        if key in input_dict:
            input_dict[key] = input_dict[key][mask]


@PIPELINES.register_module()
class BBoxRotation_MomAD_World_model_6s(BBoxRotation):
    """Rotate boxes, trajectories and explicit world states together."""

    def __call__(self, results):
        results = super().__call__(results)
        angle = results["aug_config"]["rotate_3d"]
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)
        rot_mat_t = np.array(
            [[rot_cos, rot_sin], [-rot_sin, rot_cos]], dtype=np.float32
        )

        for key in ("gt_agent_fut_trajs", "gt_ego_fut_trajs"):
            if key in results:
                results[key] = results[key] @ rot_mat_t

        for key in (
            "gt_world_model_agent_states",
            "gt_world_model_ego_states",
        ):
            if key not in results:
                continue
            states = results[key].copy()
            states[..., 0:2] = states[..., 0:2] @ rot_mat_t
            states[..., 2:4] = states[..., 2:4] @ rot_mat_t
            states[..., 5] = (
                states[..., 5] + angle + np.pi
            ) % (2 * np.pi) - np.pi
            results[key] = states
        return results


@PIPELINES.register_module()
class InstanceNameFilter_MomAD_World_model_6s(object):
    """Apply class filtering to standard and world-model agent labels."""

    def __init__(self, classes):
        self.classes = classes
        self.labels = list(range(len(classes)))

    def __call__(self, input_dict):
        labels = input_dict["gt_labels_3d"]
        mask = np.array([label in self.labels for label in labels], dtype=np.bool_)
        input_dict["gt_bboxes_3d"] = input_dict["gt_bboxes_3d"][mask]
        input_dict["gt_labels_3d"] = labels[mask]
        if "instance_inds" in input_dict:
            input_dict["instance_inds"] = input_dict["instance_inds"][mask]
        _filter_agent_fields(input_dict, mask)
        return input_dict


@PIPELINES.register_module()
class CircleObjectRangeFilter_MomAD_World_model_6s(object):
    """Range-filter all per-agent labels with one shared boolean mask."""

    def __init__(
        self,
        class_dist_thred=None,
    ):
        self.class_dist_thred = class_dist_thred or (
            [52.5] * 5 + [31.5] + [42] * 3 + [31.5]
        )

    def __call__(self, input_dict):
        boxes = input_dict["gt_bboxes_3d"]
        labels = input_dict["gt_labels_3d"]
        distance = np.sqrt(np.sum(boxes[:, :2] ** 2, axis=-1))
        mask = np.zeros(len(distance), dtype=np.bool_)
        for label_idx, threshold in enumerate(self.class_dist_thred):
            mask = np.logical_or(
                mask,
                np.logical_and(labels == label_idx, distance <= threshold),
            )
        input_dict["gt_bboxes_3d"] = boxes[mask]
        input_dict["gt_labels_3d"] = labels[mask]
        if "instance_inds" in input_dict:
            input_dict["instance_inds"] = input_dict["instance_inds"][mask]
        _filter_agent_fields(input_dict, mask)
        return input_dict


@PIPELINES.register_module()
class NuScenesSparse4DAdaptor_MomAD_World_model_6s(NuScenesSparse4DAdaptor):
    """Convert the new variable/fixed-size world labels to tensors."""

    def __call__(self, input_dict):
        input_dict = super().__call__(input_dict)
        for key in AGENT_WORLD_KEYS:
            if key in input_dict:
                input_dict[key] = DC(
                    to_tensor(input_dict[key]).float(),
                    stack=False,
                    cpu_only=False,
                )
        for key in (
            "gt_world_model_ego_states",
            "gt_world_model_ego_masks",
        ):
            if key in input_dict:
                input_dict[key] = DC(
                    to_tensor(input_dict[key]).float(),
                    stack=True,
                    cpu_only=False,
                    pad_dims=None,
                )
        return input_dict
