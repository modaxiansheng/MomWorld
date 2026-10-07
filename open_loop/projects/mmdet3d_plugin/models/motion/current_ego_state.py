"""Direct measured ego-state conditioning, separate from predicted status memory.

Field order follows the nuScenes CAN converter: acceleration xyz (m/s^2),
angular rate xyz (rad/s), velocity xyz (m/s), steering angle (rad).  These
remain in the vehicle ego frame; this module does not reinterpret them as
lidar-frame anchor velocities or read any future trajectory labels.
"""

from collections import OrderedDict
import copy

import torch
from torch import nn


class CurrentEgoStateEncoder(nn.Module):
    """Encode a batch's measured state into an additive ego-feature residual.

The final projection starts at zero to preserve the pretrained planner at
warm start.  It must be trained before claiming a learned state-conditioned
result.  Scales are fixed physical-unit scales, not validation statistics.
"""

    def __init__(self, embed_dims):
        super().__init__()
        self.register_buffer(
            "state_scale",
            torch.tensor([5., 5., 5., 1., 1., 1., 20., 20., 20., 1.]),
        )
        self.input_layer = nn.Linear(10, embed_dims)
        self.activation = nn.SiLU()
        self.output_layer = nn.Linear(embed_dims, embed_dims)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, ego_status, reference):
        if not torch.is_tensor(reference) or not reference.is_floating_point():
            raise TypeError("Ego feature must be a floating-point tensor")
        if reference.ndim != 3 or reference.shape[1:] != (1, self.output_layer.out_features):
            raise ValueError("Ego feature must have shape [batch, 1, embed_dims]")
        if not torch.is_tensor(ego_status) or not ego_status.is_floating_point():
            raise TypeError("Current ego_status must be a floating-point tensor")
        if ego_status.ndim == 3 and ego_status.shape[1] == 1:
            ego_status = ego_status[:, 0]
        if ego_status.ndim != 2 or ego_status.shape != (reference.shape[0], 10):
            raise ValueError("Current ego_status must have shape [batch, 10]")
        if not torch.isfinite(ego_status).all():
            raise ValueError("Current ego_status contains NaN or infinity")
        # Keep the encoder's parameter dtype under MMCV mixed precision, then
        # return the feature dtype. No detach: the new encoder receives grads.
        state = ego_status.to(
            device=reference.device, dtype=self.input_layer.weight.dtype,
        )
        if not torch.isfinite(state).all():
            raise ValueError("Current ego_status overflows the encoder parameter dtype")
        state = torch.asinh(state / self.state_scale)
        residual = self.output_layer(self.activation(self.input_layer(state)))
        return residual.unsqueeze(1).to(dtype=reference.dtype)


def load_current_ego_warmstart(model, pretrained_state):
    """Strictly load old weights, allowing only a wholly new ego encoder.

Use for a NEW experiment's model warm start, never to resume an old optimizer.
Unrelated missing/unexpected tensors and shape mismatches are fatal. A fully
trained state-conditioned checkpoint is also accepted without resetting it.
"""
    expected = model.state_dict()
    state = OrderedDict(pretrained_state)
    # PyTorch uses module-version metadata during state-dict loading. Do not
    # discard it when adding the new encoder's constructor-initialized keys.
    if hasattr(pretrained_state, "_metadata"):
        state._metadata = copy.deepcopy(pretrained_state._metadata)
    legacy_key = "head.motion_plan_head.last_final_planning_prediction"
    removed = []
    if legacy_key in state and legacy_key not in expected:
        owner = model
        for component in ("head", "motion_plan_head"):
            owner = owner._modules.get(component) if owner is not None else None
        cache = None if owner is None else owner._buffers.get("last_final_planning_prediction")
        if (owner is None or "last_final_planning_prediction" not in owner._non_persistent_buffers_set
                or not torch.is_tensor(cache) or not torch.is_tensor(state[legacy_key])
                or state[legacy_key].shape != cache.shape):
            raise RuntimeError("Legacy prediction cache is not the known nonpersistent buffer")
        state.pop(legacy_key)
        removed.append(legacy_key)
    encoder_keys = {
        key for key in expected
        if key.startswith("ego_state_encoder.") or ".ego_state_encoder." in key
    }
    if not encoder_keys:
        raise ValueError("Enable current ego conditioning before warm-start loading")
    missing = set(expected) - set(state)
    unexpected = set(state) - set(expected)
    if unexpected or (missing and missing != encoder_keys):
        raise RuntimeError(
            "Checkpoint mismatch outside a wholly new ego encoder: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    mismatched = [
        key for key in set(expected) & set(state)
        if expected[key].shape != state[key].shape
    ]
    if mismatched:
        raise RuntimeError(f"Checkpoint tensor shape mismatch: {sorted(mismatched)}")
    state.update({key: expected[key] for key in missing})
    model.load_state_dict(state, strict=True)
    return {
        "initialized_ego_encoder_keys": sorted(missing),
        "removed_legacy_cache_keys": removed,
        "optimizer_restored": False,
    }
