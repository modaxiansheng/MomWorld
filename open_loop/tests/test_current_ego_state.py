"""CPU-only regression tests for direct, measured current-ego conditioning.

The encoder and warm-start loader are imported from their real source file.
To avoid requiring MMCV/MMDetection, the real head constructor, helper, anchor
methods, and complete forward method are extracted from its AST. Perception,
queue, and refinement dependencies are explicit deterministic test doubles.
These are unit/dataflow tests, not an end-to-end checkpoint or metric result.

Run: python -m unittest discover -s tests -p test_current_ego_state.py -v
MOMWORLD_OPEN_LOOP may override the source root for an isolated test harness.
An in-memory harness may instead provide _CURRENT_EGO_STATE_TEST_MODULE and
_CURRENT_EGO_HEAD_TEST_SOURCE in the execution namespace; no remote writes
are then required.
"""

import ast
import importlib.util
import os
from pathlib import Path
import types
import unittest

import torch
from torch import nn


OPEN_LOOP = Path(os.environ.get("MOMWORLD_OPEN_LOOP", Path(__file__).resolve().parents[1]))
MOTION = OPEN_LOOP / "projects/mmdet3d_plugin/models/motion"
HEAD_PATH = MOTION / "motion_planning_head_MomAD_World_model_6s_v2_oracle_reg_backup_20260628.py"
ENCODER_PATH = MOTION / "current_ego_state.py"
EMBED_DIMS = 8
STEPS = 12


def _load_encoder():
    if "_CURRENT_EGO_STATE_TEST_MODULE" in globals():
        return globals()["_CURRENT_EGO_STATE_TEST_MODULE"]
    spec = importlib.util.spec_from_file_location("current_ego_state_under_test", ENCODER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENCODER_MODULE = _load_encoder()
CurrentEgoStateEncoder = ENCODER_MODULE.CurrentEgoStateEncoder
load_current_ego_warmstart = ENCODER_MODULE.load_current_ego_warmstart


class _QueueDouble(nn.Module):
    """Provide no predicted state or labels: only fixed perception features."""

    def get(self, det_output, feature_maps, metas, bs, mask, anchor_handler):
        reference = det_output["instance_feature"]
        count, dim = reference.shape[1:]
        self.period = reference.new_zeros(bs, count)
        self.anchor_queue = reference.new_zeros(bs, count, 11)
        self.ego_period = reference.new_zeros(bs, 1)
        self.ego_anchor_queue = reference.new_zeros(bs, 1, 11)
        return (
            reference.new_zeros(bs, 1, dim),
            reference.new_zeros(bs, 1, 11),
            reference.new_zeros(bs, count + 1, 1, dim),
            reference.new_zeros(bs, count + 1, 1, 11),
            torch.zeros(bs, count + 1, 1, dtype=torch.bool, device=reference.device),
        )

    def cache_motion(self, features, det_output, metas):
        self.cached_motion = features.detach().clone()

    def cache_planning(self, features, predicted_status):
        self.cached_planning = features.detach().clone()
        self.cached_predicted_status = predicted_status.detach().clone()


def _planning_values(query, ego_feature):
    # The first feature is deliberately observable so the test does not rely
    # on cancellation-sensitive averages or on actual learned refiners.
    logits = 2.0 * query[..., 0] + ego_feature[..., :1]
    deltas = logits[..., None, None].expand(*logits.shape, STEPS, 2)
    status = ego_feature[..., :1].expand(*ego_feature.shape[:-1], 10)
    return logits, deltas, status


class _RefineDouble(nn.Module):
    def forward(self, motion_query, plan_query, ego_feature, anchor_embed):
        motion_logits = motion_query[..., 0]
        motion_deltas = motion_logits[..., None, None].expand(
            *motion_logits.shape, STEPS, 2
        )
        return (motion_logits, motion_deltas, *_planning_values(plan_query, ego_feature))


class _SecondRefineDouble(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, enhanced_query, ego_feature, anchor_embed):
        return _planning_values(enhanced_query, ego_feature)


class _WorldDouble(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, **inputs):
        self.last_ego_feature = inputs["ego_feature"].detach().clone()
        enhanced_query = inputs["plan_query"] + 2.0 * inputs["ego_feature"].unsqueeze(2)
        logits, deltas, _ = _planning_values(enhanced_query, inputs["ego_feature"])
        return {
            "enhanced_plan_query": enhanced_query,
            "world_plan_logits": logits,
            "world_plan_deltas": deltas,
            "scene_flow": deltas.new_zeros(deltas.shape[0], STEPS, 2),
            "future_latents": enhanced_query,
            "residual_scale": deltas.new_tensor(0.25),
        }


class _LegacyPredictorDouble(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(dim, hidden_dim)


def _build_from_cfg(cfg, registry):
    if cfg["kind"] == "queue":
        return _QueueDouble()
    if cfg["kind"] == "refine":
        return _RefineDouble()
    raise AssertionError("Unexpected dependency requested by head constructor")


def _topk(confidence, count, *features):
    return confidence[:, :count], tuple(feature[:, :count] for feature in features)


def _sine_embedding(position):
    return position.repeat_interleave(EMBED_DIMS // 2, dim=-1)


def _anchor_file(path):
    if path == "test-motion-anchor":
        return torch.zeros(2, 1, STEPS, 2).numpy()
    if path == "test-plan-anchor":
        return torch.zeros(3, 1, STEPS, 2).numpy()
    raise AssertionError("The unit test must not read real anchor files")


def _load_head_methods():
    source = (
        globals()["_CURRENT_EGO_HEAD_TEST_SOURCE"]
        if "_CURRENT_EGO_HEAD_TEST_SOURCE" in globals()
        else HEAD_PATH.read_text(encoding="utf-8")
    )
    tree = ast.parse(source)
    original = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "MotionPlanningHead_MomAD_World_model_6s_V2"
    )
    names = {"__init__", "get_motion_anchor", "_agent2lidar", "_condition_ego_feature", "forward"}
    methods = [
        node for node in original.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in methods} != names:
        raise AssertionError("The real head must contain every tested method")
    for method in methods:
        method.decorator_list = []
    extracted = ast.ClassDef(
        name=original.name,
        bases=[ast.Name(id="BaseModule", ctx=ast.Load())],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[extracted], type_ignores=[]))
    namespace = {
        "torch": torch,
        "nn": nn,
        "np": types.SimpleNamespace(load=_anchor_file),
        "BaseModule": nn.Module,
        "CurrentEgoStateEncoder": CurrentEgoStateEncoder,
        "build_from_cfg": _build_from_cfg,
        "build_loss": lambda cfg: None,
        "Linear": nn.Linear,
        "linear_relu_ln": lambda *args: [],
        "MotionPlanning2thRefinementModule": _SecondRefineDouble,
        "LatentWorldModelMomAD6s": _WorldDouble,
        "NextTokenPredictor": _LegacyPredictorDouble,
        "topk": _topk,
        "gen_sineembed_for_position": _sine_embedding,
        "SIN_YAW": 6,
        "COS_YAW": 7,
    }
    for registry in (
        "PLUGIN_LAYERS", "BBOX_SAMPLERS", "BBOX_CODERS", "ATTENTION",
        "NORM_LAYERS", "FEEDFORWARD_NETWORK",
    ):
        namespace[registry] = object()
    exec(compile(module, str(HEAD_PATH), "exec"), namespace)
    return namespace[original.name]


HeadUnderTest = _load_head_methods()


def _head(enabled=None, world_fusion=False):
    kwargs = dict(
        motion_anchor="test-motion-anchor",
        plan_anchor="test-plan-anchor",
        embed_dims=EMBED_DIMS,
        fut_mode=1,
        ego_fut_mode=1,
        ego_fut_ts=STEPS,
        instance_queue={"kind": "queue"},
        operation_order=["refine"],
        refine_layer={"kind": "refine"},
        num_det=2,
        num_map=2,
        world_fusion_enabled=world_fusion,
    )
    if enabled is not None:
        kwargs["use_current_ego_status"] = enabled
    return HeadUnderTest(**kwargs)


def _head_inputs(batch=2):
    boxes = torch.zeros(batch, 2, 11)
    boxes[..., 7] = 1.0
    det_output = {
        "instance_feature": torch.zeros(batch, 2, EMBED_DIMS),
        "anchor_embed": torch.zeros(batch, 2, EMBED_DIMS),
        "classification": [torch.zeros(batch, 2, 2)],
        "prediction": [boxes],
    }
    map_output = {
        "instance_feature": torch.zeros(batch, 2, EMBED_DIMS),
        "anchor_embed": torch.zeros(batch, 2, EMBED_DIMS),
        "classification": [torch.zeros(batch, 2, 3)],
        "prediction": [torch.zeros(batch, 2, 40)],
    }
    return det_output, map_output


def _forward(head, metas):
    det_output, map_output = _head_inputs()
    return head(
        det_output, map_output, [], metas,
        lambda anchors: anchors.new_zeros(*anchors.shape[:-1], EMBED_DIMS),
        torch.zeros(2, dtype=torch.bool), None,
    )[1]


def _measured_state():
    # Acceleration, angular rate, velocity, steering in their original units.
    return torch.tensor([
        [0.4, -0.2, 0.1, 0.0, 0.0, 0.03, 8.0, 0.2, 0.0, 0.05],
        [-0.3, 0.1, 0.0, 0.0, 0.0, -0.02, 12.0, -0.1, 0.0, -0.08],
    ])


def _make_state_projection_observable(encoder):
    with torch.no_grad():
        encoder.input_layer.weight.zero_()
        encoder.input_layer.bias.zero_()
        encoder.input_layer.weight[0, 6] = 1.0
        encoder.output_layer.weight.zero_()
        encoder.output_layer.bias.zero_()
        encoder.output_layer.weight[0, 0] = 1.0


class CurrentEgoEncoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.encoder = CurrentEgoStateEncoder(EMBED_DIMS)
        self.reference = torch.randn(2, 1, EMBED_DIMS)

    def test_zero_initialization_both_state_shapes_and_physical_scales(self):
        state = _measured_state()
        for shape in (state, state.unsqueeze(1)):
            residual = self.encoder(shape, self.reference)
            self.assertEqual(residual.shape, self.reference.shape)
            self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))
        self.assertTrue(torch.equal(
            self.encoder.state_scale,
            torch.tensor([5., 5., 5., 1., 1., 1., 20., 20., 20., 1.]),
        ))
        self.assertIn("state_scale", dict(self.encoder.named_buffers()))
        self.assertNotIn("state_scale", dict(self.encoder.named_parameters()))

    def test_invalid_batch_rank_and_dimension_fail(self):
        for state in (
            torch.zeros(10), torch.zeros(2, 9), torch.zeros(3, 10),
            torch.zeros(2, 2, 10), torch.zeros(2, 1, 1, 10),
        ):
            with self.subTest(shape=tuple(state.shape)), self.assertRaises(ValueError):
                self.encoder(state, self.reference)

    def test_non_tensor_and_non_floating_inputs_fail(self):
        for state in ([0.] * 10, None, torch.zeros(2, 10, dtype=torch.int64)):
            with self.subTest(type=type(state).__name__), self.assertRaises(TypeError):
                self.encoder(state, self.reference)

    def test_nonfinite_inputs_fail(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            state = _measured_state()
            state[0, 9] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.encoder(state, self.reference)

    def test_invalid_reference_shape_fails(self):
        for reference in (torch.zeros(2, EMBED_DIMS), torch.zeros(2, 2, EMBED_DIMS), torch.zeros(2, 1, 9)):
            with self.subTest(shape=tuple(reference.shape)), self.assertRaises(ValueError):
                self.encoder(_measured_state(), reference)

    def test_output_matches_reference_dtype_and_cpu_device(self):
        _make_state_projection_observable(self.encoder)
        for dtype in (torch.float32, torch.float64):
            reference = self.reference.to(dtype=dtype)
            residual = self.encoder(_measured_state().double(), reference)
            self.assertEqual(residual.dtype, dtype)
            self.assertEqual(residual.device, reference.device)
            self.assertTrue(torch.isfinite(residual).all())
        double_encoder = self.encoder.double()
        residual = double_encoder(_measured_state().float(), self.reference.double())
        self.assertEqual(residual.dtype, torch.float64)

    def test_encoder_does_not_mutate_or_detach_input(self):
        _make_state_projection_observable(self.encoder)
        state = _measured_state().requires_grad_()
        before = state.detach().clone()
        self.encoder(state, self.reference).sum().backward()
        self.assertTrue(torch.equal(state.detach(), before))
        self.assertIsNotNone(state.grad)
        self.assertGreater(float(state.grad[:, 6].abs().sum()), 0.0)

    def test_one_optimizer_update_learns_state_sensitivity(self):
        state = _measured_state()
        optimizer = torch.optim.SGD(self.encoder.parameters(), lr=0.1)
        self.encoder(state, self.reference).sum().backward()
        self.assertGreater(float(self.encoder.output_layer.weight.grad.abs().sum()), 0.0)
        optimizer.step()
        changed = self.encoder(state, self.reference)
        zero_state = self.encoder(torch.zeros_like(state), self.reference)
        self.assertGreater(float((changed - zero_state).detach().abs().sum()), 0.0)
        self.assertTrue(torch.isfinite(changed).all())


class CurrentEgoHeadDataflowTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(456)

    def test_default_constructor_has_no_new_parameters(self):
        default = _head()
        disabled = _head(False)
        self.assertFalse(default.use_current_ego_status)
        self.assertFalse(hasattr(default, "ego_state_encoder"))
        self.assertEqual(set(default.state_dict()), set(disabled.state_dict()))
        self.assertFalse(any("ego_state_encoder" in key for key in default.state_dict()))

    def test_disabled_ignores_missing_or_invalid_state(self):
        head = _head(False)
        reference = torch.randn(2, 1, EMBED_DIMS)
        self.assertIs(head._condition_ego_feature(reference, {}), reference)
        self.assertIs(head._condition_ego_feature(reference, {"ego_status": None}), reference)
        baseline = _forward(head, {})
        ignored = _forward(head, {"ego_status": torch.full((2, 10), float("nan"))})
        self.assertTrue(torch.equal(baseline["final_prediction"][-1], ignored["final_prediction"][-1]))

    def test_enabled_forward_requires_valid_current_state(self):
        head = _head(True)
        with self.assertRaises(KeyError):
            _forward(head, {})
        with self.assertRaises(TypeError):
            _forward(head, {"ego_status": None})
        with self.assertRaises(ValueError):
            _forward(head, {"ego_status": torch.zeros(2, 9)})

    def test_zero_initialization_preserves_base_world_refined_and_final(self):
        for world_fusion in (False, True):
            torch.manual_seed(100)
            disabled = _head(False, world_fusion)
            torch.manual_seed(100)
            enabled = _head(True, world_fusion)
            # Encoder creation consumes RNG. Match shared parameters exactly
            # using the actual strict warm-start loader, not seed assumptions.
            load_current_ego_warmstart(enabled, disabled.state_dict())
            baseline = _forward(disabled, {})
            direct = _forward(enabled, {"ego_status": _measured_state()})
            for name in ("classification", "prediction", "status", "prediction_refined", "final_prediction", "final_status"):
                with self.subTest(fusion=world_fusion, branch=name):
                    self.assertTrue(torch.equal(baseline[name][-1], direct[name][-1]))
            self.assertTrue(torch.equal(baseline["world_prediction"], direct["world_prediction"]))

    def test_measured_state_reaches_actual_base_and_final_forward_train_and_eval(self):
        for training in (True, False):
            for world_fusion in (False, True):
                head = _head(True, world_fusion)
                head.train(training)
                _make_state_projection_observable(head.ego_state_encoder)
                zero_state = _forward(head, {"ego_status": torch.zeros(2, 10)})
                measured = _forward(head, {"ego_status": _measured_state()})
                for branch in ("prediction", "prediction_refined", "final_prediction"):
                    with self.subTest(training=training, fusion=world_fusion, branch=branch):
                        self.assertGreater(float((measured[branch][-1] - zero_state[branch][-1]).detach().abs().sum()), 0.0)
                self.assertGreater(float((measured["world_prediction"] - zero_state["world_prediction"]).detach().abs().sum()), 0.0)
                self.assertGreater(float(head.latent_world_model.last_ego_feature.abs().sum()), 0.0)

    def test_forward_needs_no_future_trajectory_labels_and_optimizer_reaches_encoder(self):
        class CurrentOnly(dict):
            def __init__(self):
                super().__init__(ego_status=_measured_state())
                self.reads = []

            def __getitem__(self, key):
                self.reads.append(key)
                if key != "ego_status":
                    raise AssertionError("Direct conditioning must not access future supervision")
                return super().__getitem__(key)

        head = _head(True)
        metas = CurrentOnly()
        optimizer = torch.optim.SGD(head.ego_state_encoder.parameters(), lr=0.01)
        initial = _forward(head, metas)["final_prediction"][-1]
        initial.sum().backward()
        self.assertGreater(float(head.ego_state_encoder.output_layer.weight.grad.abs().sum()), 0.0)
        optimizer.step()
        direct = _forward(head, metas)["final_prediction"][-1]
        counterfactual = _forward(head, {"ego_status": torch.zeros(2, 10)})["final_prediction"][-1]
        self.assertGreater(float((direct - counterfactual).detach().abs().sum()), 0.0)
        self.assertEqual(set(metas.reads), {"ego_status"})


class CurrentEgoDtypeSafetyTests(unittest.TestCase):
    def test_finite_input_that_overflows_parameter_dtype_is_rejected(self):
        encoder = CurrentEgoStateEncoder(EMBED_DIMS)
        reference = torch.zeros(1, 1, EMBED_DIMS)
        state = torch.zeros(1, 10, dtype=torch.float64)
        state[0, 0] = 1e100
        self.assertTrue(torch.isfinite(state).all())
        with self.assertRaisesRegex(ValueError, "overflows"):
            encoder(state, reference)
        half_encoder = CurrentEgoStateEncoder(EMBED_DIMS).half()
        state[0, 0] = 1e6
        with self.assertRaisesRegex(ValueError, "overflows"):
            half_encoder(state, reference.half())


class CurrentEgoFreezeHookTests(unittest.TestCase):
    def test_actual_freeze_hook_only_enables_selected_new_modules(self):
        source = (MOTION / "freeze_for_latent_world_model_MomAD_6s.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        hook_class = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        hook_class.decorator_list = []
        namespace = {"torch": torch, "Hook": object}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[hook_class], type_ignores=[])),
                     "actual_freeze_hook_under_test", "exec"), namespace)
        model = nn.Module()
        model.backbone = nn.Sequential(nn.Linear(4, EMBED_DIMS), nn.BatchNorm1d(EMBED_DIMS))
        model.head = nn.Module()
        model.head.motion_plan_head = _head(True)
        selected = ("motion_plan_head.latent_world_model", "motion_plan_head.final_fusion_bias",
                    "motion_plan_head.ego_state_encoder")
        hook = namespace["FreezeForLatentWorldModelMomAD6sHook"](train_keywords=selected)
        runner = types.SimpleNamespace(model=model, logger=types.SimpleNamespace(info=lambda *args: None))
        hook.before_run(runner)
        encoder_parameters = []
        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.requires_grad, any(keyword in name for keyword in selected), name)
            if "ego_state_encoder" in name:
                encoder_parameters.append(parameter)
        self.assertTrue(encoder_parameters)
        self.assertTrue(all(parameter.requires_grad for parameter in encoder_parameters))
        self.assertFalse(model.backbone[1].training)
        model.train()
        hook.before_train_epoch(runner)
        self.assertFalse(model.backbone[1].training)


class CurrentEgoWarmstartTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(789)
        self.model = _head(True)
        self.expected = self.model.state_dict()
        self.encoder_keys = {key for key in self.expected if key.startswith("ego_state_encoder.")}

    def test_old_checkpoint_initializes_only_wholly_new_encoder(self):
        old = {key: value.clone() for key, value in self.expected.items() if key not in self.encoder_keys}
        info = load_current_ego_warmstart(self.model, old)
        self.assertEqual(set(info["initialized_ego_encoder_keys"]), self.encoder_keys)
        self.assertEqual(info["removed_legacy_cache_keys"], [])
        self.assertIs(info["optimizer_restored"], False)
        for key, value in old.items():
            self.assertTrue(torch.equal(self.model.state_dict()[key], value))
        self.assertTrue(torch.equal(self.model.ego_state_encoder.output_layer.weight, torch.zeros_like(self.model.ego_state_encoder.output_layer.weight)))

    def test_complete_new_checkpoint_preserves_trained_encoder(self):
        _make_state_projection_observable(self.model.ego_state_encoder)
        trained = {key: value.clone() for key, value in self.model.state_dict().items()}
        target = _head(True)
        info = load_current_ego_warmstart(target, trained)
        self.assertEqual(info["initialized_ego_encoder_keys"], [])
        for key, value in trained.items():
            self.assertTrue(torch.equal(target.state_dict()[key], value))

    def test_only_exact_obsolete_prediction_cache_is_removed(self):
        wrapped = nn.Module()
        wrapped.head = nn.Module()
        wrapped.head.motion_plan_head = self.model
        state = wrapped.state_dict()
        legacy = "head.motion_plan_head.last_final_planning_prediction"
        state[legacy] = torch.ones(1, STEPS, 2)
        info = load_current_ego_warmstart(wrapped, state)
        self.assertEqual(info["removed_legacy_cache_keys"], [legacy])
        for invalid in ("last_final_planning_prediction", "head.motion_plan_head.unrelated_cache"):
            state = dict(self.expected)
            state[invalid] = torch.zeros(1)
            with self.subTest(key=invalid), self.assertRaises(RuntimeError):
                load_current_ego_warmstart(self.model, state)

    def test_cache_exception_requires_actual_nonpersistent_buffer_and_shape(self):
        legacy = "head.motion_plan_head.last_final_planning_prediction"
        state = dict(self.expected)
        state[legacy] = torch.ones(1, STEPS, 2)
        with self.assertRaises(RuntimeError):
            load_current_ego_warmstart(self.model, state)
        wrapped = nn.Module()
        wrapped.head = nn.Module()
        wrapped.head.motion_plan_head = self.model
        state = wrapped.state_dict()
        state[legacy] = torch.ones(1, 1, 2)
        with self.assertRaises(RuntimeError):
            load_current_ego_warmstart(wrapped, state)

    def test_state_dict_module_version_metadata_is_preserved(self):
        state = self.model.state_dict()
        state._metadata[""]["version"] = 73
        observed = []
        hook = self.model.register_load_state_dict_pre_hook(
            lambda model, state, prefix, metadata, *rest: observed.append(metadata["version"])
        )
        try:
            load_current_ego_warmstart(self.model, state)
        finally:
            hook.remove()
        self.assertEqual(observed, [73])
        self.assertEqual(state._metadata[""]["version"], 73)

    def test_partial_encoder_unrelated_missing_shape_mismatch_and_disabled_fail(self):
        cases = []
        partial = dict(self.expected)
        partial.pop("ego_state_encoder.output_layer.bias")
        cases.append(partial)
        unrelated = dict(self.expected)
        unrelated.pop("final_fusion_bias")
        cases.append(unrelated)
        bad_shape = dict(self.expected)
        bad_shape["ego_state_encoder.output_layer.weight"] = torch.zeros(1, 1)
        cases.append(bad_shape)
        for index, state in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(RuntimeError):
                load_current_ego_warmstart(self.model, state)
        disabled = _head(False)
        with self.assertRaises(ValueError):
            load_current_ego_warmstart(disabled, disabled.state_dict())


if __name__ == "__main__":
    unittest.main()
