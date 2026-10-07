"""CPU aggregation regression tests, not a full MMCV/nuScenes integration run.

The actual evaluator functions and metric classes are extracted from their ASTs.
Dataset, loader, progress bar, table and aggregation-input metric doubles isolate
the return/display contract; collision geometry is deliberately not simulated.
"""

import ast
import hashlib
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch


PLANNING_DIR = (Path(__file__).resolve().parents[1] / "projects" /
                "mmdet3d_plugin" / "datasets" / "evaluation" / "planning")
spec = importlib.util.spec_from_file_location(
    "planning_metric_summary", PLANNING_DIR / "planning_metric_summary.py",
)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)

EVALUATORS = {
    "roboad": ("planning_eval_roboAD_6s.py", "planning_eval_roboAD_6s"),
    "world_model": ("planning_eval_MomAD_World_model_6s.py",
                    "planning_eval_MomAD_World_model_6s"),
}


def source_nodes(kind):
    filename, _ = EVALUATORS[kind]
    tree = ast.parse((PLANNING_DIR / filename).read_text(encoding="utf-8"))
    return {node.name: node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


def actual_definition(kind, name, namespace):
    node = source_nodes(kind)[name]
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, str(PLANNING_DIR / EVALUATORS[kind][0]), "exec"), namespace)
    return namespace[name]


class TableDouble:
    def __init__(self):
        self.field_names = []
        self.rows = []

    def add_row(self, row):
        self.rows.append(row)

    def __str__(self):
        return repr((self.field_names, self.rows))


def run_actual_evaluator(kind, raw=None, samples=2, valid_mask=True):
    """Execute the actual aggregation function with declared framework doubles."""
    raw = raw or {
        "obj_col": [index / 1000 for index in range(1, 13)],
        "obj_box_col": [index / 2000 for index in range(1, 13)],
        "L2": list(range(1, 13)),
        "Consist": list(range(2, 25, 2)),
    }
    tables = []
    metric_instances = []

    class AggregationMetricDouble:
        def __init__(self, n_future=12):
            self.n_future = n_future
            self.reset()
            metric_instances.append(self)

        def reset(self):
            self.total = torch.tensor(0)
            self.consist_total = torch.tensor(0)

        def update(self, *args):
            self.total += len(args[0])
            if len(args) == 6 and args[-1]:
                self.consist_total += len(args[0])

        def compute(self):
            summary.validate_valid_samples(self.total.item())
            return {key: torch.tensor(value, dtype=torch.float64)
                    for key, value in raw.items()}

    dataset = types.SimpleNamespace(data_infos=[
        {"scene_token": "same-scene"} for _ in range(samples)
    ])
    data = [{
        "gt_ego_fut_trajs": torch.zeros(1, 12, 2),
        "gt_ego_fut_masks": torch.full((1, 12), valid_mask, dtype=torch.bool),
        "gt_ego_fut_cmd": torch.tensor([[1.0, 0.0, 0.0]]),
        "fut_boxes": [],
    } for _ in range(samples)]
    results = [{"img_bbox": {"final_planning": torch.zeros(12, 2)}}
               for _ in range(samples)]
    namespace = {
        "torch": torch, "np": np, "tqdm": lambda value: value,
        "PlanningMetric": AggregationMetricDouble,
        "build_dataset": lambda config: dataset,
        "build_dataloader": lambda *args, **kwargs: data,
        "print_log": lambda *args, **kwargs: None,
        "align_previous_plan": lambda previous, *args: previous,
        "validate_valid_samples": summary.validate_valid_samples,
        "validate_metric_values": summary.validate_metric_values,
        "cumulative_metric_summary": summary.cumulative_metric_summary,
    }
    table_module = types.ModuleType("prettytable")

    def table_factory():
        table = TableDouble()
        tables.append(table)
        return table

    table_module.PrettyTable = table_factory
    function = actual_definition(kind, EVALUATORS[kind][1], namespace)
    with patch.dict(sys.modules, {"prettytable": table_module}):
        output = function(results, {"test_double": True}, logger=None)
    return output, tables[0], raw


class PureSummaryTests(unittest.TestCase):
    def test_selects_six_cumulative_horizons_and_unrounded_means(self):
        cumulative = [0.123456789012345 + index / 7 for index in range(12)]
        result = summary.cumulative_metric_summary("L2", cumulative, 4219)
        horizons = cumulative[1::2]
        self.assertEqual(len(result), 8)
        for horizon, value in enumerate(horizons, start=1):
            self.assertEqual(result["L2_cumulative_%ds" % horizon], value)
        self.assertEqual(result["L2_cumulative_mean_1s_to_6s"], sum(horizons) / 6)
        self.assertEqual(result["L2_cumulative_mean_4s_to_6s"], sum(horizons[3:]) / 3)
        self.assertTrue(all(type(value) is float for value in result.values()))

    def test_collision_is_a_fraction_not_percent(self):
        result = summary.cumulative_metric_summary("obj_box_col", [0.002] * 12, 1)
        self.assertTrue(all(value == 0.002 for value in result.values()))

    def test_requires_twelve_half_second_values(self):
        for values in ([], [1.0] * 6, [1.0] * 11, [1.0] * 13):
            with self.subTest(length=len(values)), self.assertRaises(ValueError):
                summary.cumulative_metric_summary("L2", values, 1)

    def test_nonfinite_or_nonscalar_input_rejected(self):
        for invalid in (float("nan"), float("inf"), -float("inf"), True,
                        "1.0", [1.0]):
            values = [1.0] * 12
            values[4] = invalid
            with self.subTest(invalid=repr(invalid)), self.assertRaises(ValueError):
                summary.validate_metric_values(values, 1)

    def test_empty_or_invalid_metric_population_rejected(self):
        for count in (0, -1, float("nan"), float("inf"), 0.5, True, "1"):
            with self.subTest(count=count), self.assertRaises(ValueError):
                summary.cumulative_metric_summary("L2", [0.0] * 12, count)

    def test_summary_arithmetic_overflow_fails_closed(self):
        with self.assertRaises(ValueError):
            summary.cumulative_metric_summary("L2", [1e308] * 12, 1)

    def test_metric_name_required(self):
        for name in ("", None, 0):
            with self.subTest(name=name), self.assertRaises(ValueError):
                summary.cumulative_metric_summary(name, [1.0] * 12, 1)


class ActualEvaluatorAggregationTests(unittest.TestCase):
    def test_original_avg_and_six_horizon_summary_both_evaluators(self):
        for kind in EVALUATORS:
            with self.subTest(evaluator=kind):
                result, table, raw = run_actual_evaluator(kind)
                for key, raw_values in raw.items():
                    if kind == "roboad":
                        cumulative = [np.array(raw_values[:i + 1]).mean()
                                      for i in range(12)]
                        original_avg = sum(cumulative[i] for i in (1, 3, 5)) / 3
                    else:
                        cumulative = np.cumsum(np.asarray(raw_values, dtype=np.float64)) / np.arange(1, 13)
                        original_avg = float(np.mean(cumulative[[1, 3, 5]]))
                    self.assertEqual(result[key], original_avg)
                    for horizon, index in enumerate((1, 3, 5, 7, 9, 11), start=1):
                        self.assertEqual(result[key + "_cumulative_%ds" % horizon], float(cumulative[index]))
                    self.assertEqual(result[key + "_cumulative_mean_1s_to_6s"],
                                     sum(float(cumulative[i]) for i in (1, 3, 5, 7, 9, 11)) / 6)
                    self.assertEqual(result[key + "_cumulative_mean_4s_to_6s"],
                                     sum(float(cumulative[i]) for i in (7, 9, 11)) / 3)
                    row = next(row for row in table.rows if row[0] == key)
                    expected_avg = ("%.3f%%" % (original_avg * 100)
                                    if "col" in key else "%.4f" % original_avg)
                    self.assertEqual(row[-1], expected_avg)
                self.assertTrue(all(isinstance(value, (float, np.floating))
                                    for value in result.values()))

    def test_world_model_keeps_long_ade_and_raw_6s_return_keys(self):
        result, _, raw = run_actual_evaluator("world_model")
        for key, raw_values in raw.items():
            cumulative = np.cumsum(raw_values) / np.arange(1, 13)
            name = "ADE" if key in ("L2", "Consist") else "cumulative"
            self.assertEqual(result[key + "_" + name + "_4s_5s_6s"],
                             float(np.mean(cumulative[[7, 9, 11]])))
            self.assertEqual(result[key + "_at_6s"], raw_values[11])

    def test_collision_percent_only_in_table_exactly_times_100(self):
        for kind in EVALUATORS:
            with self.subTest(evaluator=kind):
                result, table, _ = run_actual_evaluator(kind)
                row = next(row for row in table.rows if row[0] == "obj_col")
                self.assertEqual(row[2], "0.150%")  # cumulative 1s = 0.0015
                self.assertEqual(row[12], "0.650%")  # cumulative 6s = 0.0065
                self.assertAlmostEqual(result["obj_col_cumulative_1s"], 0.0015)
                self.assertTrue(all(value < 1 for key, value in result.items()
                                    if key.startswith("obj_col")))

    def test_empty_validation_or_all_masked_samples_rejected(self):
        for kind in EVALUATORS:
            for args in ({"samples": 0}, {"valid_mask": False}):
                with self.subTest(evaluator=kind, args=args), self.assertRaises(ValueError):
                    run_actual_evaluator(kind, **args)

    def test_world_model_zero_valid_consistency_pairs_not_reported_as_zero(self):
        with self.assertRaises(ValueError):
            run_actual_evaluator("world_model", samples=1)

    def test_actual_evaluator_rejects_nonfinite_or_incomplete_metric_arrays(self):
        for kind in EVALUATORS:
            for values in ([1.0] * 11, [float("nan")] * 12, [float("inf")] * 12):
                with self.subTest(evaluator=kind, values=values), self.assertRaises(ValueError):
                    run_actual_evaluator(kind, raw={"L2": values})

    def test_actual_compute_unscaled_and_original_denominators(self):
        for kind in EVALUATORS:
            namespace = {"torch": torch, "np": np,
                         "validate_valid_samples": summary.validate_valid_samples}
            metric_class = actual_definition(kind, "PlanningMetric", namespace)
            metric = metric_class()
            vector = torch.arange(1, 13, dtype=torch.float32)
            metric.total = torch.tensor(2)
            metric.L2 = vector * 2
            metric.obj_col = vector * 0.002
            metric.obj_box_col = vector * 0.001
            metric.Consist = vector * 3
            if kind == "world_model":
                metric.consist_total = torch.tensor(3)
            result = metric.compute()
            self.assertTrue(torch.equal(result["L2"], vector))
            self.assertTrue(torch.equal(result["obj_col"], metric.obj_col / 2))
            self.assertTrue(torch.equal(result["obj_box_col"], metric.obj_box_col / 2))
            expected_consist = metric.Consist / (3 if kind == "world_model" else 2)
            self.assertTrue(torch.equal(result["Consist"], expected_consist))
            metric.reset()
            with self.assertRaises(ValueError):
                metric.compute()

    def test_collision_geometry_masking_and_tpc_scientific_methods_unchanged(self):
        # Use AST only to locate source lines, not to hash AST fields (which
        # differ between Python 3.8 and 3.12). Normalize LF and trailing space.
        # These source fingerprints protect geometry, counts, masks and cache.
        expected = {
            "common": {
                "check_collision": "18cfd633fc4ca594a7e8e279124f2a431bf920e05491a52a2582662916469afe",
                "get_yaw": "b2a456f020ad7bacabe73938c0a08174bf1e4ada1b67455bc852b8004d5e6424",
                "PlanningMetric.__init__": "106a3edb1972ec4f4f47161889c16ad932527b45ac89ee0ca736152e30f5b719",
                "PlanningMetric.compute_L2": "d9cf19859cc4ec8a0111e94c4eae5d4c517765a83df44a2c069164167d7fa044",
                "PlanningMetric.compute_Consist": "337194d53b25c5f2f847207418a79588c2b9f4950cce875678c3b3696f103be3",
            },
            "roboad": {
                "PlanningMetric.reset": "c1b5b73b8a1bd97784fb3a9a882d2d662f9e98cd3bbe760862885abeac82cbd7",
                "PlanningMetric.evaluate_single_coll": "4dfb75734c13f6c219e41dad516a0145449802a3b9009978f0f12951c87b2500",
                "PlanningMetric.evaluate_coll": "d4933f4542572d7104e853071ba2365373d85af88684d25d941a7ff17442f220",
                "PlanningMetric.update": "15563c7919f883308ef3d8fc42d9566649987234d0592b13106595e995877589",
            },
            "world_model": {
                "get_lidar_to_global": "4f236414f4580e89bd7ac81809cece3c465bd3dca0f629f9656edc48eff48f33",
                "align_previous_plan": "874aa55117962a9a523bd8856145502b5504c927d06484132451e03ac4069561",
                "PlanningMetric.reset": "e3e272d71387031670e30b286244bbd604e9ac92c45924cbc899af426ab3847e",
                "PlanningMetric.evaluate_single_coll": "84c967aa81cf2ef60d8608acc3c3a6ffb5256114926c575f447e29b228e9564e",
                "PlanningMetric.evaluate_coll": "20a983abe020bf2185fd3e43814e174b190409f06c2deb60e1943516214b1f51",
                "PlanningMetric.update": "448deab28ad76ba183a9133a873e2081ef3bb1ea4881d15b1d4099098dca208d",
            },
        }
        for kind in EVALUATORS:
            source_lines = (PLANNING_DIR / EVALUATORS[kind][0]).read_text(
                encoding="utf-8",
            ).splitlines()
            nodes = source_nodes(kind)
            nodes.update({"PlanningMetric." + method.name: method
                          for method in nodes["PlanningMetric"].body
                          if isinstance(method, ast.FunctionDef)})
            for name, sha256 in {**expected["common"], **expected[kind]}.items():
                with self.subTest(evaluator=kind, function=name):
                    node = nodes[name]
                    normalized_source = "\n".join(
                        line.rstrip() for line in
                        source_lines[node.lineno - 1:node.end_lineno]
                    ) + "\n"
                    actual = hashlib.sha256(
                        normalized_source.encode("utf-8"),
                    ).hexdigest()
                    self.assertEqual(actual, sha256)


if __name__ == "__main__":
    unittest.main()
