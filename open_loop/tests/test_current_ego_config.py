import ast
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    "prepare_current_ego_config",
    Path(__file__).resolve().parents[1] / "tools" / "prepare_current_ego_config.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ConfigTests(unittest.TestCase):
    def setUp(self):
        pipeline = [dict(type="ResizeCropFlipImage", rot3d_range=[0, 0]),
                    dict(type="Collect", keys=["img", "ego_status", "gt_ego_fut_cmd"])]
        self.base = dict(
            work_dir="/runs/archived/training", resume_from="/runs/old.pth",
            model=dict(head=dict(motion_plan_head=dict(type="MotionPlanningHead_MomAD_World_model_6s_V2"))),
            custom_imports=dict(imports=[module.SUPPORTED_HEAD_MODULE], allow_failed_imports=False),
            data={split: dict(pipeline=copy.deepcopy(pipeline), work_dir="/runs/archived/training")
                  for split in ("train", "val", "test")},
            optimizer=dict(type="AdamW", lr=1e-7, weight_decay=.001),
            lr_config=dict(policy="CosineAnnealing", warmup_iters=100),
            evaluation=dict(interval=1174),
            custom_hooks=[dict(type="FreezeForLatentWorldModelMomAD6sHook",
                               train_keywords=("motion_plan_head.latent_world_model",))],
        )

    def prepare(self):
        return module.prepare_config(self.base, "/runs/ego_direct/training", "/runs/ego_direct/init.pth")

    def test_enabled_independent_and_preserved(self):
        before = copy.deepcopy(self.base)
        result = self.prepare()
        self.assertEqual(self.base, before)
        self.assertTrue(result["model"]["head"]["motion_plan_head"]["use_current_ego_status"])
        self.assertIsNone(result["resume_from"])
        self.assertEqual(result["load_from"], "/runs/ego_direct/init.pth")
        for key in ("optimizer", "lr_config", "evaluation"):
            self.assertEqual(result[key], before[key])
        for split in ("train", "val", "test"):
            self.assertEqual(result["data"][split]["pipeline"], before["data"][split]["pipeline"])
            self.assertEqual(result["data"][split]["work_dir"], result["work_dir"])
        self.assertEqual(result["custom_hooks"][0]["train_keywords"],
                         ("motion_plan_head.latent_world_model", "motion_plan_head.ego_state_encoder"))

    def test_old_root_protected(self):
        for path in ("/runs/archived", "/runs/archived/training", "/runs/archived/new"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                module.prepare_config(self.base, path, "/new.pth")

    def test_nonabsolute_and_traversal_rejected(self):
        for path in ("relative", "/runs/new/../archived", "C:relative", "\\rooted", ""):
            with self.subTest(path=path), self.assertRaises(ValueError):
                module.prepare_config(self.base, path, "/new.pth")

    def test_relative_archive_and_absolute_posix_target(self):
        self.base["work_dir"] = "work_dirs/archived"
        for path in ("/workspace/open_loop/work_dirs/archived",
                     "/workspace/open_loop/work_dirs/archived/training"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                module.prepare_config(self.base, path, "/new.pth", path_base="/workspace/open_loop")
        result = module.prepare_config(self.base, "/workspace/new_run", "/new.pth",
                                       path_base="/workspace/open_loop")
        self.assertEqual(result["work_dir"], "/workspace/new_run")

    def test_relative_archive_uses_actual_cwd_by_default(self):
        self.base["work_dir"] = "work_dirs/archived"
        archived = Path.cwd() / "work_dirs" / "archived"
        with self.assertRaises(ValueError):
            module.prepare_config(self.base, str(archived / "training"), "/new.pth")

    def test_relative_training_directory_protects_its_parent(self):
        self.base["work_dir"] = "work_dirs/archived/training"
        with self.assertRaises(ValueError):
            module.prepare_config(self.base, "/workspace/work_dirs/archived/new", "/new.pth",
                                  path_base="/workspace")

    def test_windows_paths_case_and_relative_archive(self):
        self.base["work_dir"] = r"work_dirs\Archived\training"
        for path in (r"C:\Workspace\work_dirs\archived", "c:/workspace/WORK_DIRS/ARCHIVED/new"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                module.prepare_config(self.base, path, "C:/new.pth", path_base="C:/Workspace")
        result = module.prepare_config(self.base, "C:/new_experiment/training", "C:/new.pth",
                                       path_base="C:/Workspace")
        self.assertEqual(module.PureWindowsPath(result["work_dir"]),
                         module.PureWindowsPath("C:/new_experiment/training"))
        self.assertEqual(result["optimizer"], self.base["optimizer"])

    def test_windows_absolute_and_unc_archives(self):
        for archived, nested in ((r"D:\Runs\Old\training", "d:/runs/old/new"),
                                 (r"D:\Runs\Old\TRAINING", "d:/runs/old/new"),
                                 (r"\\server\share\runs\old\training",
                                  r"\\SERVER\SHARE\runs\OLD\new")):
            with self.subTest(archived=archived):
                self.base["work_dir"] = archived
                with self.assertRaises(ValueError):
                    module.prepare_config(self.base, nested, "D:/new.pth")

    def test_mixed_or_missing_namespace_fails_closed(self):
        for archived, target, path_base in (
            ("C:/runs/archived", "/runs/new", None),
            ("/runs/archived", "C:/runs/new", None),
            ("work_dirs/archived", "/runs/new", "C:/workspace"),
            ("", "/runs/new", None),
            ("C:archived", "C:/runs/new", None),
        ):
            with self.subTest(archived=archived, target=target), self.assertRaises(ValueError):
                self.base["work_dir"] = archived
                module.prepare_config(self.base, target, "/new.pth", path_base=path_base)

    def test_explicit_path_base_must_be_absolute(self):
        self.base["work_dir"] = "work_dirs/archived"
        for path_base in ("relative", ""):
            with self.subTest(path_base=path_base), self.assertRaises(ValueError):
                module.prepare_config(self.base, "/runs/new", "/new.pth", path_base=path_base)

    def test_native_alias_resolved_before_archive_comparison(self):
        archived = Path.cwd() / "archived"
        alias = Path.cwd() / "archive_alias"
        self.base["work_dir"] = str(archived / "training")
        original_resolve = module.Path.resolve

        def resolve(path, *args, **kwargs):
            # Pure configuration test: simulate an existing symlink/junction,
            # without writing any filesystem fixture.
            if path == alias / "new":
                return archived / "new"
            return original_resolve(path, *args, **kwargs)

        with patch.object(module.Path, "resolve", resolve), self.assertRaises(ValueError):
            module.prepare_config(self.base, str(alias / "new"), "/new.pth")

    def test_supported_head_import_required(self):
        for imports in ([], [module.UNSUPPORTED_HEAD_MODULE],
                        [module.SUPPORTED_HEAD_MODULE, module.UNSUPPORTED_HEAD_MODULE],
                        [module.SUPPORTED_HEAD_MODULE, module.SUPPORTED_HEAD_MODULE], None):
            with self.subTest(imports=imports), self.assertRaises(ValueError):
                self.base["custom_imports"]["imports"] = imports
                self.prepare()

    def test_optional_or_missing_custom_imports_rejected(self):
        self.base["custom_imports"]["allow_failed_imports"] = True
        with self.assertRaises(ValueError):
            self.prepare()
        del self.base["custom_imports"]
        with self.assertRaises(ValueError):
            self.prepare()

    def test_supported_single_import_string_is_accepted(self):
        self.base["custom_imports"]["imports"] = module.SUPPORTED_HEAD_MODULE
        self.assertTrue(self.prepare()["model"]["head"]["motion_plan_head"]["use_current_ego_status"])

    def test_real_public_configs_select_compatible_or_namesake_module(self):
        configs = Path(__file__).resolve().parents[1] / "projects" / "configs"
        prefix = "MomAD_small_stage2_MomAD_World_model_6s_v2"
        for suffix, supported in (("_oracle_mode_reg02_resume_repro.py", True),
                                  ("_fusion5.py", False)):
            # Read only the literal imports list; never execute public configs
            # or import MMCV/torch as part of these stdlib tests.
            tree = ast.parse((configs / (prefix + suffix)).read_text(encoding="utf-8"))
            assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                              and any(isinstance(t, ast.Name) and t.id == "custom_imports"
                                      for t in node.targets))
            imports = next(kw.value for kw in assignment.value.keywords if kw.arg == "imports")
            self.base["custom_imports"]["imports"] = ast.literal_eval(imports)
            with self.subTest(suffix=suffix):
                if supported:
                    self.assertTrue(self.prepare()["model"]["head"]["motion_plan_head"]
                                    ["use_current_ego_status"])
                else:
                    with self.assertRaises(ValueError):
                        self.prepare()

    def test_registered_implementation_checked_not_just_class_name(self):
        class Supported:
            def __init__(self, use_current_ego_status=False):
                pass

        class Unsupported:
            def __init__(self):
                pass

        Supported.__module__ = module.SUPPORTED_HEAD_MODULE
        Unsupported.__module__ = module.UNSUPPORTED_HEAD_MODULE
        module.validate_registered_head({module.HEAD_TYPE: Supported})
        for candidate in (None, Unsupported):
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                module.validate_registered_head({module.HEAD_TYPE: candidate})
        Unsupported.__module__ = module.SUPPORTED_HEAD_MODULE
        with self.assertRaises(ValueError):
            module.validate_registered_head({module.HEAD_TYPE: Unsupported})

    def test_state_required_in_each_input_split(self):
        for split in ("train", "val", "test"):
            with self.subTest(split=split):
                original = copy.deepcopy(self.base)
                self.base["data"][split]["pipeline"][-1]["keys"].remove("ego_status")
                with self.assertRaises(ValueError):
                    self.prepare()
                self.base = original

    def test_nonzero_rotation_rejected(self):
        self.base["data"]["train"]["pipeline"][0]["rot3d_range"] = [-.1, .1]
        with self.assertRaises(ValueError):
            self.prepare()

    def test_nested_dataset_rotation_rejected(self):
        self.base["data"]["train"]["aug_config"] = dict(rot3d_range=[-.1, .1])
        with self.assertRaises(ValueError):
            self.prepare()

    def test_wrong_head_or_missing_freeze_hook(self):
        self.base["model"]["head"]["motion_plan_head"]["type"] = "OtherHead"
        with self.assertRaises(ValueError):
            self.prepare()
        self.setUp()
        self.base["custom_hooks"] = []
        with self.assertRaises(ValueError):
            self.prepare()


    def test_public_optin_example_declares_encoder_training_and_no_resume(self):
        path = (Path(__file__).resolve().parents[1] / "projects" / "configs" /
                "MomWorld_current_ego_6s_example.py")
        # The example contains only local configuration assignments; loading
        # it does not resolve its MMCV _base_, build a model or train anything.
        namespace = {}
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
        self.assertTrue(namespace["model"]["head"]["motion_plan_head"]["use_current_ego_status"])
        self.assertIsNone(namespace["resume_from"])
        self.assertIn("motion_plan_head.ego_state_encoder",
                      namespace["custom_hooks"][0]["train_keywords"])
        for split in ("train", "val", "test"):
            self.assertEqual(namespace["data"][split]["work_dir"], namespace["work_dir"])
        for key in ("optimizer", "optimizer_config", "lr_config", "runner"):
            self.assertNotIn(key, namespace)


if __name__ == "__main__":
    unittest.main()
