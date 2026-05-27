"""
Unit tests for tools/predict_to_csv.py.

Runs with stdlib only (no pytest required):
    python -m unittest tests.test_predict_to_csv

Or with pytest if installed:
    pytest tests/test_predict_to_csv.py -v

Covers the model-loading integrity contract (F6, commit 18affe065) and
the typed exception hierarchy (F8, commit 96cf4b777) so we don't silently
regress the error-handling story.
"""
import os
import pickle
import sys
import tempfile
import unittest

# Make `import tools.predict_to_csv` resolvable from repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.join(REPO_ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)


class TestErrorClasses(unittest.TestCase):
    """F8 — typed exceptions so runtime_pipeline can branch on cause."""

    def test_data_shortage_carries_counts(self):
        from predict_to_csv import PredictDataShortageError
        err = PredictDataShortageError(available=8, required=15)
        self.assertEqual(err.available, 8)
        self.assertEqual(err.required, 15)
        # Message must mention both numbers for log-grepping convenience
        s = str(err)
        self.assertIn("8", s)
        self.assertIn("15", s)

    def test_data_shortage_is_value_error_subclass(self):
        """Subclassing ValueError keeps backwards-compat with anything that
        used to catch bare ValueError before F8 typing was introduced."""
        from predict_to_csv import PredictDataShortageError
        self.assertTrue(issubclass(PredictDataShortageError, ValueError))

    def test_model_mismatch_is_runtime_error_subclass(self):
        from predict_to_csv import PredictModelMismatchError
        self.assertTrue(issubclass(PredictModelMismatchError, RuntimeError))


class TestLoadModelIntegrityChecks(unittest.TestCase):
    """F6 — load_model() should fail fast with actionable messages."""

    def test_missing_file_raises_with_hint(self):
        import torch
        from predict_to_csv import load_model
        with self.assertRaises(FileNotFoundError) as ctx:
            load_model("/nonexistent/never/here.pth", torch.device("cpu"))
        msg = str(ctx.exception)
        # Must mention the file path AND give a remediation hint
        self.assertIn("never/here.pth", msg)
        self.assertIn("提示", msg, msg)

    def test_truncated_file_rejected(self):
        """A < 1KB file is almost certainly a write-truncated corruption."""
        import torch
        from predict_to_csv import load_model
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tf:
            tf.write(b"x" * 64)  # 64 bytes — way below the 1KB sanity floor
            tmp_path = tf.name
        try:
            with self.assertRaises(ValueError) as ctx:
                load_model(tmp_path, torch.device("cpu"))
            self.assertIn("損毀", str(ctx.exception))
        finally:
            os.unlink(tmp_path)

    def test_checkpoint_missing_required_keys_rejected(self):
        """A pickled dict without model_state_dict must be rejected with a
        descriptive error (not a downstream AttributeError on .get())."""
        import torch
        from predict_to_csv import load_model
        # Write a >1KB pickle of an empty dict so we pass the size gate
        # but fail the schema check.
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tf:
            payload = {"junk": "x" * 2048}  # ensure > 1KB
            torch.save(payload, tf.name)
            tmp_path = tf.name
        try:
            with self.assertRaises(ValueError) as ctx:
                load_model(tmp_path, torch.device("cpu"))
            self.assertIn("model_state_dict", str(ctx.exception))
        finally:
            os.unlink(tmp_path)


class TestRegisterLegacyClasses(unittest.TestCase):
    """Old checkpoints pickled GRUSequence/Log1pScaler from __main__ namespace.
    register_legacy_checkpoint_classes() must inject them so torch.load works
    when called from any module (not just train_model.py running as __main__)."""

    def test_registers_into_main_module(self):
        from predict_to_csv import (
            register_legacy_checkpoint_classes,
            Log1pScaler,
            GRUSequence,
        )
        register_legacy_checkpoint_classes()
        main_module = sys.modules.get("__main__")
        self.assertIsNotNone(main_module)
        self.assertIs(getattr(main_module, "Log1pScaler", None), Log1pScaler)
        self.assertIs(getattr(main_module, "GRUSequence", None), GRUSequence)

    def test_idempotent(self):
        """Calling twice must not raise or overwrite differently."""
        from predict_to_csv import register_legacy_checkpoint_classes, Log1pScaler
        register_legacy_checkpoint_classes()
        before = sys.modules["__main__"].Log1pScaler
        register_legacy_checkpoint_classes()
        after = sys.modules["__main__"].Log1pScaler
        self.assertIs(before, after)


class TestLoadProductionCheckpointSmoke(unittest.TestCase):
    """If the deployed pair model is present, verify load_model succeeds
    on it. Skips silently when the file isn't available (CI / clean clone)."""

    PROD_MODEL = os.path.join(REPO_ROOT, "gru_traffic_model_pair.pth")

    def test_production_pair_model_loads(self):
        if not os.path.exists(self.PROD_MODEL):
            self.skipTest("gru_traffic_model_pair.pth not present")
        import torch
        from predict_to_csv import load_model
        model, scaler, edge_ids, input_len, pred_horizon, config = load_model(
            self.PROD_MODEL, torch.device("cpu")
        )
        # Contract: 6-tuple with these specific shapes
        self.assertIsNotNone(model)
        self.assertIsNotNone(scaler)
        self.assertTrue(len(edge_ids) > 0, "edge_ids must not be empty")
        self.assertEqual(input_len, 15, "pair model uses input_len=15")
        self.assertEqual(pred_horizon, 15, "pair model uses pred_horizon=15")
        self.assertEqual(config.get("model_type"), "gru_pair_log1p_v1")
        self.assertTrue(config.get("gap_feature"), "pair model trained with gap_feature")


if __name__ == "__main__":
    unittest.main(verbosity=2)
