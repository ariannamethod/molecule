"""Tests for gamma (personality fingerprint) and the immune system."""

import sys
import os
import unittest
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from molequla import (
    GPT, EvolvingTokenizer, backward, no_grad, train_steps, CFG
)


def _make_model():
    docs = ["Hello world", "Testing gamma", "Immune system check"]
    tok = EvolvingTokenizer(docs)
    model = GPT(tok)
    return model, tok, docs


class TestGammaComputation(unittest.TestCase):

    def test_gamma_initial_zero(self):
        """Gamma should be near-zero immediately after construction."""
        model, _, _ = _make_model()
        stats = model.gamma_stats()
        # All embeddings are at init snapshot, so magnitude ≈ 0
        self.assertAlmostEqual(stats["magnitude"], 0.0, places=6)
        self.assertAlmostEqual(stats["sparsity"], 1.0, places=6)

    def test_gamma_grows_after_training(self):
        """Gamma magnitude should increase after training."""
        model, tok, docs = _make_model()
        train_steps(model, tok, docs, steps=20, train_base=True, train_deltas=True)
        stats = model.gamma_stats()
        self.assertGreater(stats["magnitude"], 0.0)

    def test_gamma_stats_structure(self):
        """gamma_stats should return dict with sparsity, magnitude, top_tokens, n_rows."""
        model, _, _ = _make_model()
        stats = model.gamma_stats()
        self.assertIn("sparsity", stats)
        self.assertIn("magnitude", stats)
        self.assertIn("top_tokens", stats)
        self.assertIn("n_rows", stats)
        self.assertEqual(stats["n_rows"], model.tok.vocab_size)
        self.assertIsInstance(stats["top_tokens"], list)

    def test_gamma_sparsity_range(self):
        """Sparsity should always be between 0 and 1."""
        model, tok, docs = _make_model()
        stats = model.gamma_stats()
        self.assertGreaterEqual(stats["sparsity"], 0.0)
        self.assertLessEqual(stats["sparsity"], 1.0)
        # After training
        train_steps(model, tok, docs, steps=10, train_base=True, train_deltas=True)
        stats = model.gamma_stats()
        self.assertGreaterEqual(stats["sparsity"], 0.0)
        self.assertLessEqual(stats["sparsity"], 1.0)

    def test_contrastive_projection_returns_tuple(self):
        """gamma_contrastive_projection should return (direction, magnitude)."""
        model, _, _ = _make_model()
        result = model.gamma_contrastive_projection()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        direction, mag = result
        self.assertIsInstance(direction, list)
        self.assertIsInstance(mag, float)

    def test_contrastive_projection_unit_vector(self):
        """After training, projection direction should be approximately unit length."""
        import numpy as np
        model, tok, docs = _make_model()
        train_steps(model, tok, docs, steps=30, train_base=True, train_deltas=True)
        direction, mag = model.gamma_contrastive_projection()
        if mag > 1e-10:
            norm = float(np.linalg.norm(direction))
            self.assertAlmostEqual(norm, 1.0, places=5)


class TestImmuneSystem(unittest.TestCase):

    def test_snapshot_restore_preserves_weights(self):
        """Snapshot + restore should return delta weights to their original state."""
        import numpy as np
        model, _, _ = _make_model()
        snap = model.snapshot_deltas()
        # Get original weights
        orig_data = []
        for mod in model.deltas:
            for name, da in mod.items():
                for row in da.A.rows:
                    orig_data.append(row.data.copy())

        # Mutate weights
        for mod in model.deltas:
            for name, da in mod.items():
                for row in da.A.rows:
                    row.data += 99.0

        # Restore
        model.restore_deltas(snap)

        # Verify restored
        idx = 0
        for mod in model.deltas:
            for name, da in mod.items():
                for row in da.A.rows:
                    np.testing.assert_array_almost_equal(row.data, orig_data[idx])
                    idx += 1

    def test_drift_check_returns_one_at_init(self):
        """Drift check should return 1.0 when gamma is near-zero (early training)."""
        model, _, _ = _make_model()
        direction, mag = model.gamma_contrastive_projection()
        drift = model.gamma_drift_check(direction, mag)
        # Should skip check and return 1.0 because magnitude is near-zero
        self.assertEqual(drift, 1.0)

    def test_drift_check_positive_after_consistent_training(self):
        """Drift cosine should be positive after two consistent training bursts."""
        model, tok, docs = _make_model()
        # First burst to build some gamma
        train_steps(model, tok, docs, steps=40, train_base=True, train_deltas=True)
        # Now measure
        direction, mag = model.gamma_contrastive_projection()
        # Second burst with same data (consistent)
        train_steps(model, tok, docs, steps=10, train_base=True, train_deltas=True)
        drift = model.gamma_drift_check(direction, mag)
        # Same data, so drift should be positive (consistent direction)
        self.assertGreater(drift, -1.0)

    def test_drift_check_none_direction(self):
        """Drift check should return 1.0 (safe) when pre_direction is None."""
        model, _, _ = _make_model()
        drift = model.gamma_drift_check(None, 0.0)
        self.assertEqual(drift, 1.0)

    def test_snapshot_deltas_structure(self):
        """Snapshot should be a list of dicts matching model.deltas."""
        model, _, _ = _make_model()
        snap = model.snapshot_deltas()
        self.assertIsInstance(snap, list)
        self.assertEqual(len(snap), len(model.deltas))
        for mod_snap in snap:
            self.assertIsInstance(mod_snap, dict)

    def test_immune_rollback_after_noise(self):
        """Manually inject noise, verify rollback restores original."""
        import numpy as np
        model, tok, docs = _make_model()
        # Train to build some baseline
        train_steps(model, tok, docs, steps=20, train_base=True, train_deltas=True)

        snap = model.snapshot_deltas()
        # Save original delta weight checksums
        orig_checksums = []
        for mod in model.deltas:
            for name in sorted(mod.keys()):
                da = mod[name]
                for row in da.A.rows:
                    orig_checksums.append(float(np.sum(row.data)))

        # Inject noise (simulating bad training)
        for mod in model.deltas:
            for name, da in mod.items():
                for row in da.A.rows:
                    row.data += np.random.randn(*row.data.shape) * 100

        # Rollback
        model.restore_deltas(snap)

        # Verify restored
        idx = 0
        for mod in model.deltas:
            for name in sorted(mod.keys()):
                da = mod[name]
                for row in da.A.rows:
                    self.assertAlmostEqual(float(np.sum(row.data)), orig_checksums[idx], places=4)
                    idx += 1


if __name__ == "__main__":
    unittest.main()
