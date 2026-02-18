#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for ontogenesis (architecture growth).
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from molecule import (MatrixParam, DeltaAdapter, GPT, EvolvingTokenizer, CFG,
                       head_types_for_n_head, save_checkpoint, load_checkpoint)


class TestGrowCols(unittest.TestCase):
    """Tests for MatrixParam.grow_cols()."""

    def test_grow_cols_extends_width(self):
        m = MatrixParam(3, 4, std=0.1)
        m.grow_cols(8)
        self.assertEqual(m.nin, 8)
        for row in m.rows:
            self.assertEqual(len(row.data), 8)

    def test_grow_cols_preserves_data(self):
        m = MatrixParam(2, 3, std=0.0)
        m.rows[0].data[:] = [1.0, 2.0, 3.0]
        m.rows[1].data[:] = [4.0, 5.0, 6.0]
        m.grow_cols(5, std=0.0)
        self.assertAlmostEqual(m.rows[0].data[0], 1.0)
        self.assertAlmostEqual(m.rows[0].data[1], 2.0)
        self.assertAlmostEqual(m.rows[0].data[2], 3.0)
        self.assertAlmostEqual(m.rows[1].data[0], 4.0)

    def test_grow_cols_noop_when_smaller(self):
        m = MatrixParam(3, 4, std=0.1)
        m.grow_cols(4)
        self.assertEqual(m.nin, 4)
        m.grow_cols(2)
        self.assertEqual(m.nin, 4)


class TestGrow(unittest.TestCase):
    """Tests for MatrixParam.grow() (both dimensions)."""

    def test_grow_both_dimensions(self):
        m = MatrixParam(3, 4, std=0.1)
        m.grow(5, 8)
        self.assertEqual(m.nout, 5)
        self.assertEqual(m.nin, 8)
        self.assertEqual(len(m.rows), 5)
        for row in m.rows:
            self.assertEqual(len(row.data), 8)

    def test_grow_preserves_top_left(self):
        m = MatrixParam(2, 3, std=0.0)
        m.rows[0].data[:] = [1.0, 2.0, 3.0]
        m.rows[1].data[:] = [4.0, 5.0, 6.0]
        m.grow(4, 6, std=0.0)
        self.assertAlmostEqual(m.rows[0].data[0], 1.0)
        self.assertAlmostEqual(m.rows[0].data[2], 3.0)
        self.assertAlmostEqual(m.rows[1].data[1], 5.0)


class TestDeltaAdapterGrowth(unittest.TestCase):

    def test_grow_dims(self):
        da = DeltaAdapter(16, 8, r=4)
        da.grow_dims(32, 16)
        self.assertEqual(da.A.nout, 32)
        self.assertEqual(da.A.nin, 4)   # rank unchanged
        self.assertEqual(da.B.nout, 4)  # rank unchanged
        self.assertEqual(da.B.nin, 16)


class TestHeadTypesForNHead(unittest.TestCase):

    def test_1_head(self):
        self.assertEqual(head_types_for_n_head(1), ("content",))

    def test_2_heads(self):
        self.assertEqual(head_types_for_n_head(2), ("content", "hybrid"))

    def test_4_heads(self):
        self.assertEqual(head_types_for_n_head(4),
                         ("content", "content", "hybrid", "hybrid"))

    def test_8_heads(self):
        result = head_types_for_n_head(8)
        self.assertEqual(len(result), 8)
        self.assertEqual(result[:4], ("content",) * 4)
        self.assertEqual(result[4:], ("hybrid",) * 4)


class TestMaybeGrowArchitecture(unittest.TestCase):

    def setUp(self):
        # Ensure embryo defaults
        CFG.n_embd = 16
        CFG.n_layer = 1
        CFG.n_head = 1
        CFG.head_types = ("content",)
        self.docs = ["Hello world."]
        self.tok = EvolvingTokenizer(self.docs)
        self.model = GPT(self.tok)

    def test_embryo_dimensions(self):
        """Model should start as embryo."""
        self.assertEqual(self.model.n_embd, 16)
        self.assertEqual(self.model.n_layer, 1)
        self.assertEqual(self.model.n_head, 1)
        self.assertEqual(self.model.current_growth_stage(), 0)

    def test_no_growth_below_threshold(self):
        """Should not grow if corpus too small."""
        grew = self.model.maybe_grow_architecture(5000)
        self.assertFalse(grew)
        self.assertEqual(self.model.n_embd, 16)

    def test_embryo_to_infant(self):
        """Should grow to infant at 20KB corpus."""
        grew = self.model.maybe_grow_architecture(25000)
        self.assertTrue(grew)
        self.assertEqual(self.model.n_embd, 32)
        self.assertEqual(self.model.n_layer, 1)
        self.assertEqual(self.model.n_head, 2)
        self.assertEqual(self.model.current_growth_stage(), 1)

    def test_embryo_to_child(self):
        """Should jump to child at 50KB+ corpus."""
        grew = self.model.maybe_grow_architecture(60000)
        self.assertTrue(grew)
        self.assertEqual(self.model.n_embd, 64)
        self.assertEqual(self.model.n_layer, 2)
        self.assertEqual(self.model.n_head, 4)
        self.assertEqual(self.model.current_growth_stage(), 2)

    def test_growth_preserves_old_weights(self):
        """Old wte values should remain in top-left corner."""
        import numpy as np
        old_wte_row0 = self.model.base["wte"].rows[0].data.copy()
        self.model.maybe_grow_architecture(25000)
        new_row0 = self.model.base["wte"].rows[0].data
        self.assertEqual(len(new_row0), 32)
        # First 16 values should be unchanged
        np.testing.assert_array_equal(new_row0[:16], old_wte_row0)

    def test_head_types_updated(self):
        """head_types should match new n_head after growth."""
        self.model.maybe_grow_architecture(25000)
        self.assertEqual(CFG.head_types, ("content", "hybrid"))
        self.assertEqual(self.model.n_head, 2)

    def test_delta_adapters_grown(self):
        """Delta adapters should have correct dimensions after growth."""
        self.model.maybe_grow_architecture(25000)
        mod = self.model.deltas[0]
        # wq adapter: A should be (32, rank), B should be (rank, 32)
        self.assertEqual(mod["l0.wq"].A.nout, 32)
        self.assertEqual(mod["l0.wq"].B.nin, 32)

    def test_new_layers_added(self):
        """Growing to child should add layer 1."""
        self.model.maybe_grow_architecture(60000)
        self.assertIn("l1.wq", self.model.base)
        self.assertIn("l1.fc_g", self.model.base)
        # Delta should also have new layer
        self.assertIn("l1.wq", self.model.deltas[0])

    def test_adam_reset_after_growth(self):
        """Adam state should be empty after growth."""
        self.model._adam = {"some_key": "some_val"}
        self.model.maybe_grow_architecture(25000)
        self.assertEqual(self.model._adam, {})

    def test_freeze_after_growth(self):
        """_growth_freeze_remaining should be set after growth."""
        self.model.maybe_grow_architecture(25000)
        self.assertEqual(self.model._growth_freeze_remaining, CFG.freeze_after_growth_steps)

    def test_gamma_snapshot_extended(self):
        """Gamma snapshot should have new dimension after growth."""
        self.model.maybe_grow_architecture(25000)
        for row in self.model._init_embed_snapshot:
            self.assertEqual(len(row), 32)

    def test_residual_alpha_updated(self):
        """residual_alpha should reflect new n_layer."""
        import math
        self.model.maybe_grow_architecture(60000)  # child: 2 layers
        expected = 1.0 / math.sqrt(2)
        self.assertAlmostEqual(self.model.residual_alpha, expected)

    def test_generation_after_growth(self):
        """Model should generate text after growth."""
        self.model.maybe_grow_architecture(25000)
        text = self.model.generate_sentence("Hello")
        self.assertIsInstance(text, str)

    def test_legacy_checkpoint_skips_growth(self):
        """Model with non-standard dimensions should skip growth."""
        self.model.n_embd = 72
        self.model.n_layer = 2
        self.model.n_head = 4
        self.assertEqual(self.model.current_growth_stage(), -1)
        grew = self.model.maybe_grow_architecture(100000)
        self.assertFalse(grew)

    def tearDown(self):
        # Restore embryo defaults for other tests
        CFG.n_embd = 16
        CFG.n_layer = 1
        CFG.n_head = 1
        CFG.head_types = ("content",)


class TestCheckpointWithGrowth(unittest.TestCase):

    def setUp(self):
        CFG.n_embd = 16
        CFG.n_layer = 1
        CFG.n_head = 1
        CFG.head_types = ("content",)
        self.docs = ["Hello world."]
        self.tok = EvolvingTokenizer(self.docs)
        self.model = GPT(self.tok)
        self.ckpt_path = "/tmp/test_growth_ckpt.json"

    def test_save_load_preserves_grown_dimensions(self):
        """Checkpoint should restore correct dimensions after growth."""
        self.model.maybe_grow_architecture(25000)  # grow to infant
        self.assertEqual(self.model.n_embd, 32)

        save_checkpoint(self.model, self.tok, self.ckpt_path)

        # Reset CFG to embryo
        CFG.n_embd = 16
        CFG.n_layer = 1
        CFG.n_head = 1
        CFG.head_types = ("content",)

        loaded_model, loaded_tok = load_checkpoint(self.docs, self.ckpt_path)
        self.assertIsNotNone(loaded_model)
        self.assertEqual(loaded_model.n_embd, 32)
        self.assertEqual(loaded_model.n_layer, 1)
        self.assertEqual(loaded_model.n_head, 2)

    def tearDown(self):
        if os.path.exists(self.ckpt_path):
            os.remove(self.ckpt_path)
        CFG.n_embd = 16
        CFG.n_layer = 1
        CFG.n_head = 1
        CFG.head_types = ("content",)


if __name__ == "__main__":
    unittest.main()
