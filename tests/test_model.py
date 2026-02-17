#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the GPT model (MatrixParam, GPT, generation).
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from macrogpt import (
    MatrixParam, VectorValue, GPT, EvolvingTokenizer, 
    DeltaAdapter, CFG, backward, rmsnorm, rope_rotate
)


class TestMatrixParam(unittest.TestCase):
    """Tests for MatrixParam (trainable weight matrix)."""

    def test_init_shape(self):
        """Matrix should have correct dimensions."""
        m = MatrixParam(10, 5, std=0.1)
        self.assertEqual(m.nout, 10)
        self.assertEqual(m.nin, 5)
        self.assertEqual(len(m.rows), 10)
        self.assertEqual(len(m.rows[0].data), 5)

    def test_matvec(self):
        """Matrix-vector multiplication should work."""
        m = MatrixParam(2, 3, std=0.0)
        # Set weights manually
        m.rows[0].data = [1.0, 0.0, 0.0]
        m.rows[1].data = [0.0, 1.0, 0.0]
        x = VectorValue([1.0, 2.0, 3.0])
        y = m.matvec(x)
        # Row 0 dot [1,2,3] = 1, Row 1 dot [1,2,3] = 2
        self.assertAlmostEqual(y.data[0], 1.0)
        self.assertAlmostEqual(y.data[1], 2.0)

    def test_grow_rows(self):
        """Growing rows should extend matrix."""
        m = MatrixParam(2, 3, std=0.1)
        m.grow_rows(5, std=0.1)
        self.assertEqual(m.nout, 5)
        self.assertEqual(len(m.rows), 5)

    def test_params_list(self):
        """params() should return all row vectors."""
        m = MatrixParam(3, 2, std=0.1)
        params = m.params()
        self.assertEqual(len(params), 3)


class TestDeltaAdapter(unittest.TestCase):
    """Tests for LoRA-style delta adapters."""

    def test_init(self):
        """Adapter should initialize with correct shapes."""
        da = DeltaAdapter(nout=16, nin=8, r=4)
        self.assertEqual(da.A.nout, 16)
        self.assertEqual(da.A.nin, 4)
        self.assertEqual(da.B.nout, 4)
        self.assertEqual(da.B.nin, 8)

    def test_apply(self):
        """Adapter should produce output of correct dimension."""
        da = DeltaAdapter(nout=16, nin=8, r=4)
        x = VectorValue([1.0] * 8)
        y = da.apply(x)
        self.assertEqual(len(y.data), 16)

    def test_grow_out(self):
        """Growing output dimension should work."""
        da = DeltaAdapter(nout=16, nin=8, r=4)
        da.maybe_grow_out(24)
        self.assertEqual(da.A.nout, 24)


class TestGPT(unittest.TestCase):
    """Tests for the GPT model."""

    def setUp(self):
        self.docs = ["Hello.", "World."]
        self.tok = EvolvingTokenizer(self.docs)
        self.model = GPT(self.tok)

    def test_init(self):
        """GPT should initialize with correct structure."""
        self.assertEqual(self.model.n_layer, CFG.n_layer)
        self.assertEqual(self.model.n_embd, CFG.n_embd)
        self.assertEqual(self.model.n_head, CFG.n_head)
        self.assertIn("wte", self.model.base)
        self.assertIn("wpe", self.model.base)
        self.assertIn("lm_head", self.model.base)

    def test_vocab_expansion(self):
        """Model should handle vocab expansion."""
        old_vocab = self.tok.vocab_size
        new_size = old_vocab + 10
        self.model.maybe_expand_vocab(new_size)
        self.assertEqual(self.model.base["wte"].nout, new_size)

    def test_add_delta_module(self):
        """Adding delta modules should work."""
        initial = len(self.model.deltas)
        self.model.add_delta_module(alpha=0.5)
        self.assertEqual(len(self.model.deltas), initial + 1)
        self.assertIn(0.5, self.model.active_alpha)

    def test_forward_step(self):
        """Forward step should produce logits."""
        keys = [[] for _ in range(self.model.n_layer)]
        values = [[] for _ in range(self.model.n_layer)]
        tok_id = self.tok.stoi[self.tok.BOS]
        logits = self.model.forward_step(tok_id, 0, keys, values)
        self.assertEqual(len(logits.data), self.tok.vocab_size)

    def test_loss_computation(self):
        """Loss computation should return a scalar."""
        ids = self.tok.encode("Hello")
        loss = self.model.loss_on_sequence(ids)
        self.assertIsInstance(loss.data, float)
        self.assertGreater(loss.data, 0)

    def test_generate_produces_text(self):
        """Generation should produce non-empty text."""
        text = self.model.generate_sentence("Hello")
        self.assertIsInstance(text, str)

    def test_all_base_params(self):
        """all_base_params should return a list."""
        params = self.model.all_base_params()
        self.assertIsInstance(params, list)
        self.assertGreater(len(params), 0)

    def test_all_delta_params(self):
        """all_delta_params should return a list."""
        params = self.model.all_delta_params()
        self.assertIsInstance(params, list)


class TestRMSNorm(unittest.TestCase):
    """Tests for RMS normalization."""

    def test_rmsnorm_normalizes(self):
        """RMSNorm should normalize vector magnitude."""
        x = VectorValue([3.0, 4.0])  # Norm = 5
        y = rmsnorm(x)
        # After normalization, mean square should be close to 1
        ms = sum(v * v for v in y.data) / len(y.data)
        self.assertAlmostEqual(ms, 1.0, places=5)


class TestRoPE(unittest.TestCase):
    """Tests for Rotary Position Embedding."""

    def test_rope_rotate_shape(self):
        """RoPE should preserve vector shape."""
        x = VectorValue([1.0, 0.0, 1.0, 0.0])
        y = rope_rotate(x, pos=0, head_dim=4)
        self.assertEqual(len(y.data), 4)

    def test_rope_position_0(self):
        """At position 0, rotation should be identity (theta=0)."""
        x = VectorValue([1.0, 0.0])
        y = rope_rotate(x, pos=0, head_dim=2)
        self.assertAlmostEqual(y.data[0], 1.0, places=5)
        self.assertAlmostEqual(y.data[1], 0.0, places=5)

    def test_rope_gradient_flows(self):
        """Gradients should flow through RoPE."""
        x = VectorValue([1.0, 1.0])
        y = rope_rotate(x, pos=5, head_dim=2)
        s = y.dot(VectorValue([1.0, 1.0]))
        backward(s)
        # Gradients should be non-zero
        self.assertTrue(any(g != 0 for g in x.grad))


if __name__ == "__main__":
    unittest.main()
