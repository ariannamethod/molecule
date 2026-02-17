#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the custom autograd engine (VectorValue, ScalarValue).
"""

import unittest
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from macrogpt import VectorValue, ScalarValue, backward


class TestVectorValue(unittest.TestCase):
    """Tests for VectorValue operations."""

    def test_add_vectors(self):
        """Vector addition should work correctly."""
        a = VectorValue([1.0, 2.0, 3.0])
        b = VectorValue([4.0, 5.0, 6.0])
        c = a + b
        self.assertEqual(c.data, [5.0, 7.0, 9.0])

    def test_add_scalar(self):
        """Vector + scalar should add to all elements."""
        a = VectorValue([1.0, 2.0, 3.0])
        c = a + 10.0
        self.assertEqual(c.data, [11.0, 12.0, 13.0])

    def test_sub_vectors(self):
        """Vector subtraction should work correctly."""
        a = VectorValue([5.0, 7.0, 9.0])
        b = VectorValue([1.0, 2.0, 3.0])
        c = a - b
        self.assertEqual(c.data, [4.0, 5.0, 6.0])

    def test_mul_vectors(self):
        """Element-wise vector multiplication."""
        a = VectorValue([2.0, 3.0, 4.0])
        b = VectorValue([1.0, 2.0, 3.0])
        c = a * b
        self.assertEqual(c.data, [2.0, 6.0, 12.0])

    def test_mul_scalar(self):
        """Vector * scalar should scale all elements."""
        a = VectorValue([1.0, 2.0, 3.0])
        c = a * 2.0
        self.assertEqual(c.data, [2.0, 4.0, 6.0])

    def test_neg(self):
        """Negation should flip signs."""
        a = VectorValue([1.0, -2.0, 3.0])
        c = -a
        self.assertEqual(c.data, [-1.0, 2.0, -3.0])

    def test_relu(self):
        """ReLU should zero out negative values."""
        a = VectorValue([-1.0, 0.0, 2.0, -3.0, 4.0])
        c = a.relu()
        self.assertEqual(c.data, [0.0, 0.0, 2.0, 0.0, 4.0])

    def test_squared_relu(self):
        """Squared ReLU should square positive values, zero negatives."""
        a = VectorValue([-1.0, 0.0, 2.0, 3.0])
        c = a.squared_relu()
        self.assertEqual(c.data, [0.0, 0.0, 4.0, 9.0])

    def test_dot_product(self):
        """Dot product should sum element-wise products."""
        a = VectorValue([1.0, 2.0, 3.0])
        b = VectorValue([4.0, 5.0, 6.0])
        c = a.dot(b)
        self.assertEqual(c.data, 32.0)  # 1*4 + 2*5 + 3*6 = 32

    def test_mean_sq(self):
        """Mean squared should compute mean of squared elements."""
        a = VectorValue([1.0, 2.0, 3.0])
        c = a.mean_sq()
        expected = (1 + 4 + 9) / 3
        self.assertAlmostEqual(c.data, expected)

    def test_slice(self):
        """Slice should extract a subvector."""
        a = VectorValue([1.0, 2.0, 3.0, 4.0, 5.0])
        c = a.slice(1, 4)
        self.assertEqual(c.data, [2.0, 3.0, 4.0])

    def test_concat(self):
        """Concat should join vectors."""
        a = VectorValue([1.0, 2.0])
        b = VectorValue([3.0, 4.0, 5.0])
        c = VectorValue.concat([a, b])
        self.assertEqual(c.data, [1.0, 2.0, 3.0, 4.0, 5.0])


class TestScalarValue(unittest.TestCase):
    """Tests for ScalarValue operations."""

    def test_add_scalars(self):
        """Scalar addition."""
        a = ScalarValue(3.0)
        b = ScalarValue(4.0)
        c = a + b
        self.assertEqual(c.data, 7.0)

    def test_mul_scalars(self):
        """Scalar multiplication."""
        a = ScalarValue(3.0)
        b = ScalarValue(4.0)
        c = a * b
        self.assertEqual(c.data, 12.0)

    def test_sub_scalars(self):
        """Scalar subtraction."""
        a = ScalarValue(10.0)
        b = ScalarValue(4.0)
        c = a - b
        self.assertEqual(c.data, 6.0)


class TestBackward(unittest.TestCase):
    """Tests for backward pass / gradient computation."""

    def test_simple_add_grad(self):
        """Gradient of sum should flow to both inputs."""
        a = VectorValue([1.0, 2.0])
        b = VectorValue([3.0, 4.0])
        c = a + b
        s = c.dot(VectorValue([1.0, 1.0]))  # Sum all elements
        backward(s)
        self.assertEqual(a.grad, [1.0, 1.0])
        self.assertEqual(b.grad, [1.0, 1.0])

    def test_mul_grad(self):
        """Gradient of product."""
        a = VectorValue([2.0, 3.0])
        b = VectorValue([4.0, 5.0])
        c = a * b
        s = c.dot(VectorValue([1.0, 1.0]))
        backward(s)
        # d(a*b)/da = b
        self.assertEqual(a.grad, [4.0, 5.0])
        self.assertEqual(b.grad, [2.0, 3.0])

    def test_relu_grad(self):
        """ReLU gradient: 1 for positive, 0 for negative."""
        a = VectorValue([-1.0, 2.0, -3.0, 4.0])
        c = a.relu()
        s = c.dot(VectorValue([1.0, 1.0, 1.0, 1.0]))
        backward(s)
        self.assertEqual(a.grad, [0.0, 1.0, 0.0, 1.0])

    def test_chain_grad(self):
        """Chain rule: (a + b) * c."""
        a = VectorValue([1.0])
        b = VectorValue([2.0])
        c = VectorValue([3.0])
        ab = a + b
        abc = ab * c
        s = abc.dot(VectorValue([1.0]))
        backward(s)
        # d/da = c = 3
        # d/db = c = 3
        # d/dc = (a+b) = 3
        self.assertAlmostEqual(a.grad[0], 3.0)
        self.assertAlmostEqual(b.grad[0], 3.0)
        self.assertAlmostEqual(c.grad[0], 3.0)


if __name__ == "__main__":
    unittest.main()
