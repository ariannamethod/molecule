#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for sampling functions (top_k, top_p, min_p, typical_p).
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from macrogpt import (
    softmax_probs_float, top_k_top_p_sample, 
    apply_min_p_filter, typical_indices, sample_with_filters
)


class TestSoftmax(unittest.TestCase):
    """Tests for softmax function."""

    def test_softmax_sums_to_one(self):
        """Softmax output should sum to 1."""
        logits = [1.0, 2.0, 3.0, 4.0]
        probs = softmax_probs_float(logits)
        self.assertAlmostEqual(sum(probs), 1.0, places=6)

    def test_softmax_positive(self):
        """All softmax values should be positive."""
        logits = [-10.0, 0.0, 10.0]
        probs = softmax_probs_float(logits)
        self.assertTrue(all(p >= 0 for p in probs))

    def test_softmax_ordering(self):
        """Higher logits should give higher probs."""
        logits = [1.0, 2.0, 3.0]
        probs = softmax_probs_float(logits)
        self.assertLess(probs[0], probs[1])
        self.assertLess(probs[1], probs[2])


class TestTopKTopP(unittest.TestCase):
    """Tests for top-k and top-p sampling."""

    def test_top_k_limits_candidates(self):
        """Top-k should consider only top k tokens."""
        probs = [0.1, 0.2, 0.3, 0.4]
        # With k=2, only indices 2 and 3 should be possible
        samples = [top_k_top_p_sample(probs, k=2, p=1.0) for _ in range(100)]
        self.assertTrue(all(s in [2, 3] for s in samples))

    def test_top_p_limits_candidates(self):
        """Top-p should consider tokens until cumsum >= p."""
        probs = [0.01, 0.04, 0.15, 0.8]  # sorted desc: [0.8, 0.15, 0.04, 0.01]
        # With p=0.9, indices 3 and 2 should be possible (0.8 + 0.15 > 0.9)
        samples = [top_k_top_p_sample(probs, k=0, p=0.9) for _ in range(100)]
        # Most samples should be 3 (highest prob)
        count_3 = samples.count(3)
        self.assertGreater(count_3, 50)

    def test_greedy_with_k1(self):
        """With k=1, should always pick argmax."""
        probs = [0.1, 0.2, 0.05, 0.65]
        samples = [top_k_top_p_sample(probs, k=1, p=1.0) for _ in range(10)]
        self.assertTrue(all(s == 3 for s in samples))


class TestMinP(unittest.TestCase):
    """Tests for min_p filtering."""

    def test_min_p_filters_low(self):
        """min_p should zero out tokens below threshold."""
        probs = [0.5, 0.3, 0.15, 0.05]
        # min_p=0.2 means threshold = 0.2 * 0.5 = 0.1
        filtered = apply_min_p_filter(probs, min_p=0.2)
        # 0.05 < 0.1, so should be zeroed
        self.assertAlmostEqual(filtered[3], 0.0)

    def test_min_p_renormalizes(self):
        """After filtering, probs should still sum to 1."""
        probs = [0.5, 0.3, 0.15, 0.05]
        filtered = apply_min_p_filter(probs, min_p=0.2)
        self.assertAlmostEqual(sum(filtered), 1.0, places=6)

    def test_min_p_zero_disabled(self):
        """min_p=0 should not filter anything."""
        probs = [0.5, 0.3, 0.15, 0.05]
        filtered = apply_min_p_filter(probs, min_p=0.0)
        for i in range(len(probs)):
            self.assertAlmostEqual(filtered[i], probs[i])


class TestTypicalSampling(unittest.TestCase):
    """Tests for typical sampling."""

    def test_typical_indices_returns_list(self):
        """Should return a list of indices."""
        probs = [0.25, 0.25, 0.25, 0.25]
        indices = typical_indices(probs, typical_p=0.9)
        self.assertIsInstance(indices, list)
        self.assertGreater(len(indices), 0)

    def test_typical_high_p_all_indices(self):
        """typical_p >= 1.0 should return all indices."""
        probs = [0.25, 0.25, 0.25, 0.25]
        indices = typical_indices(probs, typical_p=1.0)
        self.assertEqual(sorted(indices), [0, 1, 2, 3])

    def test_typical_prefers_typical_tokens(self):
        """Typical sampling should prefer tokens with surprisal near entropy."""
        # Entropy is -sum(p*log(p)); for uniform, log(1/n) per token
        probs = [0.25, 0.25, 0.25, 0.25]
        indices = typical_indices(probs, typical_p=0.5)
        # All tokens equally "typical" for uniform, so any selection valid
        self.assertGreater(len(indices), 0)


class TestSampleWithFilters(unittest.TestCase):
    """Tests for the full sampling pipeline."""

    def test_sample_returns_valid_index(self):
        """Sample should return valid token index."""
        probs = [0.1, 0.2, 0.3, 0.4]
        for _ in range(100):
            idx = sample_with_filters(probs, k=4, p=1.0, min_p=0.0, typical_p=1.0)
            self.assertIn(idx, [0, 1, 2, 3])

    def test_sample_with_all_filters(self):
        """All filters together should work."""
        probs = [0.5, 0.3, 0.15, 0.05]
        for _ in range(50):
            idx = sample_with_filters(probs, k=2, p=0.9, min_p=0.1, typical_p=0.95)
            self.assertIn(idx, range(len(probs)))


if __name__ == "__main__":
    unittest.main()
