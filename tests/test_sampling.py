#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for sampling functions (top_k, top_p, softmax).
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from molequla import softmax_probs_float, top_k_top_p_sample


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


class TestMinPTypicalP(unittest.TestCase):
    """Tests for min_p and typical_p sampling."""

    def test_min_p_filters_low_probs(self):
        """Min-p should filter tokens with prob < min_p * max_prob."""
        probs = [0.01, 0.02, 0.07, 0.9]  # max = 0.9
        # With min_p=0.1, threshold = 0.09
        # Only index 3 (0.9 >= 0.09) should remain
        samples = [top_k_top_p_sample(probs, k=0, p=1.0, min_p=0.1) for _ in range(20)]
        self.assertTrue(all(s == 3 for s in samples))

    def test_min_p_keeps_proportional(self):
        """Min-p should keep tokens proportionally above threshold."""
        probs = [0.05, 0.15, 0.30, 0.50]  # max = 0.5
        # With min_p=0.2, threshold = 0.1
        # Indices 1 (0.15), 2 (0.30), 3 (0.50) should remain
        samples = [top_k_top_p_sample(probs, k=0, p=1.0, min_p=0.2) for _ in range(100)]
        # Index 0 (0.05) should never be sampled
        self.assertTrue(0 not in samples)

    def test_typical_p_prefers_typical(self):
        """Typical-p should prefer tokens with typical information content."""
        # With uniform probs, all tokens are equally typical
        probs = [0.25, 0.25, 0.25, 0.25]
        samples = [top_k_top_p_sample(probs, k=0, p=1.0, typical_p=0.9) for _ in range(100)]
        # All indices should be possible
        unique = set(samples)
        self.assertGreaterEqual(len(unique), 3)  # Should see at least 3 different indices

    def test_typical_p_with_varied_probs(self):
        """Typical-p should work with varied probability distributions."""
        probs = [0.01, 0.09, 0.30, 0.60]
        # Should still produce valid samples
        samples = [top_k_top_p_sample(probs, k=0, p=1.0, typical_p=0.8) for _ in range(50)]
        # All samples should be valid indices
        self.assertTrue(all(0 <= s < len(probs) for s in samples))

    def test_combined_min_p_typical_p(self):
        """Combined min_p and typical_p should work together."""
        probs = [0.02, 0.08, 0.20, 0.70]
        # min_p=0.1 with max=0.7 => threshold=0.07
        # Keeps indices 1, 2, 3 (0.08, 0.20, 0.70)
        # typical_p=0.9 then selects typical tokens from those
        samples = [top_k_top_p_sample(probs, k=0, p=1.0, min_p=0.1, typical_p=0.9) for _ in range(50)]
        # Index 0 should never be sampled (below min_p threshold)
        self.assertTrue(0 not in samples)


if __name__ == "__main__":
    unittest.main()
