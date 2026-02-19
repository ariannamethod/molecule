"""Tests for QuantumBuffer — smart training trigger."""

import sys
import os
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from molequla import QuantumBuffer, EvolvingTokenizer, CFG


def _make_tok():
    docs = ["Hello world", "Testing quantum buffer"]
    return EvolvingTokenizer(docs), docs


class TestQuantumBuffer(unittest.TestCase):

    def test_initial_state(self):
        """Fresh buffer should not trigger."""
        qb = QuantumBuffer()
        self.assertEqual(qb.accumulated_bytes, 0)
        self.assertEqual(qb.total_tokens, 0)
        self.assertEqual(len(qb.unique_tokens), 0)
        self.assertFalse(qb.should_trigger())

    def test_novelty_score_zero(self):
        """Novelty score should be 0 when nothing has been fed."""
        qb = QuantumBuffer()
        self.assertEqual(qb.novelty_score(), 0.0)

    def test_feed_accumulates_bytes(self):
        """Feed should add to accumulated_bytes."""
        tok, docs = _make_tok()
        qb = QuantumBuffer()
        qb.feed(500, tok, docs)
        self.assertEqual(qb.accumulated_bytes, 500)
        qb.feed(300, tok, docs)
        self.assertEqual(qb.accumulated_bytes, 800)

    def test_feed_accumulates_tokens(self):
        """Feed should track unique and total tokens."""
        tok, docs = _make_tok()
        qb = QuantumBuffer()
        qb.feed(100, tok, docs)
        self.assertGreater(qb.total_tokens, 0)
        self.assertGreater(len(qb.unique_tokens), 0)

    def test_novelty_score_range(self):
        """Novelty should be between 0 and 1."""
        tok, docs = _make_tok()
        qb = QuantumBuffer()
        qb.feed(100, tok, docs)
        nov = qb.novelty_score()
        self.assertGreaterEqual(nov, 0.0)
        self.assertLessEqual(nov, 1.0)

    def test_trigger_requires_bytes_or_novelty(self):
        """Should trigger only when bytes >= qb_min_bytes or novelty >= qb_min_novelty."""
        tok, docs = _make_tok()
        qb = QuantumBuffer()
        qb.last_burst_time = 0  # force cooldown to be OK
        # Feed small amount
        qb.feed(10, tok, docs)
        # May or may not trigger based on novelty
        # Feed enough bytes to guarantee
        qb.feed(CFG.qb_min_bytes + 100, tok, docs)
        self.assertTrue(qb.should_trigger())

    def test_trigger_respects_cooldown(self):
        """Should not trigger during cooldown period."""
        tok, docs = _make_tok()
        qb = QuantumBuffer()
        qb.feed(CFG.qb_min_bytes + 100, tok, docs)
        # Set last burst time to now (in cooldown)
        qb.last_burst_time = time.time()
        self.assertFalse(qb.should_trigger())

    def test_trigger_after_cooldown(self):
        """Should trigger after cooldown expires."""
        tok, docs = _make_tok()
        qb = QuantumBuffer()
        qb.feed(CFG.qb_min_bytes + 100, tok, docs)
        # Set last burst time to far in the past
        qb.last_burst_time = time.time() - CFG.qb_cooldown_seconds - 10
        self.assertTrue(qb.should_trigger())

    def test_reset_clears_state(self):
        """Reset should clear accumulated state and set burst time."""
        tok, docs = _make_tok()
        qb = QuantumBuffer()
        qb.feed(1000, tok, docs)
        self.assertGreater(qb.accumulated_bytes, 0)
        qb.reset()
        self.assertEqual(qb.accumulated_bytes, 0)
        self.assertEqual(qb.total_tokens, 0)
        self.assertEqual(len(qb.unique_tokens), 0)
        self.assertGreater(qb.last_burst_time, 0)

    def test_novelty_decreases_with_repetition(self):
        """Feeding same docs repeatedly should decrease novelty (same unique, more total)."""
        tok, docs = _make_tok()
        qb = QuantumBuffer()
        qb.feed(100, tok, docs)
        nov1 = qb.novelty_score()
        # Feed same docs again (unique stays same, total increases)
        qb.feed(100, tok, docs)
        nov2 = qb.novelty_score()
        self.assertLessEqual(nov2, nov1)


if __name__ == "__main__":
    unittest.main()
