"""Tests for SyntropyTracker — mathematical self-reasoning engine."""

import sys
import os
import unittest
import sqlite3
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from molecule import (
    GPT, EvolvingTokenizer, CooccurField, SyntropyTracker,
    train_steps, init_db, CFG, no_grad
)
import numpy as np


def _make_model_and_field():
    docs = [
        "H: Who are you?",
        "A: I am molecule.",
        "H: What is syntropy?",
        "A: The opposite of decay.",
        "H: How do you learn?",
        "A: Through dialogue and training.",
        "H: What is resonance?",
        "A: Amplification through alignment.",
    ]
    tok = EvolvingTokenizer(docs)
    model = GPT(tok)
    field = CooccurField()
    field.build_from_corpus(tok, docs)
    return model, tok, docs, field


class TestFieldDeviation(unittest.TestCase):

    def test_field_deviation_returns_float(self):
        """compute_field_deviation should return a non-negative float."""
        model, tok, docs, field = _make_model_and_field()
        dev = model.compute_field_deviation(tok, field, docs)
        self.assertIsInstance(dev, float)
        self.assertGreaterEqual(dev, 0.0)

    def test_field_deviation_zero_on_empty(self):
        """Field deviation should be 0 with empty docs."""
        model, tok, _, field = _make_model_and_field()
        dev = model.compute_field_deviation(tok, field, [])
        self.assertEqual(dev, 0.0)

    def test_field_deviation_zero_empty_field(self):
        """Field deviation should be 0 with empty field."""
        model, tok, docs, _ = _make_model_and_field()
        empty_field = CooccurField()
        dev = model.compute_field_deviation(tok, empty_field, docs)
        self.assertEqual(dev, 0.0)

    def test_field_deviation_changes_after_training(self):
        """Field deviation should change after training moves the weights."""
        model, tok, docs, field = _make_model_and_field()
        dev_before = model.compute_field_deviation(tok, field, docs)
        train_steps(model, tok, docs, steps=100, train_base=True, train_deltas=True)
        dev_after = model.compute_field_deviation(tok, field, docs)
        # They should differ (training moved weights)
        self.assertNotAlmostEqual(dev_before, dev_after, places=2)


class TestModelEntropy(unittest.TestCase):

    def test_model_entropy_returns_float(self):
        """compute_model_entropy should return a non-negative float."""
        model, tok, docs, _ = _make_model_and_field()
        ent = model.compute_model_entropy(tok, docs)
        self.assertIsInstance(ent, float)
        self.assertGreaterEqual(ent, 0.0)

    def test_model_entropy_zero_on_empty(self):
        """Model entropy should be 0 with empty docs."""
        model, tok, _, _ = _make_model_and_field()
        ent = model.compute_model_entropy(tok, [])
        self.assertEqual(ent, 0.0)

    def test_model_entropy_decreases_with_training(self):
        """Entropy should generally decrease after training on same data (syntropy)."""
        model, tok, docs, _ = _make_model_and_field()
        ent_before = model.compute_model_entropy(tok, docs)
        train_steps(model, tok, docs, steps=50, train_base=True, train_deltas=True)
        ent_after = model.compute_model_entropy(tok, docs)
        # Entropy should drop (model becomes more confident on its training data)
        self.assertLess(ent_after, ent_before)


class TestPurposeVector(unittest.TestCase):

    def test_purpose_vector_returns_tuple(self):
        """compute_purpose_vector should return (vector, magnitude)."""
        model, _, _, _ = _make_model_and_field()
        result = model.compute_purpose_vector()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_purpose_vector_has_magnitude(self):
        """Purpose vector should have non-negative magnitude."""
        model, _, _, _ = _make_model_and_field()
        _, mag = model.compute_purpose_vector()
        self.assertIsInstance(mag, float)
        self.assertGreaterEqual(mag, 0.0)

    def test_purpose_vector_unit_length(self):
        """Purpose direction should be approximately unit length when magnitude > 0."""
        model, tok, docs, _ = _make_model_and_field()
        train_steps(model, tok, docs, steps=10, train_base=True, train_deltas=True)
        direction, mag = model.compute_purpose_vector()
        if direction is not None and mag > 1e-10:
            norm = float(np.linalg.norm(direction))
            self.assertAlmostEqual(norm, 1.0, places=4)


class TestPurposeGammaAlignment(unittest.TestCase):

    def test_alignment_returns_float(self):
        """purpose_gamma_alignment should return a float."""
        model, _, _, _ = _make_model_and_field()
        alignment = model.purpose_gamma_alignment()
        self.assertIsInstance(alignment, float)

    def test_alignment_in_range(self):
        """Alignment should be between -1 and 1 (cosine similarity)."""
        model, tok, docs, _ = _make_model_and_field()
        train_steps(model, tok, docs, steps=30, train_base=True, train_deltas=True)
        alignment = model.purpose_gamma_alignment()
        self.assertGreaterEqual(alignment, -1.0)
        self.assertLessEqual(alignment, 1.0)


class TestSyntropyTracker(unittest.TestCase):

    def test_tracker_creation(self):
        """SyntropyTracker should initialize with clean state."""
        st = SyntropyTracker()
        self.assertEqual(st.entropy_history, [])
        self.assertEqual(st.syntropy_trend, 0.0)
        self.assertEqual(st.field_deviation, 0.0)
        self.assertEqual(st.purpose_magnitude, 0.0)
        self.assertEqual(st.purpose_alignment, 0.0)
        self.assertEqual(st.last_action, "none")

    def test_measure_returns_dict(self):
        """measure() should return dict with all metrics."""
        model, tok, docs, field = _make_model_and_field()
        st = SyntropyTracker()
        metrics = st.measure(model, tok, field, docs)
        self.assertIn("entropy", metrics)
        self.assertIn("syntropy_trend", metrics)
        self.assertIn("field_deviation", metrics)
        self.assertIn("purpose_magnitude", metrics)
        self.assertIn("purpose_alignment", metrics)

    def test_measure_populates_entropy_history(self):
        """Each measure() call should append to entropy_history."""
        model, tok, docs, field = _make_model_and_field()
        st = SyntropyTracker()
        st.measure(model, tok, field, docs)
        self.assertEqual(len(st.entropy_history), 1)
        st.measure(model, tok, field, docs)
        self.assertEqual(len(st.entropy_history), 2)

    def test_entropy_history_bounded(self):
        """entropy_history should not exceed syntropy_window."""
        model, tok, docs, field = _make_model_and_field()
        st = SyntropyTracker()
        for _ in range(CFG.syntropy_window + 5):
            st.measure(model, tok, field, docs)
        self.assertLessEqual(len(st.entropy_history), CFG.syntropy_window)

    def test_decide_action_returns_dict(self):
        """decide_action() should return dict with lr_multiplier, delta_grow_override, action."""
        st = SyntropyTracker()
        decision = st.decide_action()
        self.assertIn("lr_multiplier", decision)
        self.assertIn("delta_grow_override", decision)
        self.assertIn("action", decision)

    def test_decide_action_steady_by_default(self):
        """With neutral metrics, action should not be destructive."""
        st = SyntropyTracker()
        st.syntropy_trend = 0.0
        st.field_deviation = 1.0  # in the sweet spot
        st.purpose_alignment = 0.0
        decision = st.decide_action()
        # lr_multiplier should be reasonable (0.5 to 2.0)
        self.assertGreater(decision["lr_multiplier"], 0.0)
        self.assertLess(decision["lr_multiplier"], 3.0)

    def test_decide_amplify_when_syntropy_rising(self):
        """When syntropy is rising and purpose aligned, should amplify."""
        st = SyntropyTracker()
        st.syntropy_trend = 0.1  # rising
        st.field_deviation = 2.0  # sweet spot
        st.purpose_alignment = 0.5  # well aligned
        decision = st.decide_action()
        self.assertEqual(decision["action"], "amplify")
        self.assertGreater(decision["lr_multiplier"], 1.0)

    def test_decide_dampen_when_syntropy_falling(self):
        """When syntropy is falling, should dampen learning."""
        st = SyntropyTracker()
        st.syntropy_trend = -0.1  # falling
        st.field_deviation = 2.0
        decision = st.decide_action()
        self.assertEqual(decision["action"], "dampen")
        self.assertLess(decision["lr_multiplier"], 1.0)

    def test_decide_ground_when_deviation_high(self):
        """When field deviation is too high, should ground."""
        st = SyntropyTracker()
        st.syntropy_trend = 0.0
        st.field_deviation = CFG.field_deviation_ceiling + 1.0
        decision = st.decide_action()
        self.assertEqual(decision["action"], "ground")

    def test_decide_explore_when_deviation_low(self):
        """When field deviation is too low, should explore."""
        st = SyntropyTracker()
        st.syntropy_trend = 0.0
        st.field_deviation = CFG.field_deviation_floor * 0.5
        decision = st.decide_action()
        self.assertEqual(decision["action"], "explore")

    def test_decide_realign_when_purpose_opposes_gamma(self):
        """When purpose opposes gamma, should realign."""
        st = SyntropyTracker()
        st.syntropy_trend = 0.0
        st.field_deviation = 2.0
        st.purpose_alignment = -0.5  # opposing identity
        decision = st.decide_action()
        self.assertEqual(decision["action"], "realign")
        self.assertLess(decision["lr_multiplier"], 1.0)


class TestSyntropyLogDB(unittest.TestCase):

    def test_log_to_db_creates_entry(self):
        """log_to_db should write to syntropy_log table."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            con = init_db(db_path)
            st = SyntropyTracker()
            st.syntropy_trend = 0.05
            st.field_deviation = 3.2
            st.purpose_magnitude = 0.8
            st.purpose_alignment = 0.4
            st.log_to_db(con, 2.5, 2.3, "boost")

            cur = con.cursor()
            cur.execute("SELECT * FROM syntropy_log")
            rows = cur.fetchall()
            self.assertEqual(len(rows), 1)
            row = rows[0]
            # id, ts, entropy_before, entropy_after, syntropy_delta,
            # field_deviation, purpose_magnitude, purpose_alignment, action_taken, note
            self.assertAlmostEqual(row[2], 2.5, places=1)  # entropy_before
            self.assertAlmostEqual(row[3], 2.3, places=1)  # entropy_after
            self.assertEqual(row[8], "boost")  # action_taken
            con.close()
        finally:
            os.unlink(db_path)

    def test_syntropy_log_table_exists(self):
        """init_db should create syntropy_log table."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            con = init_db(db_path)
            cur = con.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='syntropy_log'")
            result = cur.fetchone()
            self.assertIsNotNone(result)
            con.close()
        finally:
            os.unlink(db_path)


if __name__ == "__main__":
    unittest.main()
