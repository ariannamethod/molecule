#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for checkpointing (save/load model state).
"""

import unittest
import json
import tempfile
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from macrogpt import GPT, EvolvingTokenizer, save_checkpoint, load_checkpoint


class TestCheckpointing(unittest.TestCase):
    """Tests for save/load checkpoint functionality."""

    def setUp(self):
        self.docs = ["Hello world.", "Testing checkpoints.", "Save and load."]
        self.tok = EvolvingTokenizer(self.docs)
        self.model = GPT(self.tok)
        # Use NamedTemporaryFile for secure temp file creation
        self.tmp_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.tmp_path = self.tmp_file.name
        self.tmp_file.close()

    def tearDown(self):
        if os.path.exists(self.tmp_path):
            os.unlink(self.tmp_path)

    def test_save_creates_file(self):
        """save_checkpoint should create a JSON file."""
        save_checkpoint(self.model, self.tok, self.tmp_path)
        self.assertTrue(os.path.exists(self.tmp_path))

    def test_saved_file_is_valid_json(self):
        """Saved file should be valid JSON."""
        save_checkpoint(self.model, self.tok, self.tmp_path)
        with open(self.tmp_path, "r") as f:
            data = json.load(f)
        self.assertIn("cfg", data)
        self.assertIn("tokenizer", data)
        self.assertIn("base", data)
        self.assertIn("deltas", data)

    def test_load_restores_model(self):
        """load_checkpoint should restore a working model."""
        save_checkpoint(self.model, self.tok, self.tmp_path)
        loaded_model, loaded_tok = load_checkpoint(self.docs, self.tmp_path)

        self.assertIsNotNone(loaded_model)
        self.assertIsNotNone(loaded_tok)

    def test_loaded_model_has_correct_structure(self):
        """Loaded model should have same structure as original."""
        save_checkpoint(self.model, self.tok, self.tmp_path)
        loaded_model, loaded_tok = load_checkpoint(self.docs, self.tmp_path)

        self.assertEqual(loaded_model.n_layer, self.model.n_layer)
        self.assertEqual(loaded_model.n_embd, self.model.n_embd)
        self.assertEqual(loaded_tok.vocab_size, self.tok.vocab_size)

    def test_loaded_model_can_generate(self):
        """Loaded model should be able to generate text."""
        save_checkpoint(self.model, self.tok, self.tmp_path)
        loaded_model, loaded_tok = load_checkpoint(self.docs, self.tmp_path)

        text = loaded_model.generate_sentence("Hello")
        self.assertIsInstance(text, str)

    def test_weights_preserved(self):
        """Weights should be preserved after load."""
        # Modify a weight
        self.model.base["wte"].rows[0].data[0] = 42.0
        save_checkpoint(self.model, self.tok, self.tmp_path)
        loaded_model, _ = load_checkpoint(self.docs, self.tmp_path)

        self.assertAlmostEqual(loaded_model.base["wte"].rows[0].data[0], 42.0)

    def test_tokenizer_state_preserved(self):
        """Tokenizer state should be preserved."""
        save_checkpoint(self.model, self.tok, self.tmp_path)
        loaded_model, loaded_tok = load_checkpoint(self.docs, self.tmp_path)

        # Same tokens
        self.assertEqual(loaded_tok.tokens, self.tok.tokens)

    def test_delta_modules_preserved(self):
        """Delta modules should be preserved."""
        self.model.add_delta_module(alpha=0.5)
        num_deltas = len(self.model.deltas)

        save_checkpoint(self.model, self.tok, self.tmp_path)
        loaded_model, _ = load_checkpoint(self.docs, self.tmp_path)

        self.assertEqual(len(loaded_model.deltas), num_deltas)
        self.assertIn(0.5, loaded_model.active_alpha)

    def test_load_nonexistent_returns_none(self):
        """Loading nonexistent file should return None."""
        model, tok = load_checkpoint(self.docs, "/nonexistent/path.json")
        self.assertIsNone(model)
        self.assertIsNone(tok)


if __name__ == "__main__":
    unittest.main()
