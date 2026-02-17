#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the EvolvingTokenizer (char-level + BPE).
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from macrogpt import EvolvingTokenizer, CFG


class TestEvolvingTokenizerCharLevel(unittest.TestCase):
    """Tests for char-level tokenization (before BPE)."""

    def setUp(self):
        self.docs = ["Hello world.", "Testing tokenizer.", "The quick brown fox."]
        self.tok = EvolvingTokenizer(self.docs)

    def test_initial_vocab_contains_chars(self):
        """Vocab should contain all unique chars from docs."""
        text = "\n".join(self.docs) + "\n"
        unique_chars = set(text)
        for ch in unique_chars:
            self.assertIn(ch, self.tok.stoi, f"Char '{ch}' missing from vocab")

    def test_special_tokens_exist(self):
        """BOS, EOS, PAD should be in vocab."""
        self.assertIn(self.tok.BOS, self.tok.stoi)
        self.assertIn(self.tok.EOS, self.tok.stoi)
        self.assertIn(self.tok.PAD, self.tok.stoi)

    def test_encode_starts_with_bos(self):
        """Encoded sequence should start with BOS."""
        ids = self.tok.encode("Hello")
        self.assertEqual(ids[0], self.tok.stoi[self.tok.BOS])

    def test_encode_ends_with_eos(self):
        """Encoded sequence should end with EOS."""
        ids = self.tok.encode("Hello")
        self.assertEqual(ids[-1], self.tok.stoi[self.tok.EOS])

    def test_decode_inverts_encode(self):
        """Decode(encode(x)) should approximate x."""
        text = "Hello world"
        ids = self.tok.encode(text)
        decoded = self.tok.decode(ids)
        self.assertEqual(decoded, text)

    def test_encode_char_level(self):
        """Before BPE, each char should be one token."""
        text = "abc"
        ids = self.tok.encode(text)
        # BOS + 3 chars + EOS = 5 tokens
        self.assertEqual(len(ids), 5)

    def test_vocab_size_matches_tokens(self):
        """vocab_size should equal len(tokens)."""
        self.assertEqual(self.tok.vocab_size, len(self.tok.tokens))

    def test_stoi_itos_consistency(self):
        """stoi and itos should be inverses."""
        for tok_str, idx in self.tok.stoi.items():
            self.assertEqual(self.tok.itos[idx], tok_str)


class TestEvolvingTokenizerBPE(unittest.TestCase):
    """Tests for BPE functionality."""

    def setUp(self):
        # Create larger corpus to enable BPE
        self.docs = ["the quick brown fox"] * 500 + ["jumps over the lazy dog"] * 500
        # Lower threshold for testing
        old_threshold = CFG.enable_bpe_after_chars
        CFG.enable_bpe_after_chars = 100
        self.tok = EvolvingTokenizer(self.docs)
        self.tok.maybe_enable_bpe(self.docs)
        CFG.enable_bpe_after_chars = old_threshold

    def test_bpe_enabled(self):
        """BPE should be enabled after training."""
        self.assertTrue(self.tok.bpe_enabled)

    def test_bpe_merges_learned(self):
        """BPE should learn some merges."""
        self.assertGreater(len(self.tok.merges), 0)

    def test_vocab_expanded(self):
        """Vocab should expand with BPE merges."""
        base_docs = ["ab"]
        base_tok = EvolvingTokenizer(base_docs)
        base_size = base_tok.vocab_size
        # Our tok has merges
        self.assertGreater(self.tok.vocab_size, base_size)

    def test_encode_decode_with_bpe(self):
        """Encode/decode should work after BPE."""
        text = "the quick brown fox"
        ids = self.tok.encode(text)
        decoded = self.tok.decode(ids)
        self.assertEqual(decoded, text)

    def test_bpe_reduces_tokens(self):
        """BPE should reduce token count for repeated patterns."""
        char_tok = EvolvingTokenizer(["the quick"])
        text = "the quick"
        char_ids = char_tok.encode(text)
        bpe_ids = self.tok.encode(text)
        # BPE should use fewer tokens (or same, but not more)
        self.assertLessEqual(len(bpe_ids), len(char_ids))


class TestEvolvingTokenizerVocabGrowth(unittest.TestCase):
    """Tests for vocabulary growth (never shrinks)."""

    def test_vocab_only_grows(self):
        """Training BPE should only add tokens, never remove."""
        docs = ["hello world"] * 100
        tok = EvolvingTokenizer(docs)
        old_tokens = set(tok.tokens)

        # Simulate BPE training
        old_threshold = CFG.enable_bpe_after_chars
        CFG.enable_bpe_after_chars = 50
        tok.maybe_enable_bpe(docs)
        CFG.enable_bpe_after_chars = old_threshold

        new_tokens = set(tok.tokens)
        # All old tokens should still exist
        self.assertTrue(old_tokens.issubset(new_tokens))


if __name__ == "__main__":
    unittest.main()
