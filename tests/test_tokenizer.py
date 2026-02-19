#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the EvolvingTokenizer (byte-level BPE, GPT-3/4 style).
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from molequla import EvolvingTokenizer, CFG, _unicode_segment


class TestEvolvingTokenizerByteLevel(unittest.TestCase):
    """Tests for byte-level tokenization (before BPE)."""

    def setUp(self):
        self.docs = ["Hello world.", "Testing tokenizer.", "The quick brown fox."]
        self.tok = EvolvingTokenizer(self.docs)

    def test_initial_vocab_has_256_bytes(self):
        """Vocab should contain all 256 byte tokens."""
        for i in range(256):
            tok_name = f"0x{i:02x}"
            self.assertIn(tok_name, self.tok.stoi, f"Byte token '{tok_name}' missing from vocab")

    def test_initial_vocab_size_259(self):
        """Initial vocab = 256 bytes + BOS + EOS + PAD = 259."""
        self.assertEqual(self.tok.vocab_size, 259)

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

    def test_decode_inverts_encode_ascii(self):
        """Decode(encode(x)) == x for ASCII."""
        text = "Hello world"
        ids = self.tok.encode(text)
        decoded = self.tok.decode(ids)
        self.assertEqual(decoded, text)

    def test_decode_inverts_encode_russian(self):
        """Decode(encode(x)) == x for Russian (multi-byte UTF-8)."""
        text = "Привет мир"
        ids = self.tok.encode(text)
        decoded = self.tok.decode(ids)
        self.assertEqual(decoded, text)

    def test_decode_inverts_encode_emoji(self):
        """Decode(encode(x)) == x for emoji (4-byte UTF-8)."""
        text = "Hello 🌍"
        ids = self.tok.encode(text)
        decoded = self.tok.decode(ids)
        self.assertEqual(decoded, text)

    def test_encode_byte_level(self):
        """Before BPE, each byte should be one token."""
        text = "fox"  # 3 ASCII bytes
        ids = self.tok.encode(text)
        # BOS + 3 bytes + EOS = 5 tokens
        self.assertEqual(len(ids), 5)

    def test_encode_multibyte_char(self):
        """Multi-byte UTF-8 char produces multiple tokens before BPE."""
        text = "ё"  # 2 bytes in UTF-8 (0xd1 0x91)
        ids = self.tok.encode(text)
        # BOS + 2 bytes + EOS = 4 tokens
        self.assertEqual(len(ids), 4)

    def test_vocab_size_matches_tokens(self):
        """vocab_size should equal len(tokens)."""
        self.assertEqual(self.tok.vocab_size, len(self.tok.tokens))

    def test_stoi_itos_consistency(self):
        """stoi and itos should be inverses."""
        for tok_str, idx in self.tok.stoi.items():
            self.assertEqual(self.tok.itos[idx], tok_str)


class TestUnicodeSegmentation(unittest.TestCase):
    """Tests for _unicode_segment pre-segmentation."""

    def test_letters_separate_from_digits(self):
        segments = _unicode_segment("abc123")
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0], b"abc")
        self.assertEqual(segments[1], b"123")

    def test_whitespace_separate(self):
        segments = _unicode_segment("hello world")
        self.assertEqual(len(segments), 3)  # "hello", " ", "world"
        self.assertEqual(segments[0], b"hello")
        self.assertEqual(segments[1], b" ")
        self.assertEqual(segments[2], b"world")

    def test_punctuation_separate(self):
        segments = _unicode_segment("hello!")
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0], b"hello")
        self.assertEqual(segments[1], b"!")

    def test_russian_is_one_segment(self):
        segments = _unicode_segment("Привет")
        self.assertEqual(len(segments), 1)

    def test_empty_string(self):
        segments = _unicode_segment("")
        self.assertEqual(len(segments), 0)


class TestEvolvingTokenizerBPE(unittest.TestCase):
    """Tests for BPE functionality."""

    def setUp(self):
        self.docs = ["the quick brown fox"] * 500 + ["jumps over the lazy dog"] * 500
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
        base_tok = EvolvingTokenizer(["ab"])
        self.assertGreater(self.tok.vocab_size, base_tok.vocab_size)

    def test_encode_decode_with_bpe(self):
        """Encode/decode should work after BPE."""
        text = "the quick brown fox"
        ids = self.tok.encode(text)
        decoded = self.tok.decode(ids)
        self.assertEqual(decoded, text)

    def test_bpe_reduces_tokens(self):
        """BPE should reduce token count for repeated patterns."""
        no_bpe_tok = EvolvingTokenizer(["the quick"])
        text = "the quick"
        no_bpe_ids = no_bpe_tok.encode(text)
        bpe_ids = self.tok.encode(text)
        self.assertLessEqual(len(bpe_ids), len(no_bpe_ids))

    def test_merged_token_format(self):
        """Merged tokens should use '+' separator (e.g. '0x74+0x68')."""
        for pair, tok in self.tok.merge_to_tok.items():
            self.assertIn("+", tok)
            # All parts should be valid hex or sub-merges
            parts = tok.split("+")
            for p in parts:
                self.assertTrue(p.startswith("0x") and len(p) == 4,
                                f"Bad merge token part: {p}")


class TestEvolvingTokenizerVocabGrowth(unittest.TestCase):
    """Tests for vocabulary growth (never shrinks)."""

    def test_vocab_only_grows(self):
        """Training BPE should only add tokens, never remove."""
        docs = ["hello world"] * 100
        tok = EvolvingTokenizer(docs)
        old_tokens = set(tok.tokens)

        old_threshold = CFG.enable_bpe_after_chars
        CFG.enable_bpe_after_chars = 50
        tok.maybe_enable_bpe(docs)
        CFG.enable_bpe_after_chars = old_threshold

        new_tokens = set(tok.tokens)
        self.assertTrue(old_tokens.issubset(new_tokens))


if __name__ == "__main__":
    unittest.main()
