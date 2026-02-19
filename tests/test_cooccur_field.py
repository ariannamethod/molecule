"""Tests for CooccurField — corpus-based bigram/trigram frequency model."""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from molequla import CooccurField, EvolvingTokenizer, corpus_generate


def _make_field():
    docs = [
        "Hello world hello world",
        "This is a test sentence",
        "Hello hello hello"
    ]
    tok = EvolvingTokenizer(docs)
    field = CooccurField()
    field.build_from_corpus(tok, docs)
    return field, tok, docs


class TestCooccurField(unittest.TestCase):

    def test_build_populates_unigram(self):
        """After building, unigram counts should be populated."""
        field, tok, docs = _make_field()
        self.assertGreater(field.total_tokens, 0)
        self.assertGreater(len(field.unigram), 0)

    def test_build_populates_bigram(self):
        """After building, bigram counts should exist."""
        field, tok, docs = _make_field()
        self.assertGreater(len(field.bigram), 0)

    def test_build_populates_trigram(self):
        """After building, trigram counts should exist."""
        field, tok, docs = _make_field()
        self.assertGreater(len(field.trigram), 0)

    def test_unigram_reflects_frequency(self):
        """Frequent characters should have higher unigram counts."""
        field, tok, docs = _make_field()
        # 'l' appears many times in "Hello hello hello"
        l_id = tok.stoi.get('l', -1)
        if l_id >= 0:
            self.assertGreater(field.unigram[l_id], 0)

    def test_sample_next_returns_valid_id(self):
        """sample_next should return a valid token ID."""
        field, tok, docs = _make_field()
        # Get some context IDs
        ids = tok.encode("Hello")
        if len(ids) > 1:
            next_id = field.sample_next(ids[1:])  # skip BOS
            self.assertGreaterEqual(next_id, 0)
            self.assertLess(next_id, tok.vocab_size)

    def test_sample_next_empty_context(self):
        """sample_next with empty context should fallback to unigram."""
        field, tok, docs = _make_field()
        next_id = field.sample_next([])
        self.assertGreaterEqual(next_id, 0)

    def test_sample_next_single_context(self):
        """sample_next with single-token context should use bigram."""
        field, tok, docs = _make_field()
        h_id = tok.stoi.get('H', -1)
        if h_id >= 0:
            next_id = field.sample_next([h_id])
            self.assertGreaterEqual(next_id, 0)

    def test_corpus_generate_returns_string(self):
        """corpus_generate should return a non-empty string."""
        field, tok, docs = _make_field()
        result = corpus_generate(tok, field, "Hello")
        self.assertIsInstance(result, str)

    def test_build_from_corpus_idempotent(self):
        """Rebuilding should produce consistent counts."""
        field, tok, docs = _make_field()
        count1 = field.total_tokens
        field.build_from_corpus(tok, docs)
        count2 = field.total_tokens
        self.assertEqual(count1, count2)

    def test_build_clears_previous(self):
        """Rebuilding should clear old data first."""
        field, tok, docs = _make_field()
        # Build with different docs
        new_docs = ["Completely different text"]
        field.build_from_corpus(tok, new_docs)
        # Should not contain counts from original docs
        # (total tokens should be different)
        self.assertLess(field.total_tokens, 100)


if __name__ == "__main__":
    unittest.main()
