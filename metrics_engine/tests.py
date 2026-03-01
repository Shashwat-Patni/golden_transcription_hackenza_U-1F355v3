"""
tests.py — Team 2: Test suite for the Metrics & Scoring Engine
==============================================================
Run with:
    pytest tests.py -v

Coverage:
  - Text normalization
  - All atomic metrics
  - Metric normalization
  - CQS computation
  - Ranking & tie-breaking
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest
from metrics import (
    normalize_clean,
    normalize_punctuated,
    compute_wer,
    compute_cer,
    compute_precision_recall,
    compute_alignment_score,
    compute_completeness_score,
    compute_punctuation_score,
    normalize_metrics,
    compute_cqs,
    rank_transcriptions,
    DEFAULT_WEIGHTS,
)


# ─────────────────────────────────────────────────────────────────────────────
# TEXT NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

class TestTextNormalization:
    def test_lowercase(self):
        assert normalize_clean("Hello World") == "hello world"

    def test_punctuation_stripped(self):
        result = normalize_clean("Hello, world!")
        assert "," not in result and "!" not in result

    def test_extra_whitespace(self):
        assert normalize_clean("  hello   world  ") == "hello world"

    def test_punctuated_preserves_punct(self):
        result = normalize_punctuated("Hello, world!")
        assert "," in result and "!" in result

    def test_empty_input(self):
        assert normalize_clean("") == ""
        assert normalize_punctuated("") == ""


# ─────────────────────────────────────────────────────────────────────────────
# WER / CER
# ─────────────────────────────────────────────────────────────────────────────

class TestWER:
    def test_perfect_match(self):
        assert compute_wer("hello world", "hello world") == pytest.approx(0.0)

    def test_one_substitution(self):
        # 1 sub / 2 words = 0.5
        assert compute_wer("hello world", "hello word") == pytest.approx(0.5)

    def test_empty_reference(self):
        assert compute_wer("", "") == 0.0
        assert compute_wer("", "extra") == 1.0

    def test_empty_candidate(self):
        assert compute_wer("hello world", "") == 1.0

    def test_can_exceed_one(self):
        # More insertions than reference words
        assert compute_wer("hi", "hi there how are you today") > 1.0


class TestCER:
    def test_perfect_match(self):
        assert compute_cer("hello", "hello") == pytest.approx(0.0)

    def test_one_deletion(self):
        # "hello" vs "helo": 1 deletion / 5 chars = 0.2
        assert compute_cer("hello", "helo") == pytest.approx(0.2)

    def test_empty(self):
        assert compute_cer("", "") == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# PRECISION & RECALL
# ─────────────────────────────────────────────────────────────────────────────

class TestPrecisionRecall:
    def test_perfect(self):
        r = compute_precision_recall("the cat sat", "the cat sat")
        assert r["precision"] == pytest.approx(1.0)
        assert r["recall"] == pytest.approx(1.0)

    def test_missing_word(self):
        r = compute_precision_recall("the cat sat", "the cat")
        assert r["precision"] == pytest.approx(1.0)   # no extras
        assert r["recall"] == pytest.approx(2 / 3)    # missed "sat"

    def test_extra_word(self):
        r = compute_precision_recall("the cat", "the cat sat")
        assert r["precision"] == pytest.approx(2 / 3) # "sat" is extra
        assert r["recall"] == pytest.approx(1.0)

    def test_empty_candidate(self):
        r = compute_precision_recall("hello world", "")
        assert r["precision"] == 0.0 and r["recall"] == 0.0

    def test_repeated_words(self):
        # Multiset: ref has "cat cat", cand has "cat" — recall should be 0.5
        r = compute_precision_recall("cat cat", "cat")
        assert r["recall"] == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# ALIGNMENT
# ─────────────────────────────────────────────────────────────────────────────

class TestAlignment:
    def test_perfect(self):
        assert compute_alignment_score("a b c d", "a b c d") == pytest.approx(1.0)

    def test_reversed(self):
        score = compute_alignment_score("a b c", "c b a")
        assert score < 0.5

    def test_empty_both(self):
        assert compute_alignment_score("", "") == 1.0

    def test_empty_candidate(self):
        assert compute_alignment_score("hello world", "") == 0.0

    def test_subset_in_order(self):
        # Candidate is a subsequence of reference → high score
        score = compute_alignment_score("a b c d e", "a c e")
        assert score > 0.5


# ─────────────────────────────────────────────────────────────────────────────
# COMPLETENESS
# ─────────────────────────────────────────────────────────────────────────────

class TestCompleteness:
    def test_perfect_match(self):
        score = compute_completeness_score("the cat sat on 3 mats", "the cat sat on 3 mats")
        assert score >= 0.95

    def test_missing_number(self):
        full = compute_completeness_score("I have 3 cats", "I have 3 cats")
        miss = compute_completeness_score("I have 3 cats", "I have cats")
        assert full > miss

    def test_empty_candidate(self):
        assert compute_completeness_score("hello world", "") == 0.0

    def test_empty_both(self):
        assert compute_completeness_score("", "") >= 0.95


# ─────────────────────────────────────────────────────────────────────────────
# PUNCTUATION
# ─────────────────────────────────────────────────────────────────────────────

class TestPunctuation:
    def test_perfect_match(self):
        assert compute_punctuation_score("hello, world.", "hello, world.") == pytest.approx(1.0)

    def test_no_reference_punctuation(self):
        assert compute_punctuation_score("hello world", "hello world") == pytest.approx(0.5)

    def test_missing_all_punctuation(self):
        assert compute_punctuation_score("hello, world.", "hello world") == pytest.approx(0.0)

    def test_partial_match(self):
        score = compute_punctuation_score("hello, world.", "hello, world")
        assert 0.0 < score < 1.0


# ─────────────────────────────────────────────────────────────────────────────
# METRIC NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalization:
    def test_wer_zero_gives_one(self):
        scores = normalize_metrics({"wer": 0.0, "cer": 0.0, "precision": 1.0, "recall": 1.0,
                                    "alignment_score": 1.0, "completeness_score": 1.0,
                                    "semantic_similarity": 1.0, "fluency_score": 1.0,
                                    "punctuation_score": 1.0})
        assert scores["wer_score"] == pytest.approx(1.0)

    def test_wer_one_gives_zero(self):
        scores = normalize_metrics({"wer": 1.0})
        assert scores["wer_score"] == pytest.approx(0.0)

    def test_wer_above_one_clamped(self):
        scores = normalize_metrics({"wer": 1.5})
        assert scores["wer_score"] == pytest.approx(0.0)

    def test_semantic_rescaled(self):
        # Cosine of 1.0 → (1 + 1)/2 = 1.0
        scores = normalize_metrics({"semantic_similarity": 1.0})
        assert scores["semantic_similarity"] == pytest.approx(1.0)
        # Cosine of 0.0 → 0.5
        scores = normalize_metrics({"semantic_similarity": 0.0})
        assert scores["semantic_similarity"] == pytest.approx(0.5)

    def test_all_keys_present(self):
        scores = normalize_metrics({})
        for key in ["wer_score", "cer_score", "precision_score", "recall_score",
                    "alignment_score", "completeness_score", "semantic_similarity",
                    "fluency_score", "punctuation_score"]:
            assert key in scores
            assert 0.0 <= scores[key] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# CQS
# ─────────────────────────────────────────────────────────────────────────────

class TestCQS:
    def test_all_ones(self):
        scores = {k: 1.0 for k in ["wer_score", "cer_score", "precision_score", "recall_score",
                                    "alignment_score", "completeness_score", "semantic_similarity",
                                    "fluency_score", "punctuation_score"]}
        assert compute_cqs(scores, DEFAULT_WEIGHTS) == pytest.approx(1.0)

    def test_all_zeros(self):
        scores = {k: 0.0 for k in ["wer_score", "cer_score", "precision_score", "recall_score",
                                    "alignment_score", "completeness_score", "semantic_similarity",
                                    "fluency_score", "punctuation_score"]}
        assert compute_cqs(scores, DEFAULT_WEIGHTS) == pytest.approx(0.0)

    def test_weight_normalization(self):
        # Doubling all weights should give the same result
        scores = {"wer_score": 0.8, "cer_score": 0.6}
        w1 = {"wer": 0.5, "cer": 0.5}
        w2 = {"wer": 1.0, "cer": 1.0}
        assert compute_cqs(scores, w1) == pytest.approx(compute_cqs(scores, w2))

    def test_zero_weights_raises(self):
        with pytest.raises(ValueError):
            compute_cqs({}, {"wer": 0.0, "cer": 0.0})

    def test_single_metric_dominates(self):
        scores = {"wer_score": 1.0, "cer_score": 0.0}
        weights = {"wer": 1.0, "cer": 0.0}
        assert compute_cqs(scores, weights) == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# RANKING
# ─────────────────────────────────────────────────────────────────────────────

class TestRanker:
    def _r(self, tid, cqs, sem=0.5, comp=0.5, wer_score=0.5):
        return {
            "transcription_id": tid,
            "cqs_score": cqs,
            "metrics": {"semantic_similarity": sem, "completeness_score": comp, "wer_score": wer_score},
            "flags": [],
        }

    def test_basic_order(self):
        ranked = rank_transcriptions([self._r("t1", 0.7), self._r("t2", 0.9), self._r("t3", 0.8)])
        assert [r["transcription_id"] for r in ranked] == ["t2", "t3", "t1"]

    def test_ranks_assigned(self):
        ranked = rank_transcriptions([self._r("t1", 0.5), self._r("t2", 0.8)])
        assert ranked[0]["rank"] == 1
        assert ranked[1]["rank"] == 2

    def test_tiebreak_semantic(self):
        ranked = rank_transcriptions([self._r("t1", 0.8, sem=0.6), self._r("t2", 0.8, sem=0.9)])
        assert ranked[0]["transcription_id"] == "t2"

    def test_tiebreak_completeness(self):
        ranked = rank_transcriptions([
            self._r("t1", 0.8, sem=0.7, comp=0.5),
            self._r("t2", 0.8, sem=0.7, comp=0.9),
        ])
        assert ranked[0]["transcription_id"] == "t2"

    def test_single_candidate(self):
        ranked = rank_transcriptions([self._r("t1", 0.75)])
        assert ranked[0]["rank"] == 1
