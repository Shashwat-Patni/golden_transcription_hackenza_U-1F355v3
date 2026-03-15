"""
metrics.py — Team 2: Metrics & Scoring Engine
=============================================
Single module containing:
  - Text normalization (clean + punctuated)
  - All 9 atomic quality metrics
  - Per-metric normalization to [0, 1]
  - Composite Quality Score (CQS)
  - Ranking with tie-breaking
  - Batch / parallel processing

Usage:
    from metrics import run_scoring_pipeline

    result = run_scoring_pipeline(
        audio_id="audio_001",
        reference="the quick brown fox jumps over the lazy dog",
        candidates=[
            {"transcription_id": "t1", "text": "the quick brown fox jumped over the lazy dog"},
            {"transcription_id": "t2", "text": "quick brown fox jumps over lazy dog"},
        ],
        weights={"wer": 0.30, "cer": 0.15, ...}  # optional
    )
"""

import re
import math
import string
import copy
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
from jiwer import wer as jiwer_wer, cer as jiwer_cer
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_WEIGHTS = {
    "wer":                 0.30,
    "cer":                 0.15,
    "completeness":        0.15,
    "semantic_similarity": 0.15,
    "precision":           0.10,
    "recall":              0.05,
    "fluency":             0.05,
    "punctuation":         0.05,
}

PRESET_WEIGHTS = {
    "default": {
        "wer": 0.001, "cer": 0.2904, "precision": 0.001,
        "recall": 0.001, "completeness": 0.162,
        "semantic_similarity": 0.0249, "fluency": 0.001, "punctuation": 0.4228,
    },
    "accuracy_focused": {
        "wer": 0.40, "cer": 0.20, "completeness": 0.15,
        "semantic_similarity": 0.10, "precision": 0.10,
        "recall": 0.05, "fluency": 0.00, "punctuation": 0.00,
    },
    "readability_focused": {
        "wer": 0.10, "cer": 0.05, "completeness": 0.10,
        "semantic_similarity": 0.10, "precision": 0.05,
        "recall": 0.05, "fluency": 0.35, "punctuation": 0.20,
    },
    "semantic_fidelity": {
        "wer": 0.10, "cer": 0.05, "completeness": 0.20,
        "semantic_similarity": 0.45, "precision": 0.05,
        "recall": 0.10, "fluency": 0.05, "punctuation": 0.00,
    },
}

# Flags config
SEMANTIC_MISMATCH_THRESHOLD = 0.60
MIN_WORDS_FOR_FLUENCY = 5

# Multilingual sentence embedding model — supports 50+ languages including Arabic, Spanish, etc.
SENTENCE_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"

# Multilingual LM for perplexity-based fluency — supports 50+ languages.
# Set USE_LM_PERPLEXITY = True to enable (slower, requires ~2GB memory).
# Disabled by default — mGPT is ~2GB and can cause OOM crashes alongside Whisper large-v3.
FLUENCY_LM_NAME = "ai-forever/mGPT"
USE_LM_PERPLEXITY = False

# Lazy-loaded singletons
_sentence_model: Optional[SentenceTransformer] = None
_lm_model = None
_lm_tokenizer = None
_spacy_nlp = None


# ─────────────────────────────────────────────────────────────────────────────
# LAZY RESOURCE LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_spacy_nlp():
    global _spacy_nlp
    if _spacy_nlp is None:
        try:
            import spacy
            _spacy_nlp = spacy.load("xx_ent_wiki_sm")
        except Exception:
            _spacy_nlp = False
    return _spacy_nlp

def _get_sentence_model() -> SentenceTransformer:
    global _sentence_model
    if _sentence_model is None:
        _sentence_model = SentenceTransformer(SENTENCE_MODEL_NAME)
    return _sentence_model


def _get_lm():
    global _lm_model, _lm_tokenizer
    if _lm_model is None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            _lm_tokenizer = AutoTokenizer.from_pretrained(FLUENCY_LM_NAME)
            _lm_model = AutoModelForCausalLM.from_pretrained(FLUENCY_LM_NAME)
            _lm_model.eval()
        except Exception:
            _lm_model = False
    return _lm_model, _lm_tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# TEXT NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def normalize_clean(text: str) -> str:
    """
    Language-agnostic normalization for WER/CER/precision/recall/alignment/completeness.
    Steps: lowercase → strip punctuation → collapse whitespace.
    No filler-word removal (filler words differ per language and are not reliably detectable).
    """
    if not text:
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def normalize_punctuated(text: str) -> str:
    """Light normalization preserving punctuation for fluency/punctuation metrics."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"([.,!?;:])(?!\s)", r"\1 ", text)
    return re.sub(r"\s+", " ", text).strip()


# ─────────────────────────────────────────────────────────────────────────────
# ATOMIC METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_wer(reference: str, candidate: str) -> float:
    """Word Error Rate = (S + D + I) / N. Values > 1.0 are possible."""
    if not reference:
        return 0.0 if not candidate else 1.0
    if not candidate:
        return 1.0
    return jiwer_wer(reference, candidate)


def compute_cer(reference: str, candidate: str) -> float:
    """Character Error Rate = char_edits / reference_chars."""
    if not reference:
        return 0.0 if not candidate else 1.0
    if not candidate:
        return 1.0
    return jiwer_cer(reference, candidate)


def compute_precision_recall(reference: str, candidate: str) -> dict:
    """
    Word-level precision and recall using multiset (Counter) intersection.
    precision = correct / candidate_words  (low → hallucination)
    recall    = correct / reference_words  (low → omission)
    """
    ref_words = reference.split() if reference else []
    cand_words = candidate.split() if candidate else []

    if not ref_words and not cand_words:
        return {"precision": 1.0, "recall": 1.0}
    if not cand_words:
        return {"precision": 0.0, "recall": 0.0}
    if not ref_words:
        return {"precision": 0.0, "recall": 1.0}

    ref_c, cand_c = Counter(ref_words), Counter(cand_words)
    intersection = sum(min(ref_c[w], cand_c[w]) for w in cand_c)
    return {
        "precision": intersection / len(cand_words),
        "recall":    intersection / len(ref_words),
    }


def compute_alignment_score(reference: str, candidate: str) -> float:
    """
    LCS-based word order consistency.
    alignment = LCS_length / max(len_ref, len_cand)
    """
    ref_words = reference.split() if reference else []
    cand_words = candidate.split() if candidate else []

    if not ref_words and not cand_words:
        return 1.0
    if not ref_words or not cand_words:
        return 0.0

    m, n = len(ref_words), len(cand_words)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            curr[j] = prev[j - 1] + 1 if ref_words[i-1] == cand_words[j-1] else max(curr[j-1], prev[j])
        prev = curr

    return prev[n] / max(m, n)


def compute_completeness_score(reference: str, candidate: str) -> float:
    """
    Weighted coverage: word recall (0.6) + entity recall (0.4).
    Falls back to word (1.0) if spaCy is unavailable.
    """
    if not reference and not candidate:
        return 1.0
    if not candidate:
        return 0.0

    ref_words = reference.split() if reference else []
    cand_words = candidate.split() if candidate else []

    # Word recall
    if not ref_words:
        word_recall = 1.0 if not cand_words else 0.0
    elif not cand_words:
        word_recall = 0.0
    else:
        ref_c, cand_c = Counter(ref_words), Counter(cand_words)
        word_recall = sum(min(ref_c[w], cand_c[w]) for w in ref_c) / len(ref_words)

    # Entity recall (spaCy optional)
    nlp = _get_spacy_nlp()
    if nlp:
        try:
            ref_ents = [e.text.lower() for e in nlp(reference).ents]
            cand_ents = [e.text.lower() for e in nlp(candidate).ents]
            if not ref_ents:
                entity_recall, has_entities = 1.0, False
            else:
                ref_ec, cand_ec = Counter(ref_ents), Counter(cand_ents)
                entity_recall = sum(min(ref_ec[e], cand_ec[e]) for e in ref_ec) / len(ref_ents)
                has_entities = True
        except Exception:
            entity_recall, has_entities = 1.0, False
    else:
        entity_recall, has_entities = 1.0, False

    if has_entities:
        score = 0.60 * word_recall + 0.40 * entity_recall
    else:
        score = 1.0 * word_recall

    return round(max(0.0, min(1.0, score)), 6)


def compute_semantic_similarity_batch(
    reference: str,
    candidates: list[str],
    ref_embedding: Optional[np.ndarray] = None,
) -> tuple[list[float], np.ndarray]:
    """
    Batch cosine similarity between reference and all candidates.
    Returns (scores_list, reference_embedding) — cache the embedding for reuse.
    """
    if not candidates:
        return [], np.array([])

    model = _get_sentence_model()
    if ref_embedding is None:
        ref_embedding = model.encode([reference], convert_to_numpy=True)[0]

    cand_embs = model.encode(candidates, convert_to_numpy=True, batch_size=32)

    def cosine(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0

    return [cosine(ref_embedding, e) for e in cand_embs], ref_embedding


def compute_fluency_score(text: str) -> dict:
    """
    Fluency via LM perplexity using mGPT (ai-forever/mGPT), a multilingual causal
    language model supporting 60+ languages including Arabic, Spanish, French, etc.

    Perplexity measures how "surprised" the language model is by the text.
    Fluent, natural text has LOW perplexity; broken, ungrammatical text has HIGH perplexity.

    Set USE_LM_PERPLEXITY = True at the top of this file to enable.
    When disabled, returns a neutral 0.5 score (fluency is excluded from ranking).

    Returns dict with "fluency_perplexity" (raw, caller normalizes) or "fluency_score" (0-1).
    """
    if len(text.split()) < MIN_WORDS_FOR_FLUENCY:
        return {"fluency_score": 0.5, "mode": "neutral"}

    if not USE_LM_PERPLEXITY:
        return {"fluency_score": 0.5, "mode": "disabled"}

    import torch
    model, tokenizer = _get_lm()
    if not model:
        return {"fluency_score": 0.5, "mode": "neutral"}
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            loss = model(**inputs, labels=inputs["input_ids"]).loss.item()
        return {"fluency_perplexity": math.exp(loss), "mode": "perplexity"}
    except Exception:
        return {"fluency_score": 0.5, "mode": "neutral"}


def compute_fluency_score_batch(texts: list[str]) -> list[dict]:
    """
    Batch version of compute_fluency_score to avoid CPU contention in parallel threads.
    """
    if not USE_LM_PERPLEXITY:
        return [{"fluency_score": 0.5, "mode": "disabled"} for _ in texts]

    import torch
    model, tokenizer = _get_lm()
    if not model:
        return [{"fluency_score": 0.5, "mode": "neutral"} for _ in texts]

    results = []
    for text in texts:
        if len(text.split()) < MIN_WORDS_FOR_FLUENCY:
            results.append({"fluency_score": 0.5, "mode": "neutral"})
            continue
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
            with torch.no_grad():
                loss = model(**inputs, labels=inputs["input_ids"]).loss.item()
            # Ensure consistency with normalize_metrics: it looks for 'fluency_perplexity'
            # and then falls back to 'fluency_score'.
            results.append({"fluency_perplexity": math.exp(loss), "mode": "perplexity"})
        except Exception:
            results.append({"fluency_score": 0.5, "mode": "neutral"})
    return results


def compute_punctuation_score(ref_punct: str, cand_punct: str) -> float:
    """
    F1 score over (word_position, punct_char) tuples.
    Returns 0.5 (neutral) if reference has no punctuation.
    """
    SCORED_PUNCT = {".", ",", "!", "?", ";", ":"}

    def extract_punct_positions(text):
        tokens = re.findall(r"[a-z0-9']+|[.,!?;:]", text.lower())
        positions, word_idx = [], 0
        for t in tokens:
            if t in SCORED_PUNCT:
                positions.append((word_idx, t))
            else:
                word_idx += 1
        return set(positions)

    ref_pos = extract_punct_positions(ref_punct)
    cand_pos = extract_punct_positions(cand_punct)

    if not ref_pos:
        return 0.5  # Neutral — nothing to compare against

    tp = len(ref_pos & cand_pos)
    precision = tp / len(cand_pos) if cand_pos else 0.0
    recall = tp / len(ref_pos)

    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 6)


# ─────────────────────────────────────────────────────────────────────────────
# METRIC NORMALIZATION  (all → [0, 1] where 1 = best)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_metrics(raw: dict) -> dict:
    """
    Convert raw metric values to normalized [0, 1] scores.

    Normalization rules:
        WER, CER          : lower is better → score = max(0, 1 - value)
        Semantic sim      : cosine in [-1,1] → (sim + 1) / 2
        Fluency perplexity: lower is better → exp(-ppl / 200)
        All others        : already in [0, 1], passed through
    """
    def clamp(x): return max(0.0, min(1.0, x))

    scores = {
        "wer_score":          clamp(1.0 - raw.get("wer", 1.0)),
        "cer_score":          clamp(1.0 - raw.get("cer", 1.0)),
        "precision_score":    clamp(raw.get("precision", 0.0)),
        "recall_score":       clamp(raw.get("recall", 0.0)),
        "alignment_score":    clamp(raw.get("alignment_score", 0.0)),
        "completeness_score": clamp(raw.get("completeness_score", 0.0)),
        "punctuation_score":  clamp(raw.get("punctuation_score", 0.0)),
    }

    # Semantic similarity: cosine [-1, 1] → [0, 1]
    scores["semantic_similarity"] = clamp((raw.get("semantic_similarity", 0.0) + 1.0) / 2.0)

    # Fluency: prefer pre-computed score; fall back to perplexity scaling
    if "fluency_score" in raw:
        scores["fluency_score"] = clamp(raw["fluency_score"])
    elif "fluency_perplexity" in raw:
        scores["fluency_score"] = clamp(math.exp(-raw["fluency_perplexity"] / 200.0))
    else:
        scores["fluency_score"] = 0.5

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE QUALITY SCORE & RANKING
# ───────────────────────────────────���─────────────────────────────────────────

# Maps weight dict keys → normalized score keys
_WEIGHT_TO_SCORE = {
    "wer":                 "wer_score",
    "cer":                 "cer_score",
    "precision":           "precision_score",
    "recall":              "recall_score",
    "alignment":           "alignment_score",
    "completeness":        "completeness_score",
    "semantic_similarity": "semantic_similarity",
    "fluency":             "fluency_score",
    "punctuation":         "punctuation_score",
}


def compute_cqs(normalized_scores: dict, weights: dict) -> float:
    """CQS = Σ (weight_i × score_i). Weights are auto-normalized to sum to 1."""
    total = sum(weights.values())
    if total == 0:
        raise ValueError("All weights are zero.")
    cqs = sum(
        (w / total) * normalized_scores.get(_WEIGHT_TO_SCORE[k], 0.0)
        for k, w in weights.items()
        if k in _WEIGHT_TO_SCORE
    )
    return round(max(0.0, min(1.0, cqs)), 6)


def rank_transcriptions(scored_results: list[dict]) -> list[dict]:
    """
    Sort by CQS descending. Tie-breaks:
      1. semantic_similarity  2. completeness_score  3. wer_score
    """
    def key(item):
        m = item.get("metrics", {})
        return (
            item["cqs_score"],
            m.get("semantic_similarity", 0.0),
            m.get("completeness_score", 0.0),
            m.get("wer_score", 0.0),
        )

    ranked = sorted(scored_results, key=key, reverse=True)
    for i, item in enumerate(ranked, 1):
        item["rank"] = i
    return ranked


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE ORCHESTRATION
# ─────────────────────────────────────────────────────────────────────────────

def _score_one_candidate(
    ref_clean: str,
    cand_clean: str,
    ref_punct: str,
    cand_punct: str,
    sem_score: float,
    fluency_res: dict,
) -> tuple[dict, dict]:
    """Compute all non-embedding metrics for a single candidate. Returns (raw, normalized)."""
    pr = compute_precision_recall(ref_clean, cand_clean)

    raw = {
        "wer":                compute_wer(ref_clean, cand_clean),
        "cer":                compute_cer(ref_clean, cand_clean),
        "precision":          pr["precision"],
        "recall":             pr["recall"],
        "alignment_score":    compute_alignment_score(ref_clean, cand_clean),
        "completeness_score": compute_completeness_score(ref_clean, cand_clean),
        "semantic_similarity": sem_score,
        "punctuation_score":  compute_punctuation_score(ref_punct, cand_punct),
        **fluency_res,
    }
    return raw, normalize_metrics(raw)


def run_scoring_pipeline(
    audio_id: str,
    reference: str,
    candidates: list[dict],
    weights: Optional[dict] = None,
    max_workers: int = 4,
) -> dict:
    """
    Full pipeline: normalize → metrics → normalize → CQS → rank.

    Args:
        audio_id:    Identifier for the audio sample.
        reference:   Reference transcription from Team 1.
        candidates:  List of {"transcription_id": str, "text": str}.
        weights:     Optional CQS weight dict. Defaults to DEFAULT_WEIGHTS.
        max_workers: Thread count for parallel per-candidate metric computation.

    Returns:
        {
            "audio_id": str,
            "results": [{"transcription_id", "metrics", "cqs_score", "rank", "flags"}, ...],
            "weights_used": dict,
            "meta": {"processing_time_ms": int}
        }
    """
    start = time.time()
    weights = weights or DEFAULT_WEIGHTS

    # Step 1: Pre-normalize all (avoid redundant work in threads)
    ref_clean = normalize_clean(reference)
    ref_punct = normalize_punctuated(reference)

    candidates_data = []
    for c in candidates:
        candidates_data.append({
            "id": c["transcription_id"],
            "text": c["text"],
            "clean": normalize_clean(c["text"]),
            "punct": normalize_punctuated(c["text"])
        })

    # Step 2: Batch semantic embeddings (most expensive — done once for all candidates)
    cand_texts = [cd["text"] for cd in candidates_data]
    sem_scores, _ = compute_semantic_similarity_batch(reference, cand_texts)

    # Step 3: Batch fluency scores (also expensive, done once to avoid parallel CPU contention)
    cand_punctuated = [cd["punct"] for cd in candidates_data]
    fluency_scores = compute_fluency_score_batch(cand_punctuated)

    # Step 4: Parallel per-candidate metric computation
    def process_one(idx, cd):
        raw, normalized = _score_one_candidate(
            ref_clean,
            cd["clean"],
            ref_punct,
            cd["punct"],
            sem_scores[idx],
            fluency_scores[idx],
        )
        cqs = compute_cqs(normalized, weights)

        flags = []
        if normalized.get("semantic_similarity", 1.0) < SEMANTIC_MISMATCH_THRESHOLD:
            flags.append("semantic_mismatch_warning")

        return {
            "transcription_id": cd["id"],
            "metrics": {**raw, **normalized},
            "cqs_score": cqs,
            "flags": flags,
        }

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_one, i, cd): i for i, cd in enumerate(candidates_data)}
        for future in as_completed(futures):
            results.append(future.result())

    # Step 5: Rank
    ranked = rank_transcriptions(results)

    return {
        "audio_id": audio_id,
        "results": ranked,
        "weights_used": weights,
        "meta": {"processing_time_ms": int((time.time() - start) * 1000)},
    }


def rerank(audio_id: str, cached_results: list[dict], new_weights: dict) -> dict:
    """
    Re-rank from cached normalized scores with new weights. No metric recomputation.

    Args:
        audio_id:       Audio identifier.
        cached_results: Previous results list (must contain normalized metric scores).
        new_weights:    New weight configuration.

    Returns:
        Same structure as run_scoring_pipeline.
    """
    start = time.time()
    results = copy.deepcopy(cached_results)
    for item in results:
        item["cqs_score"] = compute_cqs(item["metrics"], new_weights)

    return {
        "audio_id": audio_id,
        "results": rank_transcriptions(results),
        "weights_used": new_weights,
        "meta": {"processing_time_ms": int((time.time() - start) * 1000)},
    }
