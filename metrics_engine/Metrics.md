# Metrics & Scoring Engine — Team 2 Documentation

## Overview

The Metrics & Scoring Engine evaluates the quality of multiple candidate transcriptions against a reference transcription (produced by Team 1's ASR pipeline). It computes a set of **atomic quality metrics**, normalizes them to a 0–1 scale, and combines them into a **Composite Quality Score (CQS)** used to rank candidates.

The engine is **language-agnostic** — it does not assume English and all models chosen explicitly support multilingual input (Arabic, Spanish, French, and 50+ other languages).

---

## File Structure

```
metrics_engine/
├── Metrics.md          ← This file
├── requirements.txt
├── metrics.py          ← All metrics, normalization, CQS, ranking, and pipeline
├── api.py              ← FastAPI app and Pydantic schemas
└── tests.py            ← Full test suite (run with: pytest tests.py -v)
```

---

## Input Format

```json
{
  "audio_id": "audio_001",
  "reference": "the quick brown fox jumps over the lazy dog",
  "candidates": [
    { "transcription_id": "t1", "text": "the quick brown fox jumped over the lazy dog" },
    { "transcription_id": "t2", "text": "quick brown fox jumps over lazy dog" }
  ],
  "weights": {
    "wer": 0.30, "cer": 0.15, "completeness": 0.15,
    "semantic_similarity": 0.15, "precision": 0.10,
    "recall": 0.05, "fluency": 0.05, "punctuation": 0.05
  }
}
```

- `reference` is the normalized output from Team 1's Whisper pipeline.
- `weights` are optional — defaults are used if omitted and auto-normalized to sum to 1.0.

---

## Output Format

```json
{
  "audio_id": "audio_001",
  "results": [
    {
      "transcription_id": "t1",
      "metrics": {
        "wer": 0.111,          "wer_score": 0.889,
        "cer": 0.042,          "cer_score": 0.958,
        "precision": 0.950,    "precision_score": 0.950,
        "recall": 1.000,       "recall_score": 1.000,
        "alignment_score": 0.980,
        "completeness_score": 0.970,
        "semantic_similarity": 0.992,
        "fluency_score": 0.500,
        "punctuation_score": 0.750
      },
      "cqs_score": 0.913,
      "rank": 1,
      "flags": []
    }
  ],
  "weights_used": { "wer": 0.30, "...": "..." },
  "meta": { "processing_time_ms": 142 }
}
```

- Raw fields are the original computed values before normalization.
- `*_score` fields are the normalized 0–1 values used in CQS.
- `flags` may include: `semantic_mismatch_warning`.

---

## Text Normalization

Before any metric is computed, both reference and candidate go through normalization. Because the engine is language-agnostic, normalization is deliberately minimal:

**Clean text** (used for WER, CER, Precision, Recall, Alignment, Completeness):
- Lowercase
- Strip punctuation
- Collapse whitespace

**Punctuated text** (used for Punctuation score only):
- Lowercase
- Normalize spacing around punctuation marks
- Collapse whitespace

> **Why no filler-word removal?** Filler words (hesitations, discourse markers) vary by language — "uh/um" in English have entirely different equivalents in Arabic, Spanish, etc. A hardcoded English filler list would silently corrupt non-English transcriptions. Filler handling is the responsibility of Team 1's language-specific ASR post-processing.

---

## Metrics Methodology

### 1. Word Error Rate (WER)

**What it measures:** Edit distance between candidate and reference at the word level.

**Formula:**
```
WER = (Substitutions + Deletions + Insertions) / len(reference_words)
```

**Library:** `jiwer`

**Normalized score:** `wer_score = max(0, 1 − WER)`

Values above 1.0 are possible (more insertions than reference words) and are clamped to 0.

---

### 2. Character Error Rate (CER)

**What it measures:** Same as WER but at the character level — sensitive to minor spelling errors and partial misrecognitions, especially useful for morphologically rich languages.

**Formula:**
```
CER = character_edit_distance / len(reference_chars)
```

**Library:** `jiwer`

**Normalized score:** `cer_score = max(0, 1 − CER)`

---

### 3. Word-Level Precision

**What it measures:** Of all words the candidate produced, what fraction were correct? Penalizes hallucinated or spurious words.

**Formula:**
```
correct_words = multiset_intersection(candidate_words, reference_words)
precision     = correct_words / len(candidate_words)
```

Computed as a **multiset** (Counter intersection) so repeated words are handled correctly.

**Normalized:** Already in [0, 1] — passed through unchanged.

---

### 4. Word-Level Recall

**What it measures:** Of all words in the reference, what fraction appear in the candidate? Penalizes omissions — words the transcription dropped entirely.

**Formula:**
```
correct_words = multiset_intersection(candidate_words, reference_words)
recall        = correct_words / len(reference_words)
```

**Example:**
- Reference: `"the cat sat on the mat"` (6 words)
- Candidate: `"the cat sat"` (3 words, all correct)
- `correct_words = 3`, `recall = 3/6 = 0.50`

A recall of 0.50 means the candidate captured only half the reference — it dropped "on the mat" entirely.

**Interpretation:**
- `recall = 1.0` → candidate contains every reference word (nothing omitted)
- `recall = 0.0` → candidate shares no words with the reference
- Low recall is a strong signal of a partial or truncated transcription

**Normalized:** Already in [0, 1] — passed through unchanged.

> **Precision vs Recall:** These are complementary. A candidate that repeats the reference twice would have recall = 1.0 but precision = 0.5 (half its words are duplicates). A candidate that says only the first word would have precision = 1.0 but recall ≈ 0.0. Both are required to detect opposite failure modes.

---

### 5. Alignment Score

**What it measures:** Whether the candidate's words appear in the same order as the reference, independent of which words are present.

**Method:** Longest Common Subsequence (LCS) ratio.
```
alignment_score = LCS_length / max(len(reference_words), len(candidate_words))
```

Score of 1.0 = perfect word order. Score of 0.0 = no common subsequence.

**Normalized:** Already in [0, 1].

---

### 6. Completeness Score

**What it measures:** Whether all key content from the reference is present — with extra weight on named entities and numbers, which are high-importance tokens that ASR systems frequently drop or mangle.

**Formula:**
```
completeness = 0.50 × word_recall + 0.30 × entity_recall + 0.20 × number_recall
```

If spaCy NER is unavailable, entity weight redistributes to word recall (0.70 / 0.30 split).

**Normalized:** Already in [0, 1].

---

### 7. Semantic Similarity

**What it measures:** Whether the candidate conveys the same *meaning* as the reference, even if different words were used (paraphrases, synonyms, reordering).

**Model:** `paraphrase-multilingual-mpnet-base-v2` (sentence-transformers)

This model was chosen because it supports **50+ languages** including Arabic, Spanish, French, German, Chinese, and more — trained on multilingual paraphrase pairs. It maps sentences from all supported languages into a shared embedding space, enabling direct semantic comparison of Arabic-to-Arabic, Spanish-to-Spanish, etc.

**Method:** Encode both texts → compute cosine similarity.
```
semantic_similarity = cosine(embed(reference), embed(candidate))
```

**Normalization:** Cosine is in [−1, 1]. Rescaled to [0, 1] via `(cosine + 1) / 2`.

**Flag:** If the rescaled score falls below 0.60, a `semantic_mismatch_warning` flag is raised — the candidate may convey a different meaning entirely.

---

### 8. Fluency Score

**What it measures:** How natural and grammatically coherent the candidate reads on its own, independent of the reference.

#### Why a separate language model is needed

The semantic similarity model (`paraphrase-multilingual-mpnet-base-v2`) answers the question *"do these two sentences mean the same thing?"* — it compares two texts against each other. It cannot judge whether a single text is fluent or broken in isolation.

Fluency requires a **generative language model** — one that has learned the probability distribution of natural language. We compute **perplexity**: how "surprised" the model is when reading the candidate word by word.

- **Low perplexity** = the model found the text predictable and natural → fluent
- **High perplexity** = the model found the text unusual or incoherent → broken

**Example of why this matters:**
- `"The doctor examined the patient carefully."` → perplexity ≈ 40 ✓ fluent
- `"Doctor patient the carefully examined."` → perplexity ≈ 800 ✗ word salad

Both sentences contain identical words, so WER = 0 and semantic similarity would be high. Only perplexity correctly identifies the second as broken. This makes fluency especially valuable for detecting ASR outputs with correct vocabulary but scrambled word order.

**Model:** `ai-forever/mGPT` — a multilingual GPT-style causal language model trained on **60+ languages**, including Arabic, Spanish, French, Russian, and others. This replaces any English-only LM (like distilgpt2) to remain language-agnostic.

**Formula:**
```
perplexity    = exp(cross_entropy_loss of candidate text under mGPT)
fluency_score = exp(−perplexity / 200)   → normalized to [0, 1]
```

The scaling constant (200) is calibrated so that perplexity ≈ 50 → score ≈ 0.78 and perplexity ≈ 500 → score ≈ 0.08.

**Enabling fluency:** Set `USE_LM_PERPLEXITY = True` in `metrics.py`. Disabled by default because mGPT requires ~2GB memory and adds latency. When disabled, fluency returns a neutral 0.5 and contributes nothing to CQS ranking.

---

### 9. Punctuation & Formatting Score

**What it measures:** How well sentence boundaries, commas, and punctuation placement in the candidate match the reference.

**Method:** Extract `(word_position, punctuation_char)` tuples from both texts. Compute F1 over matched positions.
```
punctuation_score = F1(reference_punct_positions, candidate_punct_positions)
```

Returns 0.5 (neutral) if the reference has no punctuation.

**Normalized:** F1 is already in [0, 1].

---

## Metric Normalization Summary

| Metric              | Raw Range   | Direction       | Normalized As              |
|---------------------|-------------|-----------------|----------------------------|
| WER                 | [0, ∞)      | lower = better  | `max(0, 1 − WER)`         |
| CER                 | [0, ∞)      | lower = better  | `max(0, 1 − CER)`         |
| Precision           | [0, 1]      | higher = better | unchanged                  |
| Recall              | [0, 1]      | higher = better | unchanged                  |
| Alignment           | [0, 1]      | higher = better | unchanged                  |
| Completeness        | [0, 1]      | higher = better | unchanged                  |
| Semantic Similarity | [−1, 1]     | higher = better | `(cosine + 1) / 2`        |
| Fluency             | perplexity  | lower = better  | `exp(−ppl / 200)`         |
| Punctuation         | [0, 1]      | higher = better | unchanged                  |

---

## Composite Quality Score (CQS)

```
CQS = Σ (weight_i × normalized_score_i)
```

Weights are auto-normalized to sum to 1.0 before computing.

**Default weights:**

| Metric              | Weight |
|---------------------|--------|
| WER                 | 0.30   |
| CER                 | 0.15   |
| Completeness        | 0.15   |
| Semantic Similarity | 0.15   |
| Precision           | 0.10   |
| Recall              | 0.05   |
| Fluency             | 0.05   |
| Punctuation         | 0.05   |

---

## Ranking Logic

Candidates are sorted by CQS descending. Tie-breaking order:
1. Higher `semantic_similarity`
2. Higher `completeness_score`
3. Higher `wer_score` (lower raw WER)

---

## API Endpoints

| Method | Endpoint            | Description                                           |
|--------|---------------------|-------------------------------------------------------|
| POST   | `/score`            | Full pipeline — score and rank all candidates         |
| POST   | `/score/rerank`     | Rerank with new weights (no metric recomputation)     |
| GET    | `/weights/defaults` | Return default weight configuration                   |
| GET    | `/weights/presets`  | Return all preset profiles                            |
| GET    | `/health`           | Health check                                          |

---

## Models Used

| Purpose              | Model                                   | Languages supported                |
|----------------------|-----------------------------------------|------------------------------------|
| Semantic Similarity  | `paraphrase-multilingual-mpnet-base-v2` | 50+ (Arabic, Spanish, etc.)        |
| Fluency (perplexity) | `ai-forever/mGPT`                       | 60+ (Arabic, Spanish, etc.)        |
| NER (Completeness)   | `xx_ent_wiki_sm` (spaCy)               | Multilingual (Wikipedia-based NER) |

---

## Integration Notes

**Team 1** should provide:
- `reference` — normalized reference text from Whisper
- Filler-word removal, if desired, should be handled on Team 1's side using language-specific rules

**Team 3** should:
- POST to `/score` on initial load
- POST to `/score/rerank` on every slider change — this skips all metric computation, using cached normalized scores, and is near-instant