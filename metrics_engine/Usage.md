# Usage Guide — Getting Started

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Then download the multilingual spaCy NER model (used for completeness scoring):

```bash
python -m spacy download xx_ent_wiki_sm
```

> The sentence-transformer model (`paraphrase-multilingual-mpnet-base-v2`, ~1GB) is downloaded automatically on first use.
> The fluency LM (`ai-forever/mGPT`, ~2GB) is only downloaded if you enable it — see step 4.

---

## 2. Start the API Server

```bash
uvicorn api:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.
Interactive docs (Swagger UI) at `http://localhost:8000/docs`.

---

## 3. Score Transcriptions

Send a POST request to `/score` with your reference and candidate transcriptions.

**Using curl:**
```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "audio_id": "audio_001",
    "reference": "the quick brown fox jumps over the lazy dog",
    "candidates": [
      {"transcription_id": "t1", "text": "the quick brown fox jumped over the lazy dog"},
      {"transcription_id": "t2", "text": "quick brown fox jumps over lazy dog"},
      {"transcription_id": "t3", "text": "the quick brown fox jumps over the lazy dog"}
    ]
  }'
```

**Using Python (requests):**
```python
import requests

response = requests.post("http://localhost:8000/score", json={
    "audio_id": "audio_001",
    "reference": "the quick brown fox jumps over the lazy dog",
    "candidates": [
        {"transcription_id": "t1", "text": "the quick brown fox jumped over the lazy dog"},
        {"transcription_id": "t2", "text": "quick brown fox jumps over lazy dog"},
        {"transcription_id": "t3", "text": "the quick brown fox jumps over the lazy dog"},
    ]
})

data = response.json()
for result in data["results"]:
    print(f"Rank {result['rank']}: {result['transcription_id']} — CQS: {result['cqs_score']:.4f}")
```

---

## 4. Re-rank with Different Weights (no recomputation)

After calling `/score`, you can instantly re-rank by sending new weights to `/score/rerank`.
This does **not** recompute any metrics — it just recalculates CQS from cached scores.

```python
response = requests.post("http://localhost:8000/score/rerank", json={
    "audio_id": "audio_001",   # must match a previously scored audio_id
    "weights": {
        "wer": 0.10,
        "cer": 0.05,
        "completeness": 0.10,
        "semantic_similarity": 0.10,
        "precision": 0.05,
        "recall": 0.05,
        "fluency": 0.35,
        "punctuation": 0.20
    }
})
```

You can also use a named preset instead:
```bash
# See available presets
curl http://localhost:8000/weights/presets

# Available: accuracy_focused | readability_focused | semantic_fidelity
```

---

## 5. Use the Pipeline Directly in Python (no server)

If you don't need the API, import and call the pipeline directly:

```python
from metrics import run_scoring_pipeline

result = run_scoring_pipeline(
    audio_id="audio_001",
    reference="the quick brown fox jumps over the lazy dog",
    candidates=[
        {"transcription_id": "t1", "text": "the quick brown fox jumped over the lazy dog"},
        {"transcription_id": "t2", "text": "quick brown fox jumps over lazy dog"},
    ],
    # weights are optional — omit to use defaults
    weights={"wer": 0.40, "cer": 0.20, "completeness": 0.15,
             "semantic_similarity": 0.10, "precision": 0.10,
             "recall": 0.05, "fluency": 0.00, "punctuation": 0.00}
)

for r in result["results"]:
    print(r["rank"], r["transcription_id"], r["cqs_score"])
```

---

## 6. Enable Fluency Scoring (optional)

Fluency is **disabled by default** because the mGPT model is ~2GB and adds latency.
To enable it, open `metrics.py` and change line:

```python
# Before
USE_LM_PERPLEXITY = False

# After
USE_LM_PERPLEXITY = True
```

The model will be downloaded automatically on first use. Requires ~2GB disk space and benefits from a GPU.

---

## 7. Run the Tests

```bash
pytest tests.py -v
```

---

## Summary of All Commands

```bash
# Install
pip install -r requirements.txt
python -m spacy download xx_ent_wiki_sm

# Start server
uvicorn api:app --reload --port 8000

# Run tests
pytest tests.py -v
```