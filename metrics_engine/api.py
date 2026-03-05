"""
api.py — Team 2: REST API for Metrics & Scoring Engine
=======================================================
FastAPI app exposing the scoring pipeline.

Run with:
    uvicorn api:app --reload --port 8000

Endpoints:
    POST /score             → Full pipeline (score + rank)
    POST /score/rerank      → Re-rank with new weights (no metric recomputation)
    GET  /weights/defaults  → Default weight config
    GET  /weights/presets   → All preset profiles
    GET  /health            → Health check
"""

import copy
import csv
import time
import uuid
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

from metrics import (
    run_scoring_pipeline,
    rerank,
    DEFAULT_WEIGHTS,
    PRESET_WEIGHTS,
    normalize_clean,
    compute_wer,
)

# ─────────────────────────────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class CandidateInput(BaseModel):
    transcription_id: str
    text: str


class WeightsInput(BaseModel):
    wer:                 float = Field(0.30, ge=0)
    cer:                 float = Field(0.15, ge=0)
    completeness:        float = Field(0.15, ge=0)
    semantic_similarity: float = Field(0.15, ge=0)
    precision:           float = Field(0.10, ge=0)
    recall:              float = Field(0.05, ge=0)
    fluency:             float = Field(0.05, ge=0)
    punctuation:         float = Field(0.05, ge=0)

    def to_dict(self) -> dict:
        return self.model_dump()


class ScoreRequest(BaseModel):
    audio_id: str
    reference: str
    candidates: list[CandidateInput] = Field(..., min_length=1)
    weights: Optional[WeightsInput] = None

    @model_validator(mode="after")
    def check_unique_ids(self):
        ids = [c.transcription_id for c in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate transcription_id values in candidates.")
        return self


class RerankRequest(BaseModel):
    audio_id: str
    weights: WeightsInput


class ScoredCandidate(BaseModel):
    transcription_id: str
    metrics: dict
    cqs_score: float
    rank: int
    flags: list[str] = []


class ScoreResponse(BaseModel):
    audio_id: str
    results: list[ScoredCandidate]
    weights_used: Optional[dict] = None
    meta: dict


class BatchItem(BaseModel):
    audio_id: str
    language: str
    audio: str
    reference: str
    option_1: str
    option_2: str
    option_3: str
    option_4: str
    option_5: str


class BatchScoreRequest(BaseModel):
    items: list[BatchItem]
    weights: Optional[WeightsInput] = None


class BatchScoreResponse(BaseModel):
    results: list[ScoreResponse]
    csv_file: str
    meta: dict


# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Transcription Metrics & Scoring Engine",
    description="Team 2 — CQS computation, atomic metrics, and ranking.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to Team 3's origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory result cache keyed by audio_id (for fast reranking)
_results_cache: dict[str, list[dict]] = {}


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "metrics-engine"}


@app.get("/weights/defaults")
def get_default_weights():
    """Return the default CQS weight configuration."""
    return {"weights": DEFAULT_WEIGHTS}


@app.get("/weights/presets")
def get_preset_weights():
    """Return all available preset weight profiles."""
    return {"presets": PRESET_WEIGHTS}


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest):
    """
    Full scoring pipeline:
      1. Batch-encode semantic embeddings
      2. Compute all atomic metrics in parallel
      3. Normalize metrics to [0, 1]
      4. Compute CQS
      5. Rank candidates

    Results are cached by audio_id to enable fast reranking via /score/rerank.
    """
    weights = request.weights.to_dict() if request.weights else DEFAULT_WEIGHTS
    candidates = [{"transcription_id": c.transcription_id, "text": c.text}
                  for c in request.candidates]

    response = run_scoring_pipeline(
        audio_id=request.audio_id,
        reference=request.reference,
        candidates=candidates,
        weights=weights,
    )

    # Cache a deep copy so rerank doesn't mutate stored state
    _results_cache[request.audio_id] = copy.deepcopy(response["results"])

    return response


@app.post("/score/rerank", response_model=ScoreResponse)
def rerank_endpoint(request: RerankRequest):
    """
    Rerank using new weights WITHOUT recomputing any metrics.

    Reads cached results for the given audio_id — call /score first.
    This is the fast path for Team 3's slider interactions.
    """
    cached = _results_cache.get(request.audio_id)
    if not cached:
        raise HTTPException(
            status_code=404,
            detail=f"No cached results for audio_id='{request.audio_id}'. Run /score first.",
        )

    return rerank(
        audio_id=request.audio_id,
        cached_results=copy.deepcopy(cached),
        new_weights=request.weights.to_dict(),
    )


@app.post("/score/batch", response_model=BatchScoreResponse)
def score_batch(request: BatchScoreRequest):
    """
    Batch scoring pipeline:
      1. For each item, run the scoring pipeline.
      2. Identify the highest-ranked transcription (golden reference).
      3. Calculate WER for each option compared to the golden reference.
      4. Write results to a unique CSV file in the 'outputs/' directory.
      5. Return results as JSON.
    """
    start_time = time.time()
    weights = request.weights.to_dict() if request.weights else DEFAULT_WEIGHTS
    all_results = []
    csv_rows = []

    for item in request.items:
        candidates = [
            {"transcription_id": "1", "text": item.option_1},
            {"transcription_id": "2", "text": item.option_2},
            {"transcription_id": "3", "text": item.option_3},
            {"transcription_id": "4", "text": item.option_4},
            {"transcription_id": "5", "text": item.option_5},
        ]

        # Filter out empty options
        candidates = [c for c in candidates if c["text"].strip()]

        response = run_scoring_pipeline(
            audio_id=item.audio_id,
            reference=item.reference,
            candidates=candidates,
            weights=weights,
        )
        all_results.append(response)

        # Identify golden reference (top-ranked)
        top_result = response["results"][0]
        golden_ref_id = top_result["transcription_id"]
        golden_ref_text = next(c["text"] for c in candidates if c["transcription_id"] == golden_ref_id)

        # Calculate WER wrt golden reference for each option
        norm_golden = normalize_clean(golden_ref_text)
        wer_bc = {}
        for c in candidates:
            norm_cand = normalize_clean(c["text"])
            wer_bc[c["transcription_id"]] = compute_wer(norm_golden, norm_cand)

        csv_rows.append({
            "audio_id": item.audio_id,
            "language": item.language,
            "audio": item.audio,
            "option_1": item.option_1,
            "option_2": item.option_2,
            "option_3": item.option_3,
            "option_4": item.option_4,
            "option_5": item.option_5,
            "golden_ref": golden_ref_text,
            "golden_score": top_result["cqs_score"],
            "wer_option1": wer_bc.get("1", ""),
            "wer_option2": wer_bc.get("2", ""),
            "wer_option3": wer_bc.get("3", ""),
            "wer_option4": wer_bc.get("4", ""),
            "wer_option5": wer_bc.get("5", ""),
        })

    # Ensure output directory exists
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Write unique CSV
    request_id = str(uuid.uuid4())
    csv_filename = f"batch_results_{request_id}.csv"
    csv_path = os.path.join(output_dir, csv_filename)

    fieldnames = [
        "audio_id", "language", "audio", "option_1", "option_2", "option_3",
        "option_4", "option_5", "golden_ref", "golden_score", "wer_option1",
        "wer_option2", "wer_option3", "wer_option4", "wer_option5"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    return {
        "results": all_results,
        "csv_file": csv_filename,
        "meta": {"processing_time_ms": int((time.time() - start_time) * 1000)}
    }
