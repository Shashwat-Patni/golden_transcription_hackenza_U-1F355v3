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
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

from metrics import (
    run_scoring_pipeline,
    rerank,
    DEFAULT_WEIGHTS,
    PRESET_WEIGHTS,
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
