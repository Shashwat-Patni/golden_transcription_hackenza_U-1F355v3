Objective

Rank multiple transcriptions per audio using a Composite Quality Score (CQS) derived from atomic quality metrics.

The system must:
	•	generate a reference transcription,
	•	compute quality metrics per candidate transcription,
	•	combine metrics into a weighted score,
	•	allow UI weight adjustment,
	•	re-rank in real time.

⸻

1. Data Model

Input

For each audio sample:

audio_id
audio_file
transcriptions: [t1, t2, t3, t4, t5]

Output

audio_id
transcription_id
metrics:
    WER
    CER
    precision
    recall
    alignment_score
    fluency_score
    punctuation_score
    completeness_score
    semantic_similarity
CQS_score
rank


⸻

2. Ground Truth Strategy

Use a high-quality ASR model (e.g., OpenAI Whisper large) to produce a reference transcription.

Steps
	1.	Normalize audio (16kHz mono WAV).
	2.	Run Whisper (or equivalent SOTA ASR).
	3.	Store:
	•	raw transcription
	•	timestamps (optional for alignment metrics)

Important

This is a reference, not absolute truth.
Metrics must tolerate ASR imperfections.

⸻

3. Text Normalization Pipeline

Normalization must be identical for reference and candidates.

Normalize:
	•	lowercase
	•	remove filler noise tokens (uh, um)
	•	normalize numbers (“twenty one” → “21”)
	•	strip extra whitespace
	•	standardize punctuation spacing

Maintain two versions:
	•	clean text (for WER/CER)
	•	punctuated text (for fluency/punctuation metrics)

⸻

4. Atomic Quality Metrics

4.1 Word Error Rate (WER)

Measures substitution, deletion, insertion errors.

WER = (S + D + I) / N

Use:
	•	jiwer or custom Levenshtein alignment.

Captures:
	•	word correctness
	•	missing/extra words

⸻

4.2 Character Error Rate (CER)

Useful for minor deviations.

CER = char edits / total chars

Captures:
	•	small spelling errors
	•	partial misrecognition

⸻

4.3 Precision & Recall (Word-Level)

Evaluate completeness vs over-generation.

precision = correct_words / candidate_words
recall    = correct_words / reference_words

Interpretation:
	•	low recall → missing words
	•	low precision → extra hallucinated words

⸻

4.4 Alignment Score (Timing / Order Consistency)

If timestamps available:
	•	perform forced alignment or sequence alignment
	•	penalize incorrect ordering or segment mismatch

If no timestamps:
	•	compute n-gram order preservation
	•	measure longest common subsequence ratio

⸻

4.5 Completeness Score

Measures coverage of reference content.

completeness = recall adjusted by phrase coverage

Enhancement:
	•	penalize dropped clauses
	•	detect dropped named entities / numbers

⸻

4.6 Semantic Similarity

Captures meaning preservation.

Use sentence embeddings:
	•	SBERT / Instructor / E5 embeddings
	•	cosine similarity

Helps when wording differs but meaning matches.

⸻

4.7 Fluency & Language Quality

Measures readability & grammar.

Options:
	•	perplexity using small language model
	•	grammar error detection tools
	•	sentence boundary correctness

Signals:
	•	broken phrases
	•	unnatural grammar
	•	word salad

⸻

4.8 Punctuation & Formatting Score

Compare punctuation placement.

Evaluate:
	•	sentence boundaries
	•	commas & pauses
	•	capitalization

Useful for readability ranking.

⸻

5. Metric Normalization

Each metric must be normalized to 0–1 scale.

Examples:

Metric	Raw	Normalized
WER	lower better	1 - WER
CER	lower better	1 - CER
semantic similarity	higher better	unchanged
perplexity	lower better	scaled inverse

Store normalized values.

⸻

6. Composite Quality Score (CQS)

Formula

CQS = Σ (weight_i × metric_i)

Example:

CQS =
0.30 * WER_score +
0.15 * CER_score +
0.15 * completeness +
0.15 * semantic_similarity +
0.10 * precision +
0.05 * recall +
0.05 * fluency +
0.05 * punctuation

Weights must be adjustable.

⸻

7. Ranking Logic

For each audio:

compute metrics → normalize → compute CQS → sort descending

Return ranked list.

Tie-breakers:
	1.	higher semantic similarity
	2.	higher completeness
	3.	lower WER

⸻

8. Evaluation & Calibration

Validate metric reliability
	•	manually score sample outputs
	•	compute correlation with human ranking
	•	adjust weights accordingly

Detect Whisper reference bias
	•	compare multiple ASR outputs if needed
	•	flag large divergence cases

⸻

9. Performance Considerations

Batch Processing
	•	compute embeddings in batch
	•	parallelize metric computation
	•	cache reference transcription

Storage
	•	store metrics to avoid recomputation
	•	recompute only CQS when weights change

⸻

10. UI Requirements

UI Panel Features

Weight Controls
	•	sliders for each metric
	•	total auto-normalizes to 1
	•	presets:
	•	accuracy focused
	•	readability focused
	•	semantic fidelity

Real-Time Updates
	•	recompute CQS on slider change
	•	instant re-ranking

Visualization
	•	metric breakdown per transcription
	•	highlight strengths & weaknesses

⸻

11. Edge Cases

Handle:
	•	heavy background noise
	•	partial speech
	•	non-verbal audio
	•	multilingual mixing
	•	named entities & numbers
	•	punctuation-free transcripts

Add flags:

low_confidence_reference
semantic_mismatch_warning


⸻

12. Suggested Tech Stack

ASR
	•	Whisper large / faster-whisper

Metrics
	•	jiwer (WER)
	•	rapidfuzz / Levenshtein
	•	sentence-transformers
	•	language-tool / LM perplexity

Backend
	•	Python + FastAPI
	•	vectorized metric pipeline

Frontend
	•	React + sliders + real-time updates

⸻

13. Pipeline Flow

Audio → Whisper Reference
        ↓
Normalize Text
        ↓
Compute Metrics
        ↓
Normalize Metrics
        ↓
Compute CQS
        ↓
Rank Transcriptions
        ↓
UI displays & weight tuning


⸻

14. Work Split (3 Teams)

Team 1 — ASR & Preprocessing

Responsibilities
	•	audio normalization pipeline
	•	Whisper transcription generation
	•	text normalization
	•	storage & caching
	•	timestamp extraction (optional)

Deliverables
	•	reference transcript API
	•	normalized text outputs

⸻

Team 2 — Metrics & Scoring Engine

Responsibilities
	•	implement atomic metrics
	•	normalization logic
	•	embedding similarity
	•	fluency & punctuation scoring
	•	CQS computation engine
	•	batch processing optimization

Deliverables
	•	metrics computation service
	•	scoring API
	•	evaluation & calibration tools

⸻

Team 3 — UI & Interaction Layer

Responsibilities
	•	weight adjustment panel
	•	real-time ranking updates
	•	visualization of metric breakdown
	•	preset profiles
	•	performance optimization

Deliverables
	•	interactive ranking dashboard
	•	API integration
	•	usability testing

⸻

If needed, next step options:
	•	propose default weights based on research
	•	suggest metric formulas & pseudocode
	•	outline evaluation methodology
	•	design database schema