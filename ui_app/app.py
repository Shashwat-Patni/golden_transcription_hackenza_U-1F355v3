import os
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

# Add project root to sys.path to import backend modules
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from metrics_engine.metrics import run_scoring_pipeline, PRESET_WEIGHTS, DEFAULT_WEIGHTS
from preprocessing.transcribe import build_pipeline, detect_language_for_file, MODEL_SNAPSHOT

# Page config
st.set_page_config(page_title="Transcription Assessment Dashboard", layout="wide")

# Session state for model
@st.cache_resource
def load_asr():
    return build_pipeline(MODEL_SNAPSHOT)

with st.spinner("Loading Whisper Model... This may take a minute on first run."):
    # Unpacking into local variables
    asr, model, processor, device, dtype = load_asr()

# Sidebar - Configurable Weights
st.sidebar.header("Scoring Configuration")

preset = st.sidebar.selectbox("Load Preset", ["Custom", "accuracy_focused", "readability_focused", "semantic_fidelity"])

weights = {}
if preset != "Custom":
    current_weights = PRESET_WEIGHTS[preset]
else:
    current_weights = DEFAULT_WEIGHTS

st.sidebar.subheader("Metric Weights")
for key, val in DEFAULT_WEIGHTS.items():
    weights[key] = st.sidebar.slider(f"{key.replace('_', ' ').title()}", 0.0, 1.0, current_weights.get(key, val), 0.05)

# Main UI
st.title("🎙️ Transcription Quality & Ranking Dashboard")
st.markdown("Upload a .wav file and candidate transcriptions to generate a Ground Truth and rank the candidates.")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("1. Upload Audio")
    audio_file = st.file_uploader("Choose a .wav file", type=["wav"])
    if audio_file:
        st.audio(audio_file)

with col2:
    st.header("2. Candidate Transcriptions")
    candidates_input = st.text_area(
        "Enter candidate transcriptions (one per line)",
        placeholder="Candidate 1 text...\nCandidate 2 text...",
        height=200
    )

# Improved evaluation logic flow
if st.button("Run Assessment"):
    if not audio_file or not candidates_input.strip():
        st.warning("⚠️ Please upload an audio file and provide at least one candidate transcription.")
    else:
        # Save uploaded file to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_file.getvalue())
            tmp_path = Path(tmp_file.name)

        try:
            with st.spinner("Generating Ground Truth (Whisper)..."):
                # Pass the local variables directly, NO st.session_state
                lang_code, lang_name = detect_language_for_file(
                    tmp_path,
                    model,
                    processor, 
                    device,    
                    dtype      
                )

                # Transcription
                t0 = time.time()
                result = asr(
                    str(tmp_path),
                    generate_kwargs={"language": lang_code, "task": "transcribe"}
                )
                reference_text = result.get("text", "").strip()
                elapsed_gt = time.time() - t0

                st.success(f"Ground Truth Generated in {elapsed_gt:.1f}s (Detected: {lang_name})")
                st.text_area("Generated Ground Truth", reference_text, height=100)

            with st.spinner("Computing Metrics & Ranking..."):
                candidate_lines = [line.strip() for line in candidates_input.split("\n") if line.strip()]
                candidates = [{"transcription_id": f"T{i+1}", "text": text} for i, text in enumerate(candidate_lines)]

                scoring_result = run_scoring_pipeline(
                    audio_id=audio_file.name,
                    reference=reference_text,
                    candidates=candidates,
                    weights=weights
                )

                # Display Results
                st.header("3. Assessment Results")

                results_df = []
                for res in scoring_result["results"]:
                    m = res["metrics"]
                    row = {
                        "Rank": res["rank"],
                        "ID": res["transcription_id"],
                        "CQS": res["cqs_score"],
                        "Transcription": [c["text"] for c in candidates if c["transcription_id"] == res["transcription_id"]][0],
                        "WER": m.get("wer", 0),
                        "CER": m.get("cer", 0),
                        "Semantic Sim": m.get("semantic_similarity", 0),
                        "Completeness": m.get("completeness_score", 0),
                        "Precision": m.get("precision", 0),
                        "Recall": m.get("recall", 0),
                    }
                    results_df.append(row)

                df = pd.DataFrame(results_df)
                st.dataframe(
                    df.style.highlight_max(axis=0, subset=['CQS', 'Semantic Sim', 'Completeness', 'Precision', 'Recall'], color='lightgreen')
                            .highlight_min(axis=0, subset=['WER', 'CER'], color='lightgreen'),
                    use_container_width=True
                )

                # Charts
                st.subheader("Metric Comparison")
                chart_data = df.melt(id_vars=["ID"], value_vars=["CQS", "Semantic Sim", "Completeness", "Precision", "Recall"], var_name="Metric", value_name="Score")
                st.bar_chart(chart_data, x="ID", y="Score", color="Metric", stack=False)

        finally:
            if tmp_path.exists():
                os.remove(tmp_path)