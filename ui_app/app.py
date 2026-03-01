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

import plotly.express as px
import plotly.graph_objects as go

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

# Calculate normalized weights to sum to 1
total_weight = sum(weights.values())
normalized_weights = {k: v / total_weight if total_weight > 0 else 0 for k, v in weights.items()}

# Display pie chart
st.sidebar.subheader("Effective Normalized Weights")
fig_pie = px.pie(
    names=list(normalized_weights.keys()),
    values=list(normalized_weights.values()),
    title="Normalized Weights",
    hole=0.3
)
fig_pie.update_layout(margin=dict(t=30, b=0, l=0, r=0))
st.sidebar.plotly_chart(fig_pie, use_container_width=True)

import requests
from urllib.parse import urlparse
import io

# Main UI
st.title("🎙️ Transcription Quality & Ranking Dashboard")
st.markdown("Upload a .wav file and candidate transcriptions to generate a Ground Truth and rank the candidates.")

tab1, tab2 = st.tabs(["Single Transcription", "Batch Verification"])

with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("1. Upload Audio")
        audio_file = st.file_uploader("Choose a .wav file", type=["wav"], key="single_audio")
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

                    import soundfile as sf
                    audio_data, sr = sf.read(str(tmp_path))
                    audio_input = {"raw": audio_data, "sampling_rate": sr}

                    # Transcription
                    t0 = time.time()
                    result = asr(
                            audio_input, # or whatever variable you are passing here
                            chunk_length_s=30,
                            stride_length_s=5,
                            return_timestamps=True,
                            generate_kwargs={
                                "language": lang_code,
                                "task": "transcribe",
                                "return_timestamps": True,
                            },
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
                        weights=normalized_weights
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

                    st.subheader("Polar Area Chart per Transcription")
                    radar_metrics = ["CQS", "Semantic Sim", "Completeness", "Precision", "Recall"]

                    # We can draw all transcripts on the same polar chart or individual ones
                    fig_radar = go.Figure()
                    for i, row in df.iterrows():
                        fig_radar.add_trace(go.Scatterpolar(
                            r=[row[m] for m in radar_metrics],
                            theta=radar_metrics,
                            fill='toself',
                            name=row['ID']
                        ))
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1]
                            )),
                        showlegend=True
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

            except Exception as e:
                st.error(f"Error during processing: {e}")
            finally:
                if tmp_path.exists():
                    os.remove(tmp_path)

with tab2:
    st.header("Batch Verification from CSV")
    csv_file = st.file_uploader("Upload CSV containing 'audio', 'option_1'-'option_5', and 'correct_option'", type=["csv"], key="batch_csv")

    if st.button("Run Verification"):
        if not csv_file:
            st.warning("⚠️ Please upload a CSV file.")
        else:
            try:
                import pandas as pd
                # Read CSV
                df_csv = pd.read_csv(csv_file)
                required_cols = ["audio_id", "audio", "option_1", "option_2", "option_3", "option_4", "option_5", "correct_option"]
                missing_cols = [c for c in required_cols if c not in df_csv.columns]

                if missing_cols:
                    st.error(f"Missing required columns: {', '.join(missing_cols)}")
                else:
                    st.info(f"Loaded {len(df_csv)} rows. Starting processing...")

                    correct_count = 0
                    total_count = 0
                    results_data = []

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for index, row in df_csv.iterrows():
                        audio_url = row['audio']
                        audio_id = row['audio_id']
                        correct_opt = str(row['correct_option']).strip()

                        status_text.text(f"Processing row {index + 1}/{len(df_csv)}: {audio_id}")

                        # Download audio to temp file
                        try:
                            response = requests.get(audio_url, timeout=30)
                            response.raise_for_status()
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                                tmp_file.write(response.content)
                                tmp_path = Path(tmp_file.name)

                            # Transcribe to get Ground Truth
                            lang_code, lang_name = detect_language_for_file(
                                tmp_path, model, processor, device, dtype
                            )
                            import soundfile as sf
                            audio_data, sr = sf.read(str(tmp_path))
                            audio_input = {"raw": audio_data, "sampling_rate": sr}

                            result = asr(
                                audio_input,
                                chunk_length_s=30,
                                stride_length_s=5,
                                return_timestamps=True,
                                generate_kwargs={
                                    "language": lang_code,
                                    "task": "transcribe",
                                    "return_timestamps": True,
                                },
                            )
                            reference_text = result.get("text", "").strip()

                            # Prepare candidates
                            candidates = []
                            for i in range(1, 6):
                                opt_text = row.get(f"option_{i}")
                                if pd.notna(opt_text) and str(opt_text).strip():
                                    candidates.append({
                                        "transcription_id": str(i),
                                        "text": str(opt_text).strip()
                                    })

                            if candidates:
                                scoring_result = run_scoring_pipeline(
                                    audio_id=str(audio_id),
                                    reference=reference_text,
                                    candidates=candidates,
                                    weights=normalized_weights
                                )

                                top_rank_id = scoring_result["results"][0]["transcription_id"]
                                is_correct = (top_rank_id == correct_opt)

                                if is_correct:
                                    correct_count += 1
                                total_count += 1

                                results_data.append({
                                    "Audio ID": audio_id,
                                    "Top Rank (Predicted)": top_rank_id,
                                    "Correct Option": correct_opt,
                                    "Match": is_correct
                                })

                        except Exception as e:
                            st.error(f"Error processing row {index + 1} ({audio_id}): {e}")
                        finally:
                            # Clean up tmp_path
                            if 'tmp_path' in locals() and tmp_path.exists():
                                os.remove(tmp_path)

                        progress_bar.progress((index + 1) / len(df_csv))

                    status_text.text("Processing complete.")

                    st.subheader("Verification Results")
                    st.markdown(f"**Passed:** {correct_count} / {total_count}")
                    st.dataframe(pd.DataFrame(results_data), use_container_width=True)

            except Exception as e:
                st.error(f"Failed to read CSV or error during processing: {e}")