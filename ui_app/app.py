import os
import sys
import tempfile
import time
import math
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import torch
import plotly.express as px
import plotly.graph_objects as go
import requests

# Add project root to sys.path to import backend modules
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from metrics_engine.metrics import run_scoring_pipeline, PRESET_WEIGHTS, DEFAULT_WEIGHTS
from preprocessing.transcribe import build_pipeline, detect_language_for_file, MODEL_SNAPSHOT

# ---------------------------------------------------------------------------
# Page config & custom CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Transcription Assessment Dashboard",
    page_icon="https://em-content.zobj.net/source/twitter/408/studio-microphone_1f399-fe0f.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS for a polished, modern look
st.markdown(
    """
    <style>
    /* ── Global ─────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }

    /* Main background */
    .stApp { background: linear-gradient(175deg, #0f1117 0%, #161b22 100%); }

    /* Header banner */
    .hero-banner {
        background: linear-gradient(135deg, #1e3a5f 0%, #0d253f 60%, #0a1628 100%);
        border: 1px solid rgba(56, 139, 253, 0.15);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .hero-banner h1 {
        color: #e6edf3;
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.25rem 0;
        letter-spacing: -0.5px;
    }
    .hero-banner p {
        color: #8b949e;
        font-size: 1rem;
        margin: 0;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: #161b22 !important;
        border-right: 1px solid rgba(56, 139, 253, 0.1);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 { color: #e6edf3; }

    /* Card style for metric sections */
    .metric-card {
        background: rgba(22, 27, 34, 0.85);
        border: 1px solid rgba(56, 139, 253, 0.12);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .metric-card h3 { color: #58a6ff; margin-top: 0; font-size: 1.1rem; }

    /* Weight chip */
    .weight-chip {
        display: inline-block;
        background: rgba(56, 139, 253, 0.12);
        border: 1px solid rgba(56, 139, 253, 0.3);
        border-radius: 999px;
        padding: 0.25rem 0.75rem;
        margin: 0.15rem 0.2rem;
        font-size: 0.82rem;
        color: #79c0ff;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #238636, #2ea043) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.5rem !important;
        transition: transform 0.1s, box-shadow 0.2s !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 16px rgba(46, 160, 67, 0.3) !important;
    }

    /* Dataframe */
    .stDataFrame { border-radius: 10px; overflow: hidden; }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load ASR model (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_asr():
    return build_pipeline(MODEL_SNAPSHOT)

with st.spinner("Loading Whisper Model... This may take a minute on first run."):
    asr, model, processor, device, dtype = load_asr()

# ---------------------------------------------------------------------------
# Hero banner
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <h1>Transcription Quality &amp; Ranking Dashboard</h1>
        <p>Upload audio and candidate transcriptions to generate a Ground Truth and rank candidates with configurable metrics.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR  — Interactive weight configuration
# ═══════════════════════════════════════════════════════════════════════════════
st.sidebar.markdown("## Scoring Configuration")

preset = st.sidebar.selectbox(
    "Load Preset",
    ["Custom", "accuracy_focused", "readability_focused", "semantic_fidelity"],
    help="Choose a preset or customise weights manually.",
)

# Initialise session-state weights on first run / preset change
if "weights" not in st.session_state or st.session_state.get("_last_preset") != preset:
    if preset != "Custom":
        st.session_state.weights = dict(PRESET_WEIGHTS[preset])
    else:
        st.session_state.weights = dict(DEFAULT_WEIGHTS)
    st.session_state._last_preset = preset

METRIC_LABELS = {
    "wer": "WER",
    "cer": "CER",
    "completeness": "Completeness",
    "semantic_similarity": "Semantic Similarity",
    "precision": "Precision",
    "recall": "Recall",
    "fluency": "Fluency",
    "punctuation": "Punctuation",
}

METRIC_COLORS = {
    "wer": "#f85149",
    "cer": "#ff7b72",
    "completeness": "#3fb950",
    "semantic_similarity": "#58a6ff",
    "precision": "#d2a8ff",
    "recall": "#79c0ff",
    "fluency": "#f0883e",
    "punctuation": "#e3b341",
}

st.sidebar.markdown("---")
st.sidebar.markdown("### Adjust Metric Weights")
st.sidebar.caption("Drag the sliders — weights are auto-normalised to sum to 1.")

# Sliders that write raw values into session state
for key in DEFAULT_WEIGHTS:
    st.session_state.weights[key] = st.sidebar.slider(
        METRIC_LABELS[key],
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.weights.get(key, DEFAULT_WEIGHTS[key]),
        step=0.01,
        key=f"slider_{key}",
    )

# Normalise so they sum to 1
raw_total = sum(st.session_state.weights.values())
if raw_total > 0:
    normalized_weights = {k: round(v / raw_total, 4) for k, v in st.session_state.weights.items()}
else:
    normalized_weights = {k: round(1.0 / len(DEFAULT_WEIGHTS), 4) for k in DEFAULT_WEIGHTS}

# ── Interactive donut / pie chart for normalised weights ──────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("### Effective Weight Distribution")

pie_labels = [METRIC_LABELS[k] for k in normalized_weights]
pie_values = list(normalized_weights.values())
pie_colors = [METRIC_COLORS[k] for k in normalized_weights]

fig_pie = go.Figure(
    data=[
        go.Pie(
            labels=pie_labels,
            values=pie_values,
            hole=0.45,
            marker=dict(colors=pie_colors, line=dict(color="#0d1117", width=2)),
            textinfo="label+percent",
            textposition="outside",
            textfont=dict(size=11, color="#c9d1d9"),
            hovertemplate="<b>%{label}</b><br>Weight: %{value:.2%}<extra></extra>",
            pull=[0.03] * len(pie_values),
        )
    ]
)
fig_pie.update_layout(
    showlegend=False,
    margin=dict(t=10, b=10, l=10, r=10),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=320,
    font=dict(color="#c9d1d9"),
)
st.sidebar.plotly_chart(fig_pie, use_container_width=True)

# Weight chips summary
chips_html = "".join(
    f'<span class="weight-chip">{METRIC_LABELS[k]}: {v:.0%}</span>'
    for k, v in normalized_weights.items()
    if v > 0
)
st.sidebar.markdown(chips_html, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["Single Transcription", "Batch Verification"])

# ───────────────────────────────────────────────────────────────────────────────
# TAB 1 — Single Transcription
# ───────────────────────────────────────────────────────────────────────────────
with tab1:
    col_upload, col_candidates = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown(
            '<div class="metric-card"><h3>1 &mdash; Upload Audio</h3></div>',
            unsafe_allow_html=True,
        )
        audio_file = st.file_uploader("Choose a .wav file", type=["wav"], key="single_audio")
        if audio_file:
            st.audio(audio_file)

    with col_candidates:
        st.markdown(
            '<div class="metric-card"><h3>2 &mdash; Candidate Transcriptions</h3></div>',
            unsafe_allow_html=True,
        )
        candidates_input = st.text_area(
            "Enter candidate transcriptions (one per line)",
            placeholder="Candidate 1 text...\nCandidate 2 text...",
            height=200,
        )

    # Run button
    run_single = st.button("Run Assessment", key="run_single", use_container_width=True)

    if run_single:
        if not audio_file or not candidates_input.strip():
            st.warning("Please upload an audio file and provide at least one candidate transcription.")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_file.getvalue())
                tmp_path = Path(tmp_file.name)

            try:
                # ── Generate ground truth ────────────────────────────────
                with st.spinner("Generating Ground Truth (Whisper)..."):
                    lang_code, lang_name = detect_language_for_file(
                        tmp_path, model, processor, device, dtype
                    )
                    import soundfile as sf

                    audio_data, sr = sf.read(str(tmp_path))
                    audio_input = {"raw": audio_data, "sampling_rate": sr}

                    t0 = time.time()
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
                    elapsed_gt = time.time() - t0

                st.success(f"Ground Truth generated in {elapsed_gt:.1f}s (Detected: {lang_name})")
                st.text_area("Generated Ground Truth", reference_text, height=100, disabled=True)

                # ── Compute metrics & ranking ────────────────────────────
                with st.spinner("Computing Metrics & Ranking..."):
                    candidate_lines = [
                        line.strip() for line in candidates_input.split("\n") if line.strip()
                    ]
                    candidates = [
                        {"transcription_id": f"T{i+1}", "text": text}
                        for i, text in enumerate(candidate_lines)
                    ]

                    scoring_result = run_scoring_pipeline(
                        audio_id=audio_file.name,
                        reference=reference_text,
                        candidates=candidates,
                        weights=normalized_weights,
                    )

                # ── Display results ──────────────────────────────────────
                st.markdown("---")
                st.markdown(
                    '<div class="metric-card"><h3>3 &mdash; Assessment Results</h3></div>',
                    unsafe_allow_html=True,
                )

                results_df = []
                for res in scoring_result["results"]:
                    m = res["metrics"]
                    text = next(
                        (c["text"] for c in candidates if c["transcription_id"] == res["transcription_id"]),
                        "",
                    )
                    results_df.append(
                        {
                            "Rank": res["rank"],
                            "ID": res["transcription_id"],
                            "CQS": round(res["cqs_score"], 4),
                            "Transcription": text,
                            "WER": round(m.get("wer", 0), 4),
                            "CER": round(m.get("cer", 0), 4),
                            "Semantic Sim": round(m.get("semantic_similarity", 0), 4),
                            "Completeness": round(m.get("completeness_score", 0), 4),
                            "Precision": round(m.get("precision", 0), 4),
                            "Recall": round(m.get("recall", 0), 4),
                        }
                    )

                df = pd.DataFrame(results_df)
                st.dataframe(
                    df.style.highlight_max(
                        axis=0,
                        subset=["CQS", "Semantic Sim", "Completeness", "Precision", "Recall"],
                        color="#238636",
                    ).highlight_min(axis=0, subset=["WER", "CER"], color="#238636"),
                    use_container_width=True,
                )

                # ── Charts row ───────────────────────────────────────────
                st.markdown("---")
                chart_col1, chart_col2 = st.columns(2, gap="large")

                # Bar chart comparison
                with chart_col1:
                    st.markdown("#### Metric Comparison")
                    bar_metrics = ["CQS", "Semantic Sim", "Completeness", "Precision", "Recall"]
                    bar_data = df.melt(
                        id_vars=["ID"],
                        value_vars=bar_metrics,
                        var_name="Metric",
                        value_name="Score",
                    )
                    fig_bar = px.bar(
                        bar_data,
                        x="ID",
                        y="Score",
                        color="Metric",
                        barmode="group",
                        color_discrete_sequence=["#58a6ff", "#3fb950", "#d2a8ff", "#79c0ff", "#f0883e"],
                        template="plotly_dark",
                    )
                    fig_bar.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#c9d1d9"),
                        legend=dict(orientation="h", y=-0.2),
                        margin=dict(t=20, b=60, l=40, r=20),
                        yaxis=dict(range=[0, 1]),
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                # Polar area chart per transcript
                with chart_col2:
                    st.markdown("#### Polar Area Chart per Transcript")
                    polar_metrics = ["CQS", "Semantic Sim", "Completeness", "Precision", "Recall"]
                    polar_colors = ["#f85149", "#58a6ff", "#3fb950", "#d2a8ff", "#79c0ff", "#f0883e", "#e3b341"]

                    fig_polar = go.Figure()
                    for idx, row in df.iterrows():
                        r_values = [row[m] for m in polar_metrics]
                        fig_polar.add_trace(
                            go.Barpolar(
                                r=r_values,
                                theta=polar_metrics,
                                name=row["ID"],
                                marker_color=polar_colors[idx % len(polar_colors)],
                                marker_line_color="#0d1117",
                                marker_line_width=1,
                                opacity=0.8,
                            )
                        )
                    fig_polar.update_layout(
                        template="plotly_dark",
                        polar=dict(
                            bgcolor="rgba(0,0,0,0)",
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1],
                                gridcolor="rgba(139,148,158,0.15)",
                                tickfont=dict(color="#8b949e", size=10),
                            ),
                            angularaxis=dict(
                                gridcolor="rgba(139,148,158,0.15)",
                                tickfont=dict(color="#c9d1d9", size=11),
                            ),
                        ),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#c9d1d9"),
                        legend=dict(orientation="h", y=-0.15),
                        margin=dict(t=30, b=60, l=40, r=40),
                        showlegend=True,
                    )
                    st.plotly_chart(fig_polar, use_container_width=True)

                # ── Individual polar area charts per transcript ──────────
                st.markdown("---")
                st.markdown("#### Individual Polar Area Charts")
                ind_cols = st.columns(min(len(df), 4))
                for idx, row in df.iterrows():
                    with ind_cols[idx % len(ind_cols)]:
                        r_values = [row[m] for m in polar_metrics]
                        fig_ind = go.Figure(
                            data=[
                                go.Barpolar(
                                    r=r_values,
                                    theta=polar_metrics,
                                    marker_color=polar_colors[idx % len(polar_colors)],
                                    marker_line_color="#0d1117",
                                    marker_line_width=1,
                                    opacity=0.85,
                                )
                            ]
                        )
                        fig_ind.update_layout(
                            title=dict(
                                text=f"{row['ID']} (CQS: {row['CQS']:.3f})",
                                font=dict(size=14, color="#e6edf3"),
                                x=0.5,
                            ),
                            template="plotly_dark",
                            polar=dict(
                                bgcolor="rgba(0,0,0,0)",
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 1],
                                    gridcolor="rgba(139,148,158,0.15)",
                                    tickfont=dict(color="#8b949e", size=9),
                                ),
                                angularaxis=dict(
                                    gridcolor="rgba(139,148,158,0.15)",
                                    tickfont=dict(color="#c9d1d9", size=10),
                                ),
                            ),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#c9d1d9"),
                            showlegend=False,
                            margin=dict(t=50, b=20, l=30, r=30),
                            height=300,
                        )
                        st.plotly_chart(fig_ind, use_container_width=True)

            except Exception as e:
                st.error(f"Error during processing: {e}")
            finally:
                if tmp_path.exists():
                    os.remove(tmp_path)

# ───────────────────────────────────────────────────────────────────────────────
# TAB 2 — Batch Verification
# ───────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown(
        '<div class="metric-card"><h3>Batch Verification from CSV</h3></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Upload a CSV with columns: `audio_id`, `audio` (URL to .wav), "
        "`option_1` through `option_5`, and `correct_option` (the option number, e.g. 1-5)."
    )

    csv_file = st.file_uploader(
        "Upload CSV", type=["csv"], key="batch_csv"
    )

    run_batch = st.button("Run Batch Verification", key="run_batch", use_container_width=True)

    if run_batch:
        if not csv_file:
            st.warning("Please upload a CSV file first.")
        else:
            try:
                import soundfile as sf

                df_csv = pd.read_csv(csv_file)
                required_cols = [
                    "audio_id", "audio",
                    "option_1", "option_2", "option_3", "option_4", "option_5",
                    "correct_option",
                ]
                missing_cols = [c for c in required_cols if c not in df_csv.columns]

                if missing_cols:
                    st.error(f"Missing required columns: {', '.join(missing_cols)}")
                else:
                    st.info(f"Loaded {len(df_csv)} rows. Starting batch processing...")

                    correct_count = 0
                    total_count = 0
                    results_data = []

                    progress_bar = st.progress(0, text="Starting...")
                    status_text = st.empty()

                    for index, row in df_csv.iterrows():
                        audio_url = row["audio"]
                        audio_id = row["audio_id"]
                        correct_opt = str(row["correct_option"]).strip()
                        batch_tmp_path = None

                        status_text.text(f"Processing row {index + 1}/{len(df_csv)}: {audio_id}")

                        try:
                            # Download audio to temp file
                            response = requests.get(audio_url, timeout=60)
                            response.raise_for_status()

                            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                                tmp_file.write(response.content)
                                batch_tmp_path = Path(tmp_file.name)

                            # Detect language
                            lang_code, lang_name = detect_language_for_file(
                                batch_tmp_path, model, processor, device, dtype
                            )

                            # Read audio data
                            audio_data, sr = sf.read(str(batch_tmp_path))
                            audio_input = {"raw": audio_data, "sampling_rate": sr}

                            # Transcribe (ground truth)
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

                            # Build candidates from option columns
                            candidates = []
                            for i in range(1, 6):
                                opt_text = row.get(f"option_{i}")
                                if pd.notna(opt_text) and str(opt_text).strip():
                                    candidates.append(
                                        {
                                            "transcription_id": str(i),
                                            "text": str(opt_text).strip(),
                                        }
                                    )

                            if candidates:
                                scoring_result = run_scoring_pipeline(
                                    audio_id=str(audio_id),
                                    reference=reference_text,
                                    candidates=candidates,
                                    weights=normalized_weights,
                                )

                                top_rank_id = scoring_result["results"][0]["transcription_id"]
                                is_correct = top_rank_id == correct_opt
                                if is_correct:
                                    correct_count += 1
                                total_count += 1

                                results_data.append(
                                    {
                                        "Audio ID": audio_id,
                                        "Language": lang_name,
                                        "Top Rank (Predicted)": top_rank_id,
                                        "Correct Option": correct_opt,
                                        "CQS": scoring_result["results"][0]["cqs_score"],
                                        "Match": is_correct,
                                    }
                                )

                        except Exception as e:
                            results_data.append(
                                {
                                    "Audio ID": audio_id,
                                    "Language": "-",
                                    "Top Rank (Predicted)": "-",
                                    "Correct Option": correct_opt,
                                    "CQS": 0,
                                    "Match": False,
                                }
                            )
                            st.warning(f"Row {index + 1} ({audio_id}): {e}")
                        finally:
                            if batch_tmp_path is not None and batch_tmp_path.exists():
                                os.remove(batch_tmp_path)

                        progress_bar.progress(
                            (index + 1) / len(df_csv),
                            text=f"Processed {index + 1} / {len(df_csv)}",
                        )

                    status_text.empty()
                    progress_bar.empty()

                    # ── Results display ──────────────────────────────────
                    st.markdown("---")
                    st.markdown(
                        '<div class="metric-card"><h3>Batch Results</h3></div>',
                        unsafe_allow_html=True,
                    )

                    if total_count > 0:
                        accuracy = correct_count / total_count
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Total Processed", total_count)
                        m2.metric("Correct Predictions", correct_count)
                        m3.metric("Accuracy", f"{accuracy:.1%}")

                    results_df_batch = pd.DataFrame(results_data)
                    st.dataframe(
                        results_df_batch.style.map(
                            lambda v: "background-color: #238636; color: white"
                            if v is True
                            else (
                                "background-color: #da3633; color: white"
                                if v is False
                                else ""
                            ),
                            subset=["Match"],
                        ),
                        use_container_width=True,
                    )

                    # Accuracy pie chart
                    if total_count > 0:
                        fig_acc = go.Figure(
                            data=[
                                go.Pie(
                                    labels=["Correct", "Incorrect"],
                                    values=[correct_count, total_count - correct_count],
                                    hole=0.5,
                                    marker=dict(
                                        colors=["#3fb950", "#da3633"],
                                        line=dict(color="#0d1117", width=2),
                                    ),
                                    textinfo="label+value+percent",
                                    textfont=dict(color="#e6edf3"),
                                )
                            ]
                        )
                        fig_acc.update_layout(
                            title=dict(
                                text="Prediction Accuracy",
                                font=dict(size=16, color="#e6edf3"),
                                x=0.5,
                            ),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#c9d1d9"),
                            showlegend=False,
                            height=350,
                            margin=dict(t=50, b=20, l=20, r=20),
                        )
                        st.plotly_chart(fig_acc, use_container_width=True)

            except Exception as e:
                st.error(f"Failed to process CSV: {e}")
