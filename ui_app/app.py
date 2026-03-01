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
    ["Custom", "default", "accuracy_focused", "readability_focused", "semantic_fidelity"],
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
st.sidebar.caption("Drag the boundaries between slices on the pie chart to adjust weights.")

import streamlit.components.v1 as components
import json

# Build the interactive movable pie chart HTML/JS component
_pie_keys = list(st.session_state.weights.keys())
_pie_labels = [METRIC_LABELS.get(k, k) for k in _pie_keys]
_pie_colors = [METRIC_COLORS.get(k, "#888") for k in _pie_keys]
_pie_values = [st.session_state.weights[k] for k in _pie_keys]

_movable_pie_html = """
<div id="pie-container" style="width:100%;display:flex;flex-direction:column;align-items:center;user-select:none;">
  <svg id="pie-svg" width="300" height="300" viewBox="0 0 300 300"></svg>
  <div id="pie-legend" style="margin-top:10px;width:100%;text-align:center;"></div>
</div>
<script>
(function() {
  const size = 300;
  const radius = size / 2;
  const keys = """ + json.dumps(_pie_keys) + """;
  const labels = """ + json.dumps(_pie_labels) + """;
  const colors = """ + json.dumps(_pie_colors) + """;

  // Initial slice percentages from weights (normalised to sum to 100)
  let rawValues = """ + json.dumps(_pie_values) + """;
  let rawTotal = rawValues.reduce((a,b) => a+b, 0);
  let slices = rawValues.map(v => rawTotal > 0 ? (v / rawTotal) * 100 : 100 / rawValues.length);

  const svg = document.getElementById('pie-svg');
  const legend = document.getElementById('pie-legend');
  let draggingIndex = -1;

  function toRadians(deg) { return (deg * Math.PI) / 180; }

  function polarToCartesian(cx, cy, r, angle) {
    return {
      x: cx + r * Math.cos(toRadians(angle)),
      y: cy + r * Math.sin(toRadians(angle))
    };
  }

  function describeArc(startAngle, endAngle) {
    const start = polarToCartesian(radius, radius, radius, endAngle);
    const end   = polarToCartesian(radius, radius, radius, startAngle);
    const largeArcFlag = (endAngle - startAngle) <= 180 ? "0" : "1";
    return [
      "M", radius, radius,
      "L", start.x, start.y,
      "A", radius, radius, 0, largeArcFlag, 0, end.x, end.y,
      "Z"
    ].join(" ");
  }

  function getAngles() {
    let total = 0;
    return slices.map(value => {
      const start = total;
      total += (value / 100) * 360;
      return { start, end: total };
    });
  }

  function render() {
    svg.innerHTML = '';
    const angles = getAngles();

    // Draw slices
    for (let i = 0; i < slices.length; i++) {
      const a = angles[i];
      if (a.end - a.start < 0.1) continue;

      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      path.setAttribute('d', describeArc(a.start, a.end));
      path.setAttribute('fill', colors[i]);
      path.setAttribute('stroke', '#0d1117');
      path.setAttribute('stroke-width', '1.5');

      // Hover tooltip
      path.addEventListener('mouseenter', () => {
        path.setAttribute('opacity', '0.85');
      });
      path.addEventListener('mouseleave', () => {
        path.setAttribute('opacity', '1');
      });

      svg.appendChild(path);

      // Label inside slice
      const sliceAngle = a.end - a.start;
      if (sliceAngle > 25) {
        const midAngle = a.start + sliceAngle / 2;
        const labelR = radius * 0.6;
        const lp = polarToCartesian(radius, radius, labelR, midAngle);
        const pct = slices[i].toFixed(1);

        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', lp.x);
        text.setAttribute('y', lp.y);
        text.setAttribute('text-anchor', 'middle');
        text.setAttribute('dominant-baseline', 'central');
        text.setAttribute('fill', '#fff');
        text.setAttribute('font-size', '10');
        text.setAttribute('font-weight', '600');
        text.setAttribute('pointer-events', 'none');
        text.textContent = pct + '%';
        svg.appendChild(text);
      }
    }

    // Draw draggable separator handles at each boundary
    for (let i = 0; i < slices.length; i++) {
      const boundaryAngle = angles[i].end % 360;
      const pos = polarToCartesian(radius, radius, radius, boundaryAngle);

      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circle.setAttribute('cx', pos.x);
      circle.setAttribute('cy', pos.y);
      circle.setAttribute('r', 7);
      circle.setAttribute('fill', '#e6edf3');
      circle.setAttribute('stroke', '#0d1117');
      circle.setAttribute('stroke-width', '2');
      circle.style.cursor = 'grab';
      circle.dataset.index = i;

      circle.addEventListener('mousedown', (e) => {
        e.preventDefault();
        draggingIndex = parseInt(circle.dataset.index);
      });
      circle.addEventListener('touchstart', (e) => {
        e.preventDefault();
        draggingIndex = parseInt(circle.dataset.index);
      }, {passive: false});

      svg.appendChild(circle);
    }

    // Update legend
    legend.innerHTML = keys.map((k, i) => {
      const pct = slices[i].toFixed(1);
      return '<span style="display:inline-block;background:rgba(56,139,253,0.12);border:1px solid ' + colors[i] + '40;border-radius:999px;padding:2px 8px;margin:2px;font-size:11px;color:' + colors[i] + ';white-space:nowrap;">' + labels[i] + ': ' + pct + '%</span>';
    }).join(' ');
  }

  function handleMouseMove(e) {
    if (draggingIndex < 0) return;
    const rect = svg.getBoundingClientRect();
    const cx = rect.left + radius * (rect.width / size);
    const cy = rect.top + radius * (rect.height / size);

    let angle = Math.atan2(e.clientY - cy, e.clientX - cx) * (180 / Math.PI);
    angle = (angle + 360) % 360;

    const angles = getAngles();
    const i = draggingIndex;
    const next = (i + 1) % slices.length;

    const A = angles[i].start;
    const C = angles[next].end % 360;

    let span = C - A;
    if (span <= 0) span += 360;

    let newI = angle - A;
    if (newI < 0) newI += 360;

    // Prevent collapse — minimum 2% per slice
    if (newI < 5 || newI > span - 5) return;

    const pairTotal = slices[i] + slices[next];
    slices[i] = (newI / span) * pairTotal;
    slices[next] = pairTotal - slices[i];

    render();
  }

  function handleMouseUp() {
    if (draggingIndex >= 0) {
      draggingIndex = -1;
      // Post updated weights to parent (Streamlit)
      const totalPct = slices.reduce((a,b) => a+b, 0);
      const result = {};
      keys.forEach((k, idx) => {
        result[k] = Math.round((slices[idx] / totalPct) * 10000) / 10000;
      });
      if (window.parent) {
        window.parent.postMessage({type: 'streamlit:setComponentValue', value: result}, '*');
      }
    }
  }

  document.addEventListener('mousemove', handleMouseMove);
  document.addEventListener('touchmove', (e) => {
    if (draggingIndex >= 0 && e.touches.length > 0) {
      e.preventDefault();
      handleMouseMove(e.touches[0]);
    }
  }, {passive: false});
  document.addEventListener('mouseup', handleMouseUp);
  document.addEventListener('touchend', handleMouseUp);

  render();
})();
</script>
<style>
  #pie-container { font-family: 'Inter', sans-serif; }
  #pie-svg { filter: drop-shadow(0 2px 8px rgba(0,0,0,0.3)); }
</style>
"""

# Render the movable pie chart
_pie_result = components.html(_movable_pie_html, height=420, scrolling=False)

# Since streamlit.components.v1.html doesn't return values directly,
# we provide number inputs as a compact fallback for precise control
st.sidebar.markdown("**Fine-tune values:**")
_cols = st.sidebar.columns(2)
for idx, key in enumerate(_pie_keys):
    with _cols[idx % 2]:
        st.session_state.weights[key] = st.number_input(
            METRIC_LABELS[key],
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.weights.get(key, DEFAULT_WEIGHTS[key]),
            step=0.01,
            format="%.2f",
            key=f"num_{key}",
        )

# Normalise so they sum to 1
raw_total = sum(st.session_state.weights.values())
if raw_total > 0:
    normalized_weights = {k: round(v / raw_total, 4) for k, v in st.session_state.weights.items()}
else:
    normalized_weights = {k: round(1.0 / len(DEFAULT_WEIGHTS), 4) for k in DEFAULT_WEIGHTS}

# Weight chips summary
st.sidebar.markdown("---")
st.sidebar.markdown("### Effective Weight Distribution")
chips_html = "".join(
    f'<span class="weight-chip">{METRIC_LABELS[k]}: {v:.0%}</span>'
    for k, v in normalized_weights.items()
    if v > 0
)
st.sidebar.markdown(chips_html, unsafe_allow_html=True)

# ═══════════════════════════════════���═══════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["Single Audio processing", "Batch Verification"])

# ─────────────────────────────────────────��─────────────────────────────────────
# TAB 1 — Single Audio processing
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

        if "single_candidates" not in st.session_state:
            st.session_state.single_candidates = [""]

        to_delete_idx = None
        for i in range(len(st.session_state.single_candidates)):
            c_col, d_col = st.columns([0.85, 0.15])
            with c_col:
                st.session_state.single_candidates[i] = st.text_area(
                    f"Candidate {i+1}",
                    value=st.session_state.single_candidates[i],
                    key=f"single_cand_box_{i}",
                    height=100,
                    label_visibility="visible" if i == 0 else "collapsed"
                )
            with d_col:
                if i == 0:
                    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                if st.button("🗑️", key=f"del_cand_{i}", help="Remove candidate"):
                    to_delete_idx = i

        if to_delete_idx is not None:
            st.session_state.single_candidates.pop(to_delete_idx)
            # Clear related session state keys to avoid indexing issues
            for k in list(st.session_state.keys()):
                if k.startswith("single_cand_box_"):
                    del st.session_state[k]
            st.rerun()

        if st.button("➕ Add Transcription"):
            st.session_state.single_candidates.append("")
            st.rerun()

    # Run button
    run_single = st.button("Run Assessment", key="run_single", use_container_width=True)

    if run_single:
        # Extract valid candidates
        valid_candidates = [c.strip() for c in st.session_state.single_candidates if c.strip()]

        if not audio_file or not valid_candidates:
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
                    candidates = [
                        {"transcription_id": f"T{i+1}", "text": text}
                        for i, text in enumerate(valid_candidates)
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
                            # Download audio to temp file using streaming to avoid memory issues
                            download_success = False
                            last_download_err = None
                            for attempt in range(1, 4):  # up to 3 retries
                                try:
                                    with requests.get(
                                        audio_url,
                                        stream=True,
                                        timeout=(15, 120),  # (connect, read) timeouts
                                        headers={"User-Agent": "TranscriptionApp/1.0"},
                                        allow_redirects=True,
                                    ) as dl_resp:
                                        dl_resp.raise_for_status()
                                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                                            for chunk in dl_resp.iter_content(chunk_size=65536):
                                                if chunk:
                                                    tmp_file.write(chunk)
                                            batch_tmp_path = Path(tmp_file.name)
                                    download_success = True
                                    break
                                except (requests.RequestException, IOError) as dl_err:
                                    last_download_err = dl_err
                                    if batch_tmp_path is not None and batch_tmp_path.exists():
                                        os.remove(batch_tmp_path)
                                        batch_tmp_path = None
                                    if attempt < 3:
                                        time.sleep(2 ** attempt)  # exponential back-off

                            if not download_success:
                                raise RuntimeError(
                                    f"Failed to download audio after 3 attempts: {last_download_err}"
                                )

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
