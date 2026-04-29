"""Streamlit app that visualises live agent trajectories and ROCKET predictions.

Features:
* Randomly selects a session from the SQLite database.
* Streams the agent actions in real time (using time.sleep to simulate live progression).
* Displays a gauge chart of hallucination probability at each step using Plotly.
* Uses the trained ROCKET model for real predictions.
* Shows model evaluation metrics and prediction verdict with full context.
"""

import sqlite3
import random
import sys
import time
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Make models/ importable
sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.rocket import engineer_features_for_session, FEATURE_COLS

# Configs
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "telemetry.db"
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "rocket_model.pkl"
SIM_DELAY = 0.3  # seconds between simulated steps

st.set_page_config(
    page_title="Agent Telemetry & Hallucination Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Minimal professional styling ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Base */
    html, body, [class*="css"] { font-family: 'Courier New', monospace; }

    /* Metric cards */
    .metric-card {
        background: #0d1117;
        border: 1px solid #21262d;
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 8px;
    }
    .metric-label {
        font-size: 11px;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #8b949e;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #e6edf3;
        line-height: 1.1;
    }
    .metric-sub {
        font-size: 12px;
        color: #6e7681;
        margin-top: 2px;
    }

    /* Verdict banners */
    .verdict-correct {
        background: #0d2818;
        border: 1px solid #238636;
        border-left: 4px solid #3fb950;
        border-radius: 6px;
        padding: 14px 18px;
        margin: 12px 0;
    }
    .verdict-wrong {
        background: #2d1215;
        border: 1px solid #da3633;
        border-left: 4px solid #f85149;
        border-radius: 6px;
        padding: 14px 18px;
        margin: 12px 0;
    }
    .verdict-title {
        font-size: 12px;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .verdict-body {
        font-size: 13px;
        color: #c9d1d9;
        line-height: 1.5;
    }

    /* Session header */
    .session-header {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }
    .session-id {
        font-size: 13px;
        color: #58a6ff;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .session-meta {
        font-size: 12px;
        color: #8b949e;
        margin-top: 4px;
    }

    /* Action log */
    .action-row {
        background: #0d1117;
        border: 1px solid #21262d;
        border-radius: 4px;
        padding: 10px 14px;
        margin-bottom: 6px;
        font-size: 12px;
        color: #c9d1d9;
    }
    .action-type {
        color: #79c0ff;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 10px;
        letter-spacing: 0.1em;
    }

    /* Risk level labels */
    .risk-low  { color: #3fb950; font-weight: 700; }
    .risk-mid  { color: #d29922; font-weight: 700; }
    .risk-high { color: #f85149; font-weight: 700; }

    /* Section dividers */
    .section-label {
        font-size: 10px;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #6e7681;
        border-bottom: 1px solid #21262d;
        padding-bottom: 6px;
        margin: 20px 0 12px 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model():
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    return data


def query_sessions():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT session_id, outcome FROM agent_sessions ORDER BY session_id")
    sessions = cursor.fetchall()
    conn.close()
    return sessions


def get_session_trajectory(session_id):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT action_type, exit_code, time_since_last_action,
               reasoning_length, semantic_similarity, error_keywords,
               is_repeat_command, command, args, timestamp
        FROM agent_actions
        WHERE session_id = ?
        ORDER BY timestamp
        """,
        conn,
        params=[session_id]
    )
    conn.close()

    # Clean action_type — OpenHands parquet often has null/unknown values
    # Show the command content directly rather than inferring misleading labels
    def infer_action_type(row):
        raw = str(row["action_type"]).strip().lower()
        if raw in ("unknown", "none", "nan", "", "null", "think"):
            cmd = str(row.get("command", "")).strip()
            if cmd and cmd not in ("none", "nan", "", "[]", "null"):
                return "bash"
            return "agent"
        return raw

    df["action_type"] = df.apply(infer_action_type, axis=1)
    return df


def risk_label(prob: float) -> str:
    if prob < 20:
        return '<span class="risk-low">LOW</span>'
    elif prob < 50:
        return '<span class="risk-mid">ELEVATED</span>'
    else:
        return '<span class="risk-high">HIGH</span>'


def build_gauge(halluc_prob: float, step: int, total: int) -> go.Figure:
    if halluc_prob < 20:
        bar_color = "#3fb950"
    elif halluc_prob < 50:
        bar_color = "#d29922"
    else:
        bar_color = "#f85149"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(halluc_prob, 1),
        number={"suffix": "%", "font": {"size": 48, "color": "#e6edf3"}},
        domain={"x": [0, 1], "y": [0, 1]},
        title={
            "text": f"Hallucination Probability — Step {step} of {total}",
            "font": {"size": 13, "color": "#8b949e"}
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickcolor": "#6e7681",
                "tickfont": {"size": 11, "color": "#6e7681"}
            },
            "bar": {"color": bar_color, "thickness": 0.3},
            "bgcolor": "#0d1117",
            "bordercolor": "#21262d",
            "steps": [
                {"range": [0,  20], "color": "#0d2818"},
                {"range": [20, 50], "color": "#272115"},
                {"range": [50, 100], "color": "#2d1215"},
            ],
            "threshold": {
                "line": {"color": "#e6edf3", "width": 2},
                "thickness": 0.8,
                "value": 50
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        margin=dict(t=60, b=20, l=30, r=30),
        height=280,
        font={"family": "Courier New, monospace"}
    )
    return fig


# ── Sidebar — Model Stats ─────────────────────────────────────────────────────

def render_sidebar(model_data: dict):
    st.sidebar.markdown('<div class="section-label">Model Performance</div>', unsafe_allow_html=True)

    test_acc = model_data.get("test_accuracy")
    cv_mean  = model_data.get("cv_mean")
    cv_std   = model_data.get("cv_std")

    if test_acc is not None:
        st.sidebar.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Test Accuracy (held-out)</div>
            <div class="metric-value">{test_acc:.1%}</div>
            <div class="metric-sub">+{test_acc - 0.5:.1%} above random baseline</div>
        </div>
        """, unsafe_allow_html=True)

    if cv_mean is not None and cv_std is not None:
        st.sidebar.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">5-Fold Cross-Validation</div>
            <div class="metric-value">{cv_mean:.1%}</div>
            <div class="metric-sub">Std deviation: {cv_std:.1%} across folds</div>
        </div>
        """, unsafe_allow_html=True)

    st.sidebar.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Model Architecture</div>
        <div class="metric-value" style="font-size:16px;">ROCKET</div>
        <div class="metric-sub">MiniRocket — 10,000 kernels<br>11 telemetry features per step</div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Training Dataset</div>
        <div class="metric-value" style="font-size:16px;">400</div>
        <div class="metric-sub">Sessions — 17,186 total actions<br>200 success / 200 hallucination</div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown('<div class="section-label">Threshold</div>', unsafe_allow_html=True)
    st.sidebar.markdown("""
    <div class="metric-card">
        <div class="metric-label">Intervention Threshold</div>
        <div class="metric-value" style="font-size:22px; color:#d29922;">50%</div>
        <div class="metric-sub">Flag for human review above this risk level</div>
    </div>
    """, unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────

model_data   = load_model()
all_sessions = query_sessions()
max_timesteps = model_data.get("max_timesteps", 525)

render_sidebar(model_data)

# Page title
st.markdown("## Agent Telemetry and Hallucination Predictor")
st.markdown(
    '<p style="color:#8b949e; font-size:13px; margin-top:-8px;">Real-time failure prediction on autonomous coding agent trajectories using ROCKET time-series classification.</p>',
    unsafe_allow_html=True
)

# Session selector
col_sel, col_rand = st.columns([3, 1])
with col_sel:
    session_options = [f"{sid}  |  {out}" for sid, out in all_sessions]
    selected = st.selectbox("Select session", session_options, index=0, label_visibility="collapsed")
    sess_id  = selected.split("  |  ")[0].strip()
    outcome  = selected.split("  |  ")[1].strip()

with col_rand:
    if st.button("Random Session", use_container_width=True):
        pick = random.choice(all_sessions)
        sess_id, outcome = pick[0], pick[1]

# Session header
traj = get_session_trajectory(sess_id)

st.markdown(f"""
<div class="session-header">
    <div class="session-id">{sess_id}</div>
    <div class="session-meta">
        Ground truth outcome: <strong style="color:#e6edf3;">{outcome.upper()}</strong>
        &nbsp;&nbsp;|&nbsp;&nbsp;
        Total actions: <strong style="color:#e6edf3;">{len(traj)}</strong>
        &nbsp;&nbsp;|&nbsp;&nbsp;
        Source: OpenHands SWE-bench trajectories
    </div>
</div>
""", unsafe_allow_html=True)

if len(traj) == 0:
    st.warning(f"Session {sess_id} has no recorded actions.")
    st.stop()

# Layout — gauge left, step counter right
col_gauge, col_action = st.columns([3, 2])

gauge_slot  = col_gauge.empty()
action_slot = col_action.empty()

# Initialise
pred_class  = "N/A"
halluc_prob = 0.0

# Stream predictions step by step
for step_idx in range(len(traj)):
    cur_df   = traj.iloc[:step_idx + 1].copy()
    features = engineer_features_for_session(cur_df)
    n_features, cur_steps = features.shape

    X_padded = np.zeros((1, n_features, max_timesteps))
    X_padded[0, :, :cur_steps] = features

    prob           = model_data["model"].predict_proba(X_padded)
    pred_class_idx = prob[0].argmax()
    pred_class     = model_data["label_encoder"].classes_[pred_class_idx]

    classes = list(model_data["label_encoder"].classes_)
    if "hallucination" in classes:
        hal_idx     = classes.index("hallucination")
        halluc_prob = float(prob[0, hal_idx]) * 100
    else:
        halluc_prob = 0.0

    # Gauge — use width='stretch' to suppress deprecation warning
    gauge_slot.plotly_chart(
        build_gauge(halluc_prob, step_idx + 1, len(traj)),
        width="stretch"
    )

    # Step counter — clean, no action type labels
    risk_color = "#3fb950" if halluc_prob < 20 else ("#d29922" if halluc_prob < 50 else "#f85149")
    risk_text  = "LOW" if halluc_prob < 20 else ("ELEVATED" if halluc_prob < 50 else "HIGH")

    action_slot.markdown(f"""
    <div class="action-row">
        <div style="font-size:11px; color:#8b949e;">
            Step {step_idx + 1} of {len(traj)}
            &nbsp;&nbsp;|&nbsp;&nbsp;
            Risk: <span style="color:{risk_color}; font-weight:700;">{risk_text}</span>
            &nbsp;&nbsp;|&nbsp;&nbsp;
            Probability: <span style="color:{risk_color}; font-weight:700;">{halluc_prob:.1f}%</span>
        </div>
        <div style="margin-top:10px;">
            <div style="height:4px; background:#21262d; border-radius:2px;">
                <div style="height:4px; width:{min(halluc_prob,100):.0f}%; background:{risk_color}; border-radius:2px; transition:width 0.3s;"></div>
            </div>
        </div>
        <div style="margin-top:8px; font-size:11px; color:#6e7681;">
            {"Intervention recommended — halt agent" if halluc_prob >= 50 else "Trajectory nominal — no action required"}
        </div>
    </div>
    """, unsafe_allow_html=True)

    time.sleep(SIM_DELAY)

# ── Final Summary ─────────────────────────────────────────────────────────────

prediction_correct = (pred_class.lower() == outcome.lower())

st.markdown('<div class="section-label">Session Summary</div>', unsafe_allow_html=True)
m1, m2, m3 = st.columns(3)

m1.markdown(f"""
<div class="metric-card">
    <div class="metric-label">Final Prediction</div>
    <div class="metric-value" style="font-size:18px; color:{'#f85149' if pred_class.lower() == 'hallucination' else '#3fb950'};">
        {pred_class.upper()}
    </div>
</div>
""", unsafe_allow_html=True)

m2.markdown(f"""
<div class="metric-card">
    <div class="metric-label">Hallucination Risk</div>
    <div class="metric-value" style="font-size:18px; color:{'#f85149' if halluc_prob >= 50 else '#3fb950'};">
        {halluc_prob:.1f}%
    </div>
</div>
""", unsafe_allow_html=True)

m3.markdown(f"""
<div class="metric-card">
    <div class="metric-label">Total Actions</div>
    <div class="metric-value" style="font-size:18px;">{len(traj)}</div>
    <div class="metric-sub">{"Intervention recommended" if halluc_prob >= 50 else "No intervention required"}</div>
</div>
""", unsafe_allow_html=True)