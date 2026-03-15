"""
╔══════════════════════════════════════════════════════════╗
║          AquaVision — Water Quality Dashboard            ║
║          Phase 5 of 5 — Streamlit Application            ║
║          Brainybeam Info-Tech PVT LTD                    ║
╚══════════════════════════════════════════════════════════╝
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AquaVision — Water Quality Index",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS — Professional dark-teal theme
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Main background ── */
.stApp {
    background: #0a1628;
    color: #e2e8f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f1f3d !important;
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
}

/* ── Metric cards ── */
.metric-card {
    background: #0f2744;
    border: 1px solid #1e4080;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #3b82f6; }
.metric-value {
    font-size: 2.2em;
    font-weight: 600;
    font-family: 'DM Mono', monospace;
    line-height: 1.1;
}
.metric-label {
    font-size: 0.8em;
    color: #64748b;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── WQI gauge card ── */
.wqi-card {
    background: linear-gradient(135deg, #0f2744 0%, #0a1f38 100%);
    border: 1px solid #1e4080;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
}

/* ── Section headers ── */
.section-header {
    color: #93c5fd;
    font-size: 0.75em;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid #1e3a5f;
}

/* ── Result banner ── */
.result-potable {
    background: linear-gradient(90deg, #064e3b, #065f46);
    border: 1px solid #10b981;
    border-radius: 12px;
    padding: 20px 28px;
    text-align: center;
}
.result-not-potable {
    background: linear-gradient(90deg, #7c1a0a, #991b1b);
    border: 1px solid #ef4444;
    border-radius: 12px;
    padding: 20px 28px;
    text-align: center;
}
.result-title {
    font-size: 1.8em;
    font-weight: 600;
}
.result-subtitle {
    font-size: 0.9em;
    opacity: 0.8;
    margin-top: 4px;
}

/* ── Info box ── */
.info-box {
    background: #0f2744;
    border-left: 3px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.88em;
    color: #94a3b8;
}

/* ── Slider labels ── */
.param-label {
    font-size: 0.82em;
    color: #64748b;
    margin-bottom: 2px;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab"] {
    color: #64748b !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTabs [aria-selected="true"] {
    color: #93c5fd !important;
    border-bottom-color: #3b82f6 !important;
}

/* ── Buttons ── */
.stButton button {
    background: #1e40af !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 0.5rem 1.5rem !important;
    transition: background 0.2s !important;
}
.stButton button:hover {
    background: #2563eb !important;
}

/* ── Divider ── */
hr { border-color: #1e3a5f !important; }

/* ── Header banner ── */
.header-banner {
    background: linear-gradient(135deg, #0f2744 0%, #1e3a5f 50%, #0f2744 100%);
    border: 1px solid #1e4080;
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 24px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PATHS & CONSTANTS
# ─────────────────────────────────────────────────────────────
BASE_DIR       = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR     = os.path.join(BASE_DIR, 'models')
DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')

FEATURES = [
    'ph', 'Hardness', 'Solids', 'Chloramines',
    'Sulfate', 'Conductivity', 'Organic_carbon',
    'Trihalomethanes', 'Turbidity'
]

# WQI parameters (same as Phase 3)
WQI_PARAMS = {
    'ph'              : (7.0,   8.5,   0.133),
    'Hardness'        : (0.0,   200.0, 0.100),
    'Solids'          : (0.0,   150.0, 0.077),
    'Chloramines'     : (0.0,   4.0,   0.122),
    'Sulfate'         : (0.0,   250.0, 0.100),
    'Conductivity'    : (0.0,   400.0, 0.100),
    'Organic_carbon'  : (0.0,   2.0,   0.122),
    'Trihalomethanes' : (0.0,   80.0,  0.122),
    'Turbidity'       : (0.0,   5.0,   0.122),
}

# Parameter metadata for sliders
PARAM_META = {
    'ph'              : {'label': 'pH Level',             'min': 0.0,    'max': 14.0,    'default': 7.0,    'unit': '',       'who': '6.5 – 8.5'},
    'Hardness'        : {'label': 'Hardness',             'min': 47.0,   'max': 323.0,   'default': 196.0,  'unit': 'mg/L',   'who': '< 200 mg/L'},
    'Solids'          : {'label': 'Total Dissolved Solids','min': 320.0, 'max': 61228.0, 'default': 20926.0,'unit': 'ppm',    'who': '< 500 ppm'},
    'Chloramines'     : {'label': 'Chloramines',          'min': 0.35,   'max': 13.13,   'default': 7.1,    'unit': 'ppm',    'who': '< 4 ppm'},
    'Sulfate'         : {'label': 'Sulfate',              'min': 129.0,  'max': 481.0,   'default': 333.0,  'unit': 'mg/L',   'who': '< 250 mg/L'},
    'Conductivity'    : {'label': 'Conductivity',         'min': 181.0,  'max': 753.0,   'default': 426.0,  'unit': 'μS/cm',  'who': '< 400 μS/cm'},
    'Organic_carbon'  : {'label': 'Organic Carbon',       'min': 2.2,    'max': 28.3,    'default': 14.3,   'unit': 'ppm',    'who': '< 2 ppm'},
    'Trihalomethanes' : {'label': 'Trihalomethanes',      'min': 0.74,   'max': 124.0,   'default': 66.4,   'unit': 'μg/L',   'who': '< 80 μg/L'},
    'Turbidity'       : {'label': 'Turbidity',            'min': 1.45,   'max': 6.74,    'default': 3.97,   'unit': 'NTU',    'who': '< 5 NTU'},
}

# ─────────────────────────────────────────────────────────────
# LOAD MODEL & DATA (cached for performance)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(MODELS_DIR, 'wqi_model.pkl')
    if not os.path.exists(model_path):
        return None, None, None
    model   = joblib.load(model_path)
    scaler  = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    with open(os.path.join(MODELS_DIR, 'model_metadata.json')) as f:
        meta = json.load(f)
    return model, scaler, meta

@st.cache_data
def load_dataset():
    path = os.path.join(DATA_PROCESSED, 'cleaned_data.csv')
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

model, scaler, meta = load_model()
df_data = load_dataset()

# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────
def compute_wqi_raw(values: dict) -> float:
    """Compute raw WQI score from parameter dict."""
    wqi = 0
    for param, (ideal, standard, weight) in WQI_PARAMS.items():
        val   = values[param]
        denom = standard - ideal
        qi    = abs((val - ideal) / denom) * 100 if denom != 0 else 0
        wqi  += weight * qi
    return round(wqi, 4)

def normalize_wqi(raw: float, wqi_min: float, wqi_max: float) -> float:
    """Normalize WQI to 0–100."""
    return round((raw - wqi_min) / (wqi_max - wqi_min) * 100, 2)

def wqi_category(score: float) -> tuple:
    """Return (category, color) for a normalized WQI score."""
    if score <= 25:  return "Excellent",  "#10b981"
    elif score <= 50: return "Good",       "#3b82f6"
    elif score <= 75: return "Poor",       "#f59e0b"
    elif score <= 100: return "Very Poor", "#ef4444"
    else:             return "Unsuitable", "#991b1b"

def make_gauge(score: float, category: str, color: str) -> plt.Figure:
    """Draw a semicircular WQI gauge chart."""
    fig, ax = plt.subplots(figsize=(5, 3),
                           subplot_kw=dict(aspect='equal'))
    fig.patch.set_facecolor('#0f2744')
    ax.set_facecolor('#0f2744')

    # Background arc segments
    segments = [
        (0,   25,  "#10b981"),
        (25,  50,  "#3b82f6"),
        (50,  75,  "#f59e0b"),
        (75,  100, "#ef4444"),
    ]
    for s_start, s_end, s_color in segments:
        theta1 = 180 - s_start * 1.8
        theta2 = 180 - s_end   * 1.8
        wedge  = mpatches.Wedge(
            center=(0, 0), r=1.0, theta1=theta2, theta2=theta1,
            width=0.28, facecolor=s_color, alpha=0.25
        )
        ax.add_patch(wedge)

    # Active arc
    active_theta1 = 180
    active_theta2 = 180 - score * 1.8
    if active_theta2 < active_theta1:
        wedge_active = mpatches.Wedge(
            center=(0, 0), r=1.0,
            theta1=active_theta2, theta2=active_theta1,
            width=0.28, facecolor=color, alpha=0.9
        )
        ax.add_patch(wedge_active)

    # Needle
    angle_rad = np.radians(180 - score * 1.8)
    needle_x  = 0.62 * np.cos(angle_rad)
    needle_y  = 0.62 * np.sin(angle_rad)
    ax.annotate('', xy=(needle_x, needle_y), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='white',
                                lw=2.5, connectionstyle='arc3,rad=0'))

    # Center circle
    circle = plt.Circle((0, 0), 0.12, color='#0f2744',
                         zorder=5, linewidth=1.5)
    ax.add_patch(circle)

    # Score text
    ax.text(0, -0.22, f'{score:.1f}', ha='center', va='center',
            fontsize=22, fontweight='bold', color='white',
            fontfamily='monospace')
    ax.text(0, -0.42, category, ha='center', va='center',
            fontsize=10, color=color, fontweight='600')
    ax.text(0, -0.56, 'WQI Score', ha='center', va='center',
            fontsize=8, color='#64748b')

    # Scale labels
    for val, label in [(0, '0'), (50, '50'), (100, '100')]:
        a = np.radians(180 - val * 1.8)
        lx = 1.18 * np.cos(a)
        ly = 1.18 * np.sin(a)
        ax.text(lx, ly, label, ha='center', va='center',
                fontsize=7, color='#64748b')

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-0.75, 1.2)
    ax.axis('off')
    plt.tight_layout(pad=0)
    return fig

def make_confidence_chart(conf_not_potable: float,
                           conf_potable: float) -> plt.Figure:
    """Horizontal confidence bar chart."""
    fig, ax = plt.subplots(figsize=(5, 1.6))
    fig.patch.set_facecolor('#0f2744')
    ax.set_facecolor('#0f2744')

    bars = ax.barh(
        ['Not Potable', 'Potable'],
        [conf_not_potable, conf_potable],
        color=['#ef4444', '#10b981'],
        height=0.5, alpha=0.85
    )
    for bar, val in zip(bars, [conf_not_potable, conf_potable]):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', color='white',
                fontsize=10, fontweight='500')

    ax.set_xlim(0, 115)
    ax.set_xlabel('Confidence (%)', color='#64748b', fontsize=9)
    ax.tick_params(colors='#94a3b8', labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#1e3a5f')
    ax.spines['left'].set_color('#1e3a5f')
    ax.set_facecolor('#0f2744')
    plt.tight_layout()
    return fig

def make_feature_importance_chart(importances: list,
                                   features: list) -> plt.Figure:
    """Horizontal feature importance chart."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    fig.patch.set_facecolor('#0f2744')
    ax.set_facecolor('#0f2744')

    sorted_idx = np.argsort(importances)
    sorted_feat = [features[i] for i in sorted_idx]
    sorted_imp  = [importances[i] * 100 for i in sorted_idx]

    colors = ['#3b82f6' if v < max(sorted_imp) * 0.5
              else '#10b981' if v == max(sorted_imp)
              else '#60a5fa'
              for v in sorted_imp]

    bars = ax.barh(sorted_feat, sorted_imp,
                   color=colors, height=0.6, alpha=0.9)
    for bar, val in zip(bars, sorted_imp):
        ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', color='#94a3b8', fontsize=8)

    ax.set_xlabel('Importance (%)', color='#64748b', fontsize=9)
    ax.tick_params(colors='#94a3b8', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#1e3a5f')
    ax.spines['left'].set_color('#1e3a5f')
    ax.set_facecolor('#0f2744')
    plt.tight_layout()
    return fig

def make_radar_chart(user_vals: dict, dataset_means: dict) -> plt.Figure:
    """Radar chart comparing user input vs dataset average."""
    labels  = list(user_vals.keys())
    N       = len(labels)
    angles  = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # Normalize both to 0–1 range using dataset min/max
    user_norm = []
    mean_norm = []
    for feat in labels:
        mn  = PARAM_META[feat]['min']
        mx  = PARAM_META[feat]['max']
        rng = mx - mn if mx != mn else 1
        user_norm.append((user_vals[feat] - mn) / rng)
        mean_norm.append((dataset_means[feat] - mn) / rng)

    user_norm += user_norm[:1]
    mean_norm += mean_norm[:1]

    fig, ax = plt.subplots(figsize=(4.5, 4.5),
                           subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#0f2744')
    ax.set_facecolor('#0a1628')

    ax.plot(angles, user_norm, 'o-', linewidth=2,
            color='#3b82f6', label='Your Sample')
    ax.fill(angles, user_norm, alpha=0.2, color='#3b82f6')
    ax.plot(angles, mean_norm, 'o-', linewidth=1.5,
            color='#64748b', linestyle='--', label='Dataset Mean')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=8, color='#94a3b8')
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'],
                       size=7, color='#475569')
    ax.grid(color='#1e3a5f', linestyle='--', alpha=0.6)
    ax.spines['polar'].set_color('#1e3a5f')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15),
              fontsize=8, labelcolor='#94a3b8',
              facecolor='#0f2744', edgecolor='#1e3a5f')
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────
# SIDEBAR — Parameter Inputs
# ─────────────────────────────────────────────────────────────
SAFE_PRESET = {
    'ph': 7.2, 'Hardness': 180.0, 'Solids': 15000.0,
    'Chloramines': 3.5, 'Sulfate': 200.0, 'Conductivity': 350.0,
    'Organic_carbon': 8.0, 'Trihalomethanes': 40.0, 'Turbidity': 2.5,
}
UNSAFE_PRESET = {
    'ph': 3.5, 'Hardness': 290.0, 'Solids': 50000.0,
    'Chloramines': 11.0, 'Sulfate': 430.0, 'Conductivity': 680.0,
    'Organic_carbon': 25.0, 'Trihalomethanes': 110.0, 'Turbidity': 5.8,
}

# Apply preset into session state BEFORE sliders are rendered
if 'preset' not in st.session_state:
    st.session_state['preset'] = 'none'

if st.session_state['preset'] == 'safe':
    for k, v in SAFE_PRESET.items():
        st.session_state[f'slider_{k}'] = v
    st.session_state['preset'] = 'none'
elif st.session_state['preset'] == 'unsafe':
    for k, v in UNSAFE_PRESET.items():
        st.session_state[f'slider_{k}'] = v
    st.session_state['preset'] = 'none'

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 12px 0 20px;">
        <div style="font-size:2em;">💧</div>
        <div style="font-size:1.1em; font-weight:600; color:#93c5fd;">AquaVision</div>
        <div style="font-size:0.75em; color:#475569; margin-top:2px;">Water Quality Parameters</div>
    </div>
    """, unsafe_allow_html=True)

    # Preset buttons — at TOP, before sliders
    st.markdown('<div class="section-header">Quick Presets</div>',
                unsafe_allow_html=True)
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        if st.button("🟢 Safe"):
            st.session_state['preset'] = 'safe'
            st.rerun()
    with col_p2:
        if st.button("🔴 Unsafe"):
            st.session_state['preset'] = 'unsafe'
            st.rerun()

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Input Parameters</div>',
                unsafe_allow_html=True)

    user_inputs = {}
    for feat in FEATURES:
        meta_f   = PARAM_META[feat]
        # Use slider_ prefixed key — completely separate from preset keys
        slider_key = f'slider_{feat}'
        default_val = st.session_state.get(slider_key, float(meta_f['default']))
        st.markdown(
            f'<div class="param-label">{meta_f["label"]}'
            f'{"  " + meta_f["unit"] if meta_f["unit"] else ""}'
            f' <span style="color:#334155">· WHO: {meta_f["who"]}</span></div>',
            unsafe_allow_html=True
        )
        user_inputs[feat] = st.slider(
            label=meta_f['label'],
            min_value=float(meta_f['min']),
            max_value=float(meta_f['max']),
            value=float(default_val),
            step=float((meta_f['max'] - meta_f['min']) / 200),
            label_visibility='collapsed',
            key=slider_key
        )

    st.markdown('<hr>', unsafe_allow_html=True)

    # Model info
    if meta:
        st.markdown('<div class="section-header">Model Info</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-box">
            <b>Model:</b> {meta.get('model_name','XGBoost')}<br>
            <b>Accuracy:</b> {meta.get('test_accuracy','—')}%<br>
            <b>ROC-AUC:</b> {meta.get('test_roc_auc','—')}%<br>
            <b>F1-Score:</b> {meta.get('test_f1','—')}%
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────

# ── Header ──
st.markdown("""
<div class="header-banner">
    <div style="display:flex; align-items:center; gap:20px;">
        <div style="font-size:3em;">💧</div>
        <div>
            <h1 style="margin:0; font-size:1.9em; font-weight:600;
                        color:#e2e8f0; letter-spacing:-0.02em;">
                AquaVision
            </h1>
            <p style="margin:4px 0 0; color:#64748b; font-size:0.95em;">
                Water Quality Index Prediction System &nbsp;·&nbsp;
                Brainybeam Info-Tech PVT LTD &nbsp;·&nbsp;
                Powered by XGBoost
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Check model loaded ──
if model is None:
    st.error("❌ Model not found. Please run Phase 4 notebook first to generate `models/wqi_model.pkl`")
    st.stop()

# ─────────────────────────────────────────────────────────────
# PREDICTION ENGINE
# ─────────────────────────────────────────────────────────────

# 1. Prepare input (apply sqrt to Solids — same as training)
input_engineered = user_inputs.copy()
input_engineered['Solids'] = np.sqrt(user_inputs['Solids'])

# 2. Scale
input_df     = pd.DataFrame([input_engineered], columns=FEATURES)
input_scaled = scaler.transform(input_df)
input_scaled = pd.DataFrame(input_scaled, columns=FEATURES)

# 3. Predict
prediction   = model.predict(input_scaled)[0]
confidence   = model.predict_proba(input_scaled)[0]
conf_not_pot = confidence[0] * 100
conf_pot     = confidence[1] * 100

# 4. WQI
wqi_raw_score = compute_wqi_raw(user_inputs)
wqi_norm      = normalize_wqi(
    wqi_raw_score,
    meta.get('wqi_min', 111.97),
    meta.get('wqi_max', 259.50)
)
wqi_cat, wqi_color = wqi_category(wqi_norm)

# ─────────────────────────────────────────────────────────────
# RESULT BANNER
# ─────────────────────────────────────────────────────────────
if prediction == 1:
    st.markdown(f"""
    <div class="result-potable">
        <div class="result-title">✅ POTABLE WATER</div>
        <div class="result-subtitle">
            This water sample is predicted to be safe for drinking.
            Confidence: {conf_pot:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="result-not-potable">
        <div class="result-title">⚠️ NOT POTABLE</div>
        <div class="result-subtitle">
            This water sample is predicted to be unsafe for drinking.
            Confidence: {conf_not_pot:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MAIN METRICS ROW
# ─────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:{'#10b981' if prediction==1 else '#ef4444'};">
            {'Potable' if prediction==1 else 'Not Potable'}
        </div>
        <div class="metric-label">Prediction</div>
    </div>""", unsafe_allow_html=True)

with m2:
    peak_conf = max(conf_pot, conf_not_pot)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:#93c5fd;">{peak_conf:.1f}%</div>
        <div class="metric-label">Confidence</div>
    </div>""", unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:{wqi_color};">{wqi_norm:.1f}</div>
        <div class="metric-label">WQI Score (0–100)</div>
    </div>""", unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:{wqi_color};">{wqi_cat}</div>
        <div class="metric-label">WQI Category</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Analysis",
    "🔬  Parameter Details",
    "📈  Model Insights",
    "ℹ️  About"
])

# ── TAB 1 — Analysis ──
with tab1:
    col_gauge, col_conf, col_radar = st.columns([1.2, 1, 1.2])

    with col_gauge:
        st.markdown('<div class="section-header">WQI Gauge</div>',
                    unsafe_allow_html=True)
        fig_gauge = make_gauge(wqi_norm, wqi_cat, wqi_color)
        st.pyplot(fig_gauge, use_container_width=True)
        plt.close()

    with col_conf:
        st.markdown('<div class="section-header">Prediction Confidence</div>',
                    unsafe_allow_html=True)
        fig_conf = make_confidence_chart(conf_not_pot, conf_pot)
        st.pyplot(fig_conf, use_container_width=True)
        plt.close()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">WQI Scale</div>',
                    unsafe_allow_html=True)
        scale_data = {
            "Category": ["Excellent", "Good", "Poor", "Very Poor"],
            "WQI Range": ["0 – 25", "26 – 50", "51 – 75", "76 – 100"],
            "Status":    ["✅ Safe",  "🟡 Acceptable", "🟠 Caution", "🔴 Unsafe"]
        }
        st.dataframe(
            pd.DataFrame(scale_data),
            hide_index=True,
            use_container_width=True
        )

    with col_radar:
        st.markdown('<div class="section-header">vs Dataset Average</div>',
                    unsafe_allow_html=True)
        if df_data is not None:
            dataset_means = df_data[FEATURES].mean().to_dict()
            # Reverse sqrt on Solids for display
            dataset_means['Solids'] = dataset_means['Solids'] ** 2
            fig_radar = make_radar_chart(user_inputs, dataset_means)
            st.pyplot(fig_radar, use_container_width=True)
            plt.close()
        else:
            st.info("Dataset not found for comparison.")

# ── TAB 2 — Parameter Details ──
with tab2:
    st.markdown('<div class="section-header">Parameter Values vs WHO Standards</div>',
                unsafe_allow_html=True)

    rows = []
    for feat in FEATURES:
        meta_f   = PARAM_META[feat]
        val      = user_inputs[feat]
        who_str  = meta_f['who']
        unit     = meta_f['unit']

        # Parse WHO upper limit roughly for status
        try:
            who_upper = float(who_str.split('<')[-1].strip().split()[0])
            if feat == 'ph':
                status = '✅ Normal' if 6.5 <= val <= 8.5 else '⚠️  Out of range'
            else:
                status = '✅ Normal' if val <= who_upper else '⚠️  Above limit'
        except Exception:
            status = '—'

        rows.append({
            'Parameter'     : meta_f['label'],
            'Your Value'    : f'{val:.3f} {unit}',
            'WHO Standard'  : who_str,
            'Status'        : status,
        })

    param_df = pd.DataFrame(rows)
    st.dataframe(param_df, hide_index=True, use_container_width=True,
                 height=360)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Parameter Context</div>',
                unsafe_allow_html=True)

    context_cols = st.columns(3)
    contexts = [
        ("pH", f"{user_inputs['ph']:.2f}", "Ideal drinking water is 7.0. Values below 6.5 are acidic; above 8.5 are alkaline — both can be harmful."),
        ("Sulfate", f"{user_inputs['Sulfate']:.1f} mg/L", "High sulfate causes laxative effects. WHO limit is 250 mg/L for drinking water."),
        ("Turbidity", f"{user_inputs['Turbidity']:.2f} NTU", "Measures water clarity. High turbidity indicates suspended particles and potential pathogens."),
        ("Chloramines", f"{user_inputs['Chloramines']:.2f} ppm", "Disinfectant added to water. Levels above 4 ppm can cause taste and health issues."),
        ("Organic Carbon", f"{user_inputs['Organic_carbon']:.2f} ppm", "Indicates organic contaminants. Values above 2 ppm suggest inadequate treatment."),
        ("Conductivity", f"{user_inputs['Conductivity']:.1f} μS/cm", "Measures dissolved ions. Very high conductivity indicates heavy mineral contamination."),
    ]
    for i, (name, val_str, desc) in enumerate(contexts):
        with context_cols[i % 3]:
            st.markdown(f"""
            <div class="info-box">
                <b style="color:#93c5fd;">{name}: {val_str}</b><br>
                <span>{desc}</span>
            </div>
            """, unsafe_allow_html=True)

# ── TAB 3 — Model Insights ──
with tab3:
    col_fi, col_meta = st.columns([1.4, 1])

    with col_fi:
        st.markdown('<div class="section-header">Feature Importance (XGBoost Tuned)</div>',
                    unsafe_allow_html=True)
        if meta and 'features' in meta:
            try:
                fi_values = model.feature_importances_.tolist()
                fig_fi    = make_feature_importance_chart(fi_values, FEATURES)
                st.pyplot(fig_fi, use_container_width=True)
                plt.close()
            except Exception:
                st.info("Feature importance not available.")
        else:
            st.info("Model metadata not loaded.")

    with col_meta:
        st.markdown('<div class="section-header">Model Performance Summary</div>',
                    unsafe_allow_html=True)
        if meta:
            perf_data = {
                'Metric'   : ['Accuracy', 'F1-Score', 'ROC-AUC',
                               'Precision', 'Recall'],
                'Score (%)':[meta.get('test_accuracy','—'),
                              meta.get('test_f1','—'),
                              meta.get('test_roc_auc','—'),
                              '73.83', '73.83'],
            }
            st.dataframe(pd.DataFrame(perf_data),
                         hide_index=True, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Best Hyperparameters</div>',
                    unsafe_allow_html=True)
        if meta and 'best_params' in meta:
            params_disp = {
                'Parameter': list(meta['best_params'].keys()),
                'Value'    : list(meta['best_params'].values()),
            }
            st.dataframe(pd.DataFrame(params_disp),
                         hide_index=True, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Top 3 Predictive Features</div>',
                    unsafe_allow_html=True)
        if meta and 'top3_features' in meta:
            for i, feat in enumerate(meta['top3_features']):
                medal = ['🥇', '🥈', '🥉'][i]
                st.markdown(f"""
                <div class="info-box">
                    {medal} <b style="color:#93c5fd;">{feat}</b>
                </div>
                """, unsafe_allow_html=True)

# ── TAB 4 — About ──
with tab4:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        <div class="section-header">About AquaVision</div>
        <div class="info-box">
            AquaVision is an end-to-end machine learning project for predicting
            water potability and calculating the Water Quality Index (WQI).
            Built as part of a Data Science & ML internship at
            <b>Brainybeam Info-Tech PVT LTD</b>.
        </div>
        <br>
        <div class="section-header">Project Pipeline</div>
        """, unsafe_allow_html=True)

        phases = [
            ("Phase 1", "Data Collection & Loading",      "3,276 samples · 9 features"),
            ("Phase 2", "Exploratory Data Analysis",      "Distributions · Correlations · Outliers"),
            ("Phase 3", "Feature Engineering",            "WQI calc · Imputation · Scaling"),
            ("Phase 4", "Model Building & Evaluation",    "XGBoost · 89.45% ROC-AUC"),
            ("Phase 5", "Streamlit Dashboard",            "This application"),
        ]
        for phase, name, detail in phases:
            st.markdown(f"""
            <div class="info-box">
                <b style="color:#93c5fd;">{phase}: {name}</b><br>
                <span style="color:#64748b;">{detail}</span>
            </div>
            """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="section-header">Dataset Information</div>
        """, unsafe_allow_html=True)

        dataset_info = {
            'Property'  : ['Source', 'Samples', 'Features',
                           'Target', 'Missing Values', 'Class Balance'],
            'Details'   : ['Kaggle — Water Potability', '3,276',
                           '9 physicochemical parameters',
                           'Potability (binary)', '1,434 → 0 (imputed)',
                           '61% Not Potable / 39% Potable'],
        }
        st.dataframe(pd.DataFrame(dataset_info),
                     hide_index=True, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="section-header">Technology Stack</div>
        """, unsafe_allow_html=True)

        tech_info = {
            'Tool'      : ['Python', 'pandas / numpy', 'scikit-learn',
                           'XGBoost', 'matplotlib / seaborn', 'Streamlit'],
            'Purpose'   : ['Primary language', 'Data manipulation',
                           'ML pipeline', 'Final model',
                           'Visualizations', 'Dashboard'],
        }
        st.dataframe(pd.DataFrame(tech_info),
                     hide_index=True, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<hr>
<div style="text-align:center; color:#334155; font-size:0.8em; padding:8px 0;">
    AquaVision &nbsp;·&nbsp; Brainybeam Info-Tech PVT LTD &nbsp;·&nbsp;
    Data Science & ML Internship &nbsp;·&nbsp; March 2026
</div>
""", unsafe_allow_html=True)