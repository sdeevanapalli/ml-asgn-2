# ── CONFIGURATION ─────────────────────────────────────────────────────────────
DATA_DIR = "data/"
RANDOM_STATE = 42
TEMPORAL_CUTOFF = "2020-01-01"
TEST_SIZE = 0.2
# ──────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import warnings
warnings.filterwarnings("ignore")

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              ConfusionMatrixDisplay, roc_curve, auc)
from imblearn.over_sampling import SMOTE

# ══════════════════════════════════════════════════════════════════════════════
# UI — PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Clinical Prediction · Team 13",
    page_icon="⚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "page" not in st.session_state:
    st.session_state.page = "Project Overview"

NAV_ITEMS = [
    ("Project Overview",          "01"),
    ("Exploratory Data Analysis", "02"),
    ("Model Performance",         "03"),
    ("Continual Learning",        "04"),
    ("Feature Importance",        "05"),
]

# ══════════════════════════════════════════════════════════════════════════════
# UI — GLOBAL CSS
# Design: dark charcoal base, muted amber accent, Inter/system-ui type
# No gradients, no glows, no decorative elements — clean analytics tool
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Tokens ── */
:root {
    --bg:           #171a1f;
    --bg-raised:    #1e2128;
    --bg-hover:     #252830;
    --border:       #2a2e38;
    --border-light: #323741;

    --text-primary:   #e8eaf0;
    --text-secondary: #9099aa;
    --text-muted:     #555e6e;

    --amber:      #d4a84b;
    --amber-dim:  rgba(212,168,75,0.12);
    --amber-border: rgba(212,168,75,0.25);

    --green:      #4caf82;
    --green-dim:  rgba(76,175,130,0.1);

    --red:        #e05c5c;
    --red-dim:    rgba(224,92,92,0.1);

    --font:  'Inter', system-ui, -apple-system, sans-serif;
    --mono:  'JetBrains Mono', 'Courier New', monospace;
}

/* ── Reset ── */
html, body, [class*="css"] {
    font-family: var(--font) !important;
    background:  var(--bg) !important;
    color:       var(--text-primary) !important;
}
.main .block-container {
    background:  var(--bg) !important;
    padding:     2rem 2.5rem 4rem !important;
    max-width:   1360px !important;
}
.main > div { padding-top: 0 !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background:   var(--bg-raised) !important;
    border-right: 1px solid var(--border) !important;
    min-width:    220px !important;
    max-width:    220px !important;
}
section[data-testid="stSidebar"] .block-container {
    padding: 0 !important;
}
div[data-testid="stRadio"] { display: none !important; }

/* sidebar nav buttons */
section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
    background:    transparent !important;
    border:        none !important;
    border-radius: 0 !important;
    border-left:   2px solid transparent !important;
    color:         var(--text-muted) !important;
    font-family:   var(--font) !important;
    font-size:     0.8rem !important;
    font-weight:   400 !important;
    text-align:    left !important;
    padding:       0.6rem 1.25rem !important;
    width:         100% !important;
    margin:        0 !important;
    box-shadow:    none !important;
    letter-spacing: 0 !important;
    transition:    color 0.1s, border-color 0.1s, background 0.1s !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
    background:        var(--bg-hover) !important;
    color:             var(--text-primary) !important;
    border-left-color: var(--amber) !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button:focus {
    box-shadow: none !important;
    outline:    none !important;
}

/* ── Headings ── */
h1, h2, h3, h4, h5 {
    font-family:     var(--font) !important;
    font-weight:     700 !important;
    color:           var(--text-primary) !important;
    letter-spacing:  -0.02em !important;
    line-height:     1.2 !important;
    font-style:      normal !important;
}
h1 { font-size: 1.75rem !important; }
h2 { font-size: 1.35rem !important; }
h3 { font-size: 1.1rem  !important; }

/* ── st.metric ── */
[data-testid="metric-container"] {
    background:    var(--bg-raised) !important;
    border:        1px solid var(--border) !important;
    border-radius: 6px !important;
    padding:       1.1rem 1.25rem !important;
    box-shadow:    none !important;
}
[data-testid="stMetricLabel"] {
    font-family:     var(--mono) !important;
    font-size:       0.65rem !important;
    font-weight:     500 !important;
    letter-spacing:  0.08em !important;
    text-transform:  uppercase !important;
    color:           var(--text-muted) !important;
}
[data-testid="stMetricValue"] {
    font-family:  var(--font) !important;
    font-size:    1.75rem !important;
    font-weight:  700 !important;
    color:        var(--text-primary) !important;
    line-height:  1.1 !important;
    font-style:   normal !important;
}
[data-testid="stMetricDelta"] {
    font-family: var(--mono) !important;
    font-size:   0.7rem !important;
}
/* delta color overrides */
[data-testid="stMetricDelta"][data-direction="positive"] { color: var(--green) !important; }
[data-testid="stMetricDelta"][data-direction="negative"] { color: var(--red)   !important; }

/* ── Selectbox ── */
div[data-baseweb="select"] > div {
    background:    var(--bg-raised) !important;
    border:        1px solid var(--border-light) !important;
    border-radius: 5px !important;
    font-family:   var(--font) !important;
    font-size:     0.85rem !important;
    color:         var(--text-primary) !important;
    box-shadow:    none !important;
}
div[data-baseweb="popover"] {
    background:    var(--bg-raised) !important;
    border:        1px solid var(--border-light) !important;
    border-radius: 5px !important;
}
li[role="option"] {
    background:  var(--bg-raised) !important;
    color:       var(--text-secondary) !important;
    font-size:   0.85rem !important;
    font-family: var(--font) !important;
    padding:     0.45rem 0.75rem !important;
}
li[role="option"]:hover {
    background: var(--bg-hover) !important;
    color:      var(--text-primary) !important;
}

/* ── Text input ── */
input[type="text"] {
    background:    var(--bg-raised) !important;
    border:        1px solid var(--border-light) !important;
    border-radius: 5px !important;
    font-family:   var(--mono) !important;
    font-size:     0.85rem !important;
    color:         var(--text-primary) !important;
    padding:       0.5rem 0.75rem !important;
    box-shadow:    none !important;
}
input[type="text"]:focus {
    border-color: var(--amber) !important;
    outline:      none !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    background:       var(--bg-raised) !important;
    border:           1px solid var(--border) !important;
    border-left:      3px solid var(--amber) !important;
    border-radius:    5px !important;
    color:            var(--text-secondary) !important;
    font-size:        0.85rem !important;
    font-family:      var(--font) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border:        1px solid var(--border) !important;
    border-radius: 6px !important;
    overflow:      hidden !important;
}
[data-testid="stDataFrame"] * {
    font-family: var(--mono) !important;
    font-size:   0.8rem !important;
}

/* ── Divider ── */
hr {
    border:     none !important;
    border-top: 1px solid var(--border) !important;
    margin:     1.75rem 0 !important;
    opacity:    1 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-light); border-radius: 2px; }

/* ── Spinner ── */
[data-testid="stSpinner"] > div {
    color: var(--text-muted) !important;
    font-size: 0.85rem !important;
    font-family: var(--mono) !important;
}

/* ════════════════════════════════════
   CUSTOM COMPONENTS
════════════════════════════════════ */

/* Sidebar brand block */
.sb-brand {
    padding: 1.5rem 1.25rem 1.25rem;
    border-bottom: 1px solid var(--border);
}
.sb-label {
    font-family: var(--mono);
    font-size: 0.58rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.4rem;
}
.sb-title {
    font-family: var(--font);
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.3;
}
.sb-team {
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--text-muted);
    margin-top: 0.5rem;
}
.sb-nav-section {
    padding: 1rem 0 0.25rem;
}
.sb-nav-label {
    font-family: var(--mono);
    font-size: 0.58rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-muted);
    padding: 0 1.25rem;
    margin-bottom: 0.25rem;
    display: block;
    opacity: 0.5;
}
.sb-cfg {
    border-top: 1px solid var(--border);
    padding: 1rem 1.25rem;
    margin-top: 1rem;
}
.sb-cfg-label {
    font-family: var(--mono);
    font-size: 0.58rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.6rem;
    opacity: 0.5;
}
.sb-row {
    display: flex;
    justify-content: space-between;
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--text-muted);
    padding: 0.15rem 0;
}
.sb-row span { color: var(--text-secondary); }

/* Page header */
.page-header {
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
}
.page-tag {
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--amber);
    margin-bottom: 0.5rem;
}
.page-title {
    font-family: var(--font);
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.025em;
    margin-bottom: 0.4rem;
    line-height: 1.2;
}
.page-desc {
    font-size: 0.875rem;
    color: var(--text-secondary);
    line-height: 1.65;
    max-width: 580px;
    font-weight: 400;
}

/* Section label */
.sec-label {
    font-family: var(--mono);
    font-size: 0.62rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.5rem;
    display: block;
}

/* Insight list block */
.insight-block {
    background:   var(--bg-raised);
    border:       1px solid var(--border);
    border-left:  2px solid var(--amber);
    border-radius: 0 5px 5px 0;
    padding:      1rem 1.25rem;
    margin-bottom: 0.6rem;
}
.insight-block-title {
    font-family:    var(--mono);
    font-size:      0.62rem;
    font-weight:    500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color:          var(--amber);
    margin-bottom:  0.55rem;
}
.insight-block ul {
    margin:      0;
    padding:     0;
    list-style:  none;
}
.insight-block li {
    font-size:   0.85rem;
    color:       var(--text-secondary);
    padding:     0.2rem 0;
    line-height: 1.6;
    display:     flex;
    gap:         0.55rem;
}
.insight-block li::before {
    content:    '—';
    color:      var(--text-muted);
    flex-shrink: 0;
    font-size:  0.8rem;
    margin-top: 0.05rem;
}
.insight-block.red  { border-left-color: var(--red);   }
.insight-block.red  .insight-block-title { color: var(--red); }
.insight-block.green { border-left-color: var(--green); }
.insight-block.green .insight-block-title { color: var(--green); }
.insight-block.dim  { border-left-color: var(--border-light); }
.insight-block.dim  .insight-block-title { color: var(--text-muted); }

/* Architecture grid */
.arch-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
    margin-top: 0.75rem;
}
.arch-block {
    background:    var(--bg-raised);
    border:        1px solid var(--border);
    border-radius: 5px;
    padding:       1.1rem 1.25rem;
}
.arch-block-title {
    font-family:    var(--mono);
    font-size:      0.62rem;
    font-weight:    500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color:          var(--text-muted);
    margin-bottom:  0.7rem;
    padding-bottom: 0.6rem;
    border-bottom:  1px solid var(--border);
}
.arch-block ul { margin: 0; padding: 0; list-style: none; }
.arch-block li {
    font-size:   0.83rem;
    color:       var(--text-secondary);
    padding:     0.2rem 0;
    line-height: 1.55;
}

/* Team grid */
.team-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin-top: 0.75rem;
}
.team-card {
    background:    var(--bg-raised);
    border:        1px solid var(--border);
    border-radius: 5px;
    padding:       1.1rem;
}
.team-num {
    font-family:  var(--mono);
    font-size:    0.6rem;
    color:        var(--text-muted);
    margin-bottom: 0.4rem;
}
.team-role {
    font-family:    var(--mono);
    font-size:      0.6rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color:          var(--amber);
    margin-bottom:  0.25rem;
}
.team-name {
    font-family:   var(--font);
    font-size:     0.9rem;
    font-weight:   600;
    color:         var(--text-primary);
    margin-bottom: 0.35rem;
    line-height:   1.3;
}
.team-task {
    display:       inline-block;
    font-family:   var(--mono);
    font-size:     0.58rem;
    background:    var(--bg);
    border:        1px solid var(--border);
    border-radius: 3px;
    padding:       0.15em 0.45em;
    color:         var(--text-muted);
    margin-bottom: 0.5rem;
}
.team-desc {
    font-size:   0.78rem;
    color:       var(--text-muted);
    line-height: 1.55;
}

/* Key insights block on overview */
.kib {
    background:    var(--bg-raised);
    border:        1px solid var(--border);
    border-radius: 5px;
    padding:       1.25rem 1.5rem;
    margin-bottom: 1.75rem;
}
.kib-title {
    font-family:    var(--mono);
    font-size:      0.62rem;
    font-weight:    500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color:          var(--text-muted);
    margin-bottom:  1rem;
}
.kib-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
}
.kib-item {
    display: flex;
    gap: 0.85rem;
    align-items: flex-start;
}
.kib-num {
    font-family:  var(--mono);
    font-size:    0.65rem;
    color:        var(--amber);
    flex-shrink:  0;
    margin-top:   0.15rem;
    min-width:    1.5rem;
}
.kib-text {
    font-size:   0.85rem;
    color:       var(--text-secondary);
    line-height: 1.6;
}

/* Before/after stat cells (Continual Learning) */
.cl-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin-bottom: 0.75rem;
}
.cl-cell {
    background:    var(--bg-raised);
    border:        1px solid var(--border);
    border-radius: 5px;
    padding:       1rem 1.1rem;
}
.cl-label {
    font-family:    var(--mono);
    font-size:      0.6rem;
    font-weight:    500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color:          var(--text-muted);
    margin-bottom:  0.5rem;
}
.cl-values {
    display:     flex;
    align-items: baseline;
    gap:         0.4rem;
    margin-bottom: 0.3rem;
}
.cl-before {
    font-family:  var(--font);
    font-size:    1.5rem;
    font-weight:  700;
    color:        var(--text-muted);
    line-height:  1;
}
.cl-arrow {
    font-size: 0.8rem;
    color:     var(--text-muted);
}
.cl-after {
    font-family:  var(--font);
    font-size:    1.5rem;
    font-weight:  700;
    color:        var(--text-primary);
    line-height:  1;
}
.cl-delta {
    font-family: var(--mono);
    font-size:   0.7rem;
}
.cl-delta.pos { color: var(--green); }
.cl-delta.neg { color: var(--red);   }
.cl-note {
    font-family:   var(--mono);
    font-size:     0.72rem;
    color:         var(--text-muted);
    margin-bottom: 1.5rem;
}

/* Model tags */
.model-tags {
    display:     flex;
    gap:         0.5rem;
    margin-bottom: 1.25rem;
}
.model-tag {
    font-family:    var(--mono);
    font-size:      0.68rem;
    color:          var(--text-muted);
    background:     var(--bg-raised);
    border:         1px solid var(--border);
    border-radius:  4px;
    padding:        0.2em 0.6em;
}

/* Section sep */
.sep {
    display:         flex;
    align-items:     center;
    gap:             0.75rem;
    margin:          1.75rem 0 1.25rem;
}
.sep-text {
    font-family:    var(--mono);
    font-size:      0.6rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color:          var(--text-muted);
    white-space:    nowrap;
}
.sep-line {
    flex:       1;
    height:     1px;
    background: var(--border);
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# UI — MATPLOTLIB DARK THEME (matches the dark UI)
# ══════════════════════════════════════════════════════════════════════════════
BG       = "#171a1f"
BG_CARD  = "#1e2128"
FG       = "#e8eaf0"
FG_SOFT  = "#9099aa"
FG_MUTED = "#555e6e"
BORDER   = "#2a2e38"

C_AMBER  = "#d4a84b"
C_GREEN  = "#4caf82"
C_RED    = "#e05c5c"
C_TEAL   = "#4caf82"   # reuse green for teal slots
C_INDIG  = "#7a8fc4"

matplotlib.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         9,
    "figure.facecolor":  BG_CARD,
    "axes.facecolor":    BG_CARD,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   FG_SOFT,
    "axes.titlecolor":   FG,
    "axes.titlesize":    10,
    "axes.titleweight":  "bold",
    "axes.labelsize":    8.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.spines.left":  True,
    "axes.spines.bottom":True,
    "axes.grid":         True,
    "grid.color":        BORDER,
    "grid.linestyle":    "-",
    "grid.linewidth":    0.5,
    "grid.alpha":        1,
    "xtick.color":       FG_MUTED,
    "ytick.color":       FG_MUTED,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "text.color":        FG,
    "legend.facecolor":  BG_CARD,
    "legend.edgecolor":  BORDER,
    "legend.labelcolor": FG_SOFT,
    "legend.fontsize":   8.5,
    "legend.framealpha": 1,
})

# Shorthand for inline style strings
INK_SOFT = FG_SOFT
MUTED    = FG_MUTED
INK      = FG

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE — UNTOUCHED
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def run_pipeline():
    patients     = pd.read_csv(DATA_DIR + "patients.csv", on_bad_lines="skip")
    encounters   = pd.read_csv(DATA_DIR + "encounters.csv", on_bad_lines="skip")
    observations = pd.read_csv(DATA_DIR + "observations.csv", on_bad_lines="skip")
    conditions   = pd.read_csv(DATA_DIR + "conditions.csv", on_bad_lines="skip", dayfirst=True)
    medications  = pd.read_csv(DATA_DIR + "medications.csv", on_bad_lines="skip")
    procedures   = pd.read_csv(DATA_DIR + "procedures.csv", on_bad_lines="skip")
    immunizations= pd.read_csv(DATA_DIR + "immunizations.csv", on_bad_lines="skip")
    allergies    = pd.read_csv(DATA_DIR + "allergies.csv", on_bad_lines="skip")
    careplans    = pd.read_csv(DATA_DIR + "careplans.csv", on_bad_lines="skip")
    imaging      = pd.read_csv(DATA_DIR + "imaging_studies.csv", on_bad_lines="skip")
    devices      = pd.read_csv(DATA_DIR + "devices.csv", on_bad_lines="skip")
    supplies     = pd.read_csv(DATA_DIR + "supplies.csv", on_bad_lines="skip")
    payer_trans  = pd.read_csv(DATA_DIR + "payer_transitions.csv", on_bad_lines="skip")
    claims       = pd.read_csv(DATA_DIR + "claims.csv", on_bad_lines="skip")
    claims_trans = pd.read_csv(DATA_DIR + "claims_transactions.csv", on_bad_lines="skip",
                               usecols=lambda c: c in ["PATIENTID","TYPE","AMOUNT"])

    ref_date = pd.Timestamp(TEMPORAL_CUTOFF, tz="UTC")
    patients["BIRTHDATE"] = pd.to_datetime(patients["BIRTHDATE"], errors="coerce", utc=True)
    patients["DEATHDATE"] = pd.to_datetime(patients["DEATHDATE"], errors="coerce", utc=True)
    patients["age"] = (ref_date - patients["BIRTHDATE"]).dt.days // 365
    patients["is_deceased"] = patients["DEATHDATE"].notna().astype(int)
    for col in ["GENDER","RACE","ETHNICITY","MARITAL"]:
        if col in patients.columns:
            patients[col] = LabelEncoder().fit_transform(patients[col].astype(str))
    demo_cols = [c for c in ["Id","age","is_deceased","GENDER","RACE","ETHNICITY","MARITAL","INCOME","HEALTHCARE_COVERAGE"] if c in patients.columns]
    pat_features = patients[demo_cols].rename(columns={"Id":"PATIENT"})

    encounters["START"] = pd.to_datetime(encounters["START"], errors="coerce", utc=True)
    enc_agg = encounters.groupby("PATIENT").agg(
        total_encounters=("Id","count"), unique_encounter_types=("ENCOUNTERCLASS","nunique"),
        avg_base_encounter_cost=("BASE_ENCOUNTER_COST","mean"),
        total_claim_cost=("TOTAL_CLAIM_COST","sum"), avg_payer_coverage=("PAYER_COVERAGE","mean")
    ).reset_index()

    observations["VALUE"] = pd.to_numeric(observations["VALUE"], errors="coerce")
    obs_agg = observations.dropna(subset=["VALUE"]).groupby(["PATIENT","DESCRIPTION"])["VALUE"].agg(["mean","std"]).reset_index()
    obs_agg.columns = ["PATIENT","DESC","mean","std"]
    obs_agg["DESC"] = obs_agg["DESC"].str.replace(r"[^a-zA-Z0-9]","_",regex=True).str[:40]
    obs_mean = obs_agg.pivot_table(index="PATIENT",columns="DESC",values="mean",aggfunc="mean")
    obs_std  = obs_agg.pivot_table(index="PATIENT",columns="DESC",values="std", aggfunc="mean")
    obs_mean.columns = ["obs_"+c+"_mean" for c in obs_mean.columns]
    obs_std.columns  = ["obs_"+c+"_var"  for c in obs_std.columns]
    obs_mean = obs_mean.loc[:, obs_mean.notna().mean() >= 0.05]
    obs_std  = obs_std.loc[:,  obs_std.notna().mean()  >= 0.05]
    obs_features = obs_mean.join(obs_std, how="outer").reset_index()

    med_agg = medications.groupby("PATIENT").agg(total_medications=("START","count"),unique_medications=("DESCRIPTION","nunique"),avg_medication_cost=("BASE_COST","mean"),total_dispenses=("DISPENSES","sum")).reset_index()
    proc_agg= procedures.groupby("PATIENT").agg(total_procedures=("START","count"),unique_procedures=("DESCRIPTION","nunique"),avg_procedure_cost=("BASE_COST","mean")).reset_index()
    imm_agg = immunizations.groupby("PATIENT").agg(total_immunizations=("DATE","count"),unique_vaccines=("DESCRIPTION","nunique")).reset_index()
    allergy_agg=allergies.groupby("PATIENT").agg(total_allergies=("START","count"),unique_allergy_types=("TYPE","nunique"),unique_allergy_categories=("CATEGORY","nunique")).reset_index()
    care_agg= careplans.groupby("PATIENT").agg(total_careplans=("Id","count"),unique_careplan_reasons=("REASONDESCRIPTION","nunique")).reset_index()
    img_agg = imaging.groupby("PATIENT").agg(total_imaging=("Id","count"),unique_modalities=("MODALITY_DESCRIPTION","nunique"),unique_body_sites=("BODYSITE_DESCRIPTION","nunique")).reset_index()
    dev_agg = devices.groupby("PATIENT").agg(total_devices=("START","count"),unique_device_types=("DESCRIPTION","nunique")).reset_index()
    sup_agg = supplies.groupby("PATIENT").agg(total_supplies=("DATE","count"),unique_supply_types=("DESCRIPTION","nunique")).reset_index()
    pay_agg = payer_trans.groupby("PATIENT").agg(total_payer_transitions=("START_DATE","count"),unique_payers=("PAYER","nunique")).reset_index()
    claims_cost_col = "OUTSTANDING1" if "OUTSTANDING1" in claims.columns else "Id"
    claims_agg = claims.groupby("PATIENTID").agg(total_claims=("Id","count"),avg_claim_cost=(claims_cost_col,"mean")).reset_index().rename(columns={"PATIENTID":"PATIENT"})
    if "AMOUNT" in claims_trans.columns:
        ct_agg = claims_trans.groupby("PATIENTID").agg(total_transactions=("TYPE","count"),total_transaction_amount=("AMOUNT","sum"),unique_transaction_types=("TYPE","nunique")).reset_index().rename(columns={"PATIENTID":"PATIENT"})
    else:
        ct_agg = pd.DataFrame(columns=["PATIENT"])

    df = pat_features.copy()
    for fdf in [enc_agg,obs_features,med_agg,proc_agg,imm_agg,allergy_agg,care_agg,img_agg,dev_agg,sup_agg,pay_agg,claims_agg,ct_agg]:
        if "PATIENT" in fdf.columns: df = df.merge(fdf,on="PATIENT",how="left")

    conditions["START"] = pd.to_datetime(conditions["START"],dayfirst=True,errors="coerce",utc=True)
    pos_pts = set(conditions[conditions["DESCRIPTION"].str.contains(r"\(disorder\)|\(finding\)",na=False,regex=True)]["PATIENT"].unique())
    df["label"] = df["PATIENT"].apply(lambda x: 1 if x in pos_pts else 0)

    enc_dates = encounters.groupby("PATIENT")["START"].agg(["min","max"]).reset_index()
    enc_dates.columns = ["PATIENT","first_enc","last_enc"]
    df = df.merge(enc_dates,on="PATIENT",how="left")

    cutoff = pd.Timestamp(TEMPORAL_CUTOFF,tz="UTC")
    df1 = df[df["first_enc"] < cutoff].copy()
    df2 = df[df["last_enc"]  >= cutoff].copy()
    for d in [df1,df2]: d.drop(columns=["PATIENT","first_enc","last_enc"],errors="ignore",inplace=True)

    all_X_pre = df1.drop(columns=["label"],errors="ignore")
    missing_series = all_X_pre.isna().mean().sort_values(ascending=False).head(20)

    def preprocess(df_in):
        X = df_in.drop(columns=["label"]); y = df_in["label"]
        X = X.loc[:, X.isna().mean() < 0.5]
        for col in X.columns:
            X[col] = X[col].fillna(X[col].median() if X[col].dtype in ["float64","int64"] else X[col].mode()[0])
        return X, y

    X1,y1 = preprocess(df1); X2,y2 = preprocess(df2)
    common_cols = list(set(X1.columns) & set(X2.columns))
    X1,X2 = X1[common_cols], X2[common_cols]

    X_tr1,X_te1,y_tr1,y_te1 = train_test_split(X1,y1,test_size=TEST_SIZE,stratify=y1,random_state=RANDOM_STATE)
    X_tr2,X_te2,y_tr2,y_te2 = train_test_split(X2,y2,test_size=TEST_SIZE,stratify=y2,random_state=RANDOM_STATE)

    sc1 = StandardScaler()
    X_tr1 = pd.DataFrame(sc1.fit_transform(X_tr1),columns=common_cols)
    X_te1 = pd.DataFrame(sc1.transform(X_te1),    columns=common_cols)
    sc2 = StandardScaler()
    X_tr2 = pd.DataFrame(sc2.fit_transform(X_tr2),columns=common_cols)
    X_te2 = pd.DataFrame(sc2.transform(X_te2),    columns=common_cols)

    X1s,y1s = SMOTE(random_state=RANDOM_STATE).fit_resample(X_tr1,y_tr1)
    X2s,y2s = SMOTE(random_state=RANDOM_STATE).fit_resample(X_tr2,y_tr2)

    dt_best  = GridSearchCV(DecisionTreeClassifier(class_weight="balanced",random_state=RANDOM_STATE),{"max_depth":[3,5,10,None]},scoring="f1",cv=5,n_jobs=-1).fit(X_tr1,y_tr1).best_estimator_
    svm_best = GridSearchCV(SVC(kernel="rbf",class_weight="balanced",probability=True,random_state=RANDOM_STATE),{"C":[0.1,1,10]},scoring="f1",cv=5,n_jobs=-1).fit(X_tr1,y_tr1).best_estimator_
    mlp = MLPClassifier(hidden_layer_sizes=(128,64,32),activation="relu",max_iter=500,early_stopping=True,random_state=RANDOM_STATE).fit(X1s,y1s)

    def evaluate(model,X,y,label,dataset):
        p=model.predict(X); pr=model.predict_proba(X)[:,1]
        return {"model":label,"evaluated_on":dataset,"accuracy":accuracy_score(y,p),
                "precision":precision_score(y,p,zero_division=0),"recall":recall_score(y,p,zero_division=0),
                "f1":f1_score(y,p,zero_division=0),"roc_auc":roc_auc_score(y,pr)}

    results = []
    for m,n in [(dt_best,"DT"),(svm_best,"SVM"),(mlp,"MLP")]:
        results += [evaluate(m,X_te1,y_te1,n,"D1"), evaluate(m,X_te2,y_te2,n,"D2")]
    baseline_df = pd.DataFrame(results)

    mlp_cl = MLPClassifier(hidden_layer_sizes=(128,64,32),activation="relu",max_iter=1,warm_start=False,random_state=RANDOM_STATE)
    mlp_cl.fit(X1s[:10],y1s[:10])
    mlp_cl.coefs_ = mlp.coefs_; mlp_cl.intercepts_ = mlp.intercepts_
    X2a = X2s.values if hasattr(X2s,"values") else X2s
    y2a = y2s.values if hasattr(y2s,"values") else y2s
    for _ in range(50):
        idx = np.random.permutation(len(X2a))
        for s in range(0,len(X2a),64):
            b=idx[s:s+64]; mlp_cl.partial_fit(X2a[b],y2a[b],classes=np.array([0,1]))

    continual_df = pd.DataFrame([evaluate(mlp,X_te2,y_te2,"MLP_D1","D2"),evaluate(mlp_cl,X_te2,y_te2,"MLP_CL","D2")])
    return dict(X_train_d1=X_tr1,y_train_d1=y_tr1,y_test_d1=y_te1,X_train_d2=X_tr2,y_train_d2=y_tr2,
                y_test_d2=y_te2,X_test_d1=X_te1,X_test_d2=X_te2,feature_names=common_cols,
                baseline_df=baseline_df,continual_df=continual_df,dt_best=dt_best,svm_best=svm_best,
                mlp=mlp,mlp_cl=mlp_cl,d1_size=len(df1),d2_size=len(df2),total_patients=len(df),
                missing_series=missing_series)

# ══════════════════════════════════════════════════════════════════════════════
# UI — SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-label">Internal Analytics</div>
        <div class="sb-title">Clinical Condition<br>Prediction</div>
        <div class="sb-team">BITS F464 · ML · Team 13</div>
    </div>
    <div class="sb-nav-section">
        <span class="sb-nav-label">Pages</span>
    </div>
    """, unsafe_allow_html=True)

    for full_name, num in NAV_ITEMS:
        if st.button(f"{num}  {full_name}", key=f"nav_{num}", use_container_width=True):
            st.session_state.page = full_name
            st.rerun()

    st.markdown(f"""
    <div class="sb-cfg">
        <div class="sb-cfg-label">Run Config</div>
        <div class="sb-row">cutoff      <span>{TEMPORAL_CUTOFF}</span></div>
        <div class="sb-row">test split  <span>{int(TEST_SIZE*100)}%</span></div>
        <div class="sb-row">random seed <span>{RANDOM_STATE}</span></div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE LOAD
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner("Running pipeline — first load takes ~1 min"):
    data = run_pipeline()

X_train_d1=data["X_train_d1"]; y_train_d1=data["y_train_d1"]; y_test_d1=data["y_test_d1"]
X_train_d2=data["X_train_d2"]; y_train_d2=data["y_train_d2"]; y_test_d2=data["y_test_d2"]
X_test_d1=data["X_test_d1"];   X_test_d2=data["X_test_d2"]
feature_names=data["feature_names"]; baseline_df=data["baseline_df"]
continual_df=data["continual_df"];   missing_series=data["missing_series"]
dt_best=data["dt_best"]; svm_best=data["svm_best"]; mlp=data["mlp"]; mlp_cl=data["mlp_cl"]
page = st.session_state.page

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PROJECT OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Project Overview":

    st.markdown("""
    <div class="page-header">
        <div class="page-tag">BITS F464 · Machine Learning · Team 13</div>
        <div class="page-title">Clinical Condition Prediction under Temporal Shift</div>
        <div class="page-desc">End-to-end ML pipeline on synthetic EHR data. Three classifiers trained on pre-2020 patient records, evaluated on both historical and current cohorts.</div>
    </div>
    """, unsafe_allow_html=True)

    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Patients",        f"{data['total_patients']:,}")
    c2.metric("Feature Dimensions",    len(feature_names))
    c3.metric("D1 — Pre-2020",         f"{data['d1_size']:,}")
    c4.metric("D2 — Post-2020",        f"{data['d2_size']:,}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Key insights block
    st.markdown("""
    <div class="kib">
        <div class="kib-title">Key Findings</div>
        <div class="kib-grid">
            <div class="kib-item">
                <div class="kib-num">01</div>
                <div class="kib-text">MLP with SMOTE achieves the strongest F1 on D2. Oversampling outperforms class weighting for heavily imbalanced clinical data.</div>
            </div>
            <div class="kib-item">
                <div class="kib-num">02</div>
                <div class="kib-text">Observation features (vitals and labs) dominate the top-20 importance rankings. Clinical signals outperform demographics throughout.</div>
            </div>
            <div class="kib-item">
                <div class="kib-num">03</div>
                <div class="kib-text">Continual fine-tuning via partial_fit() causes catastrophic forgetting. The D1-trained MLP generalises to D2 without retraining.</div>
            </div>
            <div class="kib-item">
                <div class="kib-num">04</div>
                <div class="kib-text">Accuracy is misleading under class imbalance. F1 and ROC-AUC are the correct metrics for evaluating clinical classifiers.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Pipeline architecture
    st.markdown("""
    <div class="sep">
        <span class="sep-text">Pipeline Architecture</span>
        <div class="sep-line"></div>
    </div>
    <div class="arch-grid">
        <div class="arch-block">
            <div class="arch-block-title">Data Pipeline · Task 2</div>
            <ul>
                <li>15 CSV files merged on patient ID</li>
                <li>Temporal split: pre / post 2020-01-01</li>
                <li>Sparse columns dropped (&gt;50% missing)</li>
                <li>Binary target: disorder / finding labels</li>
                <li>StandardScaler fit on training data only</li>
                <li>80 / 20 stratified train-test split</li>
            </ul>
        </div>
        <div class="arch-block">
            <div class="arch-block-title">Models · Task 3</div>
            <ul>
                <li>Decision Tree — GridSearchCV, balanced class weights</li>
                <li>SVM RBF kernel — GridSearchCV, balanced class weights</li>
                <li>MLP 128→64→32 — SMOTE oversampling on training set</li>
            </ul>
        </div>
        <div class="arch-block">
            <div class="arch-block-title">Class Imbalance Strategy</div>
            <ul>
                <li>Majority: label = 0 (no clinical condition)</li>
                <li>Minority: label = 1 (disorder or finding)</li>
                <li>DT + SVM: class_weight = "balanced"</li>
                <li>MLP: SMOTE synthetic minority oversampling</li>
            </ul>
        </div>
        <div class="arch-block">
            <div class="arch-block-title">Temporal Dataset Split</div>
            <ul>
                <li>D1: first encounter before 2020-01-01</li>
                <li>D2: any encounter from 2020-01-01 onward</li>
                <li>Overlap is intentional — tests generalisation</li>
                <li>All models trained on D1, evaluated on both</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Team
    st.markdown("""
    <div class="sep" style="margin-top:2rem;">
        <span class="sep-text">Team</span>
        <div class="sep-line"></div>
    </div>
    <div class="team-grid">
        <div class="team-card">
            <div class="team-num">01 / 04</div>
            <div class="team-role">Data Architect</div>
            <div class="team-name">Shriniketh Deevanapalli</div>
            <span class="team-task">Task 2 (a, b, c)</span>
            <div class="team-desc">Merged 15 CSV tables, implemented the temporal split, and built the feature dataset with StandardScaler.</div>
        </div>
        <div class="team-card">
            <div class="team-num">02 / 04</div>
            <div class="team-role">ML Engineer</div>
            <div class="team-name">Sanvi Udhan</div>
            <span class="team-task">Task 3 (a,b,c) + Task 4</span>
            <div class="team-desc">Trained DT, SVM, and MLP. Implemented continual learning via partial_fit and compiled all performance metrics.</div>
        </div>
        <div class="team-card">
            <div class="team-num">03 / 04</div>
            <div class="team-role">Full-Stack Dev</div>
            <div class="team-name">Sai Dheeraj Yadavalli</div>
            <span class="team-task">Task 1 + Task 5</span>
            <div class="team-desc">Built this Streamlit dashboard and integrated outputs from all team members into interactive visualisations.</div>
        </div>
        <div class="team-card">
            <div class="team-num">04 / 04</div>
            <div class="team-role">Data Analyst</div>
            <div class="team-name">Shambhavi Rani</div>
            <span class="team-task">Task 2(d) + Task 3(d,e,f) + Task 5</span>
            <div class="team-desc">EDA, bias-variance analysis, feature importance write-up, and final video presentation.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EXPLORATORY DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Exploratory Data Analysis":

    st.markdown("""
    <div class="page-header">
        <div class="page-tag">Task 2 · Data Exploration</div>
        <div class="page-title">Exploratory Data Analysis</div>
        <div class="page-desc">Distributions, demographics, and data quality across historical (D1) and current (D2) cohorts.</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("D1 Train Samples",  f"{len(y_train_d1):,}")
    c2.metric("D1 Positive Rate",  f"{y_train_d1.mean()*100:.1f}%")
    c3.metric("D2 Train Samples",  f"{len(y_train_d2):,}")
    c4.metric("D2 Positive Rate",  f"{y_train_d2.mean()*100:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<span class="sec-label">Select analysis</span>', unsafe_allow_html=True)
    eda_section = st.selectbox(
        "Analysis",
        ["Class Distribution","Demographics","Clinical Features",
         "Healthcare Utilization","Correlation Heatmap","Data Drift Analysis","Missing Values"],
        label_visibility="collapsed"
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── EDA sub-sections (logic untouched) ───────────────────────────────────
    if eda_section == "Class Distribution":
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, y, title, col in [
            (axes[0], y_train_d1, "D1 — Historical (pre-2020)", C_GREEN),
            (axes[1], y_train_d2, "D2 — Current (post-2020)",   C_INDIG)
        ]:
            counts = y.value_counts().sort_index()
            bars = ax.bar(["No Condition","Has Condition"], counts.values,
                          color=[col, C_RED], edgecolor="none", width=0.5)
            ax.set_title(title, pad=12)
            ax.set_ylabel("Patient Count")
            ax.bar_label(bars, fmt="%d", fontsize=10, color=FG_SOFT, padding=4)
            ax.set_ylim(0, counts.max() * 1.18)
        plt.tight_layout(pad=2)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("""
        <div class="insight-block">
            <div class="insight-block-title">Class Imbalance</div>
            <ul>
                <li>Severe imbalance is consistent across both datasets — not a temporal artifact</li>
                <li>Handled via class_weight=balanced for DT and SVM; SMOTE for MLP</li>
                <li>Accuracy alone is a misleading metric here — prioritise F1 and ROC-AUC</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    elif eda_section == "Demographics":
        demo_options = [c for c in ["age","GENDER","RACE","MARITAL","INCOME"] if c in X_train_d1.columns]
        st.markdown('<span class="sec-label">Feature</span>', unsafe_allow_html=True)
        selected_demo = st.selectbox("Feature", demo_options, label_visibility="collapsed")
        c1, c2 = st.columns(2)
        for ctx, X_tr, y_tr, title in [
            (c1, X_train_d1, y_train_d1, "D1 — Historical"),
            (c2, X_train_d2, y_train_d2, "D2 — Current")
        ]:
            with ctx:
                st.markdown(f"<div style='font-size:0.75rem;font-weight:600;color:{FG_SOFT};margin-bottom:0.5rem;font-family:var(--mono);letter-spacing:0.05em;'>{title}</div>", unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(5.5, 4))
                if selected_demo in ["age","INCOME"]:
                    ax.hist(X_tr.loc[y_tr==0, selected_demo].dropna(), bins=28, alpha=0.7, color=C_GREEN, label="No Condition", density=True, edgecolor="none")
                    ax.hist(X_tr.loc[y_tr==1, selected_demo].dropna(), bins=28, alpha=0.7, color=C_RED,   label="Has Condition", density=True, edgecolor="none")
                    ax.set_xlabel(f"{selected_demo} (scaled)")
                    ax.set_ylabel("Density")
                    ax.legend(fontsize=8)
                else:
                    tmp = X_tr[[selected_demo]].copy(); tmp["label"] = y_tr.values
                    tmp.groupby([selected_demo,"label"]).size().unstack(fill_value=0).plot(
                        kind="bar", ax=ax, color=[C_GREEN, C_RED], edgecolor="none", width=0.6)
                    ax.set_xlabel(f"{selected_demo} (encoded)")
                    ax.set_ylabel("Count")
                    ax.legend(["No Condition","Has Condition"], fontsize=8)
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
                ax.set_title(f"{selected_demo} by Label", pad=10)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
        if "INCOME" in X_train_d1.columns:
            st.markdown("---")
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            for ax, (X_t, y_t, title) in zip(axes, [(X_train_d1, y_train_d1, "D1"),(X_train_d2, y_train_d2, "D2")]):
                plot_df = X_t[["INCOME"]].copy(); plot_df["label"] = y_t.values
                ax.boxplot([plot_df[plot_df["label"]==l]["INCOME"].dropna() for l in [0,1]],
                           labels=["No condition","Has condition"], patch_artist=True, widths=0.45,
                           medianprops=dict(color=C_RED, linewidth=2),
                           boxprops=dict(facecolor=C_GREEN, alpha=0.3, linewidth=0),
                           whiskerprops=dict(color=FG_MUTED), capprops=dict(color=FG_MUTED),
                           flierprops=dict(marker='o', markerfacecolor=C_GREEN, markersize=3, alpha=0.3, linestyle='none'))
                ax.set_title(f"Income by Label — {title}", pad=10)
                ax.set_ylabel("Scaled Income")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    elif eda_section == "Clinical Features":
        clinical_cols = [c for c in feature_names if any(x in c for x in ["Body_Height","Body_Weight","BMI","Diastolic","Systolic","Heart_rate","Cholesterol"]) and "_mean" in c][:7]
        if clinical_cols:
            st.markdown('<span class="sec-label">Feature</span>', unsafe_allow_html=True)
            selected_clin = st.selectbox("Feature", clinical_cols, label_visibility="collapsed")
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            for ax, (X_t, y_t, title) in zip(axes, [(X_train_d1, y_train_d1, "D1 — Historical"),(X_train_d2, y_train_d2, "D2 — Current")]):
                if selected_clin in X_t.columns:
                    plot_df = X_t[[selected_clin]].copy(); plot_df["label"] = y_t.values
                    for label, color in [(0, C_GREEN),(1, C_RED)]:
                        parts = ax.violinplot(plot_df[plot_df["label"]==label][selected_clin].dropna(),
                                              positions=[label], showmedians=True, showextrema=True)
                        for pc in parts.get('bodies',[]): pc.set_facecolor(color); pc.set_alpha(0.5); pc.set_edgecolor("none")
                        parts['cmedians'].set_color(C_AMBER); parts['cmedians'].set_linewidth(2)
                        for p in ['cbars','cmins','cmaxes']:
                            if p in parts: parts[p].set_color(FG_MUTED)
                    ax.set_title(f"{selected_clin[:30]} — {title}", pad=10)
                    ax.set_xticks([0,1]); ax.set_xticklabels(["No condition","Has condition"])
                    ax.set_ylabel("Scaled value")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("No clinical observation features found.")

    elif eda_section == "Healthcare Utilization":
        util_map = {"Encounters":"total_encounters","Medications":"total_medications","Procedures":"total_procedures","Claims":"total_claims"}
        available = {k:v for k,v in util_map.items() if v in X_train_d1.columns}
        c1, c2 = st.columns(2)
        for i, (name, col) in enumerate(available.items()):
            with (c1 if i%2==0 else c2):
                st.markdown(f"<div style='font-size:0.75rem;font-weight:600;color:{FG_SOFT};margin-bottom:0.4rem;'>{name}</div>", unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(5.5, 4))
                ax.boxplot([X_train_d1.loc[y_train_d1==l, col].dropna() for l in [0,1]],
                           labels=["No Condition","Has Condition"], patch_artist=True, widths=0.45,
                           medianprops=dict(color=C_RED, linewidth=2),
                           boxprops=dict(facecolor=C_GREEN, alpha=0.3, linewidth=0),
                           whiskerprops=dict(color=FG_MUTED), capprops=dict(color=FG_MUTED),
                           flierprops=dict(marker='o', markerfacecolor=C_GREEN, markersize=3, alpha=0.3, linestyle='none'))
                ax.set_title(col, pad=10)
                ax.set_ylabel(f"{col} (scaled)")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

    elif eda_section == "Correlation Heatmap":
        tmp = X_train_d1.copy(); tmp["label"] = y_train_d1.values
        top30 = tmp.corr()["label"].drop("label").abs().sort_values(ascending=False).head(30).index.tolist()
        corr_mat = tmp[top30 + ["label"]].corr()
        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(corr_mat, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
        cb = plt.colorbar(im, ax=ax, shrink=0.75); cb.ax.tick_params(labelsize=8)
        labels = top30 + ["label"]
        ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title("Correlation Matrix — Top 30 Features + Label (D1)", pad=14)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.info("All correlations are weak (r < 0.1), expected with severe class imbalance. Strongest correlates: Cholesterol, Hemoglobin, Blood Pressure.")

    elif eda_section == "Data Drift Analysis":
        top10 = X_train_d1.var().sort_values(ascending=False).head(10).index.tolist()
        fig, axes = plt.subplots(2, 5, figsize=(18, 7)); axes = axes.flatten()
        for i, feat in enumerate(top10):
            ax = axes[i]
            ax.hist(X_train_d1[feat].dropna(), bins=25, alpha=0.65, color=C_GREEN, label="D1", density=True, edgecolor="none")
            d2v = X_train_d2[feat].dropna() if feat in X_train_d2.columns else pd.Series(dtype=float)
            if len(d2v): ax.hist(d2v, bins=25, alpha=0.65, color=C_RED, label="D2", density=True, edgecolor="none")
            ax.set_title(feat[:22], fontsize=8); ax.legend(fontsize=7)
        plt.suptitle("Distribution Shift: D1 vs D2 — Top 10 Features by Variance", y=1.01, fontsize=11)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.warning("Drift analysis on StandardScaled data. Distributions appear similar post-scaling (mean≈0, std≈1). Raw-space drift still exists.")

    elif eda_section == "Missing Values":
        fig, ax = plt.subplots(figsize=(12, 6))
        vals = missing_series.values; cols = missing_series.index.tolist()
        colors_bar = [C_RED if v >= 0.5 else C_GREEN for v in vals[::-1]]
        bars = ax.barh(cols[::-1], vals[::-1], color=colors_bar, edgecolor="none", height=0.6)
        ax.axvline(0.5, color=C_AMBER, linestyle="--", linewidth=1.5, label="50% drop threshold")
        ax.set_xlabel("Missing Rate")
        ax.set_title("Top 20 Columns by Missing Rate (D1 pre-filter)", pad=14)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, color=FG_SOFT, padding=4)
        ax.legend(); ax.set_xlim(0, 1.12)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.info("Columns above 50% missingness (red) are dropped before training — typically rare lab tests or sparse allergy panels.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":

    st.markdown("""
    <div class="page-header">
        <div class="page-tag">Task 3 · Evaluation</div>
        <div class="page-title">Model Performance</div>
        <div class="page-desc">Decision Tree, SVM, and MLP evaluated on both historical (D1) and current (D2) test sets. All models trained on D1.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="model-tags">
        <span class="model-tag">DT · Decision Tree</span>
        <span class="model-tag">SVM · RBF Kernel</span>
        <span class="model-tag">MLP · 128→64→32</span>
    </div>
    """, unsafe_allow_html=True)

    col_sel, _ = st.columns([1, 3])
    with col_sel:
        st.markdown('<span class="sec-label">Metric</span>', unsafe_allow_html=True)
        metric = st.selectbox("Metric", ["accuracy","precision","recall","f1","roc_auc"], label_visibility="collapsed")

    # Logic untouched
    pivot = baseline_df.pivot(index="model", columns="evaluated_on", values=metric)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(pivot)); w = 0.36
    b1 = ax.bar(x-w/2, pivot["D1"], w, label="D1 — Historical", color=C_GREEN, edgecolor="none", alpha=0.9)
    b2 = ax.bar(x+w/2, pivot["D2"], w, label="D2 — Current",    color=C_INDIG, edgecolor="none", alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(pivot.index, fontsize=10)
    ax.set_title(f"{metric.upper()} — All Models on D1 and D2 Test Sets", pad=14)
    ax.set_ylabel(metric.upper()); ax.set_ylim(0, 1.2); ax.legend()
    ax.bar_label(b1, fmt="%.3f", fontsize=8.5, color=FG_SOFT, padding=4)
    ax.bar_label(b2, fmt="%.3f", fontsize=8.5, color=FG_SOFT, padding=4)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("---")
    st.markdown('<span class="sec-label">Full metrics table</span>', unsafe_allow_html=True)
    st.dataframe(
        baseline_df.style.background_gradient(subset=["f1","roc_auc"], cmap="YlGn").format(precision=4),
        use_container_width=True
    )

    st.markdown("---")
    models_map = {"DT": dt_best, "SVM": svm_best, "MLP": mlp}
    col_map    = {"DT": C_GREEN, "SVM": C_RED,   "MLP": C_INDIG}
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<span class="sec-label">ROC Curves</span>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, (X_test, y_test, title) in zip(axes, [(X_test_d1, y_test_d1, "D1 — Historical"),(X_test_d2, y_test_d2, "D2 — Current")]):
            for name, model in models_map.items():
                probs = model.predict_proba(X_test)[:,1]; fpr, tpr, _ = roc_curve(y_test, probs)
                ax.plot(fpr, tpr, label=f"{name} ({auc(fpr,tpr):.3f})", color=col_map[name], lw=2)
            ax.plot([0,1],[0,1], linestyle="--", color=FG_MUTED, lw=1)
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title(title, pad=10); ax.legend(fontsize=8)
        plt.suptitle("ROC Curves", fontsize=11, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with c2:
        st.markdown('<span class="sec-label">Confusion Matrices</span>', unsafe_allow_html=True)
        fig, axes = plt.subplots(2, 3, figsize=(11, 7.5))
        for row, (X_test, y_test, ds) in enumerate([(X_test_d1, y_test_d1, "D1"),(X_test_d2, y_test_d2, "D2")]):
            for ci, (name, model) in enumerate(models_map.items()):
                ax = axes[row][ci]
                disp = ConfusionMatrixDisplay(confusion_matrix(y_test, model.predict(X_test)), display_labels=["Neg","Pos"])
                disp.plot(ax=ax, colorbar=False,
                          cmap=matplotlib.colors.LinearSegmentedColormap.from_list("w2g",[BG_CARD, C_GREEN], N=256))
                ax.set_title(f"{name} · {ds}", fontsize=9)
                for txt in ax.texts: txt.set_color(FG); txt.set_fontsize(10)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("---")
    st.markdown("""
    <div class="insight-block">
        <div class="insight-block-title">Key Observations</div>
        <ul>
            <li>MLP achieves the best F1 on D2, thanks to SMOTE balancing</li>
            <li>All models trained on D1, evaluated on both D1 and D2 test sets</li>
            <li>SVM ROC-AUC varies with C — sensitive to the scaling step</li>
            <li>Decision Tree is consistent but weak at depth 3–5</li>
            <li>Accuracy alone is misleading — F1 and ROC-AUC are what matter here</li>
        </ul>
    </div>
    <div class="insight-block dim">
        <div class="insight-block-title">Bias–Variance Trade-off</div>
        <ul>
            <li>Decision Tree (depth 3–5): high bias, low variance — underfitting</li>
            <li>SVM RBF: moderate balance — sensitive to C hyperparameter</li>
            <li>MLP (128→64→32): low bias, higher variance — best generalisation with SMOTE</li>
        </ul>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — CONTINUAL LEARNING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Continual Learning":

    st.markdown("""
    <div class="page-header">
        <div class="page-tag">Task 4 · Adaptation</div>
        <div class="page-title">Continual Learning</div>
        <div class="page-desc">Fine-tuning the D1-trained MLP on D2 data via partial_fit() over 50 epochs. Before vs after comparison on the D2 test set.</div>
    </div>
    """, unsafe_allow_html=True)

    # Logic untouched
    r0 = continual_df[continual_df.model=="MLP_D1"].iloc[0]
    r1 = continual_df[continual_df.model=="MLP_CL"].iloc[0]
    f1_dir = "dropped" if r1.f1 < r0.f1 else "improved"

    def _dc(a, b): return "neg" if b < a else "pos"

    # Before / after metric cells
    st.markdown(f"""
    <div class="cl-grid">
        <div class="cl-cell">
            <div class="cl-label">F1 Score · D2 Test</div>
            <div class="cl-values">
                <span class="cl-before">{r0.f1:.3f}</span>
                <span class="cl-arrow">→</span>
                <span class="cl-after">{r1.f1:.3f}</span>
            </div>
            <div class="cl-delta {_dc(r0.f1, r1.f1)}">{r1.f1-r0.f1:+.3f} after fine-tuning</div>
        </div>
        <div class="cl-cell">
            <div class="cl-label">ROC-AUC · D2 Test</div>
            <div class="cl-values">
                <span class="cl-before">{r0.roc_auc:.3f}</span>
                <span class="cl-arrow">→</span>
                <span class="cl-after">{r1.roc_auc:.3f}</span>
            </div>
            <div class="cl-delta {_dc(r0.roc_auc, r1.roc_auc)}">{r1.roc_auc-r0.roc_auc:+.3f} after fine-tuning</div>
        </div>
        <div class="cl-cell">
            <div class="cl-label">Recall · D2 Test</div>
            <div class="cl-values">
                <span class="cl-before">{r0.recall:.3f}</span>
                <span class="cl-arrow">→</span>
                <span class="cl-after">{r1.recall:.3f}</span>
            </div>
            <div class="cl-delta {_dc(r0.recall, r1.recall)}">{r1.recall-r0.recall:+.3f} after fine-tuning</div>
        </div>
        <div class="cl-cell">
            <div class="cl-label">Precision · D2 Test</div>
            <div class="cl-values">
                <span class="cl-before">{r0.precision:.3f}</span>
                <span class="cl-arrow">→</span>
                <span class="cl-after">{r1.precision:.3f}</span>
            </div>
            <div class="cl-delta {_dc(r0.precision, r1.precision)}">{r1.precision-r0.precision:+.3f} after fine-tuning</div>
        </div>
    </div>
    <div class="cl-note">before = MLP_D1 (trained D1, evaluated D2 test)  ·  after = MLP_CL (50-epoch partial_fit on D2)</div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Logic untouched — bar chart
    metrics = ["accuracy","precision","recall","f1","roc_auc"]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(metrics)); w = 0.36
    b1 = ax.bar(x-w/2, [r0[m] for m in metrics], w, label="MLP_D1 — Before", color=C_GREEN, edgecolor="none", alpha=0.9)
    b2 = ax.bar(x+w/2, [r1[m] for m in metrics], w, label="MLP_CL — After",  color=C_AMBER, edgecolor="none", alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(["Accuracy","Precision","Recall","F1","ROC-AUC"], fontsize=10)
    ax.set_ylim(0, 1.2); ax.set_ylabel("Score"); ax.legend()
    ax.set_title("Continual Learning — MLP Before vs After on D2 Test Set", pad=14)
    ax.bar_label(b1, fmt="%.3f", fontsize=8.5, color=FG_SOFT, padding=4)
    ax.bar_label(b2, fmt="%.3f", fontsize=8.5, color=FG_SOFT, padding=4)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("---")
    st.markdown('<span class="sec-label">Metrics table</span>', unsafe_allow_html=True)
    st.dataframe(continual_df.style.format(precision=4), use_container_width=True)

    st.markdown("---")
    st.markdown(f"""
    <div class="insight-block red">
        <div class="insight-block-title">Catastrophic Forgetting — What Happened</div>
        <ul>
            <li>MLP_CL F1 <strong>{f1_dir}</strong> from {r0.f1:.3f} to {r1.f1:.3f} on D2</li>
            <li>partial_fit() over 50 epochs aggressively overwrote D1 weights</li>
            <li>No regularisation applied to preserve historical knowledge</li>
            <li>Learning rate was not decayed during fine-tuning</li>
        </ul>
    </div>
    <div class="insight-block">
        <div class="insight-block-title">What Would Actually Work</div>
        <ul>
            <li>Elastic Weight Consolidation (EWC) — penalises updates to important D1 weights</li>
            <li>Learning without Forgetting (LwF) — knowledge distillation approach</li>
            <li>Progressive Neural Networks — adds D2 capacity without touching D1 weights</li>
            <li>D1 MLP already generalises to D2 — aggressive fine-tuning is counterproductive</li>
        </ul>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Feature Importance":

    st.markdown("""
    <div class="page-header">
        <div class="page-tag">Task 3 · Interpretability</div>
        <div class="page-title">Feature Importance</div>
        <div class="page-desc">What the Decision Tree found most predictive — broken down by category and searchable by name.</div>
    </div>
    """, unsafe_allow_html=True)

    # Logic untouched
    feat_imp = pd.Series(dt_best.feature_importances_, index=feature_names).sort_values(ascending=False)
    top20 = feat_imp.head(20)

    st.markdown('<span class="sec-label">Top 20 — Decision Tree</span>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(11, 7))
    alphas = [0.45 + 0.55*(1-i/20) for i in range(len(top20))]
    colors_bar = [matplotlib.colors.to_rgba(C_GREEN, a) for a in alphas]
    bars = ax.barh(top20.index[::-1], top20.values[::-1], color=colors_bar[::-1], edgecolor="none", height=0.65)
    ax.set_xlabel("Feature Importance")
    ax.set_title("Top 20 Feature Importances — Decision Tree", pad=14)
    ax.bar_label(bars, fmt="%.4f", fontsize=8, color=FG_SOFT, padding=4)
    ax.set_xlim(0, top20.max() * 1.2)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("---")

    # Logic untouched — category breakdown
    demo_f  = [f for f in feature_names if any(x in f for x in ["GENDER","RACE","ETHNICITY","INCOME","MARITAL","age","is_deceased","HEALTHCARE"])]
    enc_f   = [f for f in feature_names if any(x in f for x in ["encounter","claim_cost","payer_coverage"])]
    obs_f   = [f for f in feature_names if f.startswith("obs_")]
    util_f  = [f for f in feature_names if any(x in f for x in ["medication","procedure","immunization","careplan","imaging","device","supply","transaction","payer","claims"])]

    st.markdown('<span class="sec-label">Feature category counts</span>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Demographic",  len(demo_f))
    c2.metric("Encounter",    len(enc_f))
    c3.metric("Observation",  len(obs_f))
    c4.metric("Utilization",  len(util_f))

    st.markdown("---")

    st.markdown('<span class="sec-label">Search features by name</span>', unsafe_allow_html=True)
    search = st.text_input("", placeholder="e.g. cholesterol, BMI, systolic, age …", label_visibility="collapsed")
    filtered = [f for f in feature_names if search.lower() in f.lower()] if search else feature_names
    st.markdown(f"<div style='font-size:0.72rem;color:{FG_MUTED};margin-bottom:0.5rem;font-family:var(--mono);'>{len(filtered)} of {len(feature_names)} features</div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({"feature_name": filtered}), use_container_width=True, height=260)

    st.markdown("---")
    st.markdown("""
    <div class="insight-block">
        <div class="insight-block-title">Key Findings</div>
        <ul>
            <li>Observation-derived features (vitals and lab aggregates) dominate the top 20</li>
            <li>BMI, Blood Pressure, and Cholesterol are the most predictive clinical signals</li>
            <li>Demographics contribute but rank consistently below clinical measurements</li>
            <li>Utilisation counts (encounters, medications) add secondary signal</li>
        </ul>
    </div>
    <div class="insight-block dim">
        <div class="insight-block-title">Feature Engineering Choices That Helped</div>
        <ul>
            <li>Mean + variance aggregation captures both central tendency and patient variability</li>
            <li>Dropping columns with more than 50% missingness improved signal-to-noise</li>
            <li>StandardScaling was essential for SVM and MLP convergence</li>
        </ul>
    </div>""", unsafe_allow_html=True)
