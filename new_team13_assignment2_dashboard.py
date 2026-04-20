# ── CONFIGURATION ──────────────────────────────────────────────────────────
DATA_DIR = "data/"          # path to folder containing raw CSV files
RANDOM_STATE = 42
TEMPORAL_CUTOFF = "2020-01-01"
TEST_SIZE = 0.2
# ───────────────────────────────────────────────────────────────────────────

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

# ── THEME & STYLING ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Clinical Prediction · Team 13",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Inject global CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ─────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,300&family=DM+Mono:wght@400;500&family=Playfair+Display:wght@700;800&display=swap');

/* ── Root palette (dark medical) ──────────────────────── */
:root {
    --bg-base:      #090e1a;
    --bg-card:      #0f1929;
    --bg-elevated:  #162236;
    --border:       #1e3352;
    --border-bright:#2a4a72;
    --accent-blue:  #3d8bff;
    --accent-teal:  #00c9a7;
    --accent-orange:#ff7c45;
    --accent-purple:#a78bfa;
    --text-primary: #e8edf7;
    --text-secondary:#8ba4c8;
    --text-muted:   #4e6a8d;
    --font-display: 'Playfair Display', serif;
    --font-body:    'DM Sans', sans-serif;
    --font-mono:    'DM Mono', monospace;
    --radius:       12px;
    --radius-sm:    8px;
    --shadow:       0 4px 24px rgba(0,0,0,0.4);
    --shadow-glow:  0 0 24px rgba(61,139,255,0.15);
}

/* ── Global resets ─────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: var(--font-body) !important;
    color: var(--text-primary) !important;
    background-color: var(--bg-base) !important;
}

/* ── Main content area ─────────────────────────────────── */
.main .block-container {
    padding: 2rem 2.5rem 4rem !important;
    max-width: 1400px !important;
    background: var(--bg-base) !important;
}

/* ── Sidebar ────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
    padding-top: 0 !important;
}
section[data-testid="stSidebar"] > div:first-child {
    padding: 0 !important;
}
section[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1.25rem !important;
}

/* ── Sidebar brand block ────────────────────────────────── */
.sidebar-brand {
    background: linear-gradient(135deg, #0d1f3c 0%, #162236 100%);
    border-bottom: 1px solid var(--border);
    padding: 1.5rem 1.25rem;
    margin: -1.5rem -1.25rem 1.5rem;
}
.sidebar-brand-icon {
    font-size: 2rem;
    margin-bottom: 0.4rem;
    display: block;
}
.sidebar-brand-title {
    font-family: var(--font-display) !important;
    font-size: 1.2rem !important;
    font-weight: 800 !important;
    color: var(--text-primary) !important;
    line-height: 1.2 !important;
    margin: 0 !important;
}
.sidebar-brand-sub {
    font-size: 0.72rem !important;
    color: var(--text-muted) !important;
    margin-top: 0.3rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
.sidebar-team-badge {
    display: inline-block;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-teal));
    color: #fff !important;
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.2em 0.6em !important;
    border-radius: 99px !important;
    margin-top: 0.5rem !important;
}

/* ── Radio nav ──────────────────────────────────────────── */
div[data-testid="stRadio"] > label {
    display: none !important;
}
div[data-testid="stRadio"] > div {
    gap: 0.25rem !important;
    flex-direction: column !important;
}
div[data-testid="stRadio"] > div > label {
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.6rem 0.75rem !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    color: var(--text-secondary) !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.5rem !important;
}
div[data-testid="stRadio"] > div > label:hover {
    background: var(--bg-elevated) !important;
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
}
div[data-testid="stRadio"] > div > label[data-baseweb="radio"] input:checked + div,
div[data-testid="stRadio"] > div > label:has(input:checked) {
    background: linear-gradient(90deg, rgba(61,139,255,0.15), rgba(0,201,167,0.08)) !important;
    border-color: var(--accent-blue) !important;
    color: var(--accent-blue) !important;
}

/* ── Page titles ────────────────────────────────────────── */
h1 {
    font-family: var(--font-display) !important;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.02em !important;
    margin-bottom: 0 !important;
    line-height: 1.15 !important;
}
h2, h3 {
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.01em !important;
}
h2 { font-size: 1.35rem !important; }
h3 { font-size: 1.1rem !important; }

/* Streamlit heading elements */
[data-testid="stHeading"] > * {
    font-family: var(--font-display) !important;
}

/* ── Metric cards ───────────────────────────────────────── */
[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1.1rem 1.25rem !important;
    box-shadow: var(--shadow) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    position: relative !important;
    overflow: hidden !important;
}
[data-testid="metric-container"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-teal));
    opacity: 0.7;
}
[data-testid="metric-container"]:hover {
    border-color: var(--border-bright) !important;
    box-shadow: var(--shadow-glow) !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--font-mono) !important;
    font-size: 1.75rem !important;
    font-weight: 500 !important;
    color: var(--accent-blue) !important;
    letter-spacing: -0.02em !important;
}
[data-testid="stMetricDelta"] {
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
}

/* ── Cards & Containers ─────────────────────────────────── */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    box-shadow: var(--shadow);
    margin-bottom: 1rem;
}
.card-accent-blue  { border-top: 3px solid var(--accent-blue); }
.card-accent-teal  { border-top: 3px solid var(--accent-teal); }
.card-accent-orange{ border-top: 3px solid var(--accent-orange); }
.card-accent-purple{ border-top: 3px solid var(--accent-purple); }

/* ── Section dividers ───────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.75rem 0 !important;
    opacity: 1 !important;
}

/* ── Page header block ──────────────────────────────────── */
.page-header {
    padding: 1.5rem 0 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.75rem;
}
.page-header-eyebrow {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--accent-teal);
    margin-bottom: 0.4rem;
}
.page-header-title {
    font-family: var(--font-display) !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    color: var(--text-primary) !important;
    margin: 0 !important;
    line-height: 1.15 !important;
}
.page-header-desc {
    font-size: 0.9rem;
    color: var(--text-secondary);
    margin-top: 0.5rem;
    max-width: 680px;
    line-height: 1.6;
}

/* ── Selectbox / dropdowns ──────────────────────────────── */
div[data-baseweb="select"] > div {
    background: var(--bg-elevated) !important;
    border-color: var(--border-bright) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-size: 0.875rem !important;
}
div[data-baseweb="select"] > div:hover {
    border-color: var(--accent-blue) !important;
}
div[data-baseweb="popover"] {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-bright) !important;
    border-radius: var(--radius-sm) !important;
}
li[role="option"] {
    background: var(--bg-elevated) !important;
    color: var(--text-primary) !important;
    font-size: 0.875rem !important;
}
li[role="option"]:hover {
    background: var(--bg-card) !important;
    color: var(--accent-blue) !important;
}

/* ── Text input ─────────────────────────────────────────── */
input[type="text"] {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-bright) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.875rem !important;
    padding: 0.5rem 0.75rem !important;
}
input[type="text"]:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 2px rgba(61,139,255,0.2) !important;
}

/* ── Dataframes ─────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    overflow: hidden !important;
    background: var(--bg-card) !important;
}
[data-testid="stDataFrame"] iframe {
    border-radius: var(--radius) !important;
}

/* ── Info / warning / success banners ───────────────────── */
[data-testid="stAlert"] {
    border-radius: var(--radius-sm) !important;
    border-left-width: 3px !important;
    font-size: 0.875rem !important;
}
div[data-testid="stAlert"][kind="info"] {
    background: rgba(61,139,255,0.08) !important;
    border-color: var(--accent-blue) !important;
    color: var(--text-secondary) !important;
}
div[data-testid="stAlert"][kind="warning"] {
    background: rgba(255,124,69,0.08) !important;
    border-color: var(--accent-orange) !important;
    color: var(--text-secondary) !important;
}

/* ── Matplotlib figure backgrounds ──────────────────────── */
[data-testid="stImage"] img {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
}

/* ── Spinner ────────────────────────────────────────────── */
[data-testid="stSpinner"] > div {
    color: var(--accent-blue) !important;
}

/* ── Scrollbar ──────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border-bright); border-radius: 3px; }

/* ── Team cards ─────────────────────────────────────────── */
.team-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-top: 1rem;
}
.team-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem 1.1rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s ease, transform 0.2s ease;
}
.team-card:hover {
    border-color: var(--border-bright);
    transform: translateY(-2px);
}
.team-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.team-card:nth-child(1)::before { background: var(--accent-blue); }
.team-card:nth-child(2)::before { background: var(--accent-teal); }
.team-card:nth-child(3)::before { background: var(--accent-orange); }
.team-card:nth-child(4)::before { background: var(--accent-purple); }
.team-card .role {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.4rem;
}
.team-card .name {
    font-size: 0.97rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
    line-height: 1.3;
}
.team-card .task-badge {
    display: inline-block;
    font-family: var(--font-mono);
    font-size: 0.65rem;
    font-weight: 500;
    padding: 0.18em 0.55em;
    border-radius: 4px;
    margin-bottom: 0.6rem;
    border: 1px solid;
}
.team-card:nth-child(1) .task-badge { color: var(--accent-blue); border-color: rgba(61,139,255,0.3); background: rgba(61,139,255,0.08); }
.team-card:nth-child(2) .task-badge { color: var(--accent-teal); border-color: rgba(0,201,167,0.3); background: rgba(0,201,167,0.08); }
.team-card:nth-child(3) .task-badge { color: var(--accent-orange); border-color: rgba(255,124,69,0.3); background: rgba(255,124,69,0.08); }
.team-card:nth-child(4) .task-badge { color: var(--accent-purple); border-color: rgba(167,139,250,0.3); background: rgba(167,139,250,0.08); }
.team-card .desc {
    font-size: 0.8rem;
    color: var(--text-secondary);
    line-height: 1.55;
}

/* ── Pill badges ─────────────────────────────────────────── */
.pill {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.2em 0.65em;
    border-radius: 99px;
    border: 1px solid;
    margin-right: 0.35rem;
    margin-bottom: 0.35rem;
    font-family: var(--font-mono);
}
.pill-blue   { color: var(--accent-blue);   border-color: rgba(61,139,255,0.35);  background: rgba(61,139,255,0.09); }
.pill-teal   { color: var(--accent-teal);   border-color: rgba(0,201,167,0.35);   background: rgba(0,201,167,0.09); }
.pill-orange { color: var(--accent-orange); border-color: rgba(255,124,69,0.35);  background: rgba(255,124,69,0.09); }
.pill-purple { color: var(--accent-purple); border-color: rgba(167,139,250,0.35); background: rgba(167,139,250,0.09); }

/* ── Architecture grid ──────────────────────────────────── */
.arch-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-top: 1rem;
}
.arch-block {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 1.25rem;
}
.arch-block-title {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
}
.arch-block ul {
    margin: 0;
    padding-left: 0;
    list-style: none;
}
.arch-block ul li {
    font-size: 0.83rem;
    color: var(--text-secondary);
    padding: 0.22rem 0;
    display: flex;
    align-items: baseline;
    gap: 0.5rem;
    line-height: 1.45;
}
.arch-block ul li::before {
    content: '—';
    color: var(--text-muted);
    font-size: 0.7rem;
    flex-shrink: 0;
}

/* ── Analysis markdown blocks ────────────────────────────── */
.analysis-block {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent-blue);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    padding: 1.25rem 1.5rem;
    margin-top: 0.75rem;
}
.analysis-block h4 {
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--accent-blue) !important;
    margin: 0 0 0.6rem 0 !important;
}
.analysis-block ul {
    margin: 0;
    padding-left: 1.1rem;
}
.analysis-block ul li {
    font-size: 0.855rem;
    color: var(--text-secondary);
    padding: 0.18rem 0;
    line-height: 1.55;
}

/* ── Spinner text ────────────────────────────────────────── */
[data-testid="stSpinner"] p {
    color: var(--text-secondary) !important;
    font-size: 0.875rem !important;
}

/* ── Section label ───────────────────────────────────────── */
.section-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--accent-teal);
    margin-bottom: 0.35rem;
}

/* ── Ensure pyplot images show dark bg ───────────────────── */
[data-testid="column"] img,
.stImage img {
    background: var(--bg-card) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Matplotlib global theme ──────────────────────────────────────────────────
DARK_BG    = "#0f1929"
CARD_BG    = "#162236"
TEXT_PRI   = "#e8edf7"
TEXT_SEC   = "#8ba4c8"
TEXT_MUT   = "#4e6a8d"
BORDER_COL = "#1e3352"
ACC_BLUE   = "#3d8bff"
ACC_TEAL   = "#00c9a7"
ACC_ORANGE = "#ff7c45"
ACC_PURPLE = "#a78bfa"
PALETTE    = [ACC_BLUE, ACC_ORANGE, ACC_TEAL, ACC_PURPLE, "#f472b6", "#fbbf24"]

def apply_dark_theme(fig, ax_or_axes):
    fig.patch.set_facecolor(DARK_BG)
    axes = ax_or_axes if isinstance(ax_or_axes, (list, np.ndarray)) else [ax_or_axes]
    axes_flat = []
    for a in axes:
        if isinstance(a, np.ndarray):
            axes_flat.extend(a.flatten().tolist())
        else:
            axes_flat.append(a)
    for ax in axes_flat:
        ax.set_facecolor(CARD_BG)
        ax.tick_params(colors=TEXT_SEC, labelsize=8.5)
        ax.xaxis.label.set_color(TEXT_SEC)
        ax.yaxis.label.set_color(TEXT_SEC)
        ax.title.set_color(TEXT_PRI)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER_COL)
        ax.grid(color=BORDER_COL, linestyle='--', linewidth=0.5, alpha=0.7)

matplotlib.rcParams.update({
    'font.family':      'sans-serif',
    'font.sans-serif':  ['DejaVu Sans'],
    'axes.facecolor':   CARD_BG,
    'figure.facecolor': DARK_BG,
    'axes.edgecolor':   BORDER_COL,
    'axes.labelcolor':  TEXT_SEC,
    'xtick.color':      TEXT_SEC,
    'ytick.color':      TEXT_SEC,
    'text.color':       TEXT_PRI,
    'axes.titlecolor':  TEXT_PRI,
    'axes.grid':        True,
    'grid.color':       BORDER_COL,
    'grid.linestyle':   '--',
    'grid.linewidth':   0.5,
    'grid.alpha':       0.6,
    'legend.facecolor': DARK_BG,
    'legend.edgecolor': BORDER_COL,
    'legend.labelcolor':TEXT_SEC,
    'axes.titlesize':   11,
    'axes.labelsize':   9,
    'xtick.labelsize':  8.5,
    'ytick.labelsize':  8.5,
})

# ── PIPELINE ────────────────────────────────────────────────────────────────

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
                               usecols=lambda c: c in ["PATIENTID", "TYPE", "AMOUNT"])

    ref_date = pd.Timestamp(TEMPORAL_CUTOFF, tz="UTC")
    patients["BIRTHDATE"] = pd.to_datetime(patients["BIRTHDATE"], errors="coerce", utc=True)
    patients["DEATHDATE"] = pd.to_datetime(patients["DEATHDATE"], errors="coerce", utc=True)
    patients["age"] = (ref_date - patients["BIRTHDATE"]).dt.days // 365
    patients["is_deceased"] = patients["DEATHDATE"].notna().astype(int)

    cat_cols = ["GENDER", "RACE", "ETHNICITY", "MARITAL"]
    for col in cat_cols:
        if col in patients.columns:
            le = LabelEncoder()
            patients[col] = le.fit_transform(patients[col].astype(str))

    demo_cols = ["Id", "age", "is_deceased", "GENDER", "RACE",
                 "ETHNICITY", "MARITAL", "INCOME", "HEALTHCARE_COVERAGE"]
    demo_cols = [c for c in demo_cols if c in patients.columns]
    pat_features = patients[demo_cols].rename(columns={"Id": "PATIENT"})

    encounters["START"] = pd.to_datetime(encounters["START"], errors="coerce", utc=True)
    enc_agg = encounters.groupby("PATIENT").agg(
        total_encounters=("Id", "count"),
        unique_encounter_types=("ENCOUNTERCLASS", "nunique"),
        avg_base_encounter_cost=("BASE_ENCOUNTER_COST", "mean"),
        total_claim_cost=("TOTAL_CLAIM_COST", "sum"),
        avg_payer_coverage=("PAYER_COVERAGE", "mean")
    ).reset_index()

    observations["VALUE"] = pd.to_numeric(observations["VALUE"], errors="coerce")
    obs_numeric = observations.dropna(subset=["VALUE"])
    obs_agg = obs_numeric.groupby(["PATIENT", "DESCRIPTION"])["VALUE"].agg(
        ["mean", "std"]).reset_index()
    obs_agg.columns = ["PATIENT", "DESC", "mean", "std"]
    obs_agg["DESC"] = obs_agg["DESC"].str.replace(r"[^a-zA-Z0-9]", "_", regex=True).str[:40]

    obs_mean = obs_agg.pivot_table(index="PATIENT", columns="DESC", values="mean", aggfunc="mean")
    obs_std  = obs_agg.pivot_table(index="PATIENT", columns="DESC", values="std", aggfunc="mean")
    obs_mean.columns = ["obs_" + c + "_mean" for c in obs_mean.columns]
    obs_std.columns  = ["obs_" + c + "_var"  for c in obs_std.columns]

    threshold = 0.05
    obs_mean = obs_mean.loc[:, obs_mean.notna().mean() >= threshold]
    obs_std  = obs_std.loc[:, obs_std.notna().mean() >= threshold]
    obs_features = obs_mean.join(obs_std, how="outer").reset_index()

    med_agg = medications.groupby("PATIENT").agg(
        total_medications=("START", "count"),
        unique_medications=("DESCRIPTION", "nunique"),
        avg_medication_cost=("BASE_COST", "mean"),
        total_dispenses=("DISPENSES", "sum")
    ).reset_index()

    proc_agg = procedures.groupby("PATIENT").agg(
        total_procedures=("START", "count"),
        unique_procedures=("DESCRIPTION", "nunique"),
        avg_procedure_cost=("BASE_COST", "mean")
    ).reset_index()

    imm_agg = immunizations.groupby("PATIENT").agg(
        total_immunizations=("DATE", "count"),
        unique_vaccines=("DESCRIPTION", "nunique")
    ).reset_index()

    allergy_agg = allergies.groupby("PATIENT").agg(
        total_allergies=("START", "count"),
        unique_allergy_types=("TYPE", "nunique"),
        unique_allergy_categories=("CATEGORY", "nunique")
    ).reset_index()

    care_agg = careplans.groupby("PATIENT").agg(
        total_careplans=("Id", "count"),
        unique_careplan_reasons=("REASONDESCRIPTION", "nunique")
    ).reset_index()

    img_agg = imaging.groupby("PATIENT").agg(
        total_imaging=("Id", "count"),
        unique_modalities=("MODALITY_DESCRIPTION", "nunique"),
        unique_body_sites=("BODYSITE_DESCRIPTION", "nunique")
    ).reset_index()

    dev_agg = devices.groupby("PATIENT").agg(
        total_devices=("START", "count"),
        unique_device_types=("DESCRIPTION", "nunique")
    ).reset_index()

    sup_agg = supplies.groupby("PATIENT").agg(
        total_supplies=("DATE", "count"),
        unique_supply_types=("DESCRIPTION", "nunique")
    ).reset_index()

    pay_agg = payer_trans.groupby("PATIENT").agg(
        total_payer_transitions=("START_DATE", "count"),
        unique_payers=("PAYER", "nunique")
    ).reset_index()

    claims_cost_col = "OUTSTANDING1" if "OUTSTANDING1" in claims.columns else "Id"
    claims_agg = claims.groupby("PATIENTID").agg(
        total_claims=("Id", "count"),
        avg_claim_cost=(claims_cost_col, "mean")
    ).reset_index().rename(columns={"PATIENTID": "PATIENT"})

    if "AMOUNT" in claims_trans.columns:
        ct_agg = claims_trans.groupby("PATIENTID").agg(
            total_transactions=("TYPE", "count"),
            total_transaction_amount=("AMOUNT", "sum"),
            unique_transaction_types=("TYPE", "nunique")
        ).reset_index().rename(columns={"PATIENTID": "PATIENT"})
    else:
        ct_agg = pd.DataFrame(columns=["PATIENT"])

    df = pat_features.copy()
    for feat_df in [enc_agg, obs_features, med_agg, proc_agg, imm_agg,
                    allergy_agg, care_agg, img_agg, dev_agg, sup_agg,
                    pay_agg, claims_agg, ct_agg]:
        if "PATIENT" in feat_df.columns:
            df = df.merge(feat_df, on="PATIENT", how="left")

    conditions["START"] = pd.to_datetime(conditions["START"], dayfirst=True, errors="coerce", utc=True)
    clinical = conditions[conditions["DESCRIPTION"].str.contains(
        r"\(disorder\)|\(finding\)", na=False, regex=True)]
    positive_patients = set(clinical["PATIENT"].unique())
    df["label"] = df["PATIENT"].apply(lambda x: 1 if x in positive_patients else 0)

    enc_dates = encounters.groupby("PATIENT")["START"].agg(["min", "max"]).reset_index()
    enc_dates.columns = ["PATIENT", "first_enc", "last_enc"]
    df = df.merge(enc_dates, on="PATIENT", how="left")

    cutoff = pd.Timestamp(TEMPORAL_CUTOFF, tz="UTC")
    df1 = df[df["first_enc"] < cutoff].copy()
    df2 = df[df["last_enc"] >= cutoff].copy()

    drop_cols = ["PATIENT", "first_enc", "last_enc"]
    df1 = df1.drop(columns=[c for c in drop_cols if c in df1.columns])
    df2 = df2.drop(columns=[c for c in drop_cols if c in df2.columns])

    all_X_pre = df1.drop(columns=["label"], errors="ignore")
    missing_series = (all_X_pre.isna().mean().sort_values(ascending=False).head(20))

    def preprocess(df_in):
        X = df_in.drop(columns=["label"])
        y = df_in["label"]
        X = X.loc[:, X.isna().mean() < 0.5]
        for col in X.columns:
            if X[col].dtype in ["float64", "int64"]:
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(X[col].mode()[0])
        return X, y

    X1, y1 = preprocess(df1)
    X2, y2 = preprocess(df2)

    common_cols = list(set(X1.columns) & set(X2.columns))
    X1 = X1[common_cols]
    X2 = X2[common_cols]

    X_train_d1, X_test_d1, y_train_d1, y_test_d1 = train_test_split(
        X1, y1, test_size=TEST_SIZE, stratify=y1, random_state=RANDOM_STATE)
    X_train_d2, X_test_d2, y_train_d2, y_test_d2 = train_test_split(
        X2, y2, test_size=TEST_SIZE, stratify=y2, random_state=RANDOM_STATE)

    scaler = StandardScaler()
    X_train_d1 = pd.DataFrame(scaler.fit_transform(X_train_d1), columns=common_cols)
    X_test_d1  = pd.DataFrame(scaler.transform(X_test_d1),      columns=common_cols)

    scaler2 = StandardScaler()
    X_train_d2 = pd.DataFrame(scaler2.fit_transform(X_train_d2), columns=common_cols)
    X_test_d2  = pd.DataFrame(scaler2.transform(X_test_d2),      columns=common_cols)

    feature_names = common_cols

    smote = SMOTE(random_state=RANDOM_STATE)
    X1_smote, y1_smote = smote.fit_resample(X_train_d1, y_train_d1)

    smote2 = SMOTE(random_state=RANDOM_STATE)
    X2_smote, y2_smote = smote2.fit_resample(X_train_d2, y_train_d2)

    dt_grid = GridSearchCV(
        DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        {"max_depth": [3, 5, 10, None]}, scoring="f1", cv=5, n_jobs=-1)
    dt_grid.fit(X_train_d1, y_train_d1)
    dt_best = dt_grid.best_estimator_

    svm_grid = GridSearchCV(
        SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=RANDOM_STATE),
        {"C": [0.1, 1, 10]}, scoring="f1", cv=5, n_jobs=-1)
    svm_grid.fit(X_train_d1, y_train_d1)
    svm_best = svm_grid.best_estimator_

    mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation="relu",
                        max_iter=500, early_stopping=True, random_state=RANDOM_STATE)
    mlp.fit(X1_smote, y1_smote)

    def evaluate(model, X, y, label, dataset):
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]
        return {
            "model": label, "evaluated_on": dataset,
            "accuracy":  accuracy_score(y, preds),
            "precision": precision_score(y, preds, zero_division=0),
            "recall":    recall_score(y, preds, zero_division=0),
            "f1":        f1_score(y, preds, zero_division=0),
            "roc_auc":   roc_auc_score(y, probs),
        }

    results = []
    for model, name in [(dt_best, "DT"), (svm_best, "SVM"), (mlp, "MLP")]:
        results.append(evaluate(model, X_test_d1, y_test_d1, name, "D1"))
        results.append(evaluate(model, X_test_d2, y_test_d2, name, "D2"))
    baseline_df = pd.DataFrame(results)

    mlp_cl = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation="relu",
                           max_iter=1, warm_start=False, random_state=RANDOM_STATE)
    mlp_cl.fit(X1_smote[:10], y1_smote[:10])
    mlp_cl.coefs_      = mlp.coefs_
    mlp_cl.intercepts_ = mlp.intercepts_

    X2_arr = X2_smote.values if hasattr(X2_smote, "values") else X2_smote
    y2_arr = y2_smote.values if hasattr(y2_smote, "values") else y2_smote
    classes = np.array([0, 1])

    for _ in range(50):
        idx = np.random.permutation(len(X2_arr))
        for start in range(0, len(X2_arr), 64):
            batch = idx[start:start + 64]
            mlp_cl.partial_fit(X2_arr[batch], y2_arr[batch], classes=classes)

    r_before = evaluate(mlp,    X_test_d2, y_test_d2, "MLP_D1", "D2")
    r_after  = evaluate(mlp_cl, X_test_d2, y_test_d2, "MLP_CL", "D2")
    continual_df = pd.DataFrame([r_before, r_after])

    return {
        "X_train_d1": X_train_d1, "y_train_d1": y_train_d1,
        "y_test_d1": y_test_d1,
        "X_train_d2": X_train_d2, "y_train_d2": y_train_d2,
        "y_test_d2": y_test_d2,
        "X_test_d1": X_test_d1,  "X_test_d2": X_test_d2,
        "feature_names": feature_names,
        "baseline_df": baseline_df, "continual_df": continual_df,
        "dt_best": dt_best, "svm_best": svm_best,
        "mlp": mlp, "mlp_cl": mlp_cl,
        "d1_size": len(df1), "d2_size": len(df2),
        "total_patients": len(df),
        "missing_series": missing_series,
    }


# ── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <span class="sidebar-brand-icon">🏥</span>
        <div class="sidebar-brand-title">Clinical<br>Prediction</div>
        <div class="sidebar-brand-sub">BITS F464 · Machine Learning</div>
        <span class="sidebar-team-badge">Team 13 · Sem 2 2025-26</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("Navigate", [
        "📋  Project Overview",
        "🔬  Exploratory Data Analysis",
        "📊  Model Performance",
        "🔄  Continual Learning",
        "🎯  Feature Importance"
    ])

    st.markdown("---")
    st.markdown(f"""
    <div style="font-size:0.72rem; color: var(--text-muted); line-height:1.6;">
        <div style="color:var(--text-secondary); font-weight:600; margin-bottom:0.4rem;">Pipeline Config</div>
        <div><span style="color:var(--accent-teal);">Cutoff</span> · {TEMPORAL_CUTOFF}</div>
        <div><span style="color:var(--accent-teal);">Test Split</span> · {int(TEST_SIZE*100)}%</div>
        <div><span style="color:var(--accent-teal);">Seed</span> · {RANDOM_STATE}</div>
    </div>
    """, unsafe_allow_html=True)

# ── Strip emoji prefix for logic ─────────────────────────────────────────────
page_clean = page.split("  ")[-1]

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("⚙️  Running ML pipeline — first load may take a few minutes…"):
    data = run_pipeline()

X_train_d1    = data["X_train_d1"]
y_train_d1    = data["y_train_d1"]
y_test_d1     = data["y_test_d1"]
X_train_d2    = data["X_train_d2"]
y_train_d2    = data["y_train_d2"]
y_test_d2     = data["y_test_d2"]
X_test_d1     = data["X_test_d1"]
X_test_d2     = data["X_test_d2"]
feature_names = data["feature_names"]
baseline_df   = data["baseline_df"]
continual_df  = data["continual_df"]
dt_best       = data["dt_best"]
svm_best      = data["svm_best"]
mlp           = data["mlp"]
mlp_cl        = data["mlp_cl"]
missing_series = data["missing_series"]


# ── PAGE 1: PROJECT OVERVIEW ──────────────────────────────────────────────────

if page_clean == "Project Overview":
    st.markdown("""
    <div class="page-header">
        <div class="page-header-eyebrow">Automated ML Pipeline</div>
        <div class="page-header-title">Clinical Prediction under Temporal Shift</div>
        <div class="page-header-desc">
            An end-to-end machine learning pipeline on synthetic Electronic Health Records (EHR) data —
            predicting clinically significant conditions across historical and current patient cohorts.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Patients",        f"{data['total_patients']:,}")
    col2.metric("Feature Dimensions",    len(feature_names))
    col3.metric("D1 Historical Patients",f"{data['d1_size']:,}")
    col4.metric("D2 Current Patients",   f"{data['d2_size']:,}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Pipeline Architecture</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="arch-grid">
        <div class="arch-block">
            <div class="arch-block-title">📦 Data Pipeline · Task 2</div>
            <ul>
                <li>15 CSV files merged on patient identifier</li>
                <li>Temporal split: pre / post 2020-01-01</li>
                <li>Sparse columns dropped (>50% missing)</li>
                <li>Binary target: has (disorder) / (finding)</li>
                <li>StandardScaler fit on training only</li>
                <li>80/20 stratified train-test split</li>
            </ul>
        </div>
        <div class="arch-block">
            <div class="arch-block-title">🤖 Models Trained · Task 3</div>
            <ul>
                <li>Decision Tree — GridSearchCV, class_weight=balanced</li>
                <li>SVM RBF kernel — GridSearchCV, class_weight=balanced</li>
                <li>MLP Neural Network — SMOTE oversampling</li>
            </ul>
        </div>
        <div class="arch-block">
            <div class="arch-block-title">⚖️ Class Imbalance Strategy</div>
            <ul>
                <li>Majority: label=0 (no clinical condition)</li>
                <li>Minority: label=1 (has disorder / finding)</li>
                <li>DT + SVM: class_weight="balanced"</li>
                <li>MLP: SMOTE synthetic oversampling</li>
            </ul>
        </div>
        <div class="arch-block">
            <div class="arch-block-title">⏱ Temporal Dataset Split</div>
            <ul>
                <li>D1 (Historical): first encounter before 2020</li>
                <li>D2 (Current): any encounter from 2020 onward</li>
                <li>Patients may overlap across datasets</li>
                <li>Models trained on D1, evaluated on both D1 & D2</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Team</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="team-grid">
        <div class="team-card">
            <div class="role">Data Architect</div>
            <div class="name">Shriniketh Deevanapalli</div>
            <span class="task-badge">Task 2 (a, b, c)</span>
            <div class="desc">Merged 15 CSV tables, implemented the temporal train/test split,
            and engineered the feature dataset with StandardScaler.</div>
        </div>
        <div class="team-card">
            <div class="role">ML Engineer</div>
            <div class="name">Sanvi Udhan</div>
            <span class="task-badge">Task 3 (a,b,c) + Task 4</span>
            <div class="desc">Trained Decision Tree, SVM, and MLP models.
            Implemented continual learning via partial_fit and compiled all performance metrics.</div>
        </div>
        <div class="team-card">
            <div class="role">Full-Stack Developer</div>
            <div class="name">Sai Dheeraj Yadavalli</div>
            <span class="task-badge">Task 1 + Task 5</span>
            <div class="desc">Built this Streamlit dashboard and integrated outputs
            from all team members into interactive visualizations.</div>
        </div>
        <div class="team-card">
            <div class="role">Data Analyst</div>
            <div class="name">Shambhavi Rani</div>
            <span class="task-badge">Task 2(d) + Task 3(d,e,f) + Task 5</span>
            <div class="desc">Performed EDA, wrote the bias-variance and feature importance analysis,
            and produced the final video presentation.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── PAGE 2: EDA ───────────────────────────────────────────────────────────────

elif page_clean == "Exploratory Data Analysis":
    st.markdown("""
    <div class="page-header">
        <div class="page-header-eyebrow">Data Exploration</div>
        <div class="page-header-title">Exploratory Data Analysis</div>
        <div class="page-header-desc">
            Statistical summaries, distributions, and quality diagnostics across both
            historical (D1) and current (D2) patient cohorts.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("D1 Train Size",     f"{len(y_train_d1):,}")
    col2.metric("D1 Positive Rate",  f"{y_train_d1.mean()*100:.1f}%")
    col3.metric("D2 Train Size",     f"{len(y_train_d2):,}")
    col4.metric("D2 Positive Rate",  f"{y_train_d2.mean()*100:.1f}%")

    st.markdown("---")
    st.markdown('<div class="section-label">Select EDA Section</div>', unsafe_allow_html=True)
    eda_section = st.selectbox("EDA Section", [
        "Class Distribution", "Demographics", "Clinical Features",
        "Healthcare Utilization", "Correlation Heatmap",
        "Data Drift Analysis", "Missing Values",
    ], label_visibility="collapsed")
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Class Distribution ─────────────────────────────────────────────────
    if eda_section == "Class Distribution":
        st.markdown('<div class="section-label">Class Distribution</div>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, y, title, col in [
            (axes[0], y_train_d1, "Dataset 1 — Historical", ACC_BLUE),
            (axes[1], y_train_d2, "Dataset 2 — Current",   ACC_TEAL),
        ]:
            counts = y.value_counts().sort_index()
            bars = ax.bar(["No Condition", "Has Condition"],
                          counts.values,
                          color=[col, ACC_ORANGE],
                          edgecolor="none", width=0.55)
            ax.set_title(title, fontweight="600", pad=12)
            ax.set_ylabel("Patient Count")
            ax.bar_label(bars, fmt="%d", fontsize=10, color=TEXT_SEC, padding=4)
            ax.set_ylim(0, counts.max() * 1.18)
            ax.grid(axis='x', alpha=0)
        plt.tight_layout(pad=2)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.info("⚠️ **Severe class imbalance** is consistent across both D1 and D2 — confirming this is not a temporal artifact. Addressed via class_weight=balanced and SMOTE.")

    # ── Demographics ────────────────────────────────────────────────────────
    elif eda_section == "Demographics":
        st.markdown('<div class="section-label">Demographic Analysis</div>', unsafe_allow_html=True)
        demo_options = [c for c in ["age", "GENDER", "RACE", "MARITAL", "INCOME"]
                        if c in X_train_d1.columns]
        selected_demo = st.selectbox("Select demographic feature", demo_options)

        col1, col2 = st.columns(2)
        for col_ctx, X_tr, y_tr, title in [
            (col1, X_train_d1, y_train_d1, "Dataset 1 — Historical"),
            (col2, X_train_d2, y_train_d2, "Dataset 2 — Current"),
        ]:
            with col_ctx:
                st.markdown(f"<div style='font-size:0.8rem;font-weight:600;color:{TEXT_SEC};margin-bottom:0.5rem;'>{title}</div>", unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(5.5, 4))
                if selected_demo in ["age", "INCOME"]:
                    d0 = X_tr.loc[y_tr == 0, selected_demo].dropna()
                    d1 = X_tr.loc[y_tr == 1, selected_demo].dropna()
                    ax.hist(d0, bins=28, alpha=0.65, color=ACC_BLUE,   label="No Condition", density=True, edgecolor="none")
                    ax.hist(d1, bins=28, alpha=0.65, color=ACC_ORANGE, label="Has Condition", density=True, edgecolor="none")
                    ax.set_xlabel(f"{selected_demo} (scaled)")
                    ax.set_ylabel("Density")
                    ax.legend(fontsize=8)
                else:
                    tmp = X_tr[[selected_demo]].copy()
                    tmp["label"] = y_tr.values
                    gd = tmp.groupby([selected_demo, "label"]).size().unstack(fill_value=0)
                    gd.plot(kind="bar", ax=ax, color=[ACC_BLUE, ACC_ORANGE], edgecolor="none", width=0.65)
                    ax.set_xlabel(f"{selected_demo} (encoded)")
                    ax.set_ylabel("Count")
                    ax.legend(["No Condition", "Has Condition"], fontsize=8)
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
                ax.set_title(f"{selected_demo} by Label", fontweight="600")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        if "INCOME" in X_train_d1.columns:
            st.markdown("---")
            st.markdown('<div class="section-label">Income Distribution by Label</div>', unsafe_allow_html=True)
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            for ax, (X_t, y_t, title) in zip(axes, [
                (X_train_d1, y_train_d1, "D1 — Historical"),
                (X_train_d2, y_train_d2, "D2 — Current")
            ]):
                plot_df = X_t[["INCOME"]].copy()
                plot_df["label"] = y_t.values
                groups = [plot_df[plot_df["label"]==l]["INCOME"].dropna() for l in [0, 1]]
                bp = ax.boxplot(groups, labels=["No condition", "Has condition"],
                               patch_artist=True, widths=0.45,
                               medianprops=dict(color=ACC_ORANGE, linewidth=2),
                               boxprops=dict(facecolor=ACC_BLUE, alpha=0.55, linewidth=0),
                               whiskerprops=dict(color=TEXT_MUT),
                               capprops=dict(color=TEXT_MUT),
                               flierprops=dict(marker='o', markerfacecolor=ACC_BLUE,
                                               markersize=3, alpha=0.4, linestyle='none'))
                ax.set_title(f"Income by Label — {title}", fontweight="600")
                ax.set_ylabel("Scaled Income")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # ── Clinical Features ───────────────────────────────────────────────────
    elif eda_section == "Clinical Features":
        st.markdown('<div class="section-label">Clinical Feature Distributions</div>', unsafe_allow_html=True)
        clinical_cols = [c for c in feature_names if any(x in c for x in
            ["Body_Height", "Body_Weight", "BMI", "Diastolic", "Systolic",
             "Heart_rate", "Cholesterol"]) and "_mean" in c][:7]

        if clinical_cols:
            selected_clin = st.selectbox("Select clinical feature", clinical_cols)
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            for ax, (X_t, y_t, title) in zip(axes, [
                (X_train_d1, y_train_d1, "D1 — Historical"),
                (X_train_d2, y_train_d2, "D2 — Current")
            ]):
                if selected_clin in X_t.columns:
                    plot_df = X_t[[selected_clin]].copy()
                    plot_df["label"] = y_t.values
                    for label, color in [(0, ACC_BLUE), (1, ACC_ORANGE)]:
                        subset = plot_df[plot_df["label"]==label][selected_clin].dropna()
                        parts = ax.violinplot(subset, positions=[label], showmedians=True,
                                              showextrema=True)
                        for pc in parts.get('bodies', []):
                            pc.set_facecolor(color)
                            pc.set_alpha(0.6)
                            pc.set_edgecolor("none")
                        parts['cmedians'].set_color(ACC_ORANGE if label==1 else ACC_TEAL)
                        parts['cmedians'].set_linewidth(2)
                        for part in ['cbars','cmins','cmaxes']:
                            if part in parts:
                                parts[part].set_color(TEXT_MUT)
                    ax.set_title(f"{selected_clin[:30]} — {title}", fontweight="600")
                    ax.set_xticks([0, 1])
                    ax.set_xticklabels(["No condition", "Has condition"])
                    ax.set_ylabel("Scaled value")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("No clinical observation features found in this dataset.")

    # ── Healthcare Utilization ───────────────────────────────────────────────
    elif eda_section == "Healthcare Utilization":
        st.markdown('<div class="section-label">Healthcare Utilization by Label</div>', unsafe_allow_html=True)
        util_map = {
            "Encounters":  "total_encounters",
            "Medications": "total_medications",
            "Procedures":  "total_procedures",
            "Claims":      "total_claims",
        }
        available = {k: v for k, v in util_map.items() if v in X_train_d1.columns}
        col1, col2 = st.columns(2)
        items = list(available.items())
        for i, (name, col) in enumerate(items):
            ctx = col1 if i % 2 == 0 else col2
            with ctx:
                st.markdown(f"<div style='font-size:0.8rem;font-weight:600;color:{TEXT_SEC};margin-bottom:0.5rem;'>{name}</div>", unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(5.5, 4))
                d0 = X_train_d1.loc[y_train_d1 == 0, col].dropna()
                d1 = X_train_d1.loc[y_train_d1 == 1, col].dropna()
                bp = ax.boxplot([d0, d1], labels=["No Condition", "Has Condition"],
                               patch_artist=True, widths=0.45,
                               medianprops=dict(color=ACC_ORANGE, linewidth=2),
                               boxprops=dict(facecolor=ACC_BLUE, alpha=0.55, linewidth=0),
                               whiskerprops=dict(color=TEXT_MUT),
                               capprops=dict(color=TEXT_MUT),
                               flierprops=dict(marker='o', markerfacecolor=ACC_BLUE,
                                               markersize=3, alpha=0.4, linestyle='none'))
                ax.set_title(f"{col} by Label", fontweight="600")
                ax.set_ylabel(f"{col} (scaled)")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

    # ── Correlation Heatmap ─────────────────────────────────────────────────
    elif eda_section == "Correlation Heatmap":
        st.markdown('<div class="section-label">Correlation Heatmap — Top 30 Features vs Label (D1)</div>', unsafe_allow_html=True)
        tmp = X_train_d1.copy()
        tmp["label"] = y_train_d1.values
        corr_with_label = tmp.corr()["label"].drop("label").abs()
        top30 = corr_with_label.sort_values(ascending=False).head(30).index.tolist()
        corr_mat = tmp[top30 + ["label"]].corr()

        fig, ax = plt.subplots(figsize=(14, 12))
        import matplotlib.colors as mcolors
        cmap = plt.cm.RdBu_r
        im = ax.imshow(corr_mat, cmap=cmap, aspect="auto", vmin=-1, vmax=1)
        cbar = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
        cbar.ax.tick_params(colors=TEXT_SEC, labelsize=8)
        cbar.outline.set_edgecolor(BORDER_COL)
        labels = top30 + ["label"]
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7, color=TEXT_SEC)
        ax.set_yticklabels(labels, fontsize=7, color=TEXT_SEC)
        ax.set_title("Correlation Matrix — Top 30 Features + Label", fontsize=12, fontweight="600", pad=16)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER_COL)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.info("All correlations are weak (r < 0.1), expected given the severe class imbalance. Strongest correlates include Cholesterol, Hemoglobin, and Blood Pressure features.")

    # ── Data Drift ──────────────────────────────────────────────────────────
    elif eda_section == "Data Drift Analysis":
        st.markdown('<div class="section-label">Data Drift — D1 vs D2 Distribution Comparison</div>', unsafe_allow_html=True)
        top10 = X_train_d1.var().sort_values(ascending=False).head(10).index.tolist()
        fig, axes = plt.subplots(2, 5, figsize=(18, 7))
        axes = axes.flatten()
        for i, feat in enumerate(top10):
            ax = axes[i]
            d1_vals = X_train_d1[feat].dropna()
            d2_vals = X_train_d2[feat].dropna() if feat in X_train_d2.columns else pd.Series(dtype=float)
            ax.hist(d1_vals, bins=25, alpha=0.6, color=ACC_BLUE,   label="D1", density=True, edgecolor="none")
            if len(d2_vals) > 0:
                ax.hist(d2_vals, bins=25, alpha=0.6, color=ACC_ORANGE, label="D2", density=True, edgecolor="none")
            ax.set_title(feat[:22], fontsize=8, fontweight="600")
            ax.legend(fontsize=7)
        plt.suptitle("Distribution Shift: D1 vs D2 — Top 10 Features by Variance",
                     y=1.01, fontsize=11, color=TEXT_PRI, fontweight="600")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.warning("Drift analysis performed on StandardScaled data. Distributions appear similar post-scaling (mean≈0, std≈1). Drift exists in the raw feature space.")

    # ── Missing Values ──────────────────────────────────────────────────────
    elif eda_section == "Missing Values":
        st.markdown('<div class="section-label">Missing Value Analysis — Top 20 Sparse Columns</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(12, 6))
        cols = missing_series.index.tolist()
        vals = missing_series.values
        colors_bar = [ACC_ORANGE if v >= 0.5 else ACC_BLUE for v in vals[::-1]]
        bars = ax.barh(cols[::-1], vals[::-1], color=colors_bar, edgecolor="none", height=0.65)
        ax.axvline(0.5, color=ACC_ORANGE, linestyle="--", linewidth=1.5,
                   label="50% drop threshold", alpha=0.8)
        ax.set_xlabel("Missing Rate")
        ax.set_title("Top 20 Columns by Missing Rate (D1 pre-filter)", fontweight="600", pad=14)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, color=TEXT_SEC, padding=4)
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1.12)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.info("Columns exceeding the 50% missingness threshold (shown in orange) are dropped before training. These typically represent rare lab tests or sparse allergy panels.")


# ── PAGE 3: MODEL PERFORMANCE ─────────────────────────────────────────────────

elif page_clean == "Model Performance":
    st.markdown("""
    <div class="page-header">
        <div class="page-header-eyebrow">Evaluation</div>
        <div class="page-header-title">Model Performance</div>
        <div class="page-header-desc">
            Cross-model comparison of Decision Tree, SVM, and MLP across historical (D1)
            and current (D2) test sets, with ROC curves and confusion matrices.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Metric selector + bar chart
    st.markdown('<div class="section-label">Metric Comparison</div>', unsafe_allow_html=True)
    metric = st.selectbox("Select metric",
                          ["accuracy", "precision", "recall", "f1", "roc_auc"],
                          label_visibility="collapsed")

    pivot = baseline_df.pivot(index="model", columns="evaluated_on", values=metric)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(pivot))
    w = 0.36
    b1 = ax.bar(x - w/2, pivot["D1"], w, label="D1 — Historical",
                color=ACC_BLUE, edgecolor="none", alpha=0.9)
    b2 = ax.bar(x + w/2, pivot["D2"], w, label="D2 — Current",
                color=ACC_TEAL, edgecolor="none", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, fontsize=10)
    ax.set_title(f"{metric.upper()} — All Models on D1 and D2 Test Sets",
                 fontweight="600", pad=14)
    ax.set_ylabel(metric.upper())
    ax.set_ylim(0, 1.18)
    ax.legend()
    ax.bar_label(b1, fmt="%.3f", fontsize=8.5, color=TEXT_SEC, padding=4)
    ax.bar_label(b2, fmt="%.3f", fontsize=8.5, color=TEXT_SEC, padding=4)
    ax.grid(axis='x', alpha=0)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Full metrics table
    st.markdown("---")
    st.markdown('<div class="section-label">Full Metrics Table</div>', unsafe_allow_html=True)
    styled = baseline_df.style.background_gradient(subset=["f1", "roc_auc"], cmap="YlGn") \
                               .format(precision=4)
    st.dataframe(styled, use_container_width=True)

    # ROC + Confusion
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    models_map = {"DT": dt_best, "SVM": svm_best, "MLP": mlp}
    colours_m  = {"DT": ACC_BLUE, "SVM": ACC_ORANGE, "MLP": ACC_TEAL}

    with col1:
        st.markdown('<div class="section-label">ROC Curves</div>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, (X_test, y_test, title) in zip(
                axes,
                [(X_test_d1, y_test_d1, "D1 — Historical"),
                 (X_test_d2, y_test_d2, "D2 — Current")]):
            for name, model in models_map.items():
                probs = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, probs)
                ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.3f})",
                        color=colours_m[name], lw=2)
            ax.plot([0, 1], [0, 1], linestyle="--", color=TEXT_MUT, lw=1)
            ax.fill_between([0,1],[0,1], alpha=0.03, color=TEXT_MUT)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(title, fontweight="600")
            ax.legend(loc="lower right", fontsize=8)
        plt.suptitle("ROC Curves", fontsize=11, fontweight="600")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col2:
        st.markdown('<div class="section-label">Confusion Matrices</div>', unsafe_allow_html=True)
        fig, axes = plt.subplots(2, 3, figsize=(11, 7.5))
        for row, (X_test, y_test, ds) in enumerate([
                (X_test_d1, y_test_d1, "D1"), (X_test_d2, y_test_d2, "D2")]):
            for col_i, (name, model) in enumerate(models_map.items()):
                ax = axes[row][col_i]
                cm = confusion_matrix(y_test, model.predict(X_test))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                              display_labels=["Neg", "Pos"])
                disp.plot(ax=ax, colorbar=False,
                          cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
                              "custom", [CARD_BG, ACC_BLUE], N=256))
                ax.set_title(f"{name} · {ds}", fontsize=9, fontweight="600")
                ax.set_facecolor(CARD_BG)
                for text in ax.texts:
                    text.set_color(TEXT_PRI)
                    text.set_fontsize(10)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Analysis
    st.markdown("---")
    st.markdown('<div class="section-label">Analysis</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="analysis-block">
        <h4>Key Observations</h4>
        <ul>
            <li><strong>MLP</strong> achieves the best overall performance on D2 thanks to SMOTE oversampling</li>
            <li>All models are trained on D1 and evaluated on both D1 and D2 test sets</li>
            <li><strong>SVM</strong> may show variable ROC-AUC — sensitive to C tuning and scaling</li>
            <li><strong>Decision Tree</strong> is consistent but weak — expected with limited depth from grid search</li>
            <li>High accuracy is misleading due to class imbalance — F1 and ROC-AUC are the meaningful metrics</li>
        </ul>
    </div>
    <div class="analysis-block" style="border-left-color: var(--accent-purple); margin-top:0.75rem;">
        <h4 style="color: var(--accent-purple) !important;">Bias–Variance Trade-off</h4>
        <ul>
            <li>Decision Tree (max_depth 3–5): High bias, low variance — underfitting regime</li>
            <li>SVM RBF: Moderate bias-variance balance — sensitive to C tuning</li>
            <li>MLP (128→64→32): Low bias, higher variance — best generalization with SMOTE</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# ── PAGE 4: CONTINUAL LEARNING ────────────────────────────────────────────────

elif page_clean == "Continual Learning":
    st.markdown("""
    <div class="page-header">
        <div class="page-header-eyebrow">Adaptation</div>
        <div class="page-header-title">Continual Learning Analysis</div>
        <div class="page-header-desc">
            Fine-tuning the D1-trained MLP on new D2 data via <code style="color:var(--accent-teal);background:rgba(0,201,167,0.1);padding:0.1em 0.35em;border-radius:3px;">partial_fit()</code>
            over 50 epochs using mini-batches — and evaluating the effect on D2 test performance.
        </div>
    </div>
    """, unsafe_allow_html=True)

    mlp_d1_row = continual_df[continual_df.model == "MLP_D1"].iloc[0]
    mlp_cl_row = continual_df[continual_df.model == "MLP_CL"].iloc[0]

    st.markdown('<div class="section-label">Performance Shift on D2 Test Set</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)

    for col_ctx, label, before, after in [
        (col1, "F1 Score",  mlp_d1_row.f1,      mlp_cl_row.f1),
        (col2, "ROC-AUC",  mlp_d1_row.roc_auc,  mlp_cl_row.roc_auc),
        (col3, "Recall",   mlp_d1_row.recall,    mlp_cl_row.recall),
        (col4, "Precision",mlp_d1_row.precision, mlp_cl_row.precision),
    ]:
        delta = after - before
        col_ctx.metric(
            label=f"{label}",
            value=f"{after:.3f}",
            delta=f"{delta:+.3f}"
        )

    # Before / After bar chart
    st.markdown("---")
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    labels_display = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    before_vals = [mlp_d1_row[m] for m in metrics]
    after_vals  = [mlp_cl_row[m] for m in metrics]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(metrics))
    w = 0.36
    b1 = ax.bar(x - w/2, before_vals, w, label="MLP_D1 — Before (trained on D1)",
                color=ACC_BLUE, edgecolor="none", alpha=0.9)
    b2 = ax.bar(x + w/2, after_vals,  w, label="MLP_CL — After (continual learning)",
                color=ACC_ORANGE, edgecolor="none", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_display, fontsize=10)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Score")
    ax.set_title("Continual Learning — MLP Before vs After on D2 Test Set",
                 fontweight="600", pad=14)
    ax.legend(fontsize=9)
    ax.bar_label(b1, fmt="%.3f", fontsize=8.5, color=TEXT_SEC, padding=4)
    ax.bar_label(b2, fmt="%.3f", fontsize=8.5, color=TEXT_SEC, padding=4)
    ax.grid(axis='x', alpha=0)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Metrics table
    st.markdown("---")
    st.markdown('<div class="section-label">Metrics Table</div>', unsafe_allow_html=True)
    st.dataframe(continual_df.style.format(precision=4), use_container_width=True)

    # Analysis
    st.markdown("---")
    f1_before = mlp_d1_row.f1
    f1_after  = mlp_cl_row.f1
    direction = "dropped" if f1_after < f1_before else "improved"

    st.markdown(f"""
    <div class="analysis-block" style="border-left-color: var(--accent-orange);">
        <h4 style="color: var(--accent-orange) !important;">⚠️ Catastrophic Forgetting Detected</h4>
        <ul>
            <li>MLP_CL F1 <strong>{direction}</strong> from {f1_before:.3f} → {f1_after:.3f} on D2 — a classic continual learning failure mode</li>
            <li><code>partial_fit()</code> over 50 epochs aggressively overwrote D1-learned weights</li>
            <li>No regularization was applied to preserve D1 knowledge</li>
            <li>The learning rate was not decayed during fine-tuning</li>
        </ul>
    </div>
    <div class="analysis-block" style="border-left-color: var(--accent-teal); margin-top:0.75rem;">
        <h4 style="color: var(--accent-teal) !important;">Recommendation</h4>
        <ul>
            <li>The D1-trained MLP already generalizes to D2 — aggressive fine-tuning is counterproductive</li>
            <li>Elastic Weight Consolidation (EWC) would protect critical D1 weights during adaptation</li>
            <li>Learning without Forgetting (LwF) offers knowledge distillation as an alternative</li>
            <li>Progressive Neural Networks could add D2-specific capacity without overwriting D1</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# ── PAGE 5: FEATURE IMPORTANCE ────────────────────────────────────────────────

elif page_clean == "Feature Importance":
    st.markdown("""
    <div class="page-header">
        <div class="page-header-eyebrow">Interpretability</div>
        <div class="page-header-title">Feature Importance & Model Interpretation</div>
        <div class="page-header-desc">
            Decision Tree feature importances, feature category breakdown,
            and model-specific interpretation analysis.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Top 20 feature bar chart
    importances = dt_best.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    top20 = feat_imp.head(20)

    st.markdown('<div class="section-label">Top 20 Features — Decision Tree</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(11, 7))
    colors_grad = [matplotlib.colors.to_rgba(ACC_BLUE, alpha=0.5 + 0.5 * (1 - i/20))
                   for i in range(len(top20))]
    bars = ax.barh(top20.index[::-1], top20.values[::-1],
                   color=colors_grad[::-1], edgecolor="none", height=0.7)
    ax.set_xlabel("Feature Importance")
    ax.set_title("Top 20 Feature Importances — Decision Tree", fontweight="600", pad=14)
    ax.bar_label(bars, fmt="%.4f", fontsize=8, color=TEXT_SEC, padding=4)
    ax.set_xlim(0, top20.max() * 1.18)
    ax.grid(axis='y', alpha=0)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Feature category summary
    st.markdown("---")
    st.markdown('<div class="section-label">Feature Categories</div>', unsafe_allow_html=True)

    demo_feats      = [f for f in feature_names if any(x in f for x in
                       ["GENDER","RACE","ETHNICITY","INCOME","MARITAL","age","is_deceased","HEALTHCARE"])]
    encounter_feats = [f for f in feature_names if any(x in f for x in
                       ["encounter","claim_cost","payer_coverage"])]
    obs_feats       = [f for f in feature_names if f.startswith("obs_")]
    util_feats      = [f for f in feature_names if any(x in f for x in
                       ["medication","procedure","immunization","careplan",
                        "imaging","device","supply","transaction","payer","claims"])]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Demographic",   len(demo_feats))
    col2.metric("Encounter",     len(encounter_feats))
    col3.metric("Observation",   len(obs_feats))
    col4.metric("Utilization",   len(util_feats))

    # Feature search
    st.markdown("---")
    st.markdown('<div class="section-label">Explore All Features</div>', unsafe_allow_html=True)
    search = st.text_input("🔍  Search features by name", placeholder="e.g. cholesterol, BMI, age…")
    filtered = [f for f in feature_names if search.lower() in f.lower()] if search else feature_names
    st.markdown(f"<div style='font-size:0.78rem;color:{TEXT_MUT};margin-bottom:0.5rem;'>Showing {len(filtered)} of {len(feature_names)} features</div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({"feature_name": filtered}), use_container_width=True, height=280)

    # Interpretation
    st.markdown("---")
    st.markdown("""
    <div class="analysis-block">
        <h4>Key Findings</h4>
        <ul>
            <li>Observation-derived features (vitals and lab aggregates) dominate the top 20</li>
            <li>Clinical measurements like BMI, Blood Pressure, and Cholesterol are most predictive</li>
            <li>Demographic features contribute but are secondary to clinical indicators</li>
            <li>Utilization features (encounter counts, medication counts) provide additional signal</li>
        </ul>
    </div>
    <div class="analysis-block" style="border-left-color: var(--accent-teal); margin-top:0.75rem;">
        <h4 style="color: var(--accent-teal) !important;">Model-Specific Behavior</h4>
        <ul>
            <li><strong>Decision Tree</strong> — uses a small subset at each split; interpretable but limited</li>
            <li><strong>SVM</strong> — all features via kernel trick; less interpretable but more powerful</li>
            <li><strong>MLP</strong> — learns complex non-linear combinations; most powerful but black-box</li>
        </ul>
    </div>
    <div class="analysis-block" style="border-left-color: var(--accent-purple); margin-top:0.75rem;">
        <h4 style="color: var(--accent-purple) !important;">Feature Engineering Impact</h4>
        <ul>
            <li>Aggregating observations as mean + variance captures central tendency and variability</li>
            <li>Dropping columns with >50% missingness improved signal-to-noise ratio</li>
            <li>StandardScaling was critical for SVM and MLP convergence</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
