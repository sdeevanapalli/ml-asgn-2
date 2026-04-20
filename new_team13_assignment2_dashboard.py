# ── CONFIGURATION ──────────────────────────────────────────────────────────
DATA_DIR = "data/"
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

st.set_page_config(
    page_title="Clinical Prediction · Team 13",
    page_icon="🏥",
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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Instrument+Sans:ital,wght@0,400;0,500;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --ink:        #1a1f2e;
    --ink-soft:   #3d4a5c;
    --paper:      #f5f0e8;
    --paper-warm: #ede8dc;
    --paper-deep: #ddd6c8;
    --white:      #ffffff;
    --sb-bg:      #1b2132;
    --sb-border:  #262f45;
    --teal:  #2a9d8f;
    --red:   #e05a3a;
    --amber: #e9a84c;
    --indig: #5c6bc0;
    --muted: #8a96a8;
    --font-d: 'Syne', sans-serif;
    --font-b: 'Instrument Sans', sans-serif;
    --font-m: 'JetBrains Mono', monospace;
}

html, body, [class*="css"] {
    font-family: var(--font-b) !important;
    background: var(--paper) !important;
    color: var(--ink) !important;
}
.main .block-container {
    background: var(--paper) !important;
    padding: 2rem 2.5rem 4rem !important;
    max-width: 1380px !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--sb-bg) !important;
    border-right: 1px solid var(--sb-border) !important;
}
section[data-testid="stSidebar"] .block-container { padding: 0 !important; }

/* Hide default radio */
div[data-testid="stRadio"] { display: none !important; }

/* Sidebar buttons */
section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 7px !important;
    color: #7a8ba8 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    text-align: left !important;
    padding: 0.55rem 0.8rem !important;
    width: 100% !important;
    margin-bottom: 2px !important;
    box-shadow: none !important;
    transition: all 0.12s ease !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
    background: rgba(255,255,255,0.06) !important;
    color: #c8d5e8 !important;
    border-color: rgba(255,255,255,0.08) !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button:focus {
    box-shadow: none !important; outline: none !important;
}

/* Headings */
h1 { font-family: var(--font-d) !important; font-size: 2.3rem !important; font-weight: 800 !important; color: var(--ink) !important; letter-spacing: -0.025em !important; line-height: 1.1 !important; }
h2, h3 { font-family: var(--font-d) !important; color: var(--ink) !important; font-weight: 700 !important; }

/* Metrics */
[data-testid="metric-container"] {
    background: var(--white) !important;
    border: 1.5px solid var(--paper-deep) !important;
    border-radius: 10px !important;
    padding: 1rem 1.15rem !important;
    box-shadow: 2px 3px 0 var(--paper-deep) !important;
}
[data-testid="stMetricLabel"] { font-family: var(--font-m) !important; font-size: 0.63rem !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; color: var(--muted) !important; }
[data-testid="stMetricValue"] { font-family: var(--font-d) !important; font-size: 1.85rem !important; font-weight: 800 !important; color: var(--ink) !important; }

/* Selectbox */
div[data-baseweb="select"] > div { background: var(--white) !important; border: 1.5px solid var(--paper-deep) !important; border-radius: 8px !important; font-family: var(--font-b) !important; font-size: 0.9rem !important; color: var(--ink) !important; box-shadow: 1px 2px 0 var(--paper-deep) !important; }
div[data-baseweb="popover"] { background: var(--white) !important; border: 1.5px solid var(--paper-deep) !important; border-radius: 8px !important; }
li[role="option"] { background: var(--white) !important; color: var(--ink) !important; font-size: 0.875rem !important; }
li[role="option"]:hover { background: var(--paper) !important; }

/* Text input */
input[type="text"] { background: var(--white) !important; border: 1.5px solid var(--paper-deep) !important; border-radius: 8px !important; font-family: var(--font-m) !important; font-size: 0.875rem !important; color: var(--ink) !important; box-shadow: 1px 2px 0 var(--paper-deep) !important; }

/* Alerts */
[data-testid="stAlert"] { border-radius: 8px !important; border-left-width: 3px !important; font-size: 0.875rem !important; background: var(--white) !important; color: var(--ink-soft) !important; }

/* Dataframe */
[data-testid="stDataFrame"] { border: 1.5px solid var(--paper-deep) !important; border-radius: 10px !important; overflow: hidden !important; box-shadow: 2px 3px 0 var(--paper-deep) !important; }

/* HR */
hr { border: none !important; border-top: 1.5px dashed var(--paper-deep) !important; margin: 1.5rem 0 !important; opacity: 1 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--paper); }
::-webkit-scrollbar-thumb { background: var(--paper-deep); border-radius: 3px; }

/* Page header */
.ph { margin-bottom: 1.75rem; padding-bottom: 1.25rem; border-bottom: 1.5px dashed var(--paper-deep); }
.ph-eye { font-family: var(--font-m); font-size: 0.65rem; color: var(--teal); letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 0.3rem; }
.ph-title { font-family: var(--font-d); font-size: 2rem; font-weight: 800; color: var(--ink); line-height: 1.1; letter-spacing: -0.02em; }
.ph-desc { font-size: 0.9rem; color: var(--muted); margin-top: 0.5rem; line-height: 1.65; max-width: 620px; font-style: italic; }

/* Sec label */
.sl { font-family: var(--font-m); font-size: 0.62rem; font-weight: 500; letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted); margin-bottom: 0.45rem; }

/* Arch grid */
.ag { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 0.75rem; }
.ab { background: var(--white); border: 1.5px solid var(--paper-deep); border-radius: 10px; padding: 1.2rem 1.3rem; box-shadow: 2px 3px 0 var(--paper-deep); }
.ab-t { font-family: var(--font-m); font-size: 0.62rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); margin-bottom: 0.7rem; }
.ab ul { margin: 0; padding-left: 0; list-style: none; }
.ab li { font-size: 0.84rem; color: var(--ink-soft); padding: 0.18rem 0; display: flex; gap: 0.5rem; line-height: 1.5; }
.ab li::before { content: '→'; color: var(--teal); font-size: 0.72rem; flex-shrink: 0; margin-top: 0.07rem; }

/* Team grid */
.tg { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-top: 0.75rem; }
.tc { background: var(--white); border: 1.5px solid var(--paper-deep); border-radius: 10px; padding: 1.2rem; box-shadow: 2px 3px 0 var(--paper-deep); }
.tc-num { font-family: var(--font-m); font-size: 0.62rem; color: var(--muted); margin-bottom: 0.5rem; }
.tc-role { font-family: var(--font-m); font-size: 0.6rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--teal); margin-bottom: 0.2rem; }
.tc-name { font-family: var(--font-d); font-size: 0.95rem; font-weight: 700; color: var(--ink); margin-bottom: 0.4rem; }
.tc-task { display: inline-block; font-family: var(--font-m); font-size: 0.58rem; background: var(--paper); border: 1px solid var(--paper-deep); border-radius: 4px; padding: 0.15em 0.5em; color: var(--muted); margin-bottom: 0.5rem; }
.tc-desc { font-size: 0.79rem; color: var(--muted); line-height: 1.55; font-style: italic; }

/* Note blocks */
.nb { background: var(--white); border: 1.5px solid var(--paper-deep); border-left: 3px solid var(--teal); border-radius: 0 8px 8px 0; padding: 1.1rem 1.3rem; margin-bottom: 0.75rem; box-shadow: 2px 3px 0 var(--paper-deep); }
.nb h4 { font-family: var(--font-m); font-size: 0.62rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--teal); margin: 0 0 0.6rem; }
.nb ul { margin: 0; padding-left: 1rem; }
.nb li { font-size: 0.85rem; color: var(--ink-soft); padding: 0.15rem 0; line-height: 1.55; }
.nb.red  { border-left-color: var(--red);   } .nb.red  h4 { color: var(--red);   }
.nb.amb  { border-left-color: var(--amber); } .nb.amb  h4 { color: var(--amber); }
.nb.ind  { border-left-color: var(--indig); } .nb.ind  h4 { color: var(--indig); }

/* Sidebar layout */
.sb-brand { padding: 1.5rem 1.3rem 1.1rem; border-bottom: 1px solid #262f45; }
.sb-icon  { font-size: 1.5rem; display: block; margin-bottom: 0.45rem; }
.sb-title { font-family: 'Syne', sans-serif; font-size: 1.15rem; font-weight: 800; color: #eee8dc; line-height: 1.2; }
.sb-sub   { font-size: 0.67rem; color: #4a5a76; margin-top: 0.25rem; }
.sb-badge { display: inline-block; margin-top: 0.6rem; background: rgba(42,157,143,0.15); color: #2a9d8f; font-family: 'JetBrains Mono', monospace; font-size: 0.58rem; letter-spacing: 0.1em; text-transform: uppercase; padding: 0.25em 0.6em; border-radius: 4px; border: 1px solid rgba(42,157,143,0.28); }
.sb-nav-wrap { padding: 0.9rem 0.85rem 0.5rem; }
.sb-nl { font-family: 'JetBrains Mono', monospace; font-size: 0.55rem; letter-spacing: 0.18em; text-transform: uppercase; color: #353f55; padding: 0 0.4rem; margin-bottom: 0.45rem; }
.sb-cfg { padding: 0.9rem 1.3rem 1.3rem; border-top: 1px solid #262f45; }
.sb-cfg-t { font-family: 'JetBrains Mono', monospace; font-size: 0.55rem; letter-spacing: 0.15em; text-transform: uppercase; color: #353f55; margin-bottom: 0.55rem; }
.sb-cfg-r { font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: #5a6a88; display: flex; justify-content: space-between; padding: 0.12rem 0; }
.sb-cfg-r span { color: #8a9ab8; }
</style>
""", unsafe_allow_html=True)

# ── Matplotlib warm theme ────────────────────────────────────────────────────
WARM_BG = "#f9f5ef"; CARD_BG = "#ffffff"; INK = "#1a1f2e"; INK_SOFT = "#3d4a5c"
MUTED = "#9aa5b8"; BORDER = "#ddd6c8"
C_TEAL = "#2a9d8f"; C_RED = "#e05a3a"; C_AMBER = "#e9a84c"; C_INDIG = "#5c6bc0"

matplotlib.rcParams.update({
    'font.family':'sans-serif', 'axes.facecolor':CARD_BG, 'figure.facecolor':WARM_BG,
    'axes.edgecolor':BORDER, 'axes.labelcolor':INK_SOFT, 'xtick.color':MUTED, 'ytick.color':MUTED,
    'text.color':INK, 'axes.titlecolor':INK, 'axes.grid':True, 'grid.color':BORDER,
    'grid.linestyle':'--', 'grid.linewidth':0.6, 'grid.alpha':0.8,
    'legend.facecolor':CARD_BG, 'legend.edgecolor':BORDER, 'legend.labelcolor':INK_SOFT,
    'axes.titlesize':11, 'axes.labelsize':9, 'axes.titleweight':'bold',
    'axes.spines.top':False, 'axes.spines.right':False,
})

# ── PIPELINE ─────────────────────────────────────────────────────────────────
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

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <span class="sb-icon">🏥</span>
        <div class="sb-title">Clinical<br>Prediction</div>
        <div class="sb-sub">BITS F464 · Machine Learning</div>
        <span class="sb-badge">Team 13 · Sem 2 2025–26</span>
    </div>
    <div class="sb-nav-wrap">
        <div class="sb-nl">Pages</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="padding: 0 0.85rem;">', unsafe_allow_html=True)
    for full_name, num in NAV_ITEMS:
        if st.button(f"{num}  {full_name}", key=f"nav_{num}", use_container_width=True):
            st.session_state.page = full_name
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sb-cfg">
        <div class="sb-cfg-t">Config</div>
        <div class="sb-cfg-r">cutoff <span>{TEMPORAL_CUTOFF}</span></div>
        <div class="sb-cfg-r">test split <span>{int(TEST_SIZE*100)}%</span></div>
        <div class="sb-cfg-r">random seed <span>{RANDOM_STATE}</span></div>
    </div>
    """, unsafe_allow_html=True)

# ── LOAD ─────────────────────────────────────────────────────────────────────
with st.spinner("Running the pipeline — grab a coffee, first load takes a minute ☕"):
    data = run_pipeline()

X_train_d1=data["X_train_d1"]; y_train_d1=data["y_train_d1"]; y_test_d1=data["y_test_d1"]
X_train_d2=data["X_train_d2"]; y_train_d2=data["y_train_d2"]; y_test_d2=data["y_test_d2"]
X_test_d1=data["X_test_d1"];   X_test_d2=data["X_test_d2"]
feature_names=data["feature_names"]; baseline_df=data["baseline_df"]
continual_df=data["continual_df"];   missing_series=data["missing_series"]
dt_best=data["dt_best"]; svm_best=data["svm_best"]; mlp=data["mlp"]; mlp_cl=data["mlp_cl"]
page = st.session_state.page

# ── PAGE 1 ────────────────────────────────────────────────────────────────────
if page == "Project Overview":
    st.markdown("""
    <div class="ph">
        <div class="ph-eye">BITS F464 · Machine Learning · Team 13</div>
        <div class="ph-title">Clinical Prediction<br>under Temporal Shift</div>
        <div class="ph-desc">An end-to-end ML pipeline on synthetic EHR data — predicting clinically
        significant conditions across historical and current patient cohorts.</div>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Patients",       f"{data['total_patients']:,}")
    c2.metric("Feature Dimensions",   len(feature_names))
    c3.metric("D1 — Historical",      f"{data['d1_size']:,}")
    c4.metric("D2 — Current",         f"{data['d2_size']:,}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sl">Pipeline Architecture</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="ag">
        <div class="ab"><div class="ab-t">Data Pipeline · Task 2</div><ul>
            <li>15 CSV files merged on patient ID</li>
            <li>Temporal split: pre / post 2020-01-01</li>
            <li>Sparse columns dropped (&gt;50% missing)</li>
            <li>Binary target: disorder / finding labels</li>
            <li>StandardScaler fit on training data only</li>
            <li>80/20 stratified train-test split</li>
        </ul></div>
        <div class="ab"><div class="ab-t">Models · Task 3</div><ul>
            <li>Decision Tree — GridSearchCV, balanced weights</li>
            <li>SVM RBF kernel — GridSearchCV, balanced weights</li>
            <li>MLP 128→64→32 — SMOTE oversampling</li>
        </ul></div>
        <div class="ab"><div class="ab-t">Class Imbalance Strategy</div><ul>
            <li>Majority label=0 (no clinical condition)</li>
            <li>Minority label=1 (disorder / finding)</li>
            <li>DT + SVM: class_weight="balanced"</li>
            <li>MLP: SMOTE synthetic oversampling</li>
        </ul></div>
        <div class="ab"><div class="ab-t">Temporal Dataset Split</div><ul>
            <li>D1: first encounter before 2020</li>
            <li>D2: any encounter from 2020 onward</li>
            <li>Overlap intentional — tests generalization</li>
            <li>Models trained on D1, evaluated on both</li>
        </ul></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sl">The Team</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="tg">
        <div class="tc"><div class="tc-num">01 / 04</div><div class="tc-role">Data Architect</div>
            <div class="tc-name">Shriniketh Deevanapalli</div>
            <span class="tc-task">Task 2 (a, b, c)</span>
            <div class="tc-desc">Merged 15 CSV tables, implemented the temporal split, and engineered the feature dataset with StandardScaler.</div></div>
        <div class="tc"><div class="tc-num">02 / 04</div><div class="tc-role">ML Engineer</div>
            <div class="tc-name">Sanvi Udhan</div>
            <span class="tc-task">Task 3 (a,b,c) + Task 4</span>
            <div class="tc-desc">Trained DT, SVM, and MLP. Implemented continual learning via partial_fit and compiled all performance metrics.</div></div>
        <div class="tc"><div class="tc-num">03 / 04</div><div class="tc-role">Full-Stack Dev</div>
            <div class="tc-name">Sai Dheeraj Yadavalli</div>
            <span class="tc-task">Task 1 + Task 5</span>
            <div class="tc-desc">Built this Streamlit dashboard and integrated outputs from all team members into interactive visualizations.</div></div>
        <div class="tc"><div class="tc-num">04 / 04</div><div class="tc-role">Data Analyst</div>
            <div class="tc-name">Shambhavi Rani</div>
            <span class="tc-task">Task 2(d) + Task 3(d,e,f) + Task 5</span>
            <div class="tc-desc">EDA, bias-variance analysis, feature importance write-up, and the final video presentation.</div></div>
    </div>""", unsafe_allow_html=True)

# ── PAGE 2 ────────────────────────────────────────────────────────────────────
elif page == "Exploratory Data Analysis":
    st.markdown("""
    <div class="ph">
        <div class="ph-eye">Task 2 · Data Exploration</div>
        <div class="ph-title">Exploratory Data Analysis</div>
        <div class="ph-desc">Distributions, demographics, and data quality checks across both historical (D1) and current (D2) cohorts.</div>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("D1 Train Size",    f"{len(y_train_d1):,}")
    c2.metric("D1 Positive Rate", f"{y_train_d1.mean()*100:.1f}%")
    c3.metric("D2 Train Size",    f"{len(y_train_d2):,}")
    c4.metric("D2 Positive Rate", f"{y_train_d2.mean()*100:.1f}%")

    st.markdown("---")
    st.markdown('<div class="sl">Choose a section</div>', unsafe_allow_html=True)
    eda_section = st.selectbox("EDA", ["Class Distribution","Demographics","Clinical Features",
        "Healthcare Utilization","Correlation Heatmap","Data Drift Analysis","Missing Values"],
        label_visibility="collapsed")
    st.markdown("<br>", unsafe_allow_html=True)

    if eda_section == "Class Distribution":
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, y, title, col in [(axes[0],y_train_d1,"Dataset 1 — Historical",C_TEAL),(axes[1],y_train_d2,"Dataset 2 — Current",C_INDIG)]:
            counts = y.value_counts().sort_index()
            bars = ax.bar(["No Condition","Has Condition"],counts.values,color=[col,C_RED],edgecolor="none",width=0.5)
            ax.set_title(title,pad=12); ax.set_ylabel("Patient Count")
            ax.bar_label(bars,fmt="%d",fontsize=10,color=INK_SOFT,padding=4); ax.set_ylim(0,counts.max()*1.18)
        plt.tight_layout(pad=2); st.pyplot(fig,use_container_width=True); plt.close(fig)
        st.info("Severe class imbalance is consistent across both datasets — not a temporal artifact. Handled via class_weight=balanced and SMOTE.")

    elif eda_section == "Demographics":
        demo_options = [c for c in ["age","GENDER","RACE","MARITAL","INCOME"] if c in X_train_d1.columns]
        selected_demo = st.selectbox("Feature", demo_options)
        c1,c2 = st.columns(2)
        for ctx,X_tr,y_tr,title in [(c1,X_train_d1,y_train_d1,"D1 — Historical"),(c2,X_train_d2,y_train_d2,"D2 — Current")]:
            with ctx:
                st.markdown(f"<div style='font-size:0.78rem;font-weight:600;color:{INK_SOFT};margin-bottom:0.4rem;'>{title}</div>", unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(5.5,4))
                if selected_demo in ["age","INCOME"]:
                    ax.hist(X_tr.loc[y_tr==0,selected_demo].dropna(),bins=28,alpha=0.65,color=C_TEAL,label="No Condition",density=True,edgecolor="none")
                    ax.hist(X_tr.loc[y_tr==1,selected_demo].dropna(),bins=28,alpha=0.65,color=C_RED, label="Has Condition",density=True,edgecolor="none")
                    ax.set_xlabel(f"{selected_demo} (scaled)"); ax.set_ylabel("Density"); ax.legend(fontsize=8)
                else:
                    tmp=X_tr[[selected_demo]].copy(); tmp["label"]=y_tr.values
                    tmp.groupby([selected_demo,"label"]).size().unstack(fill_value=0).plot(kind="bar",ax=ax,color=[C_TEAL,C_RED],edgecolor="none",width=0.6)
                    ax.set_xlabel(f"{selected_demo} (encoded)"); ax.set_ylabel("Count")
                    ax.legend(["No Condition","Has Condition"],fontsize=8); ax.set_xticklabels(ax.get_xticklabels(),rotation=0)
                ax.set_title(f"{selected_demo} by Label",pad=10)
                plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)
        if "INCOME" in X_train_d1.columns:
            st.markdown("---")
            fig,axes = plt.subplots(1,2,figsize=(14,5))
            for ax,(X_t,y_t,title) in zip(axes,[(X_train_d1,y_train_d1,"D1"),(X_train_d2,y_train_d2,"D2")]):
                plot_df=X_t[["INCOME"]].copy(); plot_df["label"]=y_t.values
                ax.boxplot([plot_df[plot_df["label"]==l]["INCOME"].dropna() for l in [0,1]],labels=["No condition","Has condition"],
                           patch_artist=True,widths=0.45,medianprops=dict(color=C_RED,linewidth=2),
                           boxprops=dict(facecolor=C_TEAL,alpha=0.4,linewidth=0),whiskerprops=dict(color=MUTED),capprops=dict(color=MUTED),
                           flierprops=dict(marker='o',markerfacecolor=C_TEAL,markersize=3,alpha=0.3,linestyle='none'))
                ax.set_title(f"Income by Label — {title}",pad=10); ax.set_ylabel("Scaled Income")
            plt.tight_layout(); st.pyplot(fig); plt.close()

    elif eda_section == "Clinical Features":
        clinical_cols=[c for c in feature_names if any(x in c for x in ["Body_Height","Body_Weight","BMI","Diastolic","Systolic","Heart_rate","Cholesterol"]) and "_mean" in c][:7]
        if clinical_cols:
            selected_clin = st.selectbox("Feature", clinical_cols)
            fig,axes = plt.subplots(1,2,figsize=(14,5))
            for ax,(X_t,y_t,title) in zip(axes,[(X_train_d1,y_train_d1,"D1"),(X_train_d2,y_train_d2,"D2")]):
                if selected_clin in X_t.columns:
                    plot_df=X_t[[selected_clin]].copy(); plot_df["label"]=y_t.values
                    for label,color in [(0,C_TEAL),(1,C_RED)]:
                        parts=ax.violinplot(plot_df[plot_df["label"]==label][selected_clin].dropna(),positions=[label],showmedians=True,showextrema=True)
                        for pc in parts.get('bodies',[]): pc.set_facecolor(color); pc.set_alpha(0.5); pc.set_edgecolor("none")
                        parts['cmedians'].set_color(C_AMBER); parts['cmedians'].set_linewidth(2)
                        for p in ['cbars','cmins','cmaxes']:
                            if p in parts: parts[p].set_color(MUTED)
                    ax.set_title(f"{selected_clin[:30]} — {title}",pad=10)
                    ax.set_xticks([0,1]); ax.set_xticklabels(["No condition","Has condition"]); ax.set_ylabel("Scaled value")
            plt.tight_layout(); st.pyplot(fig); plt.close()
        else:
            st.info("No clinical observation features found.")

    elif eda_section == "Healthcare Utilization":
        util_map={"Encounters":"total_encounters","Medications":"total_medications","Procedures":"total_procedures","Claims":"total_claims"}
        available={k:v for k,v in util_map.items() if v in X_train_d1.columns}
        c1,c2=st.columns(2)
        for i,(name,col) in enumerate(available.items()):
            with (c1 if i%2==0 else c2):
                st.markdown(f"<div style='font-size:0.78rem;font-weight:600;color:{INK_SOFT};margin-bottom:0.4rem;'>{name}</div>", unsafe_allow_html=True)
                fig,ax=plt.subplots(figsize=(5.5,4))
                ax.boxplot([X_train_d1.loc[y_train_d1==l,col].dropna() for l in [0,1]],labels=["No Condition","Has Condition"],
                           patch_artist=True,widths=0.45,medianprops=dict(color=C_RED,linewidth=2),
                           boxprops=dict(facecolor=C_TEAL,alpha=0.4,linewidth=0),whiskerprops=dict(color=MUTED),capprops=dict(color=MUTED),
                           flierprops=dict(marker='o',markerfacecolor=C_TEAL,markersize=3,alpha=0.3,linestyle='none'))
                ax.set_title(col,pad=10); ax.set_ylabel(f"{col} (scaled)")
                plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)

    elif eda_section == "Correlation Heatmap":
        tmp=X_train_d1.copy(); tmp["label"]=y_train_d1.values
        top30=tmp.corr()["label"].drop("label").abs().sort_values(ascending=False).head(30).index.tolist()
        corr_mat=tmp[top30+["label"]].corr()
        fig,ax=plt.subplots(figsize=(14,12))
        im=ax.imshow(corr_mat,cmap="RdBu_r",aspect="auto",vmin=-1,vmax=1)
        cb=plt.colorbar(im,ax=ax,shrink=0.75); cb.ax.tick_params(labelsize=8)
        labels=top30+["label"]
        ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels,rotation=90,fontsize=7); ax.set_yticklabels(labels,fontsize=7)
        ax.set_title("Correlation Matrix — Top 30 Features + Label (D1)",pad=14)
        plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)
        st.info("All correlations are weak (r < 0.1), expected with severe class imbalance. Strongest correlates are Cholesterol, Hemoglobin, and Blood Pressure features.")

    elif eda_section == "Data Drift Analysis":
        top10=X_train_d1.var().sort_values(ascending=False).head(10).index.tolist()
        fig,axes=plt.subplots(2,5,figsize=(18,7)); axes=axes.flatten()
        for i,feat in enumerate(top10):
            ax=axes[i]
            ax.hist(X_train_d1[feat].dropna(),bins=25,alpha=0.6,color=C_TEAL,label="D1",density=True,edgecolor="none")
            d2v=X_train_d2[feat].dropna() if feat in X_train_d2.columns else pd.Series(dtype=float)
            if len(d2v): ax.hist(d2v,bins=25,alpha=0.6,color=C_RED,label="D2",density=True,edgecolor="none")
            ax.set_title(feat[:22],fontsize=8); ax.legend(fontsize=7)
        plt.suptitle("Distribution Shift: D1 vs D2 — Top 10 Features by Variance",y=1.01,fontsize=11)
        plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)
        st.warning("Drift analysis on StandardScaled data. Distributions appear similar post-scaling (mean≈0, std≈1). Raw-space drift still exists.")

    elif eda_section == "Missing Values":
        fig,ax=plt.subplots(figsize=(12,6))
        vals=missing_series.values; cols=missing_series.index.tolist()
        colors_bar=[C_RED if v>=0.5 else C_TEAL for v in vals[::-1]]
        bars=ax.barh(cols[::-1],vals[::-1],color=colors_bar,edgecolor="none",height=0.6)
        ax.axvline(0.5,color=C_AMBER,linestyle="--",linewidth=1.5,label="50% drop threshold")
        ax.set_xlabel("Missing Rate"); ax.set_title("Top 20 Columns by Missing Rate (D1 pre-filter)",pad=14)
        ax.bar_label(bars,fmt="%.2f",fontsize=8,color=INK_SOFT,padding=4); ax.legend(); ax.set_xlim(0,1.12)
        plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)
        st.info("Columns above 50% missingness (red) are dropped before training — typically rare lab tests or sparse allergy panels.")

# ── PAGE 3 ────────────────────────────────────────────────────────────────────
elif page == "Model Performance":
    st.markdown("""
    <div class="ph">
        <div class="ph-eye">Task 3 · Evaluation</div>
        <div class="ph-title">Model Performance</div>
        <div class="ph-desc">Decision Tree, SVM, and MLP compared across historical (D1) and current (D2) test sets.</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sl">Pick a metric</div>', unsafe_allow_html=True)
    metric = st.selectbox("Metric",["accuracy","precision","recall","f1","roc_auc"],label_visibility="collapsed")
    pivot = baseline_df.pivot(index="model",columns="evaluated_on",values=metric)
    fig,ax=plt.subplots(figsize=(9,4.5))
    x=np.arange(len(pivot)); w=0.36
    b1=ax.bar(x-w/2,pivot["D1"],w,label="D1 — Historical",color=C_TEAL, edgecolor="none",alpha=0.9)
    b2=ax.bar(x+w/2,pivot["D2"],w,label="D2 — Current",   color=C_INDIG,edgecolor="none",alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(pivot.index,fontsize=10)
    ax.set_title(f"{metric.upper()} — All Models on D1 and D2 Test Sets",pad=14)
    ax.set_ylabel(metric.upper()); ax.set_ylim(0,1.2); ax.legend()
    ax.bar_label(b1,fmt="%.3f",fontsize=8.5,color=INK_SOFT,padding=4)
    ax.bar_label(b2,fmt="%.3f",fontsize=8.5,color=INK_SOFT,padding=4)
    plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)

    st.markdown("---")
    st.markdown('<div class="sl">Full metrics table</div>', unsafe_allow_html=True)
    st.dataframe(baseline_df.style.background_gradient(subset=["f1","roc_auc"],cmap="YlGn").format(precision=4),use_container_width=True)

    st.markdown("---")
    models_map={"DT":dt_best,"SVM":svm_best,"MLP":mlp}
    col_map={"DT":C_TEAL,"SVM":C_RED,"MLP":C_INDIG}
    c1,c2=st.columns(2)
    with c1:
        st.markdown('<div class="sl">ROC Curves</div>', unsafe_allow_html=True)
        fig,axes=plt.subplots(1,2,figsize=(11,4.5))
        for ax,(X_test,y_test,title) in zip(axes,[(X_test_d1,y_test_d1,"D1 — Historical"),(X_test_d2,y_test_d2,"D2 — Current")]):
            for name,model in models_map.items():
                probs=model.predict_proba(X_test)[:,1]; fpr,tpr,_=roc_curve(y_test,probs)
                ax.plot(fpr,tpr,label=f"{name} ({auc(fpr,tpr):.3f})",color=col_map[name],lw=2)
            ax.plot([0,1],[0,1],linestyle="--",color=MUTED,lw=1)
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title(title,pad=10); ax.legend(fontsize=8)
        plt.suptitle("ROC Curves",fontsize=11,fontweight="bold")
        plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)
    with c2:
        st.markdown('<div class="sl">Confusion Matrices</div>', unsafe_allow_html=True)
        fig,axes=plt.subplots(2,3,figsize=(11,7.5))
        for row,(X_test,y_test,ds) in enumerate([(X_test_d1,y_test_d1,"D1"),(X_test_d2,y_test_d2,"D2")]):
            for ci,(name,model) in enumerate(models_map.items()):
                ax=axes[row][ci]
                disp=ConfusionMatrixDisplay(confusion_matrix(y_test,model.predict(X_test)),display_labels=["Neg","Pos"])
                disp.plot(ax=ax,colorbar=False,cmap=matplotlib.colors.LinearSegmentedColormap.from_list("w2t",[CARD_BG,C_TEAL],N=256))
                ax.set_title(f"{name} · {ds}",fontsize=9)
                for txt in ax.texts: txt.set_color(INK); txt.set_fontsize(10)
        plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)

    st.markdown("---")
    st.markdown("""
    <div class="nb">
        <h4>Key Observations</h4>
        <ul>
            <li><strong>MLP</strong> achieves the best F1 on D2, thanks to SMOTE balancing</li>
            <li>All models trained on D1, evaluated on both D1 and D2 test sets</li>
            <li><strong>SVM</strong> ROC-AUC varies with C — sensitive to the scaling step</li>
            <li><strong>Decision Tree</strong> is consistent but weak at depth 3–5</li>
            <li>Accuracy alone is misleading — F1 and ROC-AUC are what matter here</li>
        </ul>
    </div>
    <div class="nb ind">
        <h4>Bias–Variance Trade-off</h4>
        <ul>
            <li>Decision Tree (depth 3–5): high bias, low variance — underfitting</li>
            <li>SVM RBF: moderate balance — sensitive to C hyperparameter</li>
            <li>MLP (128→64→32): low bias, higher variance — best generalization with SMOTE</li>
        </ul>
    </div>""", unsafe_allow_html=True)

# ── PAGE 4 ────────────────────────────────────────────────────────────────────
elif page == "Continual Learning":
    st.markdown("""
    <div class="ph">
        <div class="ph-eye">Task 4 · Adaptation</div>
        <div class="ph-title">Continual Learning</div>
        <div class="ph-desc">Fine-tuning the D1-trained MLP on new D2 data via partial_fit() over 50 epochs — and what went wrong.</div>
    </div>""", unsafe_allow_html=True)

    r0=continual_df[continual_df.model=="MLP_D1"].iloc[0]
    r1=continual_df[continual_df.model=="MLP_CL"].iloc[0]
    c1,c2,c3,c4=st.columns(4)
    for ctx,lbl,bv,av in [(c1,"F1 Score",r0.f1,r1.f1),(c2,"ROC-AUC",r0.roc_auc,r1.roc_auc),(c3,"Recall",r0.recall,r1.recall),(c4,"Precision",r0.precision,r1.precision)]:
        ctx.metric(lbl,f"{av:.3f}",delta=f"{av-bv:+.3f}")

    st.markdown("---")
    metrics=["accuracy","precision","recall","f1","roc_auc"]
    fig,ax=plt.subplots(figsize=(11,4.5))
    x=np.arange(len(metrics)); w=0.36
    b1=ax.bar(x-w/2,[r0[m] for m in metrics],w,label="MLP_D1 — Before",color=C_TEAL, edgecolor="none",alpha=0.9)
    b2=ax.bar(x+w/2,[r1[m] for m in metrics],w,label="MLP_CL — After", color=C_AMBER,edgecolor="none",alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(["Accuracy","Precision","Recall","F1","ROC-AUC"],fontsize=10)
    ax.set_ylim(0,1.2); ax.set_ylabel("Score"); ax.legend()
    ax.set_title("Continual Learning — MLP Before vs After on D2 Test Set",pad=14)
    ax.bar_label(b1,fmt="%.3f",fontsize=8.5,color=INK_SOFT,padding=4)
    ax.bar_label(b2,fmt="%.3f",fontsize=8.5,color=INK_SOFT,padding=4)
    plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)

    st.markdown("---")
    st.markdown('<div class="sl">Metrics table</div>', unsafe_allow_html=True)
    st.dataframe(continual_df.style.format(precision=4),use_container_width=True)

    f1_dir="dropped" if r1.f1 < r0.f1 else "improved"
    st.markdown("---")
    st.markdown(f"""
    <div class="nb red">
        <h4>Catastrophic Forgetting — What Happened</h4>
        <ul>
            <li>MLP_CL F1 <strong>{f1_dir}</strong> from {r0.f1:.3f} → {r1.f1:.3f} on D2</li>
            <li>partial_fit() over 50 epochs aggressively overwrote D1 weights</li>
            <li>No regularization to preserve historical knowledge</li>
            <li>Learning rate was not decayed during fine-tuning</li>
        </ul>
    </div>
    <div class="nb amb">
        <h4>What Would Actually Work</h4>
        <ul>
            <li>Elastic Weight Consolidation (EWC) — penalizes updates to important D1 weights</li>
            <li>Learning without Forgetting (LwF) — knowledge distillation approach</li>
            <li>Progressive Neural Networks — adds D2 capacity without touching D1 weights</li>
            <li>The D1 MLP already generalizes to D2 — aggressive fine-tuning is counterproductive</li>
        </ul>
    </div>""", unsafe_allow_html=True)

# ── PAGE 5 ────────────────────────────────────────────────────────────────────
elif page == "Feature Importance":
    st.markdown("""
    <div class="ph">
        <div class="ph-eye">Task 3 · Interpretability</div>
        <div class="ph-title">Feature Importance</div>
        <div class="ph-desc">What the Decision Tree found most useful, broken down by category and searchable by name.</div>
    </div>""", unsafe_allow_html=True)

    feat_imp=pd.Series(dt_best.feature_importances_,index=feature_names).sort_values(ascending=False)
    top20=feat_imp.head(20)
    st.markdown('<div class="sl">Top 20 — Decision Tree</div>', unsafe_allow_html=True)
    fig,ax=plt.subplots(figsize=(11,7))
    alphas=[0.5+0.5*(1-i/20) for i in range(len(top20))]
    colors_bar=[matplotlib.colors.to_rgba(C_TEAL,a) for a in alphas]
    bars=ax.barh(top20.index[::-1],top20.values[::-1],color=colors_bar[::-1],edgecolor="none",height=0.65)
    ax.set_xlabel("Feature Importance"); ax.set_title("Top 20 Feature Importances — Decision Tree",pad=14)
    ax.bar_label(bars,fmt="%.4f",fontsize=8,color=INK_SOFT,padding=4); ax.set_xlim(0,top20.max()*1.2)
    plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)

    st.markdown("---")
    st.markdown('<div class="sl">Feature category breakdown</div>', unsafe_allow_html=True)
    demo_f =[f for f in feature_names if any(x in f for x in ["GENDER","RACE","ETHNICITY","INCOME","MARITAL","age","is_deceased","HEALTHCARE"])]
    enc_f  =[f for f in feature_names if any(x in f for x in ["encounter","claim_cost","payer_coverage"])]
    obs_f  =[f for f in feature_names if f.startswith("obs_")]
    util_f =[f for f in feature_names if any(x in f for x in ["medication","procedure","immunization","careplan","imaging","device","supply","transaction","payer","claims"])]
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Demographic",len(demo_f)); c2.metric("Encounter",len(enc_f))
    c3.metric("Observation",len(obs_f)); c4.metric("Utilization",len(util_f))

    st.markdown("---")
    st.markdown('<div class="sl">Search features</div>', unsafe_allow_html=True)
    search=st.text_input("",placeholder="e.g. cholesterol, BMI, age…",label_visibility="collapsed")
    filtered=[f for f in feature_names if search.lower() in f.lower()] if search else feature_names
    st.markdown(f"<div style='font-size:0.75rem;color:{MUTED};margin-bottom:0.4rem;'>{len(filtered)} of {len(feature_names)} features</div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({"feature_name":filtered}),use_container_width=True,height=260)

    st.markdown("---")
    st.markdown("""
    <div class="nb">
        <h4>Key Findings</h4>
        <ul>
            <li>Observation-derived features (vitals + lab aggregates) dominate the top 20</li>
            <li>BMI, Blood Pressure, and Cholesterol are the most predictive clinical signals</li>
            <li>Demographics contribute but rank below clinical measurements</li>
            <li>Utilization counts (encounters, medications) add secondary signal</li>
        </ul>
    </div>
    <div class="nb ind">
        <h4>Feature Engineering Choices That Helped</h4>
        <ul>
            <li>Mean + variance aggregation captures both central tendency and patient variability</li>
            <li>Dropping >50% missing columns improved signal-to-noise noticeably</li>
            <li>StandardScaling was essential for SVM and MLP convergence</li>
        </ul>
    </div>""", unsafe_allow_html=True)
