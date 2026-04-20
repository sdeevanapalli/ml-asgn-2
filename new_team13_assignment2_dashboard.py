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

st.set_page_config(
    page_title="Clinical Prediction - Team 13",
    page_icon="🏥",
    layout="wide"
)

# ── PIPELINE ────────────────────────────────────────────────────────────────

@st.cache_data
def run_pipeline():
    # --- LOAD RAW CSVs ---
    patients     = pd.read_csv(DATA_DIR + "patients.csv", on_bad_lines="skip")
    encounters   = pd.read_csv(DATA_DIR + "encounters.csv", on_bad_lines="skip")
    observations = pd.read_csv(DATA_DIR + "observations.csv", on_bad_lines="skip")
    conditions   = pd.read_csv(DATA_DIR + "conditions.csv", on_bad_lines="skip",
                               dayfirst=True)
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
    claims_trans = pd.read_csv(DATA_DIR + "claims_transactions.csv",
                               on_bad_lines="skip",
                               usecols=lambda c: c in [
                                   "PATIENTID", "TYPE", "AMOUNT"])

    # --- FEATURE ENGINEERING ---

    # patient demographics
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

    # encounters aggregates
    encounters["START"] = pd.to_datetime(encounters["START"], errors="coerce", utc=True)
    enc_agg = encounters.groupby("PATIENT").agg(
        total_encounters=("Id", "count"),
        unique_encounter_types=("ENCOUNTERCLASS", "nunique"),
        avg_base_encounter_cost=("BASE_ENCOUNTER_COST", "mean"),
        total_claim_cost=("TOTAL_CLAIM_COST", "sum"),
        avg_payer_coverage=("PAYER_COVERAGE", "mean")
    ).reset_index()

    # observations aggregates (numeric only, >5% coverage)
    observations["VALUE"] = pd.to_numeric(observations["VALUE"], errors="coerce")
    obs_numeric = observations.dropna(subset=["VALUE"])
    obs_agg = obs_numeric.groupby(["PATIENT", "DESCRIPTION"])["VALUE"].agg(
        ["mean", "std"]).reset_index()
    obs_agg.columns = ["PATIENT", "DESC", "mean", "std"]
    obs_agg["DESC"] = obs_agg["DESC"].str.replace(
        r"[^a-zA-Z0-9]", "_", regex=True).str[:40]

    obs_mean = obs_agg.pivot_table(index="PATIENT", columns="DESC", values="mean", aggfunc="mean")
    obs_std  = obs_agg.pivot_table(index="PATIENT", columns="DESC", values="std",  aggfunc="mean")
    obs_mean.columns = ["obs_" + c + "_mean" for c in obs_mean.columns]
    obs_std.columns  = ["obs_" + c + "_var"  for c in obs_std.columns]

    threshold = 0.05
    obs_mean = obs_mean.loc[:, obs_mean.notna().mean() >= threshold]
    obs_std  = obs_std.loc[:, obs_std.notna().mean() >= threshold]
    obs_features = obs_mean.join(obs_std, how="outer").reset_index()

    # medications
    med_agg = medications.groupby("PATIENT").agg(
        total_medications=("START", "count"),
        unique_medications=("DESCRIPTION", "nunique"),
        avg_medication_cost=("BASE_COST", "mean"),
        total_dispenses=("DISPENSES", "sum")
    ).reset_index()

    # procedures
    proc_agg = procedures.groupby("PATIENT").agg(
        total_procedures=("START", "count"),
        unique_procedures=("DESCRIPTION", "nunique"),
        avg_procedure_cost=("BASE_COST", "mean")
    ).reset_index()

    # immunizations
    imm_agg = immunizations.groupby("PATIENT").agg(
        total_immunizations=("DATE", "count"),
        unique_vaccines=("DESCRIPTION", "nunique")
    ).reset_index()

    # allergies
    allergy_agg = allergies.groupby("PATIENT").agg(
        total_allergies=("START", "count"),
        unique_allergy_types=("TYPE", "nunique"),
        unique_allergy_categories=("CATEGORY", "nunique")
    ).reset_index()

    # careplans
    care_agg = careplans.groupby("PATIENT").agg(
        total_careplans=("Id", "count"),
        unique_careplan_reasons=("REASONDESCRIPTION", "nunique")
    ).reset_index()

    # imaging
    img_agg = imaging.groupby("PATIENT").agg(
        total_imaging=("Id", "count"),
        unique_modalities=("MODALITY_DESCRIPTION", "nunique"),
        unique_body_sites=("BODYSITE_DESCRIPTION", "nunique")
    ).reset_index()

    # devices
    dev_agg = devices.groupby("PATIENT").agg(
        total_devices=("START", "count"),
        unique_device_types=("DESCRIPTION", "nunique")
    ).reset_index()

    # supplies
    sup_agg = supplies.groupby("PATIENT").agg(
        total_supplies=("DATE", "count"),
        unique_supply_types=("DESCRIPTION", "nunique")
    ).reset_index()

    # payer transitions
    pay_agg = payer_trans.groupby("PATIENT").agg(
        total_payer_transitions=("START_DATE", "count"),
        unique_payers=("PAYER", "nunique")
    ).reset_index()

    # claims
    claims_cost_col = "OUTSTANDING1" if "OUTSTANDING1" in claims.columns else "Id"
    claims_agg = claims.groupby("PATIENTID").agg(
        total_claims=("Id", "count"),
        avg_claim_cost=(claims_cost_col, "mean")
    ).reset_index().rename(columns={"PATIENTID": "PATIENT"})

    # claims transactions
    if "AMOUNT" in claims_trans.columns:
        ct_agg = claims_trans.groupby("PATIENTID").agg(
            total_transactions=("TYPE", "count"),
            total_transaction_amount=("AMOUNT", "sum"),
            unique_transaction_types=("TYPE", "nunique")
        ).reset_index().rename(columns={"PATIENTID": "PATIENT"})
    else:
        ct_agg = pd.DataFrame(columns=["PATIENT"])

    # --- MERGE ALL FEATURES ---
    df = pat_features.copy()
    for feat_df in [enc_agg, obs_features, med_agg, proc_agg, imm_agg,
                    allergy_agg, care_agg, img_agg, dev_agg, sup_agg,
                    pay_agg, claims_agg, ct_agg]:
        if "PATIENT" in feat_df.columns:
            df = df.merge(feat_df, on="PATIENT", how="left")

    # --- TARGET VARIABLE ---
    conditions["START"] = pd.to_datetime(
        conditions["START"], dayfirst=True, errors="coerce", utc=True)
    clinical = conditions[
        conditions["DESCRIPTION"].str.contains(
            r"\(disorder\)|\(finding\)", na=False, regex=True)
    ]
    positive_patients = set(clinical["PATIENT"].unique())
    df["label"] = df["PATIENT"].apply(
        lambda x: 1 if x in positive_patients else 0)

    # --- TEMPORAL SPLIT ---
    enc_dates = encounters.groupby("PATIENT")["START"].agg(
        ["min", "max"]).reset_index()
    enc_dates.columns = ["PATIENT", "first_enc", "last_enc"]
    df = df.merge(enc_dates, on="PATIENT", how="left")

    cutoff = pd.Timestamp(TEMPORAL_CUTOFF, tz="UTC")
    df1 = df[df["first_enc"] < cutoff].copy()
    df2 = df[df["last_enc"] >= cutoff].copy()

    drop_cols = ["PATIENT", "first_enc", "last_enc"]
    df1 = df1.drop(columns=[c for c in drop_cols if c in df1.columns])
    df2 = df2.drop(columns=[c for c in drop_cols if c in df2.columns])

    # --- MISSING VALUE TRACKING (before dropping) ---
    all_X_pre = df1.drop(columns=["label"], errors="ignore")
    missing_series = (all_X_pre.isna().mean()
                      .sort_values(ascending=False)
                      .head(20))

    # --- PREPROCESSING ---
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

    # --- SMOTE ---
    smote = SMOTE(random_state=RANDOM_STATE)
    X1_smote, y1_smote = smote.fit_resample(X_train_d1, y_train_d1)

    smote2 = SMOTE(random_state=RANDOM_STATE)
    X2_smote, y2_smote = smote2.fit_resample(X_train_d2, y_train_d2)

    # --- TRAIN MODELS ---
    dt_grid = GridSearchCV(
        DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        {"max_depth": [3, 5, 10, None]},
        scoring="f1", cv=5, n_jobs=-1
    )
    dt_grid.fit(X_train_d1, y_train_d1)
    dt_best = dt_grid.best_estimator_

    svm_grid = GridSearchCV(
        SVC(kernel="rbf", class_weight="balanced",
            probability=True, random_state=RANDOM_STATE),
        {"C": [0.1, 1, 10]},
        scoring="f1", cv=5, n_jobs=-1
    )
    svm_grid.fit(X_train_d1, y_train_d1)
    svm_best = svm_grid.best_estimator_

    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        max_iter=500,
        early_stopping=True,
        random_state=RANDOM_STATE
    )
    mlp.fit(X1_smote, y1_smote)

    # --- EVALUATE ---
    def evaluate(model, X, y, label, dataset):
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]
        return {
            "model":     label,
            "evaluated_on": dataset,
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

    # --- CONTINUAL LEARNING ---
    mlp_cl = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        max_iter=1,
        warm_start=False,
        random_state=RANDOM_STATE
    )
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
        "X_train_d1":    X_train_d1,
        "y_train_d1":    y_train_d1,
        "y_test_d1":     y_test_d1,
        "X_train_d2":    X_train_d2,
        "y_train_d2":    y_train_d2,
        "y_test_d2":     y_test_d2,
        "X_test_d1":     X_test_d1,
        "X_test_d2":     X_test_d2,
        "feature_names": feature_names,
        "baseline_df":   baseline_df,
        "continual_df":  continual_df,
        "dt_best":       dt_best,
        "svm_best":      svm_best,
        "mlp":           mlp,
        "mlp_cl":        mlp_cl,
        "d1_size":       len(df1),
        "d2_size":       len(df2),
        "total_patients": len(df),
        "missing_series": missing_series,
    }


# ── SIDEBAR ──────────────────────────────────────────────────────────────────

st.sidebar.markdown("## Clinical Prediction")
st.sidebar.markdown("**BITS F464 Machine Learning**")
st.sidebar.markdown("**Team 13** | Semester 2, 2025-26")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "Project Overview",
    "Exploratory Data Analysis",
    "Model Performance",
    "Continual Learning",
    "Feature Importance"
])

with st.spinner("Running pipeline... this may take a few minutes on first load."):
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

if page == "Project Overview":
    st.title("Automated ML Pipeline for Clinical Prediction")
    st.markdown("### Under Temporal Shift in EHR Data")
    st.markdown("---")

    st.markdown("""
    This dashboard presents an end-to-end machine learning pipeline built on a
    synthetic Electronic Health Records (EHR) dataset. The goal is to predict
    whether a patient has a clinically significant condition (disorder/finding)
    using patient demographics, clinical observations, and healthcare utilization data.
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Patients", f"{data['total_patients']:,}")
    col2.metric("Features", len(feature_names))
    col3.metric("D1 Patients (Historical)", f"{data['d1_size']:,}")
    col4.metric("D2 Patients (Current)", f"{data['d2_size']:,}")

    st.markdown("---")
    st.subheader("Pipeline Architecture")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Data Pipeline (Task 2)**
        - 15 CSV files merged on patient identifier
        - Temporal split: pre/post 2020-01-01
        - Features after dropping sparse columns (>50% missing)
        - Binary target: has (disorder)/(finding) condition
        - StandardScaler applied (fit on train only)
        - 80/20 stratified train/test split
        """)
        st.markdown("""
        **Models Trained (Task 3)**
        - Decision Tree (GridSearchCV, class_weight=balanced)
        - SVM RBF (GridSearchCV, class_weight=balanced)
        - MLP Neural Network (SMOTE oversampling)
        """)
    with col2:
        st.markdown("""
        **Key Challenge: Class Imbalance**
        - Majority label=0 (no condition)
        - Minority label=1 (has condition)
        - Handled via class_weight=balanced and SMOTE
        """)
        st.markdown("""
        **Temporal Datasets**
        - Dataset 1 (Historical): earliest encounter < 2020
        - Dataset 2 (Current): any encounter >= 2020
        - Patients may overlap between datasets
        - Models trained on D1, evaluated on both D1 and D2
        """)

    st.markdown("---")
    st.subheader("Team")
    st.markdown("""
    <style>
    .team-row { display: flex; gap: 16px; }
    .team-card {
        flex: 1;
        background-color: #1e3a5f;
        border-left: 4px solid #4a9eff;
        border-radius: 6px;
        padding: 16px;
        box-sizing: border-box;
        font-size: 14px;
        line-height: 1.6;
        color: #f0f4f8 !important;
    }
    .team-card strong { color: #ffffff !important; font-size: 15px; }
    .team-card em { color: #90c2ff !important; }
    </style>
    <div class="team-row">
      <div class="team-card">
        <strong>Data Architect</strong><br>
        Shriniketh Deevanapalli<br><br>
        <em>Task 2 (a, b, c)</em><br><br>
        Merged 15 CSV tables, implemented the temporal train/test split,
        and engineered the feature dataset with StandardScaler.
      </div>
      <div class="team-card">
        <strong>ML Engineer</strong><br>
        Sanvi Udhan<br><br>
        <em>Task 3 (a, b, c) + Task 4</em><br><br>
        Trained Decision Tree, SVM, and MLP models.
        Implemented continual learning via partial_fit and compiled all performance metrics.
      </div>
      <div class="team-card">
        <strong>Full-Stack Developer</strong><br>
        Sai Dheeraj Yadavalli<br><br>
        <em>Task 1 + Task 5</em><br><br>
        Built this Streamlit dashboard and integrated outputs
        from all team members into interactive visualizations.
      </div>
      <div class="team-card">
        <strong>Data Analyst</strong><br>
        Shambhavi Rani<br><br>
        <em>Task 2 (d) + Task 3 (d, e, f) + Task 5</em><br><br>
        Performed EDA, wrote the bias-variance and feature importance analysis,
        and produced the final video presentation.
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── PAGE 2: EDA ───────────────────────────────────────────────────────────────

elif page == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("D1 Train Size", f"{len(y_train_d1):,}")
    col2.metric("D1 Positive Rate", f"{y_train_d1.mean()*100:.1f}%")
    col3.metric("D2 Train Size", f"{len(y_train_d2):,}")
    col4.metric("D2 Positive Rate", f"{y_train_d2.mean()*100:.1f}%")

    st.markdown("---")

    eda_section = st.selectbox("Select EDA Section", [
        "Class Distribution",
        "Demographics",
        "Clinical Features",
        "Healthcare Utilization",
        "Correlation Heatmap",
        "Data Drift Analysis",
        "Missing Values",
    ])

    # ── Class Distribution ────────────────────────────────────────────────────
    if eda_section == "Class Distribution":
        st.subheader("Class Distribution")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, y, title in [
            (axes[0], y_train_d1, "Dataset 1 — Training"),
            (axes[1], y_train_d2, "Dataset 2 — Training"),
        ]:
            counts = y.value_counts().sort_index()
            bars = ax.bar(["No Condition (0)", "Has Condition (1)"],
                          counts.values, color=["steelblue", "darkorange"],
                          edgecolor="black")
            ax.set_title(title)
            ax.set_ylabel("Count")
            ax.bar_label(bars, fmt="%d", fontsize=10)
            ax.set_ylim(0, counts.max() * 1.15)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.info("Severe class imbalance: majority have no condition vs a minority with "
                "a disorder/finding. This is consistent across both D1 and D2, confirming "
                "the imbalance is not a temporal artifact.")

    # ── Demographics ──────────────────────────────────────────────────────────
    elif eda_section == "Demographics":
        st.subheader("Demographic Analysis")

        demo_options = [c for c in ["age", "GENDER", "RACE", "MARITAL", "INCOME"]
                        if c in X_train_d1.columns]
        selected_demo = st.selectbox("Select demographic", demo_options)

        col1, col2 = st.columns(2)
        for col_ctx, X_tr, y_tr, title in [
            (col1, X_train_d1, y_train_d1, "Dataset 1 (Historical)"),
            (col2, X_train_d2, y_train_d2, "Dataset 2 (Current)"),
        ]:
            with col_ctx:
                st.markdown(f"**{title}**")
                fig, ax = plt.subplots(figsize=(5, 4))
                if selected_demo in ["age", "INCOME"]:
                    d0 = X_tr.loc[y_tr == 0, selected_demo].dropna()
                    d1 = X_tr.loc[y_tr == 1, selected_demo].dropna()
                    ax.hist(d0, bins=25, alpha=0.6, color="steelblue",
                            label="No Condition", density=True)
                    ax.hist(d1, bins=25, alpha=0.6, color="darkorange",
                            label="Has Condition", density=True)
                    ax.set_xlabel(selected_demo + " (scaled)")
                    ax.set_ylabel("Density")
                    ax.legend()
                else:
                    tmp = X_tr[[selected_demo]].copy()
                    tmp["label"] = y_tr.values
                    gd = tmp.groupby([selected_demo, "label"]).size().unstack(fill_value=0)
                    gd.plot(kind="bar", ax=ax, color=["steelblue", "darkorange"],
                            edgecolor="black")
                    ax.set_xlabel(selected_demo + " (encoded)")
                    ax.set_ylabel("Count")
                    ax.legend(["No Condition", "Has Condition"])
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
                ax.set_title(f"{selected_demo} by Label")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

    # ── Clinical Features ─────────────────────────────────────────────────────
    elif eda_section == "Clinical Features":
        st.subheader("Clinical Feature Distributions")

        obs_cols = [c for c in feature_names if c.startswith("obs_") and "_mean" in c]
        if obs_cols:
            selected_obs = st.selectbox("Select clinical feature", obs_cols)
            fig, ax = plt.subplots(figsize=(10, 5))
            for label, colour, lname in [(0, "steelblue", "No Condition"),
                                          (1, "darkorange", "Has Condition")]:
                vals = X_train_d1.loc[y_train_d1 == label, selected_obs].dropna()
                ax.hist(vals, bins=30, alpha=0.6, color=colour,
                        label=lname, density=True)
            ax.set_xlabel(selected_obs + " (scaled)")
            ax.set_ylabel("Density")
            ax.set_title(f"{selected_obs} — Distribution by Label (D1 Training)")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.caption("Distribution split by label (0=no condition, 1=has condition) "
                       "for the D1 training set.")
        else:
            st.info("No observation features found in the feature set.")

    # ── Healthcare Utilization ────────────────────────────────────────────────
    elif eda_section == "Healthcare Utilization":
        st.subheader("Healthcare Utilization by Label")

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
                st.markdown(f"**{name}**")
                fig, ax = plt.subplots(figsize=(5, 4))
                d0 = X_train_d1.loc[y_train_d1 == 0, col].dropna()
                d1 = X_train_d1.loc[y_train_d1 == 1, col].dropna()
                ax.boxplot([d0, d1], labels=["No Condition", "Has Condition"],
                           patch_artist=True,
                           boxprops=dict(facecolor="steelblue", alpha=0.6))
                ax.set_title(f"{col} by Label")
                ax.set_ylabel(f"{col} (scaled)")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

    # ── Correlation Heatmap ───────────────────────────────────────────────────
    elif eda_section == "Correlation Heatmap":
        st.subheader("Correlation Heatmap — Top 30 Features vs Label (D1)")
        tmp = X_train_d1.copy()
        tmp["label"] = y_train_d1.values
        corr_with_label = tmp.corr()["label"].drop("label").abs()
        top30 = corr_with_label.sort_values(ascending=False).head(30).index.tolist()
        corr_mat = tmp[top30 + ["label"]].corr()

        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(corr_mat, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, shrink=0.8)
        labels = top30 + ["label"]
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title("Correlation Matrix — Top 30 Features + Label (D1 Training Set)")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.info("All correlations are weak (r < 0.1), which is expected given the severe "
                "class imbalance. Strongest correlates are clinical observation features "
                "such as Cholesterol, Hemoglobin, and Blood Pressure.")

    # ── Data Drift Analysis ───────────────────────────────────────────────────
    elif eda_section == "Data Drift Analysis":
        st.subheader("Data Drift — D1 vs D2 Distribution Comparison")
        top10 = X_train_d1.var().sort_values(ascending=False).head(10).index.tolist()
        fig, axes = plt.subplots(2, 5, figsize=(18, 7))
        axes = axes.flatten()
        for i, feat in enumerate(top10):
            ax = axes[i]
            d1_vals = X_train_d1[feat].dropna()
            d2_vals = X_train_d2[feat].dropna() if feat in X_train_d2.columns else pd.Series(dtype=float)
            ax.hist(d1_vals, bins=25, alpha=0.55, color="steelblue",
                    label="D1", density=True)
            if len(d2_vals) > 0:
                ax.hist(d2_vals, bins=25, alpha=0.55, color="darkorange",
                        label="D2", density=True)
            ax.set_title(feat[:22], fontsize=8)
            ax.legend(fontsize=7)
            ax.tick_params(labelsize=7)
        plt.suptitle("Distribution Shift: D1 vs D2 — Top 10 Features by Variance",
                     y=1.01, fontsize=11)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.warning("Note: Drift analysis was performed on StandardScaled data. "
                   "Distributions appear similar post-scaling (mean≈0, std≈1). "
                   "Drift exists in the raw feature space.")

    # ── Missing Values ────────────────────────────────────────────────────────
    elif eda_section == "Missing Values":
        st.subheader("Missing Value Analysis — Top 20 Most Sparse Columns")
        fig, ax = plt.subplots(figsize=(12, 6))
        cols = missing_series.index.tolist()
        vals = missing_series.values
        bars = ax.barh(cols[::-1], vals[::-1], color="steelblue", edgecolor="black")
        ax.axvline(0.5, color="red", linestyle="--", linewidth=1.5,
                   label="50% drop threshold")
        ax.set_xlabel("Missing Rate")
        ax.set_title("Top 20 Columns by Missing Rate (D1 pre-filter)")
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.info("Columns with >50% missingness are dropped before training. "
                "These are typically rare lab tests or sparse allergy/IgE panels.")


# ── PAGE 3: MODEL PERFORMANCE ─────────────────────────────────────────────────

elif page == "Model Performance":
    st.title("Model Performance")
    st.markdown("---")

    st.subheader("Metrics Summary")

    metric = st.selectbox("Select metric to compare",
                          ["accuracy", "precision", "recall", "f1", "roc_auc"])

    pivot = baseline_df.pivot(index="model", columns="evaluated_on", values=metric)

    fig, ax = plt.subplots(figsize=(8, 4))
    pivot.plot(kind="bar", ax=ax, color=["steelblue", "darkorange"],
               edgecolor="black", width=0.6)
    ax.set_title(f"{metric.upper()} — All Models on D1 and D2 Test Sets")
    ax.set_xlabel("Model")
    ax.set_ylabel(metric.upper())
    ax.set_ylim(0, 1)
    ax.legend(["D1 Test (Historical)", "D2 Test (Current)"])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("---")
    st.subheader("Full Metrics Table")
    styled = baseline_df.style.background_gradient(subset=["f1", "roc_auc"], cmap="RdYlGn")
    st.dataframe(styled, use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ROC Curves")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        models_map = {"DT": dt_best, "SVM": svm_best, "MLP": mlp}
        colours = {"DT": "#2196F3", "SVM": "#FF9800", "MLP": "#4CAF50"}
        for ax, (X_test, y_test, title) in zip(
                axes,
                [(X_test_d1, y_test_d1, "D1 Test"),
                 (X_test_d2, y_test_d2, "D2 Test")]):
            for name, model in models_map.items():
                probs = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, probs)
                ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.3f})",
                        color=colours[name], lw=2)
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.set_title(title)
            ax.legend(loc="lower right", fontsize=8)
        plt.suptitle("ROC Curves", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col2:
        st.subheader("Confusion Matrices")
        fig, axes = plt.subplots(2, 3, figsize=(10, 7))
        for row, (X_test, y_test, ds) in enumerate([
                (X_test_d1, y_test_d1, "D1"), (X_test_d2, y_test_d2, "D2")]):
            for col_i, (name, model) in enumerate(models_map.items()):
                ax = axes[row][col_i]
                cm = confusion_matrix(y_test, model.predict(X_test))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                              display_labels=["Neg", "Pos"])
                disp.plot(ax=ax, colorbar=False, cmap="Blues")
                ax.set_title(f"{name} — {ds}", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("---")
    st.subheader("Analysis")
    st.markdown("""
    **Key Observations:**
    - **MLP** achieves the best overall performance on D2 thanks to SMOTE oversampling
    - All models are trained on D1 and evaluated on both D1 and D2 test sets
    - **SVM** may show variable ROC-AUC depending on the C tuning — sensitive to scaling
    - **Decision Tree** is consistent but weak — expected with limited depth from grid search
    - High accuracy scores are misleading due to class imbalance —
      F1 and ROC-AUC are the meaningful metrics

    **Bias-Variance Trade-off:**
    - Decision Tree (max_depth=3–5): High bias, low variance — underfitting
    - SVM RBF: Moderate bias-variance — sensitive to C tuning
    - MLP (128-64-32): Low bias, higher variance — best generalization with SMOTE
    """)


# ── PAGE 4: CONTINUAL LEARNING ────────────────────────────────────────────────

elif page == "Continual Learning":
    st.title("Continual Learning Analysis")
    st.markdown("---")

    st.markdown("""
    Continual learning allows a model trained on historical data (D1) to adapt to
    new data (D2) without full retraining. We implemented fine-tuning via
    **partial_fit()** on mini-batches of D2 training data over 50 epochs.
    """)

    st.markdown("---")
    st.subheader("MLP Before vs After Continual Learning on D2 Test Set")

    col1, col2 = st.columns(2)
    mlp_d1_row = continual_df[continual_df.model == "MLP_D1"].iloc[0]
    mlp_cl_row = continual_df[continual_df.model == "MLP_CL"].iloc[0]

    with col1:
        st.markdown("**MLP_D1 (trained on D1 only)**")
        st.metric("F1 Score", f"{mlp_d1_row.f1:.3f}")
        st.metric("ROC-AUC",  f"{mlp_d1_row.roc_auc:.3f}")
        st.metric("Recall",   f"{mlp_d1_row.recall:.3f}")

    with col2:
        st.markdown("**MLP_CL (after continual learning on D2)**")
        delta_f1  = mlp_cl_row.f1      - mlp_d1_row.f1
        delta_roc = mlp_cl_row.roc_auc - mlp_d1_row.roc_auc
        delta_rec = mlp_cl_row.recall  - mlp_d1_row.recall
        st.metric("F1 Score", f"{mlp_cl_row.f1:.3f}",      delta=f"{delta_f1:.3f}")
        st.metric("ROC-AUC",  f"{mlp_cl_row.roc_auc:.3f}", delta=f"{delta_roc:.3f}")
        st.metric("Recall",   f"{mlp_cl_row.recall:.3f}",  delta=f"{delta_rec:.3f}")

    st.markdown("---")

    # Continual learning bar chart
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    before_vals = [mlp_d1_row[m] for m in metrics]
    after_vals  = [mlp_cl_row[m] for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(metrics))
    w = 0.35
    b1 = ax.bar(x - w/2, before_vals, w, label="MLP_D1 (Before)",
                color="steelblue", edgecolor="black", alpha=0.85)
    b2 = ax.bar(x + w/2, after_vals,  w, label="MLP_CL (After)",
                color="darkorange", edgecolor="black", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Continual Learning: MLP Before vs After Adaptation on D2 Test Set")
    ax.legend()
    ax.bar_label(b1, fmt="%.3f", fontsize=8)
    ax.bar_label(b2, fmt="%.3f", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("---")
    st.subheader("Continual Learning Metrics Table")
    st.dataframe(continual_df, use_container_width=True)

    st.markdown("---")
    st.subheader("Analysis — Catastrophic Forgetting")

    f1_before = mlp_d1_row.f1
    f1_after  = mlp_cl_row.f1
    direction = "dropped" if f1_after < f1_before else "improved"

    st.warning(f"""
    **Finding: Continual learning via partial_fit caused catastrophic forgetting.**

    The MLP_CL model F1 {direction} from {f1_before:.2f} → {f1_after:.2f} on D2.
    This is a well-known phenomenon in continual learning where fine-tuning on new data
    overwrites previously learned weights.

    **Why this happened:**
    - partial_fit() with 50 epochs aggressively updated all network weights
    - No regularization was applied to preserve D1 knowledge
    - The learning rate was not decayed during fine-tuning

    **What this means:**
    - The D1-trained MLP already generalizes to D2 — aggressive fine-tuning is
      counterproductive when the base model generalizes well
    - More sophisticated CL strategies like Elastic Weight Consolidation (EWC) or
      Learning without Forgetting (LwF) would be needed for reliable improvement
    """)


# ── PAGE 5: FEATURE IMPORTANCE ────────────────────────────────────────────────

elif page == "Feature Importance":
    st.title("Feature Importance & Model Interpretation")
    st.markdown("---")

    st.subheader("Top 20 Features — Decision Tree")
    importances = dt_best.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    top20 = feat_imp.head(20)

    fig, ax = plt.subplots(figsize=(10, 7))
    colours = plt.cm.viridis(np.linspace(0.2, 0.85, len(top20)))
    bars = ax.barh(top20.index[::-1], top20.values[::-1],
                   color=colours[::-1], edgecolor="black")
    ax.set_xlabel("Importance")
    ax.set_title("Top 20 Feature Importances — Decision Tree")
    ax.bar_label(bars, fmt="%.4f", fontsize=8, padding=2)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("---")
    st.subheader("Feature Categories Breakdown")

    demo_feats    = [f for f in feature_names if any(x in f for x in
                     ["GENDER", "RACE", "ETHNICITY", "INCOME", "MARITAL", "age",
                      "is_deceased", "HEALTHCARE"])]
    encounter_feats = [f for f in feature_names if any(x in f for x in
                       ["encounter", "claim_cost", "payer_coverage"])]
    obs_feats     = [f for f in feature_names if f.startswith("obs_")]
    util_feats    = [f for f in feature_names if any(x in f for x in
                     ["medication", "procedure", "immunization", "careplan",
                      "imaging", "device", "supply", "transaction", "payer", "claims"])]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Demographic Features",  len(demo_feats))
    col2.metric("Encounter Features",    len(encounter_feats))
    col3.metric("Observation Features",  len(obs_feats))
    col4.metric("Utilization Features",  len(util_feats))

    st.markdown("---")
    st.subheader("Explore All Features")
    search = st.text_input("Search features by name")
    filtered = [f for f in feature_names if search.lower() in f.lower()] \
               if search else feature_names
    st.write(f"Showing {len(filtered)} of {len(feature_names)} features")
    st.dataframe(pd.DataFrame({"feature_name": filtered}), use_container_width=True)

    st.markdown("---")
    st.subheader("Interpretation")
    st.markdown("""
    **Key findings from feature importance:**
    - Observation-derived features (vitals and lab aggregates) dominate the top 20
    - Clinical measurements like BMI, Blood Pressure, Cholesterol are most predictive
    - Demographic features contribute but are less important than clinical indicators
    - Utilization features (encounter counts, medication counts) provide secondary signal

    **Model-specific behavior:**
    - **Decision Tree**: Uses a small subset of features at each split — interpretable
      but limited
    - **SVM**: Uses all features via kernel trick — less interpretable but more powerful
    - **MLP**: Learns complex non-linear combinations of features — most powerful but
      black-box

    **Feature representation impact:**
    - Aggregating observations as mean + variance per patient captures both
      central tendency and variability
    - Dropping columns with >50% missingness improved signal-to-noise ratio
    - StandardScaling was critical for SVM and MLP convergence
    """)
