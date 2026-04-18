import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import os

st.set_page_config(
    page_title="Clinical Prediction Dashboard",
    page_icon="🏥",
    layout="wide"
)

st.sidebar.markdown("## 🏥 Clinical Prediction")
st.sidebar.markdown("**BITS F464 Machine Learning**")
st.sidebar.markdown("**Team XX** | Semester 2, 2025-26")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "🏠 Project Overview",
    "📊 Exploratory Data Analysis",
    "🤖 Model Performance",
    "🔄 Continual Learning",
    "🔍 Feature Importance"
])

@st.cache_data
def load_metrics():
    baseline = pd.read_csv("models/metrics_baseline.csv")
    continual = pd.read_csv("models/metrics_continual.csv")
    return baseline, continual

@st.cache_data
def load_splits_info():
    y_train_d1 = pd.read_pickle("data/processed/y_train_d1.pkl")
    y_test_d1  = pd.read_pickle("data/processed/y_test_d1.pkl")
    y_train_d2 = pd.read_pickle("data/processed/y_train_d2.pkl")
    y_test_d2  = pd.read_pickle("data/processed/y_test_d2.pkl")
    X_train_d1 = pd.read_pickle("data/processed/X_train_d1.pkl")
    feature_names = joblib.load("data/processed/feature_names.pkl")
    return y_train_d1, y_test_d1, y_train_d2, y_test_d2, X_train_d1, feature_names

baseline_df, continual_df = load_metrics()
y_train_d1, y_test_d1, y_train_d2, y_test_d2, X_train_d1, feature_names = load_splits_info()

# ── PAGE 1 ──────────────────────────────────────────────────────────────────
if page == "🏠 Project Overview":
    st.title("🏥 Automated ML Pipeline for Clinical Prediction")
    st.markdown("### Under Temporal Shift in EHR Data")
    st.markdown("---")

    st.markdown("""
    This dashboard presents an end-to-end machine learning pipeline built on a
    synthetic Electronic Health Records (EHR) dataset. The goal is to predict
    whether a patient has a clinically significant condition (disorder/finding)
    using patient demographics, clinical observations, and healthcare utilization data.
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Patients", "2,823")
    col2.metric("Features", "104")
    col3.metric("D1 Patients (Historical)", "2,152")
    col4.metric("D2 Patients (Current)", "1,975")

    st.markdown("---")
    st.subheader("📋 Pipeline Architecture")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Data Pipeline (Task 2)**
        - 17 CSV files merged on patient identifier
        - Temporal split: pre/post 2020-01-01
        - 104 features after dropping 132 sparse columns
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
        - ~95% label=0 (no condition)
        - ~5% label=1 (has condition)
        - Handled via class_weight=balanced and SMOTE
        """)
        st.markdown("""
        **Temporal Datasets**
        - Dataset 1 (Historical): earliest encounter < 2020
        - Dataset 2 (Current): any encounter ≥ 2020
        - 1,852 patients overlap between datasets
        - Models trained on D1, evaluated on both D1 and D2
        """)

    st.markdown("---")
    st.subheader("👥 Team")
    team_col1, team_col2, team_col3, team_col4 = st.columns(4)
    team_col1.info("**Data Architect**\nShriniketh\nTask 2")
    team_col2.info("**ML Engineer**\nSanvi\nTask 3 & 4")
    team_col3.info("**Full Stack Dev**\nTask 1 & 5")
    team_col4.info("**Analyst**\nTask 2d & Video")

# ── PAGE 2 ──────────────────────────────────────────────────────────────────
elif page == "📊 Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis")
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
        "Missing Values"
    ])

    if eda_section == "Class Distribution":
        st.subheader("Class Distribution")
        img = Image.open("data/eda/eda_class_distribution.png")
        st.image(img, use_container_width=True)
        st.info("Severe class imbalance: ~95% no condition vs ~5% has condition. "
                "This is consistent across both D1 and D2, confirming the imbalance "
                "is not a temporal artifact.")

    elif eda_section == "Demographics":
        st.subheader("Demographic Analysis")
        demo_plots = {
            "Age Distribution": ["data/eda/eda_age_by_label_d1.png",
                                  "data/eda/eda_age_by_label_d2.png"],
            "Gender": ["data/eda/eda_gender_by_label_d1.png",
                       "data/eda/eda_gender_by_label_d2.png"],
            "Race": ["data/eda/eda_race_by_label_d1.png",
                     "data/eda/eda_race_by_label_d2.png"],
            "Marital Status": ["data/eda/eda_marital_by_label_d1.png",
                                "data/eda/eda_marital_by_label_d2.png"],
        }
        selected_demo = st.selectbox("Select demographic", list(demo_plots.keys()))
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Dataset 1 (Historical)**")
            st.image(Image.open(demo_plots[selected_demo][0]), use_container_width=True)
        with col2:
            st.markdown("**Dataset 2 (Current)**")
            st.image(Image.open(demo_plots[selected_demo][1]), use_container_width=True)

        st.subheader("Income Distribution by Label")
        st.image(Image.open("data/eda/eda_income_by_label.png"), use_container_width=True)

    elif eda_section == "Clinical Features":
        st.subheader("Clinical Feature Distributions")
        clinical_plots = [f for f in os.listdir("data/eda/")
                          if f.startswith("eda_clinical_") and f.endswith(".png")]
        clinical_plots = sorted(clinical_plots)
        selected_clinical = st.selectbox("Select clinical feature", clinical_plots)
        st.image(Image.open(f"data/eda/{selected_clinical}"), use_container_width=True)
        st.caption("Violin plots show distribution split by label (0=no condition, 1=has condition) "
                   "for both D1 and D2.")

    elif eda_section == "Healthcare Utilization":
        st.subheader("Healthcare Utilization by Label")
        util_plots = {
            "Encounters": "data/eda/eda_utilization_total_encounters.png",
            "Medications": "data/eda/eda_utilization_total_medications.png",
            "Procedures": "data/eda/eda_utilization_total_procedures.png",
            "Claims": "data/eda/eda_utilization_total_claims.png",
        }
        col1, col2 = st.columns(2)
        items = list(util_plots.items())
        for i, (name, path) in enumerate(items):
            col = col1 if i % 2 == 0 else col2
            with col:
                st.markdown(f"**{name}**")
                st.image(Image.open(path), use_container_width=True)

    elif eda_section == "Correlation Heatmap":
        st.subheader("Correlation Heatmap — Top 30 Features vs Label (D1)")
        st.image(Image.open("data/eda/eda_correlation_heatmap_d1.png"),
                 use_container_width=True)
        st.info("All correlations are weak (r < 0.1), which is expected given the severe "
                "class imbalance. Strongest correlates: Cholesterol HDL, Hemoglobin, "
                "Systolic BP.")

    elif eda_section == "Data Drift Analysis":
        st.subheader("Data Drift — D1 vs D2 Distribution Comparison")
        st.image(Image.open("data/eda/eda_drift_kde_top10.png"), use_container_width=True)
        st.warning("Note: Drift analysis was performed on StandardScaled data. "
                   "Distributions appear similar post-scaling (mean≈0, std≈1). "
                   "Drift exists in the raw feature space.")

    elif eda_section == "Missing Values":
        st.subheader("Missing Value Analysis — Top 20 Most Sparse Columns")
        st.image(Image.open("data/eda/eda_missing_values_top20.png"),
                 use_container_width=True)
        st.info("132 columns were dropped for >50% missingness, mostly IgE allergy panels "
                "and rare lab tests. 104 features survived.")

# ── PAGE 3 ──────────────────────────────────────────────────────────────────
elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance")
    st.markdown("---")

    st.subheader("📈 Metrics Summary")

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
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("📋 Full Metrics Table")

    styled = baseline_df.style.background_gradient(subset=["f1", "roc_auc"],
                                                    cmap="RdYlGn")
    st.dataframe(styled, use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ROC Curves")
        st.image(Image.open("models/roc_curves.png"), use_container_width=True)
    with col2:
        st.subheader("Confusion Matrices")
        st.image(Image.open("models/confusion_matrices.png"), use_container_width=True)

    st.markdown("---")
    st.subheader("🧠 Analysis")
    st.markdown("""
    **Key Observations:**
    - **MLP** achieves the best overall performance (F1=0.78, ROC-AUC=0.90 on D2)
    - All models perform comparably or better on D2 than D1, suggesting the current
      data is more separable
    - **SVM** shows near-random performance on D1 (ROC-AUC=0.47) but recovers on D2
    - **Decision Tree** is consistent but weak — expected with shallow depth
    - High accuracy scores are misleading due to 95/5 class imbalance —
      F1 and ROC-AUC are the meaningful metrics

    **Bias-Variance Trade-off:**
    - Decision Tree (max_depth=3-5): High bias, low variance — underfitting
    - SVM RBF: Moderate bias-variance — sensitive to C tuning
    - MLP (128-64-32): Low bias, higher variance — best generalization with SMOTE
    """)

# ── PAGE 4 ──────────────────────────────────────────────────────────────────
elif page == "🔄 Continual Learning":
    st.title("🔄 Continual Learning Analysis")
    st.markdown("---")

    st.markdown("""
    Continual learning allows a model trained on historical data (D1) to adapt to
    new data (D2) without full retraining. We implemented fine-tuning via
    **partial_fit()** on mini-batches of D2 training data over 50 epochs.
    """)

    st.markdown("---")
    st.subheader("📊 MLP Before vs After Continual Learning on D2 Test Set")

    col1, col2 = st.columns(2)

    mlp_d1_row = continual_df[continual_df.model == "MLP_D1"].iloc[0]
    mlp_cl_row = continual_df[continual_df.model == "MLP_CL"].iloc[0]

    with col1:
        st.markdown("**MLP_D1 (trained on D1 only)**")
        st.metric("F1 Score", f"{mlp_d1_row.f1:.3f}")
        st.metric("ROC-AUC", f"{mlp_d1_row.roc_auc:.3f}")
        st.metric("Recall", f"{mlp_d1_row.recall:.3f}")

    with col2:
        st.markdown("**MLP_CL (after continual learning on D2)**")
        delta_f1 = mlp_cl_row.f1 - mlp_d1_row.f1
        delta_roc = mlp_cl_row.roc_auc - mlp_d1_row.roc_auc
        delta_rec = mlp_cl_row.recall - mlp_d1_row.recall
        st.metric("F1 Score", f"{mlp_cl_row.f1:.3f}", delta=f"{delta_f1:.3f}")
        st.metric("ROC-AUC", f"{mlp_cl_row.roc_auc:.3f}", delta=f"{delta_roc:.3f}")
        st.metric("Recall", f"{mlp_cl_row.recall:.3f}", delta=f"{delta_rec:.3f}")

    st.markdown("---")
    st.image(Image.open("models/continual_learning_comparison.png"),
             use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Continual Learning Metrics Table")
    st.dataframe(continual_df, use_container_width=True)

    st.markdown("---")
    st.subheader("🧠 Analysis — Catastrophic Forgetting")
    st.warning("""
    **Finding: Continual learning via partial_fit caused catastrophic forgetting.**

    The MLP_CL model performed worse than MLP_D1 on D2 (F1 dropped from 0.78 → 0.23).
    This is a well-known phenomenon in continual learning where fine-tuning on new data
    overwrites previously learned weights.

    **Why this happened:**
    - partial_fit() with 50 epochs aggressively updated all network weights
    - No regularization was applied to preserve D1 knowledge
    - The learning rate was not decayed during fine-tuning

    **What this means:**
    - The D1-trained MLP already generalizes well to D2 (F1=0.78, ROC-AUC=0.90)
    - Aggressive fine-tuning is counterproductive when the base model generalizes well
    - More sophisticated CL strategies like Elastic Weight Consolidation (EWC) or
      Learning without Forgetting (LwF) would be needed for improvement
    """)

# ── PAGE 5 ──────────────────────────────────────────────────────────────────
elif page == "🔍 Feature Importance":
    st.title("🔍 Feature Importance & Model Interpretation")
    st.markdown("---")

    st.subheader("Top 20 Features — Decision Tree")
    st.image(Image.open("models/feature_importance_dt.png"), use_container_width=True)

    st.markdown("---")
    st.subheader("Feature Categories Breakdown")

    demo_feats = [f for f in feature_names if any(x in f for x in
                  ["GENDER", "RACE", "ETHNICITY", "INCOME", "MARITAL", "age",
                   "is_deceased", "HEALTHCARE"])]
    encounter_feats = [f for f in feature_names if any(x in f for x in
                       ["encounter", "claim_cost", "payer_coverage"])]
    obs_feats = [f for f in feature_names if f.startswith("obs_")]
    util_feats = [f for f in feature_names if any(x in f for x in
                  ["medication", "procedure", "immunization", "careplan",
                   "imaging", "device", "supply", "transaction", "payer", "claims"])]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Demographic Features", len(demo_feats))
    col2.metric("Encounter Features", len(encounter_feats))
    col3.metric("Observation Features", len(obs_feats))
    col4.metric("Utilization Features", len(util_feats))

    st.markdown("---")

    st.subheader("🔎 Explore All Features")
    search = st.text_input("Search features by name")
    filtered = [f for f in feature_names if search.lower() in f.lower()] \
               if search else feature_names
    st.write(f"Showing {len(filtered)} of {len(feature_names)} features")
    st.dataframe(pd.DataFrame({"feature_name": filtered}), use_container_width=True)

    st.markdown("---")
    st.subheader("🧠 Interpretation")
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
    - Dropping 132 sparse columns (>50% missing) improved signal-to-noise ratio
    - StandardScaling was critical for SVM and MLP convergence
    """)

# ── SIDEBAR FOOTER ───────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("*BITS Pilani Hyderabad Campus*")
st.sidebar.markdown("*BITS F464 Machine Learning*")
st.sidebar.markdown("*Second Semester 2025-26*")
