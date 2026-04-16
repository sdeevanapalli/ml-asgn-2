"""
Comprehensive EDA on processed EHR feature matrices.
Outputs: PNG plots + CSV summary stats → data/eda/
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
EDA_DIR = "data/eda"
PROC_DIR = "data/processed"
os.makedirs(EDA_DIR, exist_ok=True)

C0, C1 = "#4C72B0", "#DD8452"   # blue=label 0, orange=label 1
PALETTE = [C0, C1]              # list — seaborn stringifies x keys, list avoids mismatch
sns.set_theme(style="whitegrid", font_scale=1.1)
SAVED = []

def savefig(name):
    path = os.path.join(EDA_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    SAVED.append(path)
    print(f"  saved: {name}")

# ── Load processed data ───────────────────────────────────────────────────────
def load(name):
    with open(os.path.join(PROC_DIR, name), "rb") as f:
        return pickle.load(f)

X_train_d1 = load("X_train_d1.pkl")
X_test_d1  = load("X_test_d1.pkl")
y_train_d1 = load("y_train_d1.pkl")
y_test_d1  = load("y_test_d1.pkl")

X_train_d2 = load("X_train_d2.pkl")
X_test_d2  = load("X_test_d2.pkl")
y_train_d2 = load("y_train_d2.pkl")
y_test_d2  = load("y_test_d2.pkl")

feature_names = load("feature_names.pkl")

# Build full (train+test) sets with labels for EDA
def full_set(Xtr, Xte, ytr, yte):
    X = pd.concat([Xtr, Xte], ignore_index=True)
    y = np.concatenate([ytr, yte])
    X["label"] = y
    return X

d1 = full_set(X_train_d1, X_test_d1, y_train_d1, y_test_d1)
d2 = full_set(X_train_d2, X_test_d2, y_train_d2, y_test_d2)

print(f"D1 full: {d1.shape},  D2 full: {d2.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. CLASS DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] Class Distribution")

for ds_name, ds in [("D1", d1), ("D2", d2)]:
    vc = ds["label"].value_counts().sort_index()
    pct = vc / len(ds) * 100
    print(f"  {ds_name}: label=0 → {vc[0]} ({pct[0]:.1f}%)  |  "
          f"label=1 → {vc[1]} ({pct[1]:.1f}%)")

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for ax, (ds_name, ds) in zip(axes, [("Dataset 1 (Historical)", d1),
                                      ("Dataset 2 (Current)", d2)]):
    vc = ds["label"].value_counts().sort_index()
    bars = ax.bar(["label=0\n(No condition)", "label=1\n(Has condition)"],
                  vc.values, color=[C0, C1], edgecolor="white", width=0.5)
    for bar, val, pct in zip(bars, vc.values, vc.values / len(ds) * 100):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                f"{val}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=11)
    ax.set_title(ds_name, fontsize=13, fontweight="bold")
    ax.set_ylabel("Patient Count")
    ax.set_xlabel("Label")
    ax.set_ylim(0, max(vc.values) * 1.2)
fig.suptitle("Class Distribution — D1 vs D2", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
savefig("eda_class_distribution.png")

# ─────────────────────────────────────────────────────────────────────────────
# 2. DEMOGRAPHIC ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Demographic Analysis")

# 2a. Age distribution — histogram + KDE by label
for ds_name, ds, tag in [("D1 (Historical)", d1, "d1"),
                          ("D2 (Current)",    d2, "d2")]:
    fig, ax = plt.subplots(figsize=(10, 6))
    for lbl, color, name in [(0, C0, "label=0"), (1, C1, "label=1")]:
        sub = ds[ds["label"] == lbl]["age"].dropna()
        ax.hist(sub, bins=30, alpha=0.4, color=color, density=True, label=f"{name} (n={len(sub)})")
        sub.plot.kde(ax=ax, color=color, linewidth=2)
    ax.set_title(f"Age Distribution by Label — {ds_name}", fontweight="bold")
    ax.set_xlabel("Age (years at 2020-01-01)")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    savefig(f"eda_age_by_label_{tag}.png")

# 2b. Gender
for ds_name, ds, tag in [("D1 (Historical)", d1, "d1"),
                          ("D2 (Current)",    d2, "d2")]:
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_df = ds.groupby(["GENDER", "label"]).size().reset_index(name="count")
    pivot = plot_df.pivot(index="GENDER", columns="label", values="count").fillna(0)
    pivot.plot(kind="bar", ax=ax, color=[C0, C1], edgecolor="white", rot=0)
    ax.set_title(f"Gender Distribution by Label — {ds_name}", fontweight="bold")
    ax.set_xlabel("Gender (encoded)")
    ax.set_ylabel("Patient Count")
    ax.legend(["label=0", "label=1"], title="Label")
    plt.tight_layout()
    savefig(f"eda_gender_by_label_{tag}.png")

# 2c. Race
for ds_name, ds, tag in [("D1 (Historical)", d1, "d1"),
                          ("D2 (Current)",    d2, "d2")]:
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df = ds.groupby(["RACE", "label"]).size().reset_index(name="count")
    pivot = plot_df.pivot(index="RACE", columns="label", values="count").fillna(0)
    pivot.plot(kind="bar", ax=ax, color=[C0, C1], edgecolor="white", rot=0)
    ax.set_title(f"Race Distribution by Label — {ds_name}", fontweight="bold")
    ax.set_xlabel("Race (encoded)")
    ax.set_ylabel("Patient Count")
    ax.legend(["label=0", "label=1"], title="Label")
    plt.tight_layout()
    savefig(f"eda_race_by_label_{tag}.png")

# 2d. Income boxplot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, (ds_name, ds) in zip(axes, [("D1 (Historical)", d1), ("D2 (Current)", d2)]):
    sns.boxplot(data=ds, x="label", y="INCOME", palette=PALETTE, ax=ax,
                order=[0, 1], width=0.5, flierprops={"marker": ".", "markersize": 3})
    ax.set_title(ds_name, fontweight="bold")
    ax.set_xlabel("Label")
    ax.set_ylabel("Income")
    ax.set_xticklabels(["label=0", "label=1"])
fig.suptitle("Income Distribution by Label", fontsize=14, fontweight="bold")
plt.tight_layout()
savefig("eda_income_by_label.png")

# 2e. Marital status
for ds_name, ds, tag in [("D1 (Historical)", d1, "d1"),
                          ("D2 (Current)",    d2, "d2")]:
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df = ds.groupby(["MARITAL", "label"]).size().reset_index(name="count")
    pivot = plot_df.pivot(index="MARITAL", columns="label", values="count").fillna(0)
    pivot.plot(kind="bar", ax=ax, color=[C0, C1], edgecolor="white", rot=0)
    ax.set_title(f"Marital Status Distribution by Label — {ds_name}", fontweight="bold")
    ax.set_xlabel("Marital Status (encoded)")
    ax.set_ylabel("Patient Count")
    ax.legend(["label=0", "label=1"], title="Label")
    plt.tight_layout()
    savefig(f"eda_marital_by_label_{tag}.png")

# ─────────────────────────────────────────────────────────────────────────────
# 3. CLINICAL FEATURE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Clinical Feature Analysis")

CLINICAL_FEATURES = {
    "Body Height":           "obs_Body_Height_mean",
    "Body Weight":           "obs_Body_Weight_mean",
    "BMI":                   "obs_Body_mass_index__BMI___Ratio_mean",
    "Diastolic BP":          "obs_Diastolic_Blood_Pressure_mean",
    "Systolic BP":           "obs_Systolic_Blood_Pressure_mean",
    "Heart Rate":            "obs_Heart_rate_mean",
    "Total Cholesterol":     "obs_Cholesterol__Mass_volume__in_Serum_or_Plasma_mean",
}

for label, col in CLINICAL_FEATURES.items():
    if col not in d1.columns:
        print(f"  SKIP {col} — not in feature matrix")
        continue
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (ds_name, ds) in zip(axes, [("D1 (Historical)", d1), ("D2 (Current)", d2)]):
        sns.violinplot(data=ds, x="label", y=col, palette=PALETTE, ax=ax,
                       order=[0, 1], inner="box", density_norm="width", cut=0)
        ax.set_title(ds_name, fontweight="bold")
        ax.set_xlabel("Label")
        ax.set_ylabel(label)
        ax.set_xticklabels(["label=0", "label=1"])
    fig.suptitle(f"{label} Distribution by Label", fontsize=14, fontweight="bold")
    plt.tight_layout()
    safe = col.replace("obs_", "").replace("__", "_").replace("(", "").replace(")", "")[:40]
    savefig(f"eda_clinical_{safe}.png")

# ─────────────────────────────────────────────────────────────────────────────
# 4. ENCOUNTER & UTILIZATION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] Encounter & Utilization Analysis")

UTIL_FEATURES = [
    ("total_encounters",  "Total Encounters"),
    ("total_medications", "Total Medications"),
    ("total_procedures",  "Total Procedures"),
    ("total_claims",      "Total Claims"),
]

for col, title in UTIL_FEATURES:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (ds_name, ds) in zip(axes, [("D1 (Historical)", d1), ("D2 (Current)", d2)]):
        # Cap at 99th pct to avoid extreme outliers squashing the plot
        cap = ds[col].quantile(0.99)
        plot_data = ds.copy()
        plot_data[col] = plot_data[col].clip(upper=cap)
        sns.boxplot(data=plot_data, x="label", y=col, palette=PALETTE, ax=ax,
                    order=[0, 1], width=0.5, flierprops={"marker": ".", "markersize": 3})
        ax.set_title(ds_name, fontweight="bold")
        ax.set_xlabel("Label")
        ax.set_ylabel(f"{title} (capped at 99th pct)")
        ax.set_xticklabels(["label=0", "label=1"])
    fig.suptitle(f"{title} by Label", fontsize=14, fontweight="bold")
    plt.tight_layout()
    savefig(f"eda_utilization_{col}.png")

# ─────────────────────────────────────────────────────────────────────────────
# 5. CORRELATION HEATMAP (D1 train, top 30 features correlated with label)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] Correlation Heatmap")

corr_df = X_train_d1.copy()
corr_df["label"] = y_train_d1

full_corr = corr_df.corr()
label_corr = full_corr["label"].drop("label").abs().sort_values(ascending=False)
top30 = label_corr.head(30).index.tolist()
top30_with_label = top30 + ["label"]

heatmap_data = corr_df[top30_with_label].corr()

fig, ax = plt.subplots(figsize=(16, 14))
mask = np.zeros_like(heatmap_data, dtype=bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(
    heatmap_data, mask=mask, ax=ax,
    cmap="RdBu_r", center=0, vmin=-1, vmax=1,
    annot=True, fmt=".2f", annot_kws={"size": 7},
    linewidths=0.3, linecolor="white",
    cbar_kws={"shrink": 0.7, "label": "Pearson r"}
)
ax.set_title("Correlation Heatmap — Top 30 Features Correlated with Label\n(D1 Train Set)",
             fontsize=13, fontweight="bold")
# Shorten tick labels
short_labels = [t.get_text().replace("obs_", "").replace("__Mass_volume__in_Serum_or_Plasma", "")
                             .replace("__volume__in_Blood_by_Automated_count", "")
                             .replace("__Entitic_", "_")[:35]
                for t in ax.get_xticklabels()]
ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(short_labels, rotation=0, fontsize=8)
plt.tight_layout()
savefig("eda_correlation_heatmap_d1.png")

print(f"  Top 10 features correlated with label (D1):")
for feat, val in label_corr.head(10).items():
    print(f"    {val:.3f}  {feat}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. DATASET DRIFT ANALYSIS (D1 train vs D2 train)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] Dataset Drift Analysis")

# Identify top 10 numeric features by absolute correlation with label in D1
numeric_features = [c for c in feature_names if c != "label"]
top10_drift = label_corr.head(10).index.tolist()

print("\n  Feature mean/std comparison (D1 train vs D2 train):")
print(f"  {'Feature':<55} {'D1 mean':>10} {'D1 std':>10} {'D2 mean':>10} {'D2 std':>10}")
print("  " + "-" * 100)

drift_rows = []
for feat in top10_drift:
    d1_vals = X_train_d1[feat].dropna()
    d2_vals = X_train_d2[feat].dropna()
    row = {
        "Feature": feat,
        "D1 mean": round(d1_vals.mean(), 3),
        "D1 std":  round(d1_vals.std(), 3),
        "D2 mean": round(d2_vals.mean(), 3),
        "D2 std":  round(d2_vals.std(), 3),
    }
    drift_rows.append(row)
    print(f"  {feat:<55} {row['D1 mean']:>10} {row['D1 std']:>10} "
          f"{row['D2 mean']:>10} {row['D2 std']:>10}")

drift_table = pd.DataFrame(drift_rows)
drift_table.to_csv(os.path.join(EDA_DIR, "eda_drift_stats.csv"), index=False)
SAVED.append(os.path.join(EDA_DIR, "eda_drift_stats.csv"))
print("  saved: eda_drift_stats.csv")

# KDE plots — top 10 features
fig, axes = plt.subplots(2, 5, figsize=(22, 9))
axes = axes.flatten()
for ax, feat in zip(axes, top10_drift):
    d1_vals = X_train_d1[feat].dropna()
    d2_vals = X_train_d2[feat].dropna()
    d1_vals.plot.kde(ax=ax, color=C0, linewidth=2, label="D1 train")
    d2_vals.plot.kde(ax=ax, color=C1, linewidth=2, label="D2 train")
    short = feat.replace("obs_", "")[:30]
    ax.set_title(short, fontsize=9, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
fig.suptitle("Dataset Drift — Top 10 Features: D1 Train vs D2 Train KDE",
             fontsize=14, fontweight="bold")
plt.tight_layout()
savefig("eda_drift_kde_top10.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7. MISSING VALUE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7] Missing Value Analysis")

# The 132 dropped columns — reconstruct from the full pre-drop matrix
# We know from pipeline: obs vars with >50% missing + 3 allergy cols
# Rebuild missingness from raw observations (representative subset)
raw_obs_df = pd.read_csv("data/observations.csv", on_bad_lines="skip")
raw_obs_df["VALUE_NUMERIC"] = pd.to_numeric(raw_obs_df["VALUE"], errors="coerce")
total_patients = 2823

obs_desc_counts = raw_obs_df[raw_obs_df["VALUE_NUMERIC"].notna()]\
    .groupby("DESCRIPTION")["PATIENT"].nunique()
obs_miss_pct = (1 - obs_desc_counts / total_patients) * 100
obs_miss_pct = obs_miss_pct.sort_values(ascending=False)

# Top 20 most-missing numeric observation types
top20_miss = obs_miss_pct.head(20)

fig, ax = plt.subplots(figsize=(14, 7))
bars = ax.barh(range(len(top20_miss)), top20_miss.values, color="#E07070", edgecolor="white")
ax.set_yticks(range(len(top20_miss)))
short_names = [n.replace(" [Mass/volume] in Serum or Plasma", "")
                .replace(" [Moles/volume] in Serum or Plasma", "")
                .replace(" by Automated count", "")[:55]
               for n in top20_miss.index]
ax.set_yticklabels(short_names, fontsize=9)
ax.invert_yaxis()
ax.axvline(50, color="red", linestyle="--", linewidth=1.5, label="50% threshold")
ax.set_xlabel("Missingness % (across all 2823 patients)")
ax.set_title("Top 20 Most-Missing Observation Types (Pre-Imputation)",
             fontsize=13, fontweight="bold")
ax.legend()
for bar, val in zip(bars, top20_miss.values):
    ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{val:.0f}%", va="center", fontsize=8)
plt.tight_layout()
savefig("eda_missing_values_top20.png")

# Also show the 3 allergy columns that were dropped
print(f"  Observation types with >50% missing: "
      f"{(obs_miss_pct > 50).sum()} of {len(obs_miss_pct)}")
print(f"  Allergy features also dropped: total_allergies, "
      f"unique_allergy_types, unique_allergy_categories (82% missing)")

# ─────────────────────────────────────────────────────────────────────────────
# 8. SUMMARY STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8] Summary Statistics")

stats_d1 = d1.describe().T
stats_d1.to_csv(os.path.join(EDA_DIR, "summary_stats_d1.csv"))
SAVED.append(os.path.join(EDA_DIR, "summary_stats_d1.csv"))
print("  saved: summary_stats_d1.csv")

stats_d2 = d2.describe().T
stats_d2.to_csv(os.path.join(EDA_DIR, "summary_stats_d2.csv"))
SAVED.append(os.path.join(EDA_DIR, "summary_stats_d2.csv"))
print("  saved: summary_stats_d2.csv")

# ─────────────────────────────────────────────────────────────────────────────
# BONUS: Label-stratified feature means table for clinical features
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Bonus] Label-stratified clinical means (D1)")
clinical_cols = list(CLINICAL_FEATURES.values()) + [
    "total_encounters", "total_medications", "total_procedures",
    "total_claims", "age", "INCOME"
]
clinical_cols = [c for c in clinical_cols if c in d1.columns]
strat_means = d1.groupby("label")[clinical_cols].mean().T
strat_means.columns = ["label=0 mean", "label=1 mean"]
strat_means["delta (1-0)"] = strat_means["label=1 mean"] - strat_means["label=0 mean"]
strat_means["delta %"] = (strat_means["delta (1-0)"] / strat_means["label=0 mean"].replace(0, np.nan) * 100).round(1)
strat_means = strat_means.round(3)
print(strat_means.to_string())
strat_means.to_csv(os.path.join(EDA_DIR, "eda_clinical_stratified_means_d1.csv"))
SAVED.append(os.path.join(EDA_DIR, "eda_clinical_stratified_means_d1.csv"))
print("  saved: eda_clinical_stratified_means_d1.csv")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL FILE LIST
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("ALL SAVED FILES:")
print("="*60)
for path in sorted(SAVED):
    size = os.path.getsize(path) / 1024
    print(f"  {os.path.basename(path):<50}  {size:>8.1f} KB")
print(f"\nTotal: {len(SAVED)} files saved to {EDA_DIR}/")
