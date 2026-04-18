"""
EHR Data Pipeline — Data Architect
Produces clean train/test splits for Dataset 1 (Historical) and Dataset 2 (Current).
"""

import os
import re
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings("ignore")

OUTPUT_DIR = "data/processed"
TEMPORAL_CUTOFF = pd.Timestamp("2020-01-01")
MIN_OBS_PATIENT_FRACTION = 0.05  # keep observation types seen in >=5% of patients
MISSING_THRESHOLD = 0.50         # drop columns with >50% missing
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD AND CLEAN ALL 17 TABLES
# ─────────────────────────────────────────────────────────────────────────────
print("\nstep 1 — loading and cleaning all 17 tables")

patients = pd.read_csv("data/patients.csv", on_bad_lines="skip")
for col in ["BIRTHDATE", "DEATHDATE"]:
    patients[col] = pd.to_datetime(patients[col], errors="coerce")
print(f"  patients: {patients.shape}")

encounters = pd.read_csv("data/encounters.csv", on_bad_lines="skip")
for col in ["START", "STOP"]:
    encounters[col] = pd.to_datetime(encounters[col], errors="coerce", utc=True).dt.tz_localize(None)
print(f"  encounters: {encounters.shape}")

# conditions has DD-MM-YYYY dates and unnamed garbage columns
conditions = pd.read_csv("data/conditions.csv", on_bad_lines="skip")
conditions.drop(columns=[c for c in conditions.columns if c.startswith("Unnamed")], inplace=True)
for col in ["START", "STOP"]:
    conditions[col] = pd.to_datetime(conditions[col], dayfirst=True, errors="coerce")
print(f"  conditions: {conditions.shape}")

observations = pd.read_csv("data/observations.csv", on_bad_lines="skip")
observations["DATE"] = pd.to_datetime(observations["DATE"], errors="coerce")
observations["VALUE_NUMERIC"] = pd.to_numeric(observations["VALUE"], errors="coerce")
print(f"  observations: {observations.shape}")

medications = pd.read_csv("data/medications.csv", on_bad_lines="skip")
for col in ["START", "STOP"]:
    medications[col] = pd.to_datetime(medications[col], errors="coerce")
print(f"  medications: {medications.shape}")

procedures = pd.read_csv("data/procedures.csv", on_bad_lines="skip")
for col in ["START", "STOP"]:
    procedures[col] = pd.to_datetime(procedures[col], errors="coerce")
print(f"  procedures: {procedures.shape}")

immunizations = pd.read_csv("data/immunizations.csv", on_bad_lines="skip")
immunizations["DATE"] = pd.to_datetime(immunizations["DATE"], errors="coerce")
print(f"  immunizations: {immunizations.shape}")

allergies = pd.read_csv("data/allergies.csv", on_bad_lines="skip")
allergies["START"] = pd.to_datetime(allergies["START"], errors="coerce")
print(f"  allergies: {allergies.shape}")

careplans = pd.read_csv("data/careplans.csv", on_bad_lines="skip")
for col in ["START", "STOP"]:
    careplans[col] = pd.to_datetime(careplans[col], errors="coerce")
print(f"  careplans: {careplans.shape}")

imaging_studies = pd.read_csv("data/imaging_studies.csv", on_bad_lines="skip")
imaging_studies["DATE"] = pd.to_datetime(imaging_studies["DATE"], errors="coerce")
print(f"  imaging_studies: {imaging_studies.shape}")

devices = pd.read_csv("data/devices.csv", on_bad_lines="skip")
for col in ["START", "STOP"]:
    devices[col] = pd.to_datetime(devices[col], errors="coerce")
print(f"  devices: {devices.shape}")

supplies = pd.read_csv("data/supplies.csv", on_bad_lines="skip")
supplies["DATE"] = pd.to_datetime(supplies["DATE"], errors="coerce")
print(f"  supplies: {supplies.shape}")

payer_transitions = pd.read_csv("data/payer_transitions.csv", on_bad_lines="skip")
for col in ["START_DATE", "END_DATE"]:
    payer_transitions[col] = pd.to_datetime(payer_transitions[col], errors="coerce")
print(f"  payer_transitions: {payer_transitions.shape}")

claims = pd.read_csv("data/claims.csv", on_bad_lines="skip")
for col in ["SERVICEDATE", "CURRENTILLNESSDATE"]:
    claims[col] = pd.to_datetime(claims[col], errors="coerce")
print(f"  claims: {claims.shape}")

# claims_transactions is large — only load patient-relevant cols
CT_KEEP = ["ID", "CLAIMID", "PATIENTID", "TYPE", "AMOUNT",
           "FROMDATE", "TODATE", "PAYMENTS", "ADJUSTMENTS",
           "TRANSFERS", "OUTSTANDING"]
claims_transactions = pd.read_csv(
    "data/claims_transactions.csv",
    usecols=CT_KEEP,
    on_bad_lines="skip"
)
for col in ["FROMDATE", "TODATE"]:
    claims_transactions[col] = pd.to_datetime(claims_transactions[col], errors="coerce")
print(f"  claims_trans: {claims_transactions.shape}  (trimmed cols)")

# orgs/providers are reference only — not used in features
organizations = pd.read_csv("data/organizations.csv", on_bad_lines="skip")
providers = pd.read_csv("data/providers.csv", on_bad_lines="skip")
print(f"  organizations: {organizations.shape}")
print(f"  providers: {providers.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — BUILD UNIFIED PATIENT-LEVEL FEATURE MATRIX
# ─────────────────────────────────────────────────────────────────────────────
print("\nstep 2 — building patient-level feature matrix")

all_patient_ids = patients["Id"].unique()
feat = pd.DataFrame({"PATIENT": all_patient_ids})

# ── 2a. patients.csv demographics ──────────────────────────────────────────
ref_date = pd.Timestamp("2020-01-01")
p = patients[["Id", "BIRTHDATE", "DEATHDATE", "GENDER", "RACE",
              "ETHNICITY", "INCOME", "HEALTHCARE_COVERAGE", "MARITAL"]].copy()
p = p.rename(columns={"Id": "PATIENT"})
p["age"] = ((ref_date - p["BIRTHDATE"]).dt.days / 365.25).round(1)
p["is_deceased"] = p["DEATHDATE"].notna().astype(int)

for col in ["GENDER", "RACE", "ETHNICITY", "MARITAL"]:
    le = LabelEncoder()
    p[col] = le.fit_transform(p[col].fillna("Unknown").astype(str))

p = p.drop(columns=["BIRTHDATE", "DEATHDATE"])
feat = feat.merge(p, on="PATIENT", how="left")
print(f"  After demographics      : {feat.shape}")

# ── 2b. encounters.csv ──────────────────────────────────────────────────────
enc_agg = encounters.groupby("PATIENT").agg(
    total_encounters=("Id", "count"),
    unique_encounter_types=("ENCOUNTERCLASS", "nunique"),
    avg_base_encounter_cost=("BASE_ENCOUNTER_COST", "mean"),
    total_claim_cost=("TOTAL_CLAIM_COST", "sum"),
    avg_payer_coverage=("PAYER_COVERAGE", "mean"),
).reset_index()
feat = feat.merge(enc_agg, on="PATIENT", how="left")
print(f"  After encounters        : {feat.shape}")

# ── 2c. observations.csv — pivot numeric types ──────────────────────────────
obs_numeric = observations[observations["VALUE_NUMERIC"].notna()].copy()

# Determine which observation descriptions are common enough (≥5% of patients)
total_patients = len(all_patient_ids)
obs_desc_counts = obs_numeric.groupby("DESCRIPTION")["PATIENT"].nunique()
min_patients = int(np.ceil(MIN_OBS_PATIENT_FRACTION * total_patients))
common_obs = obs_desc_counts[obs_desc_counts >= min_patients].index.tolist()
obs_numeric = obs_numeric[obs_numeric["DESCRIPTION"].isin(common_obs)]
print(f"  Observation types ≥5% coverage: {len(common_obs)} of "
      f"{obs_desc_counts.shape[0]} total")

obs_mean = (
    obs_numeric.groupby(["PATIENT", "DESCRIPTION"])["VALUE_NUMERIC"]
    .mean()
    .unstack("DESCRIPTION")
)
obs_var = (
    obs_numeric.groupby(["PATIENT", "DESCRIPTION"])["VALUE_NUMERIC"]
    .var()
    .unstack("DESCRIPTION")
)

def sanitize(name):
    """Clean description string for use as a column name."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", str(name)).strip("_")

obs_mean.columns = [f"obs_{sanitize(c)}_mean" for c in obs_mean.columns]
obs_var.columns  = [f"obs_{sanitize(c)}_var"  for c in obs_var.columns]

obs_pivot = obs_mean.join(obs_var, how="outer").reset_index()
feat = feat.merge(obs_pivot, on="PATIENT", how="left")
print(f"  After observations      : {feat.shape}")

# ── 2d. medications.csv ──────────────────────────────────────────────────────
med_agg = medications.groupby("PATIENT").agg(
    total_medications=("DESCRIPTION", "count"),
    unique_medications=("DESCRIPTION", "nunique"),
    avg_medication_cost=("BASE_COST", "mean"),
    total_dispenses=("DISPENSES", "sum"),
).reset_index()
feat = feat.merge(med_agg, on="PATIENT", how="left")
print(f"  After medications       : {feat.shape}")

# ── 2e. procedures.csv ──────────────────────────────────────────────────────
proc_agg = procedures.groupby("PATIENT").agg(
    total_procedures=("DESCRIPTION", "count"),
    unique_procedures=("DESCRIPTION", "nunique"),
    avg_procedure_cost=("BASE_COST", "mean"),
).reset_index()
feat = feat.merge(proc_agg, on="PATIENT", how="left")
print(f"  After procedures        : {feat.shape}")

# ── 2f. immunizations.csv ────────────────────────────────────────────────────
imm_agg = immunizations.groupby("PATIENT").agg(
    total_immunizations=("DESCRIPTION", "count"),
    unique_vaccines=("DESCRIPTION", "nunique"),
).reset_index()
feat = feat.merge(imm_agg, on="PATIENT", how="left")
print(f"  After immunizations     : {feat.shape}")

# ── 2g. allergies.csv ────────────────────────────────────────────────────────
alg_agg = allergies.groupby("PATIENT").agg(
    total_allergies=("DESCRIPTION", "count"),
    unique_allergy_types=("TYPE", "nunique"),
    unique_allergy_categories=("CATEGORY", "nunique"),
).reset_index()
feat = feat.merge(alg_agg, on="PATIENT", how="left")
print(f"  After allergies         : {feat.shape}")

# ── 2h. careplans.csv ────────────────────────────────────────────────────────
cp_agg = careplans.groupby("PATIENT").agg(
    total_careplans=("DESCRIPTION", "count"),
    unique_careplan_reasons=("REASONDESCRIPTION", "nunique"),
).reset_index()
feat = feat.merge(cp_agg, on="PATIENT", how="left")
print(f"  After careplans         : {feat.shape}")

# ── 2i. imaging_studies.csv ──────────────────────────────────────────────────
img_agg = imaging_studies.groupby("PATIENT").agg(
    total_imaging_studies=("Id", "count"),
    unique_modalities=("MODALITY_DESCRIPTION", "nunique"),
    unique_body_sites=("BODYSITE_DESCRIPTION", "nunique"),
).reset_index()
feat = feat.merge(img_agg, on="PATIENT", how="left")
print(f"  After imaging studies   : {feat.shape}")

# ── 2j. devices.csv ──────────────────────────────────────────────────────────
dev_agg = devices.groupby("PATIENT").agg(
    total_devices=("DESCRIPTION", "count"),
    unique_device_types=("DESCRIPTION", "nunique"),
).reset_index()
feat = feat.merge(dev_agg, on="PATIENT", how="left")
print(f"  After devices           : {feat.shape}")

# ── 2k. supplies.csv ─────────────────────────────────────────────────────────
sup_agg = supplies.groupby("PATIENT").agg(
    total_supplies=("DESCRIPTION", "count"),
    unique_supply_types=("DESCRIPTION", "nunique"),
).reset_index()
feat = feat.merge(sup_agg, on="PATIENT", how="left")
print(f"  After supplies          : {feat.shape}")

# ── 2l. payer_transitions.csv ────────────────────────────────────────────────
pay_agg = payer_transitions.groupby("PATIENT").agg(
    total_payer_transitions=("PAYER", "count"),
    unique_payers=("PAYER", "nunique"),
).reset_index()
feat = feat.merge(pay_agg, on="PATIENT", how="left")
print(f"  After payer transitions : {feat.shape}")

# ── 2m. claims.csv (uses PATIENTID) ──────────────────────────────────────────
clm_agg = claims.groupby("PATIENTID").agg(
    total_claims=("Id", "count"),
).reset_index().rename(columns={"PATIENTID": "PATIENT"})
feat = feat.merge(clm_agg, on="PATIENT", how="left")
print(f"  After claims            : {feat.shape}")

# ── 2n. claims_transactions.csv (uses PATIENTID) ─────────────────────────────
ct_agg = claims_transactions.groupby("PATIENTID").agg(
    total_transactions=("ID", "count"),
    total_transaction_amount=("AMOUNT", "sum"),
    unique_transaction_types=("TYPE", "nunique"),
).reset_index().rename(columns={"PATIENTID": "PATIENT"})
feat = feat.merge(ct_agg, on="PATIENT", how="left")
print(f"  After claims_trans      : {feat.shape}")

print(f"\nfeature matrix shape: {feat.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — ASSIGN BINARY LABELS
# ─────────────────────────────────────────────────────────────────────────────
print("\nstep 3 — assigning binary labels")

labelled = conditions[
    conditions["DESCRIPTION"].str.contains(r"\(disorder\)|\(finding\)", na=False)
]["PATIENT"].unique()

feat["label"] = feat["PATIENT"].isin(labelled).astype(int)
print(f"  Patients with label=1 : {feat['label'].sum()} "
      f"({feat['label'].mean()*100:.1f}%)")
print(f"  Patients with label=0 : {(feat['label']==0).sum()} "
      f"({(feat['label']==0).mean()*100:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — TEMPORAL SPLIT
# ─────────────────────────────────────────────────────────────────────────────
print("\nstep 4 — temporal split")

enc_dates = encounters.groupby("PATIENT")["START"].agg(
    earliest_enc="min", latest_enc="max"
).reset_index()

feat = feat.merge(enc_dates, on="PATIENT", how="left")

# Dataset 1 — Historical: earliest encounter before 2020-01-01
d1_mask = feat["earliest_enc"] < TEMPORAL_CUTOFF
# Dataset 2 — Current: at least one encounter on or after 2020-01-01
d2_mask = feat["latest_enc"] >= TEMPORAL_CUTOFF

print(f"  D1 (historical, earliest enc < 2020): {d1_mask.sum()} patients")
print(f"  D2 (current, latest enc >= 2020): {d2_mask.sum()} patients")
print(f"  overlap: {(d1_mask & d2_mask).sum()} patients")

d1 = feat[d1_mask].copy()
d2 = feat[d2_mask].copy()

# Drop the helper date columns — not model features
d1 = d1.drop(columns=["earliest_enc", "latest_enc"])
d2 = d2.drop(columns=["earliest_enc", "latest_enc"])


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — PREPROCESS EACH DATASET (drop sparse cols, impute, scale, split)
# ─────────────────────────────────────────────────────────────────────────────
print("\nstep 5 — preprocessing, imputation, scaling, train/test split")

FEATURE_COLS_CACHE = None  # will be set on first dataset

def preprocess(df, dataset_name):
    global FEATURE_COLS_CACHE

    df = df.copy()
    y = df["label"].values
    X = df.drop(columns=["PATIENT", "label"])

    # ── drop columns with >50% missing ──────────────────────────────────────
    missing_frac = X.isnull().mean()
    dropped_cols = missing_frac[missing_frac > MISSING_THRESHOLD].index.tolist()
    X = X.drop(columns=dropped_cols)
    print(f"\n  [{dataset_name}] Dropped {len(dropped_cols)} cols (>50% missing):")
    if dropped_cols:
        for c in dropped_cols:
            print(f"    - {c}  ({missing_frac[c]*100:.0f}% missing)")
    else:
        print("    (none)")

    # cache feature names from first dataset; align second to match
    if FEATURE_COLS_CACHE is None:
        FEATURE_COLS_CACHE = X.columns.tolist()
    else:
        for c in FEATURE_COLS_CACHE:
            if c not in X.columns:
                X[c] = 0.0
        X = X[FEATURE_COLS_CACHE]

    # ── impute: median for numeric, mode for categorical ────────────────────
    # All columns at this stage are numeric (label encoding was applied earlier)
    for col in X.columns:
        if X[col].isnull().any():
            if X[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(X[col].mode()[0])

    # ── train/test split ─────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"  [{dataset_name}] X_train={X_train.shape}, X_test={X_test.shape}")
    print(f"  [{dataset_name}] y_train label dist: "
          f"0={np.sum(y_train==0)}, 1={np.sum(y_train==1)}")
    print(f"  [{dataset_name}] y_test  label dist: "
          f"0={np.sum(y_test==0)},  1={np.sum(y_test==1)}")

    # ── standard scale numeric features ─────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, dropped_cols

X_train_d1, X_test_d1, y_train_d1, y_test_d1, scaler_d1, dropped_d1 = preprocess(d1, "D1")
X_train_d2, X_test_d2, y_train_d2, y_test_d2, scaler_d2, dropped_d2 = preprocess(d2, "D2")



# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — SAVE ALL OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────
print("\nstep 6 — saving outputs to data/processed/")

def save_pkl(obj, name):
    path = os.path.join(OUTPUT_DIR, name)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  saved {name}")

save_pkl(X_train_d1, "X_train_d1.pkl")
save_pkl(X_test_d1,  "X_test_d1.pkl")
save_pkl(y_train_d1, "y_train_d1.pkl")
save_pkl(y_test_d1,  "y_test_d1.pkl")

save_pkl(X_train_d2, "X_train_d2.pkl")
save_pkl(X_test_d2,  "X_test_d2.pkl")
save_pkl(y_train_d2, "y_train_d2.pkl")
save_pkl(y_test_d2,  "y_test_d2.pkl")

save_pkl(scaler_d1, "scaler_d1.pkl")
save_pkl(scaler_d2, "scaler_d2.pkl")
save_pkl(FEATURE_COLS_CACHE, "feature_names.pkl")

for df, name in [
    (X_train_d1, "X_train_d1.csv"), (X_test_d1, "X_test_d1.csv"),
    (X_train_d2, "X_train_d2.csv"), (X_test_d2, "X_test_d2.csv"),
]:
    path = os.path.join(OUTPUT_DIR, name)
    df.to_csv(path, index=False)
    print(f"  saved {name}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────
print("\nstep 7 — pipeline summary")

datasets = {
    "D1 train": (X_train_d1, y_train_d1),
    "D1 test" : (X_test_d1,  y_test_d1),
    "D2 train": (X_train_d2, y_train_d2),
    "D2 test" : (X_test_d2,  y_test_d2),
}

print(f"\n{'Split':<12} {'Shape':<25} {'label=0':>10} {'label=1':>10} "
      f"{'% pos':>8} {'Mem (MB)':>10}")
print("-" * 80)
for name, (X, y) in datasets.items():
    n0 = np.sum(y == 0)
    n1 = np.sum(y == 1)
    pct = n1 / len(y) * 100
    mem = X.memory_usage(deep=True).sum() / 1e6
    print(f"  {name:<10} {str(X.shape):<25} {n0:>10} {n1:>10} {pct:>7.1f}% {mem:>10.2f}")

print(f"\nfeature count: {len(FEATURE_COLS_CACHE)}")

print(f"\n{'─'*60}")
print("feature names:")
print(f"{'─'*60}")
for i, name in enumerate(FEATURE_COLS_CACHE, 1):
    print(f"  {i:>4}. {name}")

print(f"\n{'─'*60}")
print("dropped columns (>50% missing):")
print(f"{'─'*60}")
print(f"  Dataset 1 : {len(dropped_d1)} column(s)")
for c in dropped_d1:
    print(f"    - {c}")
print(f"  Dataset 2 : {len(dropped_d2)} column(s)")
for c in dropped_d2:
    print(f"    - {c}")

print(f"\n{'─'*60}")
print("output files (data/processed/):")
print(f"{'─'*60}")
for f in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
    print(f"  {f:<30} {size/1e6:>8.2f} MB")

print("\npipeline complete.")
