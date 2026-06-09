"""
retrain_combined.py
───────────────────
Jalankan script ini SEKALI untuk generate ulang model dari dataset combined.
Output: models/flood_classifier.joblib + models/flood_depth_regressor.joblib

Cara pakai:
    cd Aol_softeng_v3
    pip install -r mvp/requirements_mvp.txt
    python sourcecode/retrain_combined.py

Model akan disimpan ke folder models/ secara otomatis.
"""

import sys, os
from pathlib import Path

# Resolve paths relative to project root (Aol_softeng_v3/)
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH  = ROOT / "dataset" / "jakarta_flood_combined.xlsx"
MODEL_DIR  = ROOT / "models"
CLF_OUT    = MODEL_DIR / "flood_classifier.joblib"
REG_OUT    = MODEL_DIR / "flood_depth_regressor.joblib"

print(f"Project root : {ROOT}")
print(f"Dataset      : {DATA_PATH}")
print(f"Model output : {MODEL_DIR}")
print()

if not DATA_PATH.exists():
    sys.exit(f"ERROR: Dataset tidak ditemukan di {DATA_PATH}")

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

# ── Feature engineering ───────────────────────────────────────────────────────
def add_time_features(df):
    ts = pd.to_datetime(df["timestamp"])
    df = df.copy()
    df["hour_sin"]   = np.sin(2*np.pi*ts.dt.hour/24.0)
    df["hour_cos"]   = np.cos(2*np.pi*ts.dt.hour/24.0)
    df["doy_sin"]    = np.sin(2*np.pi*ts.dt.dayofyear/365.25)
    df["doy_cos"]    = np.cos(2*np.pi*ts.dt.dayofyear/365.25)
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    return df

def add_lag_roll_features(df, max_lag=6):
    df = df.copy().sort_values(["wilayah","kelurahan","timestamp"]).reset_index(drop=True)
    base = ["rain_mm_1h","rain_mm_3h","rain_mm_24h","river_level_cm","tide_level_cm",
            "soil_moisture","drainage_blockage_idx","temp_c","humidity_pct","wind_mps","pump_active"]
    grp   = ["wilayah","kelurahan"]
    combo = df["wilayah"] + "||" + df["kelurahan"]
    for col in base:
        for k in range(1, max_lag+1):
            df[f"{col}_lag{k}"] = df.groupby(grp)[col].shift(k)
    for col in ["rain_mm_1h","river_level_cm","tide_level_cm"]:
        shifted = df.groupby(grp)[col].shift(1)
        df[f"{col}_roll3_sum"]  = shifted.groupby(combo).transform(lambda x: x.rolling(3,  min_periods=3).sum())
        df[f"{col}_roll6_sum"]  = shifted.groupby(combo).transform(lambda x: x.rolling(6,  min_periods=6).sum())
        df[f"{col}_roll24_sum"] = shifted.groupby(combo).transform(lambda x: x.rolling(24, min_periods=24).sum())
        df[f"{col}_roll6_max"]  = shifted.groupby(combo).transform(lambda x: x.rolling(6,  min_periods=6).max())
    df["river_level_delta1"] = df.groupby(grp)["river_level_cm"].diff()
    df["tide_level_delta1"]  = df.groupby(grp)["tide_level_cm"].diff()
    return df

# ── Load & prepare ────────────────────────────────────────────────────────────
MAX_LAG = 6

print("Loading dataset...")
df = pd.read_excel(DATA_PATH, sheet_name="data")
df["timestamp"] = pd.to_datetime(df["timestamp"])
print(f"  Rows loaded: {len(df):,}")
print(f"  Wilayah    : {sorted(df['wilayah'].unique())}")
print(f"  Kelurahan  : {df['kelurahan'].nunique()} unique")

print("\nEncoding area labels...")
le_wilayah   = LabelEncoder().fit(df["wilayah"])
le_kelurahan = LabelEncoder().fit(df["kelurahan"])
df["wilayah_enc"]   = le_wilayah.transform(df["wilayah"])
df["kelurahan_enc"] = le_kelurahan.transform(df["kelurahan"])

print("Adding time & lag features (this may take ~1–2 min)...")
df = add_time_features(df)
df = add_lag_roll_features(df, max_lag=MAX_LAG)
df = df.dropna().reset_index(drop=True)
print(f"  Rows after dropna: {len(df):,}")

# Train/test split by time
cutoff = pd.Timestamp("2024-11-01")
train  = df[df["timestamp"] < cutoff]
print(f"  Train rows: {len(train):,} | Test rows: {len(df)-len(train):,}")

DROP = {"timestamp","wilayah","kota_code","kelurahan","flood_now","flood_depth_cm","flood_next_6h"}
feature_cols = [c for c in df.columns if c not in DROP]

X_train  = train[feature_cols].astype(float)
y_clf    = train["flood_next_6h"].astype(int)
y_dep    = train["flood_depth_cm"].astype(float)

# ── Train classifier ──────────────────────────────────────────────────────────
print("\nTraining flood classifier...")
clf = HistGradientBoostingClassifier(
    learning_rate=0.08, max_depth=6, max_iter=150,
    l2_regularization=0.1, random_state=42, class_weight="balanced"
)
clf.fit(X_train, y_clf)
print("  Classifier done.")

# ── Train depth regressor ─────────────────────────────────────────────────────
print("Training depth regressor (flood rows only)...")
mask = y_dep > 0
reg = HistGradientBoostingRegressor(
    learning_rate=0.08, max_depth=6, max_iter=150,
    l2_regularization=0.1, random_state=42
)
reg.fit(X_train[mask], y_dep[mask])
print("  Regressor done.")

# ── Save ──────────────────────────────────────────────────────────────────────
clf_bundle = {
    "model":             clf,
    "features":          feature_cols,
    "threshold":         0.45,
    "max_lag":           MAX_LAG,
    "le_wilayah":        le_wilayah,
    "le_kelurahan":      le_kelurahan,
    "wilayah_classes":   le_wilayah.classes_.tolist(),
    "kelurahan_classes": le_kelurahan.classes_.tolist(),
}
reg_bundle = {
    "model":        reg,
    "features":     feature_cols,
    "max_lag":      MAX_LAG,
    "le_wilayah":   le_wilayah,
    "le_kelurahan": le_kelurahan,
}

joblib.dump(clf_bundle, CLF_OUT)
joblib.dump(reg_bundle, REG_OUT)

print(f"\n✅ Saved:")
print(f"   {CLF_OUT}")
print(f"   {REG_OUT}")
print("\nSekarang jalankan app-nya:")
print("   cd mvp")
print("   streamlit run app_streamlit.py")
