import streamlit as st
import pandas as pd
import numpy as np
import joblib, io
from datetime import timedelta

st.set_page_config(
    page_title="Jakarta FEWS",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [data-testid="stApp"] {
    background: #080c12 !important;
    color: #dce4f0;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stSidebar"] {
    background: #0c1018 !important;
    border-right: 1px solid #161f2e;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #b0bdd0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
h1 { font-size:1.65rem !important; font-weight:700; letter-spacing:-0.03em; }
[data-testid="metric-container"] {
    background:#0f1620; border:1px solid #1a2740;
    border-radius:12px; padding:16px 20px !important;
}
[data-testid="stMetricValue"] {
    font-family:'DM Mono',monospace !important;
    font-size:1.75rem !important; font-weight:500 !important;
}
[data-testid="stMetricLabel"] {
    font-size:0.68rem !important; color:#4a6080 !important;
    text-transform:uppercase; letter-spacing:.1em;
}
[data-testid="stMetricDelta"] { font-size:0.75rem !important; }
[data-baseweb="select"] > div {
    background:#0f1620 !important; border:1px solid #1a2740 !important;
    border-radius:8px !important;
}
[data-testid="stFileUploader"] {
    background:#0f1620; border:1.5px dashed #1a2740;
    border-radius:10px; padding:8px;
}
[data-testid="stDataFrame"] { border:1px solid #1a2740; border-radius:10px; overflow:hidden; }
[data-testid="stExpander"]  { border:1px solid #1a2740 !important; border-radius:10px; background:#0c1018; }
[data-testid="stDownloadButton"] > button {
    background:#0f1620 !important; border:1px solid #1a2740 !important;
    color:#dce4f0 !important; border-radius:8px !important;
    font-family:'DM Sans',sans-serif !important; font-weight:500 !important; font-size:.84rem !important;
}
hr { border-color:#161f2e !important; }

.pred-box {
    background:linear-gradient(135deg,#0c1624,#0f1c2e);
    border:1px solid #1a2d4a; border-radius:14px; padding:20px 26px; margin:8px 0 16px;
}
.ts-line {
    font-family:'DM Mono',monospace; font-size:.72rem;
    color:#4a6080; letter-spacing:.06em; margin-bottom:8px;
}
.ts-arrow { color:#2563eb; margin:0 6px; }
.area-pill {
    display:inline-block; background:#0e2040; color:#60a5fa;
    border:1px solid #1e3a6a; border-radius:6px;
    padding:2px 11px; font-size:.72rem; font-weight:600;
    letter-spacing:.06em; margin-bottom:12px;
}
.badge {
    padding:3px 12px; border-radius:20px; font-size:.82rem; font-weight:700;
    font-family:'DM Mono',monospace; display:inline-block;
}
.badge-ok     { background:#0d3321; color:#30d158; border:1px solid #1a6640; }
.badge-minor  { background:#2d2200; color:#ffd60a; border:1px solid #665500; }
.badge-mod    { background:#2d1800; color:#ff9500; border:1px solid #663300; }
.badge-high   { background:#2d0a0a; color:#ff453a; border:1px solid #661515; }
.badge-severe { background:#220028; color:#bf5af2; border:1px solid #551166; }
.sec {
    font-size:.66rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase;
    color:#2a3a50; margin:22px 0 8px; border-bottom:1px solid #161f2e; padding-bottom:5px;
}
/* Risk rank card */
.rank-card {
    background:#0f1620; border:1px solid #1a2740; border-radius:10px;
    padding:12px 16px; margin-bottom:8px; display:flex;
    align-items:center; gap:14px;
}
.rank-num {
    font-family:'DM Mono',monospace; font-size:1.3rem; font-weight:700;
    color:#334a65; width:28px; flex-shrink:0; text-align:center;
}
.rank-num.top { color:#ff453a; }
.rank-name { font-weight:600; font-size:.92rem; color:#dce4f0; }
.rank-sub  { font-size:.72rem; color:#4a6080; margin-top:2px; }
.rank-bar-wrap { flex:1; background:#161f2e; border-radius:4px; height:6px; overflow:hidden; }
.rank-bar { height:6px; border-radius:4px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
AREA_MAP = {
    "DKI Jakarta":      ["Semua Wilayah"],
    "Jakarta Utara":    ["Semua Wilayah","Kelapa Gading","Penjaringan","Pluit","Tanjung Priok","Pademangan"],
    "Jakarta Barat":    ["Semua Wilayah","Kemanggisan","Grogol","Tambora","Cengkareng","Kalideres"],
    "Jakarta Selatan":  ["Semua Wilayah","Kebayoran Baru","Mampang Prapatan","Tebet","Jagakarsa","Pesanggrahan"],
    "Jakarta Timur":    ["Semua Wilayah","Jatinegara","Kramat Jati","Cakung","Duren Sawit","Matraman"],
    "Jakarta Pusat":    ["Semua Wilayah","Menteng","Gambir","Senen","Kemayoran","Tanah Abang"],
}

# Bar color per risk level
def risk_color(pct):
    if pct >= 0.6:  return "#ff453a"
    if pct >= 0.4:  return "#ff9500"
    if pct >= 0.2:  return "#ffd60a"
    return "#30d158"

def severity_label(cm):
    if cm <= 0:  return "Tidak Banjir",    "badge-ok",     "✅"
    if cm < 10:  return "Minor (<10cm)",   "badge-minor",  "⚠️"
    if cm < 30:  return "Sedang (10–30cm)","badge-mod",    "🟠"
    if cm < 50:  return "Tinggi (30–50cm)","badge-high",   "🚨"
    return            "Parah (≥50cm)",      "badge-severe", "🔴"

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

@st.cache_data(show_spinner="⚙️ Menyiapkan fitur dataset...")
def prepare_combined(_df_raw, max_lag):
    df = _df_raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    needed = ["timestamp","wilayah","kelurahan","rain_mm_1h","river_level_cm","tide_level_cm",
              "soil_moisture","drainage_blockage_idx","temp_c","humidity_pct","wind_mps","pump_active"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom tidak ditemukan: {missing}")
    if "rain_mm_3h"  not in df.columns:
        df["rain_mm_3h"]  = df.groupby(["wilayah","kelurahan"])["rain_mm_1h"].transform(lambda x: x.rolling(3,  min_periods=1).sum())
    if "rain_mm_24h" not in df.columns:
        df["rain_mm_24h"] = df.groupby(["wilayah","kelurahan"])["rain_mm_1h"].transform(lambda x: x.rolling(24, min_periods=1).sum())
    for c in ["rain_mm_1h","rain_mm_3h","rain_mm_24h","river_level_cm","tide_level_cm",
              "soil_moisture","drainage_blockage_idx","temp_c","humidity_pct","wind_mps","pump_active"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = add_time_features(df)
    df = add_lag_roll_features(df, max_lag=max_lag)
    return df.dropna().reset_index(drop=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌧️ Jakarta FEWS")

    st.markdown('<div class="sec">① Classifier Model</div>', unsafe_allow_html=True)
    up_clf = st.file_uploader("flood_classifier.joblib", type=["joblib"], key="clf")
    if up_clf:
        st.success("✅ Classifier siap", icon=None)

    st.markdown('<div class="sec">② Depth Regressor</div>', unsafe_allow_html=True)
    up_reg = st.file_uploader("flood_depth_regressor.joblib", type=["joblib"], key="reg")
    if up_reg:
        st.success("✅ Regressor siap", icon=None)

    st.markdown('<div class="sec">③ Dataset</div>', unsafe_allow_html=True)
    up_data = st.file_uploader("jakarta_flood_combined.xlsx", type=["xlsx","csv"], key="data")
    if up_data:
        st.success("✅ Dataset siap", icon=None)

    st.markdown('<div class="sec">④ Pilih Area</div>', unsafe_allow_html=True)
    wilayah_choice   = st.selectbox("Wilayah",   list(AREA_MAP.keys()))
    kelurahan_choice = st.selectbox("Kelurahan", AREA_MAP[wilayah_choice])

# ── Header ────────────────────────────────────────────────────────────────────
area_label = wilayah_choice if kelurahan_choice == "Semua Wilayah" else f"{kelurahan_choice}, {wilayah_choice}"
st.markdown("# 🌧️ Jakarta Flood Early Warning System")
st.markdown(f'<div class="area-pill">📍 {area_label}</div>', unsafe_allow_html=True)
st.caption("Upload 2 model + 1 dataset → pilih area dari sidebar → prediksi banjir 6 jam ke depan")

# ── Gate ──────────────────────────────────────────────────────────────────────
missing_files = []
if not up_clf:  missing_files.append("**flood_classifier.joblib**")
if not up_reg:  missing_files.append("**flood_depth_regressor.joblib**")
if not up_data: missing_files.append("**jakarta_flood_combined.xlsx**")
if missing_files:
    st.info("👈 Upload file berikut dari sidebar:\n\n" + "\n\n".join(f"- {f}" for f in missing_files))
    st.stop()

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🤖 Loading model...")
def load_model_from_bytes(clf_bytes, reg_bytes):
    return joblib.load(io.BytesIO(clf_bytes)), joblib.load(io.BytesIO(reg_bytes))

clf_bundle, reg_bundle = load_model_from_bytes(up_clf.read(), up_reg.read())
clf       = clf_bundle["model"]
clf_feat  = clf_bundle["features"]
threshold = float(clf_bundle.get("threshold", 0.45))
max_lag   = int(clf_bundle.get("max_lag", 6))
le_wil    = clf_bundle["le_wilayah"]
le_kel    = clf_bundle["le_kelurahan"]
reg       = reg_bundle["model"]
reg_feat  = reg_bundle["features"]

st.markdown(
    f"<small>✅ Model loaded · threshold <code>{threshold:.3f}</code> · max_lag <code>{max_lag}</code></small>",
    unsafe_allow_html=True
)

# ── Load dataset ──────────────────────────────────────────────────────────────
with st.spinner("📂 Membaca dataset..."):
    name   = up_data.name.lower()
    df_raw = pd.read_csv(up_data) if name.endswith(".csv") else pd.read_excel(up_data, sheet_name="data")

try:
    df_full = prepare_combined(df_raw, max_lag)
except Exception as e:
    st.error(f"Data error: {e}")
    st.stop()

# Encode area
w_map = {c: i for i,c in enumerate(le_wil.classes_)}
k_map = {c: i for i,c in enumerate(le_kel.classes_)}
df_full = df_full.copy()
df_full["wilayah_enc"]   = df_full["wilayah"].map(w_map).fillna(0).astype(int)
df_full["kelurahan_enc"] = df_full["kelurahan"].map(k_map).fillna(0).astype(int)

def filter_area(df, wilayah, kelurahan):
    if wilayah == "DKI Jakarta":
        return df[df["kelurahan"] == "Semua Wilayah"].reset_index(drop=True)
    mask = df["wilayah"] == wilayah
    if kelurahan != "Semua Wilayah":
        mask &= df["kelurahan"] == kelurahan
    return df[mask].reset_index(drop=True)

df = filter_area(df_full, wilayah_choice, kelurahan_choice)
if len(df) == 0:
    st.warning(f"Tidak ada data untuk area **{area_label}**.")
    st.stop()

# ── TAB LAYOUT ────────────────────────────────────────────────────────────────
tab_pred, tab_history, tab_ranking = st.tabs([
    "🔮 Prediksi",
    "📊 Riwayat Banjir",
    "🏆 Ranking Risiko",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – PREDIKSI
# ══════════════════════════════════════════════════════════════════════════════
with tab_pred:
    st.markdown('<div class="sec">Prediksi Banjir</div>', unsafe_allow_html=True)

    idx        = st.slider("Pilih baris / timestamp (default = paling baru)", 0, len(df)-1, len(df)-1)
    row        = df.iloc[[idx]]
    prob       = float(clf.predict_proba(row[clf_feat].astype(float))[:,1][0])
    pred_flood = int(prob >= threshold)
    depth_est  = float(np.maximum(reg.predict(row[reg_feat].astype(float))[0], 0.0)) if pred_flood else 0.0
    sev_label, sev_cls, sev_icon = severity_label(depth_est)

    ts_data = pd.to_datetime(df["timestamp"].iloc[idx])
    ts_pred = ts_data + timedelta(hours=6)

    st.markdown(f"""
    <div class="pred-box">
      <div class="ts-line">
        📅 Data &nbsp;<strong>{ts_data.strftime('%A, %d %B %Y · %H:%M')}</strong>
        <span class="ts-arrow">→</span>
        🕐 Prediksi &nbsp;<strong>{ts_pred.strftime('%A, %d %B %Y · %H:%M')}</strong>
      </div>
      <div style="font-size:.78rem;color:#334a65;margin-bottom:12px;">
        Prediksi kondisi banjir 6 jam ke depan · <strong>{area_label}</strong>
      </div>
      <span class="badge {sev_cls}">{sev_icon} {sev_label}</span>
      &nbsp;
      <span style="font-family:'DM Mono',monospace;font-size:.82rem;color:#4a8fcc;">
        prob = {prob:.1%} &nbsp;|&nbsp; est. depth = {depth_est:.1f} cm
      </span>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    delta_label = "⬆ Waspada" if prob > 0.6 else ("↔ Normal" if prob > 0.3 else "⬇ Aman")
    delta_color = "inverse" if prob > 0.6 else "normal"
    c1.metric("🌊 Prob Banjir (6h)", f"{prob:.1%}", delta=delta_label, delta_color=delta_color)
    c2.metric("🚨 Status",           "BANJIR" if pred_flood else "AMAN")
    c3.metric("📏 Est. Kedalaman",   f"{depth_est:.1f} cm")
    c4.metric("🕐 Waktu Prediksi",   ts_pred.strftime("%d %b %H:%M"))

    with st.expander("📡 Data sensor baris terpilih"):
        sensor_cols = ["rain_mm_1h","rain_mm_3h","rain_mm_24h","river_level_cm","tide_level_cm",
                       "soil_moisture","drainage_blockage_idx","temp_c","humidity_pct","wind_mps","pump_active"]
        avail = [c for c in sensor_cols if c in df.columns]
        st.dataframe(row[avail].rename(columns={
            "rain_mm_1h":"Hujan 1h (mm)","rain_mm_3h":"Hujan 3h (mm)","rain_mm_24h":"Hujan 24h (mm)",
            "river_level_cm":"Sungai (cm)","tide_level_cm":"Pasang (cm)","soil_moisture":"Kelembaban Tanah",
            "drainage_blockage_idx":"Drainase Idx","temp_c":"Suhu (°C)",
            "humidity_pct":"Kelembaban Udara (%)","wind_mps":"Angin (m/s)","pump_active":"Pompa Aktif",
        }), use_container_width=True, hide_index=True)

    st.markdown('<div class="sec">Tabel Prediksi Batch</div>', unsafe_allow_html=True)
    col_n, col_f = st.columns([2,1])
    with col_n: n_rows    = st.number_input("Tampilkan N baris terakhir", 10, 1000, 48, 10)
    with col_f: flood_only = st.checkbox("Hanya BANJIR")

    tail    = df.tail(int(n_rows)).copy()
    prob_a  = clf.predict_proba(tail[clf_feat].astype(float))[:,1]
    pred_a  = (prob_a >= threshold).astype(int)
    dep_a   = np.where(pred_a==1, np.maximum(reg.predict(tail[reg_feat].astype(float)), 0.0), 0.0)
    pred_ts = pd.to_datetime(tail["timestamp"]) + timedelta(hours=6)

    out = pd.DataFrame({
        "📅 Timestamp Data":      tail["timestamp"].dt.strftime("%d/%m/%Y %H:%M").values,
        "🕐 Prediksi (+6h)":      pred_ts.dt.strftime("%d/%m/%Y %H:%M").values,
        "📍 Wilayah":             tail["wilayah"].values,
        "🏘️ Kelurahan":           tail["kelurahan"].values,
        "🌊 Prob Banjir":         [f"{p:.1%}" for p in prob_a],
        "🚨 Status":               ["BANJIR" if p else "AMAN" for p in pred_a],
        "📏 Kedalaman Est. (cm)": [f"{d:.1f}" for d in dep_a],
        "⚠️ Severity":            [severity_label(d)[0] for d in dep_a],
    })
    if flood_only:
        out = out[out["🚨 Status"] == "BANJIR"]
    st.dataframe(out, use_container_width=True, hide_index=True)

    fname = f"prediksi_{area_label.replace(' ','_').replace(',','').lower()}.csv"
    st.download_button("⬇️ Download CSV", out.to_csv(index=False).encode(), file_name=fname, mime="text/csv")

    st.markdown('<div class="sec">Ringkasan</div>', unsafe_allow_html=True)
    pct_flood  = pred_a.mean()
    avg_depth  = dep_a[dep_a>0].mean() if (dep_a>0).any() else 0.0
    max_depth  = dep_a.max()
    worst_time = pred_ts.iloc[dep_a.argmax()].strftime("%d %b %Y %H:%M")
    s1,s2,s3,s4,s5 = st.columns(5)
    s1.metric("📊 % Jam Berisiko",      f"{pct_flood:.1%}")
    s2.metric("🔢 Total Jam Banjir",    f"{int(pred_a.sum())}")
    s3.metric("📏 Kedalaman Rata-rata", f"{avg_depth:.1f} cm")
    s4.metric("⬆️ Kedalaman Maks",      f"{max_depth:.1f} cm")
    s5.metric("🕐 Puncak Prediksi",     worst_time)

    if df["kelurahan"].nunique() > 1:
        st.markdown('<div class="sec">Breakdown per Kelurahan</div>', unsafe_allow_html=True)
        tail2  = df.tail(int(n_rows)).copy()
        prob_b = clf.predict_proba(tail2[clf_feat].astype(float))[:,1]
        pred_b = (prob_b >= threshold).astype(int)
        dep_b  = np.where(pred_b==1, np.maximum(reg.predict(tail2[reg_feat].astype(float)),0.0), 0.0)
        tail2["_prob"] = prob_b; tail2["_pred"] = pred_b; tail2["_dep"] = dep_b
        smry = (
            tail2.groupby("kelurahan")
            .agg(total=("_pred","count"), banjir=("_pred","sum"),
                 prob_avg=("_prob","mean"),
                 dep_avg=("_dep", lambda x: x[x>0].mean() if (x>0).any() else 0),
                 dep_max=("_dep","max"))
            .reset_index().sort_values("banjir", ascending=False)
        )
        smry["% Banjir"]       = (smry["banjir"]/smry["total"]).map("{:.1%}".format)
        smry["Prob Rata-rata"] = smry["prob_avg"].map("{:.1%}".format)
        smry["Kedalaman Avg"]  = smry["dep_avg"].map("{:.1f} cm".format)
        smry["Kedalaman Maks"] = smry["dep_max"].map("{:.1f} cm".format)
        smry = smry.rename(columns={"kelurahan":"🏘️ Kelurahan","total":"Total Jam","banjir":"Jam Banjir"})
        st.dataframe(
            smry[["🏘️ Kelurahan","Total Jam","Jam Banjir","% Banjir","Prob Rata-rata","Kedalaman Avg","Kedalaman Maks"]],
            use_container_width=True, hide_index=True
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – RIWAYAT BANJIR
# ══════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.markdown('<div class="sec">Riwayat Kejadian Banjir di Dataset</div>', unsafe_allow_html=True)
    st.caption("Berdasarkan kolom `flood_now` di dataset historis — bukan hasil prediksi model.")

    # hanya pakai baris yang ada flood_now
    if "flood_now" not in df_full.columns:
        st.warning("Kolom `flood_now` tidak ditemukan di dataset.")
    else:
        hist_scope = df_full[df_full["kelurahan"] != "Semua Wilayah"].copy()
        if len(hist_scope) == 0:
            hist_scope = df_full.copy()

        # Filter wilayah context
        if wilayah_choice != "DKI Jakarta":
            hist_scope = hist_scope[hist_scope["wilayah"] == wilayah_choice]
        if kelurahan_choice != "Semua Wilayah":
            hist_scope = hist_scope[hist_scope["kelurahan"] == kelurahan_choice]

        hist_scope["timestamp"] = pd.to_datetime(hist_scope["timestamp"])
        hist_scope["bulan"]     = hist_scope["timestamp"].dt.to_period("M").astype(str)
        hist_scope["tahun"]     = hist_scope["timestamp"].dt.year

        # Monthly flood count
        monthly = (
            hist_scope.groupby(["bulan","kelurahan"])["flood_now"]
            .sum().reset_index()
            .rename(columns={"flood_now":"Jam Banjir","bulan":"Bulan","kelurahan":"Kelurahan"})
        )
        monthly_pivot = monthly.pivot_table(index="Bulan", columns="Kelurahan", values="Jam Banjir", fill_value=0)

        st.markdown("#### 📅 Jam Banjir per Bulan")
        st.dataframe(monthly_pivot, use_container_width=True)

        # Yearly summary
        st.markdown("#### 📆 Ringkasan per Tahun")
        yearly = (
            hist_scope.groupby(["tahun","kelurahan"])
            .agg(
                jam_banjir     = ("flood_now","sum"),
                total_jam      = ("flood_now","count"),
                depth_avg      = ("flood_depth_cm", lambda x: x[x>0].mean() if (x>0).any() else 0),
                depth_max      = ("flood_depth_cm","max"),
            )
            .reset_index()
        )
        yearly["% Waktu Banjir"] = (yearly["jam_banjir"]/yearly["total_jam"]).map("{:.1%}".format)
        yearly["depth_avg"]      = yearly["depth_avg"].map("{:.1f} cm".format)
        yearly["depth_max"]      = yearly["depth_max"].map("{:.1f} cm".format)
        yearly = yearly.rename(columns={
            "tahun":"Tahun","kelurahan":"Kelurahan",
            "jam_banjir":"Jam Banjir","total_jam":"Total Jam",
            "depth_avg":"Depth Avg","depth_max":"Depth Maks",
        }).sort_values(["Tahun","Jam Banjir"], ascending=[True,False])
        st.dataframe(
            yearly[["Tahun","Kelurahan","Jam Banjir","% Waktu Banjir","Depth Avg","Depth Maks"]],
            use_container_width=True, hide_index=True
        )

        # Worst flood events (individual hours)
        st.markdown("#### 🚨 10 Kejadian Banjir Terparah")
        worst = (
            hist_scope[hist_scope["flood_now"]==1]
            .nlargest(10, "flood_depth_cm")
            [["timestamp","wilayah","kelurahan","flood_depth_cm","rain_mm_24h","river_level_cm"]]
            .copy()
        )
        worst["timestamp"] = worst["timestamp"].dt.strftime("%d %b %Y %H:%M")
        worst = worst.rename(columns={
            "timestamp":"Waktu","wilayah":"Wilayah","kelurahan":"Kelurahan",
            "flood_depth_cm":"Kedalaman (cm)","rain_mm_24h":"Hujan 24h (mm)","river_level_cm":"Sungai (cm)",
        })
        st.dataframe(worst, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – RANKING RISIKO
# ══════════════════════════════════════════════════════════════════════════════
with tab_ranking:
    st.markdown('<div class="sec">Ranking Risiko Banjir per Kelurahan</div>', unsafe_allow_html=True)
    st.caption("Dihitung dari % kejadian banjir historis di dataset. Gunakan ini untuk tahu area mana yang perlu paling sering dipantau.")

    # Compute flood rate per kelurahan across all data
    rank_df = df_full[df_full["kelurahan"] != "Semua Wilayah"].copy()
    if "flood_now" not in rank_df.columns:
        st.warning("Kolom `flood_now` tidak tersedia.")
    else:
        rank_df["flood_now"] = pd.to_numeric(rank_df["flood_now"], errors="coerce").fillna(0)
        if "flood_depth_cm" not in rank_df.columns:
            rank_df["flood_depth_cm"] = 0.0

        ranking = (
            rank_df.groupby(["wilayah","kelurahan"])
            .agg(
                total       = ("flood_now","count"),
                jam_banjir  = ("flood_now","sum"),
                depth_avg   = ("flood_depth_cm", lambda x: x[x>0].mean() if (x>0).any() else 0),
                depth_max   = ("flood_depth_cm","max"),
            )
            .reset_index()
        )
        ranking["flood_rate"] = ranking["jam_banjir"] / ranking["total"]
        ranking = ranking.sort_values("flood_rate", ascending=False).reset_index(drop=True)

        # Filter by selected wilayah (unless DKI = show all)
        if wilayah_choice != "DKI Jakarta":
            ranking_show = ranking[ranking["wilayah"] == wilayah_choice].reset_index(drop=True)
            scope_label  = wilayah_choice
        else:
            ranking_show = ranking.copy()
            scope_label  = "DKI Jakarta (semua wilayah)"

        st.markdown(f"**Scope: {scope_label}** — {len(ranking_show)} kelurahan")
        st.markdown("")

        max_rate = ranking_show["flood_rate"].max() if len(ranking_show) else 1.0

        for i, row_r in ranking_show.iterrows():
            rank_num    = i + 1
            pct         = row_r["flood_rate"]
            bar_w       = int(pct / max(max_rate, 0.001) * 100)
            color       = risk_color(pct)
            top_cls     = "top" if rank_num <= 3 else ""
            dep_avg_str = f"{row_r['depth_avg']:.1f} cm avg" if row_r['depth_avg'] > 0 else "—"
            dep_max_str = f"{row_r['depth_max']:.0f} cm maks" if row_r['depth_max'] > 0 else "—"

            st.markdown(f"""
            <div class="rank-card">
              <div class="rank-num {top_cls}">#{rank_num}</div>
              <div style="flex:1;min-width:0;">
                <div class="rank-name">{row_r['kelurahan']}</div>
                <div class="rank-sub">{row_r['wilayah']} &nbsp;·&nbsp; {int(row_r['jam_banjir'])} jam banjir &nbsp;·&nbsp; {dep_avg_str} &nbsp;·&nbsp; {dep_max_str}</div>
                <div style="margin-top:6px;">
                  <div class="rank-bar-wrap">
                    <div class="rank-bar" style="width:{bar_w}%;background:{color};"></div>
                  </div>
                </div>
              </div>
              <div style="font-family:'DM Mono',monospace;font-size:.9rem;color:{color};font-weight:600;flex-shrink:0;">
                {pct:.1%}
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Summary table
        st.markdown("")
        with st.expander("📋 Lihat sebagai tabel"):
            tbl = ranking_show.copy()
            tbl["% Waktu Banjir"] = tbl["flood_rate"].map("{:.1%}".format)
            tbl["Depth Avg"]      = tbl["depth_avg"].map("{:.1f} cm".format)
            tbl["Depth Maks"]     = tbl["depth_max"].map("{:.1f} cm".format)
            tbl = tbl.rename(columns={"wilayah":"Wilayah","kelurahan":"Kelurahan","jam_banjir":"Jam Banjir","total":"Total Jam"})
            st.dataframe(
                tbl[["Wilayah","Kelurahan","Jam Banjir","Total Jam","% Waktu Banjir","Depth Avg","Depth Maks"]],
                use_container_width=True, hide_index=True
            )
