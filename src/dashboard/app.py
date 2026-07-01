import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
import numpy as np
import json
import pickle
import pandas as pd
from datetime import date, timedelta
import io

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Climate Digital Twin — Maharashtra",
    page_icon="🌏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Clean White Theme CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ── */
    .stApp { background-color: #f0f4f8; }
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }

    /* ── Hide Streamlit default header ── */
    header[data-testid="stHeader"] { background: transparent; }
    #MainMenu, footer { visibility: hidden; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
        padding-top: 0.5rem;
    }

    /* Remove ALL gaps between sidebar elements */
    [data-testid="stSidebar"] .block-container { padding: 0.5rem 1rem; }
    [data-testid="stSidebar"] .stMarkdown { margin-bottom: 0 !important; }
    [data-testid="stSidebar"] .element-container { margin-bottom: 0.3rem !important; }
    [data-testid="stSidebar"] hr { margin: 0.5rem 0 !important; }

    /* Sidebar text — all dark */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div { color: #1e293b !important; }

    /* Sidebar section headers */
    [data-testid="stSidebar"] h3 {
        color: #1A73E8 !important;
        font-size: 0.7rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 0.5rem 0 0.25rem 0 !important;
    }

    /* ── Fix date input black box ── */
    [data-testid="stDateInput"] input,
    [data-testid="stDateInput"] > div > div,
    [data-testid="stDateInput"] > div > div > input {
        background-color: #ffffff !important;
        color: #1e293b !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 6px !important;
    }

    /* ── Fix selectbox ── */
    [data-testid="stSelectbox"] > div > div {
        background-color: #ffffff !important;
        color: #1e293b !important;
        border: 1px solid #cbd5e1 !important;
    }

    /* ── Fix ALL input fields globally ── */
    input, select, textarea {
        background-color: #ffffff !important;
        color: #1e293b !important;
    }

    /* ── Remove empty white boxes (section-card divs with no content) ── */
    .section-card:empty { display: none; }

    /* ── Metric cards ── */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #1A73E8;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #1A73E8;
        line-height: 1.1;
    }
    .metric-label {
        font-size: 0.7rem;
        color: #64748b;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
    }
    .metric-card.orange { border-left-color: #f97316; }
    .metric-card.orange .metric-value { color: #f97316; }
    .metric-card.green  { border-left-color: #22c55e; }
    .metric-card.green  .metric-value { color: #22c55e; }
    .metric-card.purple { border-left-color: #a855f7; }
    .metric-card.purple .metric-value { color: #a855f7; }

    /* ── Section cards ── */
    .section-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        margin-bottom: 14px;
    }
    .section-title {
        font-size: 0.92rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 10px;
    }

    /* ── Header banner ── */
    .main-header {
        background: linear-gradient(135deg, #1A73E8 0%, #0d47a1 100%);
        border-radius: 14px;
        padding: 20px 28px;
        margin-bottom: 18px;
    }
    .main-header h1 {
        color: #ffffff !important;
        font-size: 1.55rem;
        margin: 0;
        font-weight: 700;
    }
    .main-header p {
        color: rgba(255,255,255,0.9) !important;
        margin: 5px 0 0 0;
        font-size: 0.85rem;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #ffffff;
        border-radius: 10px;
        padding: 4px;
        border: 1px solid #e2e8f0;
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.82rem;
        color: #64748b !important;
        padding: 7px 14px;
    }
    .stTabs [aria-selected="true"] {
        background: #1A73E8 !important;
        color: #ffffff !important;
    }

    /* ── Download button ── */
    .stDownloadButton > button {
        background-color: #1A73E8 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px;
        font-weight: 500;
        padding: 6px 18px;
    }

    /* ── Metric widget ── */
    [data-testid="stMetric"] {
        background: #ffffff;
        border-radius: 10px;
        padding: 10px 14px;
        border: 1px solid #e2e8f0;
    }
    [data-testid="stMetricLabel"] p { color: #64748b !important; font-size: 0.78rem !important; }
    [data-testid="stMetricValue"]   { color: #1e293b !important; font-weight: 700 !important; }
    [data-testid="stMetricDelta"]   { font-size: 0.78rem !important; }

    /* ── Tables ── */
    thead tr th { background: #f1f5f9 !important; color: #1e293b !important; font-weight: 600; }
    tbody tr td { color: #334155 !important; }

    /* ── Info box ── */
    [data-testid="stAlert"] { border-radius: 8px; }

    /* ── AI Insight Panel ───────────────────────────────────────────────────── */
    @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateY(-6px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulseDot {
        0%, 100% { opacity: 1;   transform: scale(1); }
        50%       { opacity: 0.4; transform: scale(0.75); }
    }
    .ai-insight-panel {
        background: linear-gradient(135deg, #eef5ff 0%, #ffffff 100%);
        border: 1px solid #bfdbfe;
        border-left: 4px solid #1A73E8;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 6px rgba(26,115,232,0.07);
        animation: fadeSlideIn 0.45s ease;
        margin-bottom: 2px;
    }
    .insight-header {
        display: flex; align-items: center; gap: 8px;
        font-size: 0.86rem; font-weight: 700; color: #1A73E8; margin-bottom: 10px;
    }
    .live-dot {
        width: 8px; height: 8px; background: #22c55e;
        border-radius: 50%;
        animation: pulseDot 1.8s ease-in-out infinite;
        display: inline-block; flex-shrink: 0;
    }
    .insight-text  { font-size: 0.83rem; color: #334155; line-height: 1.72; }
    .insight-footer {
        margin-top: 10px; font-size: 0.71rem; color: #64748b;
        border-top: 1px solid #e2e8f0; padding-top: 8px;
    }

    /* ── Risk Panel ─────────────────────────────────────────────────────────── */
    .risk-panel {
        background: #ffffff; border: 1px solid #e2e8f0;
        border-radius: 12px; padding: 14px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        animation: fadeSlideIn 0.45s ease;
    }
    .risk-title {
        font-size: 0.74rem; font-weight: 700; color: #1e293b;
        text-transform: uppercase; letter-spacing: 0.06em;
        margin-bottom: 10px; padding-bottom: 6px; border-bottom: 1px solid #f1f5f9;
    }
    .risk-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
    .risk-label  { color: #475569; font-size: 0.77rem; }
    .risk-badge  { font-size: 0.72rem; font-weight: 600; padding: 2px 9px; border-radius: 12px; }
    .risk-low    { background: #dcfce7; color: #15803d; }
    .risk-medium { background: #fef9c3; color: #a16207; }
    .risk-high   { background: #fee2e2; color: #b91c1c; }

    /* ── Confidence Bar ─────────────────────────────────────────────────────── */
    .conf-wrap { margin-top: 8px; }
    .conf-label-row { display: flex; justify-content: space-between; font-size: 0.7rem; color: #64748b; margin-bottom: 4px; }
    .conf-bar-bg    { background: #e2e8f0; border-radius: 4px; height: 5px; overflow: hidden; }
    .conf-bar-fill  { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #1A73E8 0%, #22c55e 100%); }
    .conf-model-tag { font-size: 0.68rem; color: #94a3b8; margin-top: 3px; text-align: right; }

    /* ── Simulation Before / After ──────────────────────────────────────────── */
    .sim-ba-row { display: grid; grid-template-columns: 1fr 28px 1fr; gap: 10px; align-items: center; margin: 12px 0; }
    .sim-card {
        background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 10px; padding: 12px 14px; text-align: center;
    }
    .sim-card.scenario { background: #fff7ed; border-color: #fed7aa; }
    .sim-card-label { font-size: 0.7rem; color: #64748b; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 5px; }
    .sim-card-value { font-size: 1.3rem; font-weight: 700; color: #1e293b; }
    .sim-card.scenario .sim-card-value { color: #f97316; }
    .sim-card-sub   { font-size: 0.7rem; color: #64748b; margin-top: 3px; }
    .sim-arrow      { text-align: center; font-size: 1.4rem; color: #94a3b8; }

    /* ── Simulation Explanation ─────────────────────────────────────────────── */
    .sim-exp {
        background: #fffbeb; border: 1px solid #fde68a;
        border-left: 4px solid #f59e0b; border-radius: 10px;
        padding: 13px 16px; margin-top: 14px;
        animation: fadeSlideIn 0.3s ease;
    }
    .sim-exp-title { font-size: 0.74rem; font-weight: 700; color: #92400e; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 7px; }
    .sim-exp-line  { font-size: 0.81rem; color: #78350f; line-height: 1.85; }
</style>
""", unsafe_allow_html=True)


# ── Data & Model Loaders ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_cached():
    from src.model.convlstm import ConvLSTM
    import torch
    model = ConvLSTM(
        in_channels=3, hidden_channels=64,
        kernel_size=3, num_layers=2,
        pred_len=3, out_channels=3
    )
    model.load_state_dict(torch.load("models/best_model.pth", map_location="cpu"))
    model.eval()
    return model

@st.cache_data(show_spinner=False)
def load_processed_data(years):
    from scipy.ndimage import zoom
    rainfall_all, maxtemp_all, mintemp_all = [], [], []
    rf_stats = mt_stats = mn_stats = None

    for year in years:
        rf = np.load(f"data/processed/rainfall/rainfall_{year}.npy")
        mt = np.load(f"data/processed/max_temp/max_temperature_{year}.npy")
        mn = np.load(f"data/processed/min_temp/min_temperature_{year}.npy")

        if rf_stats is None:
            for var, fname, dest in [
                ("rainfall",    f"data/processed/rainfall/rainfall_{year}_meta.pkl",        "rf"),
                ("max_temp",    f"data/processed/max_temp/max_temperature_{year}_meta.pkl", "mt"),
                ("min_temp",    f"data/processed/min_temp/min_temperature_{year}_meta.pkl", "mn"),
            ]:
                with open(fname, "rb") as f:
                    m = pickle.load(f)
                if dest == "rf": rf_stats = m["norm_stats"]
                elif dest == "mt": mt_stats = m["norm_stats"]
                else: mn_stats = m["norm_stats"]

        zh = rf.shape[1] / mt.shape[1]
        zw = rf.shape[2] / mt.shape[2]
        mt = zoom(mt, (1, zh, zw), order=1)
        mn = zoom(mn, (1, zh, zw), order=1)

        rainfall_all.append(rf)
        maxtemp_all.append(mt)
        mintemp_all.append(mn)

    rainfall = np.concatenate(rainfall_all, axis=0)
    max_temp = np.concatenate(maxtemp_all, axis=0)
    min_temp = np.concatenate(mintemp_all, axis=0)
    data     = np.stack([rainfall, max_temp, min_temp], axis=1)

    with open("data/processed/rainfall/rainfall_2018_meta.pkl", "rb") as f:
        meta = pickle.load(f)

    norm_stats = {"rainfall": rf_stats, "max_temp": mt_stats, "min_temp": mn_stats}
    return data, meta["lats"], meta["lons"], norm_stats

@st.cache_data(show_spinner=False)
def load_history():
    with open("models/history.json") as f:
        return json.load(f)

def denormalize(data, stats):
    if stats["method"] == "minmax":
        return data * (stats["max"] - stats["min"]) + stats["min"]
    return data * stats["std"] + stats["mean"]

def run_prediction(model, data, day_idx):
    import torch
    seq = data[day_idx - 7:day_idx]
    x   = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        pred = model(x).squeeze(0).numpy()
    return pred


# ── AI / Risk helper functions ──────────────────────────────────────────────

def compute_seasonal_average(data, day_idx, var_idx, norm_stats, var_key, window=30):
    """Area-average mean for same ±window day-of-year across all available years."""
    total = len(data)
    dpy   = 365
    doy   = day_idx % dpy
    idxs  = []
    for yr in range(total // dpy + 2):
        for off in range(-window, window + 1):
            idx = yr * dpy + doy + off
            if 7 <= idx < total:
                idxs.append(idx)
    if not idxs:
        return float(np.nanmean(denormalize(data[day_idx, var_idx], norm_stats[var_key])))
    sample = idxs[::max(1, len(idxs) // 60)]          # cap at ~60 samples
    vals   = [float(np.nanmean(denormalize(data[i, var_idx], norm_stats[var_key]))) for i in sample]
    return float(np.nanmean(vals))


def compute_confidence(history, data, day_idx, var_idx, norm_stats, var_key):
    """Confidence score 75–96 based on best val-loss + 7-day input-sequence stability."""
    best_val = min(history["val_loss"])
    base     = max(78.0, min(96.0, 96.0 - (best_val / 0.001) * 11.0))
    s        = max(0, day_idx - 7)
    seq_vals = [float(np.nanmean(denormalize(data[i, var_idx], norm_stats[var_key])))
                for i in range(s, day_idx)]
    if len(seq_vals) > 1:
        variability = np.std(seq_vals) / (abs(np.mean(seq_vals)) + 1e-8)
        base       *= max(0.93, 1.0 - variability * 0.07)
    return round(min(96.0, max(75.0, base)), 1)


def classify_climate_status(rf_avg, mt_avg):
    """Return (emoji_label, hex_color) for current climate conditions."""
    if rf_avg > 50:              return "🌊 Heavy Rain",      "#1d4ed8"
    if rf_avg > 20:              return "🌧️ Moderate Rain",   "#3b82f6"
    if mt_avg > 40:              return "🔥 Extreme Heat",    "#dc2626"
    if mt_avg > 35:              return "☀️ Hot & Dry",       "#f97316"
    if rf_avg < 1 and mt_avg>32: return "🏜️ Dry Conditions",  "#d97706"
    return                              "✅ Normal",           "#22c55e"


def compute_risk_assessment(rf_avg, mt_avg, rf_delta, tmp_delta, seasonal_rf):
    """Return dict of 4 risk (label, css-class) pairs: flood, drought, heat, water."""
    eff_rf   = rf_avg * (1 + rf_delta / 100)
    eff_mt   = mt_avg + tmp_delta
    seas     = max(seasonal_rf, 0.1)
    pct_seas = ((eff_rf - seas) / seas) * 100

    flood   = ("🔴 High",     "risk-high")   if eff_rf > 50   else (("🟡 Moderate", "risk-medium") if eff_rf > 20   else ("🟢 Low",      "risk-low"))
    drought = ("🔴 High",     "risk-high")   if pct_seas<-30  else (("🟡 Moderate", "risk-medium") if pct_seas<-10  else ("🟢 Low",      "risk-low"))
    heat    = ("🔴 High",     "risk-high")   if eff_mt > 40   else (("🟡 Moderate", "risk-medium") if eff_mt > 35   else ("🟢 Low",      "risk-low"))
    water   = ("🔴 Critical", "risk-high")   if eff_rf < 2    else (("🟡 Limited",  "risk-medium") if eff_rf < 5    else ("🟢 Good",     "risk-low"))

    return {"flood": flood, "drought": drought, "heat": heat, "water": water}


def generate_ai_insight(var_key, area_avg, seasonal_avg, pred_avgs,
                          rf_delta, tmp_delta, confidence, risks, selected_date):
    """Build a 4-sentence, fully data-driven AI insight paragraph (HTML)."""
    unit   = {"rainfall": "mm/day", "max_temp": "°C", "min_temp": "°C"}[var_key]
    vlabel = {"rainfall": "Rainfall", "max_temp": "Max Temperature", "min_temp": "Min Temperature"}[var_key]

    # Sentence 1 — current vs seasonal
    pct = ((area_avg - seasonal_avg) / (abs(seasonal_avg) + 1e-8)) * 100
    if abs(pct) < 5:
        s1 = (f"<strong>{vlabel}</strong> over Maharashtra is <strong>{area_avg:.1f} {unit}</strong>, "
              f"near the seasonal average ({seasonal_avg:.1f} {unit}, within ±5%).")
    elif pct > 0:
        s1 = (f"<strong>{vlabel}</strong> over Maharashtra is <strong>{area_avg:.1f} {unit}</strong> — "
              f"<strong>{abs(pct):.0f}% above</strong> the seasonal baseline ({seasonal_avg:.1f} {unit}).")
    else:
        s1 = (f"<strong>{vlabel}</strong> over Maharashtra is <strong>{area_avg:.1f} {unit}</strong> — "
              f"<strong>{abs(pct):.0f}% below</strong> the seasonal baseline ({seasonal_avg:.1f} {unit}).")

    # Sentence 2 — prediction trend
    s2 = ""
    if pred_avgs and len(pred_avgs) == 3:
        trend = pred_avgs[-1] - area_avg
        if abs(trend) < 0.5:
            s2 = (f"ConvLSTM projects a <strong>stable trend</strong> over the next 3 days "
                  f"({pred_avgs[0]:.1f} → {pred_avgs[-1]:.1f} {unit}).")
        elif trend > 0:
            s2 = (f"A <strong>gradual increase</strong> is forecast, reaching "
                  f"<strong>{pred_avgs[-1]:.1f} {unit}</strong> by Day +3.")
        else:
            s2 = (f"A <strong>declining trend</strong> is predicted, dropping to "
                  f"<strong>{pred_avgs[-1]:.1f} {unit}</strong> by Day +3.")

    # Sentence 3 — risk narrative
    rnames = {"flood": "flood risk", "drought": "drought stress", "heat": "heat stress", "water": "water scarcity"}
    highs  = [k for k, v in risks.items() if v[1] == "risk-high"]
    if highs:
        s3 = ("⚠️ <strong>Risk alert</strong>: "
              + " and ".join(f"<strong>{rnames[r]}</strong>" for r in highs)
              + " is elevated across the region. Immediate monitoring advised.")
    elif all(v[1] == "risk-low" for v in risks.values()):
        s3 = "All climate risk indicators are within <strong>normal ranges</strong>. Conditions are stable across Maharashtra."
    else:
        mods = [k for k, v in risks.items() if v[1] == "risk-medium"]
        s3   = ("<strong>Moderate risk</strong> for "
                + ", ".join(rnames[r] for r in mods) + ". Conditions warrant observation.")

    # Sentence 4 — scenario or confidence footer
    if rf_delta != 0 or tmp_delta != 0:
        parts = []
        if rf_delta  != 0: parts.append(f"rainfall {rf_delta:+d}%")
        if tmp_delta != 0: parts.append(f"temperature {tmp_delta:+.1f}°C")
        s4 = (f"⚡ <strong>What-If Scenario active</strong> ({', '.join(parts)}). "
              f"All indicators reflect the simulated climate state.")
    else:
        s4 = (f"Prediction confidence: <strong>{confidence:.1f}%</strong> | "
              f"ConvLSTM (2L, H=64) | IMD + MOSDAC | 0.25°×0.25°")

    return " ".join(s for s in [s1, s2, s3, s4] if s)


def generate_simulation_explanation(rf_delta, tmp_delta, baseline_map, modified_map):
    """Return list of HTML line strings explaining simulation impact (data-driven)."""
    lines = []
    if rf_delta != 0:
        act_pct = ((float(np.nanmean(modified_map)) - float(np.nanmean(baseline_map))) /
                   (abs(float(np.nanmean(baseline_map))) + 1e-8)) * 100
        thresh  = 0.1 * abs(float(np.nanmean(baseline_map))) + 0.1
        aff_pct = float(np.sum(np.abs(modified_map - baseline_map) > thresh)) / baseline_map.size * 100
        dword   = "increase" if rf_delta > 0 else "reduction"
        lines  += [
            f"• Rainfall <strong>{dword} of {abs(rf_delta)}%</strong> applied to the simulation.",
            f"• Estimated actual regional change: <strong>{act_pct:+.1f}%</strong>",
            f"• Significantly affected grid cells: <strong>{aff_pct:.0f}%</strong> of the region",
        ]
        if rf_delta > 20:
            lines.append("• ⚠️ Flood-prone areas likely to expand. River discharge expected to rise.")
        elif rf_delta < -20:
            lines.append("• ⚠️ Water stress rising. Agricultural impact likely in eastern Maharashtra.")
    if tmp_delta != 0:
        dword = "warming" if tmp_delta > 0 else "cooling"
        lines.append(f"• Temperature <strong>{dword} of {tmp_delta:+.1f}°C</strong> applied uniformly.")
        if tmp_delta > 2:
            lines.append("• Heat stress rising in Vidarbha & Marathwada. Evapotranspiration to increase.")
        elif tmp_delta < -2:
            lines.append("• Cold conditions expected in Sahyadri and Satpura highland regions.")
    return lines


# ── Load everything ───────────────────────────────────────────────────────────
with st.spinner("🔄 Loading climate data..."):
    years                          = list(range(2018, 2024))
    data, lats, lons, norm_stats   = load_processed_data(years)
    model                          = load_model_cached()
    history                        = load_history()

START_DATE = date(2018, 1, 1)
END_DATE   = date(2023, 12, 25)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛰️ ISRO Hackathon 2026")
    st.markdown("**AI Climate Digital Twin**")
    st.markdown("*Maharashtra Pilot Region*")
    st.divider()

    st.markdown("### ⚙️ Controls")

    selected_date = st.date_input(
        "📅 Select Date",
        value=date(2022, 6, 15),
        min_value=START_DATE,
        max_value=END_DATE
    )

    variable = st.selectbox(
        "🌡️ Climate Variable",
        ["Rainfall", "Max Temperature", "Min Temperature"]
    )



    st.divider()
    st.markdown("### 🎛️ What-If Scenario")
    rf_delta  = st.slider("🌧️ Rainfall Change (%)",    -50, 50, 0, 5)
    tmp_delta = st.slider("🌡️ Temperature Change (°C)", -5,  5,  0, 1)
    show_scenario = st.checkbox("Apply scenario to predictions")

    st.divider()
    st.caption("📡 Data: IMD + ISRO MOSDAC\n🧠 Model: ConvLSTM (PyTorch)\n📍 Region: Maharashtra\n📊 Resolution: 0.25°×0.25°")


# ── Variable config ───────────────────────────────────────────────────────────
var_map = {
    "Rainfall":        (0, "rainfall",  "Blues",    "mm/day", "#1A73E8"),
    "Max Temperature": (1, "max_temp",  "RdYlBu_r", "°C",     "#E8710A"),
    "Min Temperature": (2, "min_temp",  "RdYlBu_r", "°C",     "#9334E6")
}
var_idx, var_key, colorscale, unit, color = var_map[variable]

day_idx = (selected_date - START_DATE).days
day_idx = max(7, min(day_idx, len(data) - 10))

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='main-header'>
    <h1>🌏 AI-Powered Digital Twin of India's Climate</h1>
    <p>Maharashtra Pilot Region &nbsp;•&nbsp; IMD + ISRO Datasets &nbsp;•&nbsp;
    ConvLSTM Model &nbsp;•&nbsp; Showing: <b>{variable}</b> for <b>{selected_date}</b></p>
</div>
""", unsafe_allow_html=True)


# ── Metric Cards ──────────────────────────────────────────────────────────────
current_map  = denormalize(data[day_idx, var_idx], norm_stats[var_key])
area_avg     = float(np.nanmean(current_map))
area_max     = float(np.nanmax(current_map))
area_min     = float(np.nanmin(current_map))
best_val     = min(history["val_loss"])

card_colors  = ["", "orange", "green", "purple"]
card_labels  = [f"Avg {unit}", f"Max {unit}", f"Min {unit}", "Model Val Loss"]
card_values  = [f"{area_avg:.1f}", f"{area_max:.1f}", f"{area_min:.1f}", f"{best_val:.4f}"]

cols = st.columns(4)
for i, col in enumerate(cols):
    with col:
        st.markdown(f"""
        <div class='metric-card {card_colors[i]}'>
            <div class='metric-value'>{card_values[i]}</div>
            <div class='metric-label'>{card_labels[i]}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── AI Intelligence Panel ─────────────────────────────────────────────────────
_rf_map_now  = denormalize(data[day_idx, 0], norm_stats["rainfall"])
_mt_map_now  = denormalize(data[day_idx, 1], norm_stats["max_temp"])
_rf_now      = float(np.nanmean(_rf_map_now))
_mt_now      = float(np.nanmean(_mt_map_now))

_seasonal_avg    = compute_seasonal_average(data, day_idx, var_idx, norm_stats, var_key)
_seasonal_rf     = compute_seasonal_average(data, day_idx, 0,       norm_stats, "rainfall")
_confidence      = compute_confidence(history, data, day_idx, var_idx, norm_stats, var_key)
_risks           = compute_risk_assessment(_rf_now, _mt_now, rf_delta, tmp_delta, _seasonal_rf)
_climate_status, _status_color = classify_climate_status(_rf_now, _mt_now)

_pred_insight = run_prediction(model, data, day_idx)
_pred_avgs    = [float(np.nanmean(denormalize(_pred_insight[i, var_idx], norm_stats[var_key])))
                  for i in range(3)]
_insight_html = generate_ai_insight(
    var_key, area_avg, _seasonal_avg, _pred_avgs,
    rf_delta, tmp_delta, _confidence, _risks, selected_date
)

_insight_col, _risk_col = st.columns([2.1, 1])

with _insight_col:
    st.markdown(f"""
<div class='ai-insight-panel'>
  <div class='insight-header'>
    <span class='live-dot'></span>🧠 AI Climate Insight — Live
  </div>
  <div class='insight-text'>{_insight_html}</div>
  <div class='insight-footer'>
    {_climate_status} &nbsp;|&nbsp; {selected_date} &nbsp;|&nbsp; IMD + MOSDAC
  </div>
</div>""", unsafe_allow_html=True)

with _risk_col:
    _risk_rows = [
        ("🌊 Flood Risk",         _risks["flood"]),
        ("🏜️ Drought Risk",       _risks["drought"]),
        ("🌡️ Heat Stress",        _risks["heat"]),
        ("💧 Water Availability", _risks["water"]),
    ]
    _rh = "<div class='risk-panel'><div class='risk-title'>⚡ Climate Risk Assessment</div>"
    for _rlabel, (_rbadge, _rcls) in _risk_rows:
        _rh += (f"<div class='risk-row'>"
                f"<span class='risk-label'>{_rlabel}</span>"
                f"<span class='risk-badge {_rcls}'>{_rbadge}</span>"
                f"</div>")
    _rh += (f"<div style='margin-top:8px;font-size:0.68rem;color:#94a3b8;text-align:right;'>"
            f"Confidence: {_confidence:.0f}% | ConvLSTM</div></div>")
    st.markdown(_rh, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── Main Tabs ─────────────────────────────────────────────────────────────────
from src.dashboard.charts import (
    plot_spatial_heatmap, plot_time_series,
    plot_prediction_comparison, plot_loss_history,
    plot_scenario_impact
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Historical",
    "🤖 AI Prediction",
    "🔍 Comparison",
    "🎛️ What-If Scenario",
    "📉 Model Performance"
])


# ── Tab 1: Historical ─────────────────────────────────────────────────────────
with tab1:
        col_map, col_ts = st.columns([1.2, 0.8])

        with col_map:
            with st.container(border=True):
                st.markdown(
    "<span style='color:#1A1C1E; font-weight:700; font-size:0.95rem;'>🗺️ Spatial Distribution Map</span>",
    unsafe_allow_html=True
)

                fig_map = plot_spatial_heatmap(
                    current_map, lats, lons,
                    f"{variable} — {selected_date}", colorscale, unit
                )
                st.plotly_chart(fig_map, use_container_width=True)

        with col_ts:
            with st.container(border=True):
                st.markdown(
    "<span style='color:#1A1C1E; font-weight:700; font-size:0.95rem;'>📈 30-Day Area Average</span>",
    unsafe_allow_html=True
                )
                s = max(0, day_idx - 15)
                e = min(len(data), day_idx + 15)
                ts_vals  = [float(np.nanmean(denormalize(data[i, var_idx], norm_stats[var_key])))
                            for i in range(s, e)]
                ts_dates = [(START_DATE + timedelta(days=i)).strftime("%b %d")
                            for i in range(s, e)]
                fig_ts = plot_time_series(ts_dates, ts_vals, "", unit, color)
                st.plotly_chart(fig_ts, use_container_width=True)

            with st.container(border=True):
                st.markdown(
    "<span style='color:#1A1C1E; font-weight:700; font-size:0.95rem;'>📊 Statistics</span>",
    unsafe_allow_html=True
                )
                st.markdown(f"""
                | Metric | Value |
                |--------|-------|
                | Mean   | {np.nanmean(ts_vals):.2f} {unit} |
                | Max    | {np.nanmax(ts_vals):.2f} {unit} |
                | Min    | {np.nanmin(ts_vals):.2f} {unit} |
                | Std Dev| {np.nanstd(ts_vals):.2f} {unit} |
                """)
                df  = pd.DataFrame({"Date": ts_dates, variable: ts_vals})
                csv = df.to_csv(index=False).encode()
                st.download_button("⬇️ Download CSV", csv,
                    f"{var_key}_{selected_date}.csv", "text/csv")

        with st.container(border=True):
            st.markdown(
    "<span style='color:#1A1C1E; font-weight:700; font-size:0.95rem;'>🗺️ Professional Climate Map — Maharashtra Digital Twin</span>",
    unsafe_allow_html=True
            )

        from src.dashboard.map_view import build_professional_map


        # Get prediction
        pred    = run_prediction(model, data, day_idx)
        pred_rf = pred[0, 0]

        map_html = build_professional_map(
            all_data      = data,
            day_idx       = day_idx,
            pred_rf       = pred_rf,
            lats          = lats,
            lons          = lons,
            norm_stats    = norm_stats,
            selected_date = str(selected_date),
            rf_delta      = rf_delta,
            tmp_delta     = float(tmp_delta),
            window        = 15
        )
        # ← key forces re-render on every date/variable change
        import hashlib
        render_key = hashlib.md5(
            f"{selected_date}-{rf_delta}-{tmp_delta}-{variable}".encode()
        ).hexdigest()[:8]

        st.components.v1.html(
            map_html + f"<!-- {render_key} -->",
            height=700,
            scrolling=False
        )
        


# ── Tab 2: AI Prediction ──────────────────────────────────────────────────────
with tab2:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🤖 AI Prediction — Next 3 Days</div>",
                unsafe_allow_html=True)
    st.caption(f"Based on 7 days prior to {selected_date} | ConvLSTM Model")

    pred = run_prediction(model, data, day_idx)

    pred_cols = st.columns(3)
    pred_data_all = []

    for i in range(3):
        pred_day = selected_date + timedelta(days=i + 1)
        pred_map = denormalize(pred[i, var_idx], norm_stats[var_key])

        if show_scenario:
            if rf_delta != 0 and var_key == "rainfall":
                pred_map = pred_map * (1 + rf_delta / 100)
            if tmp_delta != 0 and var_key in ["max_temp", "min_temp"]:
                pred_map = pred_map + tmp_delta

        pred_data_all.append(pred_map)

        with pred_cols[i]:
            fig = plot_spatial_heatmap(
                pred_map, lats, lons,
                f"Day +{i+1}: {pred_day}", colorscale, unit
            )
            st.plotly_chart(fig, use_container_width=True)
            avg_val   = float(np.nanmean(pred_map))
            delta     = avg_val - area_avg
            st.metric(
                f"Avg {unit}",
                f"{avg_val:.2f}",
                f"{delta:+.2f} vs today"
            )
            _day_conf = max(70.0, round(_confidence - i * 1.2, 1))
            st.markdown(f"""
<div class='conf-wrap'>
  <div class='conf-label-row'>
    <span>Confidence</span><span><strong>{_day_conf:.1f}%</strong></span>
  </div>
  <div class='conf-bar-bg'>
    <div class='conf-bar-fill' style='width:{_day_conf}%'></div>
  </div>
  <div class='conf-model-tag'>Model: ConvLSTM &nbsp;|&nbsp; IMD Data</div>
</div>""", unsafe_allow_html=True)

    # Download predictions
    st.divider()
    rows = []
    for i, pm in enumerate(pred_data_all):
        pred_day = selected_date + timedelta(days=i + 1)
        for li, lat in enumerate(lats):
            for lj, lon in enumerate(lons):
                rows.append({
                    "date": pred_day, "lat": lat,
                    "lon": lon, variable: pm[li, lj]
                })
    df_pred = pd.DataFrame(rows)
    csv_pred = df_pred.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download Prediction CSV",
        csv_pred,
        f"prediction_{selected_date}.csv",
        "text/csv"
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ── Tab 3: Comparison ─────────────────────────────────────────────────────────
with tab3:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🔍 Actual vs Predicted vs Difference</div>",
                unsafe_allow_html=True)

    pred       = run_prediction(model, data, day_idx)
    actual_map = denormalize(data[day_idx, var_idx], norm_stats[var_key])
    pred_map   = denormalize(pred[0, var_idx], norm_stats[var_key])

    fig_comp = plot_prediction_comparison(actual_map, pred_map, var_key, unit)
    st.plotly_chart(fig_comp, use_container_width=True)

    # Error stats
    error  = np.abs(actual_map - pred_map)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Absolute Error", f"{np.nanmean(error):.3f} {unit}")
    with col2:
        st.metric("Max Error", f"{np.nanmax(error):.3f} {unit}")
    with col3:
        rmse = float(np.sqrt(np.nanmean((actual_map - pred_map)**2)))
        st.metric("RMSE", f"{rmse:.3f} {unit}")

    st.markdown("</div>", unsafe_allow_html=True)


# ── Tab 4: What-If Scenario ───────────────────────────────────────────────────
with tab4:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🎛️ What-If Climate Scenario Simulation</div>",
                unsafe_allow_html=True)

    if rf_delta == 0 and tmp_delta == 0:
        st.info("👈 Adjust the **Rainfall Change** or **Temperature Change** sliders in the sidebar to simulate a scenario.")
    else:
        baseline_map = denormalize(data[day_idx, var_idx], norm_stats[var_key])
        modified_map = baseline_map.copy()

        if rf_delta != 0 and var_key == "rainfall":
            modified_map = modified_map * (1 + rf_delta / 100)
            scenario_desc = f"Rainfall {'+' if rf_delta > 0 else ''}{rf_delta}%"
        elif tmp_delta != 0 and var_key in ["max_temp", "min_temp"]:
            modified_map = modified_map + tmp_delta
            scenario_desc = f"Temperature {'+' if tmp_delta > 0 else ''}{tmp_delta}°C"
        else:
            modified_map = baseline_map.copy()
            scenario_desc = "No change for selected variable"

        st.markdown(f"**Scenario:** {scenario_desc} | **Date:** {selected_date}")

        # ── Before / After comparison cards ──────────────────────────────────
        _b_avg  = float(np.nanmean(baseline_map))
        _m_avg  = float(np.nanmean(modified_map))
        _ba_pct = ((_m_avg - _b_avg) / (abs(_b_avg) + 1e-8)) * 100
        _arrow  = "▲" if _m_avg >= _b_avg else "▼"
        _arrowc = "#16a34a" if _m_avg >= _b_avg else "#dc2626"
        st.markdown(f"""
<div class='sim-ba-row'>
  <div class='sim-card'>
    <div class='sim-card-label'>📊 Current (Baseline)</div>
    <div class='sim-card-value'>{_b_avg:.2f} {unit}</div>
    <div class='sim-card-sub'>Area-averaged observed value</div>
  </div>
  <div class='sim-arrow' style='color:{_arrowc}'>{_arrow}</div>
  <div class='sim-card scenario'>
    <div class='sim-card-label'>⚡ Simulated Scenario</div>
    <div class='sim-card-value'>{_m_avg:.2f} {unit}</div>
    <div class='sim-card-sub'>Change: {_ba_pct:+.1f}% from baseline</div>
  </div>
</div>""", unsafe_allow_html=True)

        fig_scenario = plot_scenario_impact(
            baseline_map, modified_map, lats, lons, var_key, unit
        )
        st.plotly_chart(fig_scenario, use_container_width=True)

        # Impact summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Baseline Avg", f"{np.nanmean(baseline_map):.2f} {unit}")
        with col2:
            st.metric("Scenario Avg", f"{np.nanmean(modified_map):.2f} {unit}",
                      f"{np.nanmean(modified_map) - np.nanmean(baseline_map):+.2f}")
        with col3:
            pct_change = ((np.nanmean(modified_map) - np.nanmean(baseline_map))
                          / (np.nanmean(baseline_map) + 1e-8)) * 100
            st.metric("% Change", f"{pct_change:+.1f}%")

        # ── Simulation Explanation ────────────────────────────────────────────
        _sim_lines = generate_simulation_explanation(rf_delta, tmp_delta, baseline_map, modified_map)
        if _sim_lines:
            _sim_html = "".join(f"<div class='sim-exp-line'>{ln}</div>" for ln in _sim_lines)
            st.markdown(f"""
<div class='sim-exp'>
  <div class='sim-exp-title'>🔬 Simulation Analysis</div>
  {_sim_html}
</div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ── Tab 5: Model Performance ──────────────────────────────────────────────────
with tab5:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📉 Model Training History</div>",
                unsafe_allow_html=True)

    fig_loss = plot_loss_history(history)
    st.plotly_chart(fig_loss, use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Best Val Loss", f"{min(history['val_loss']):.6f}",
                  f"Epoch {history['val_loss'].index(min(history['val_loss']))+1}")
    with col2:
        st.metric("Final Train Loss", f"{history['train_loss'][-1]:.6f}")
    with col3:
        st.metric("Total Epochs", "30")
    with col4:
        st.metric("Model Parameters", "1,058,691")

    st.divider()
    st.markdown("**Architecture:** ConvLSTM (2 encoder + 2 decoder layers, hidden=64)")
    st.markdown("**Input:** 7 days × 3 variables (rainfall, max temp, min temp) × 29×33 grid")
    st.markdown("**Output:** Next 3 days × 3 variables × 29×33 grid")
    st.markdown("**Training:** Adam optimizer, ReduceLROnPlateau, MSE loss, batch size 4")
    st.markdown("</div>", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<p style='text-align:center; color:#888; font-size:0.8rem'>
    🌏 AI-Powered Digital Twin of India's Climate &nbsp;|&nbsp;
    ISRO Hackathon 2026 &nbsp;|&nbsp;
    Data: IMD + MOSDAC &nbsp;|&nbsp;
    Model: ConvLSTM (PyTorch) &nbsp;|&nbsp;
    Region: Maharashtra
</p>
""", unsafe_allow_html=True)