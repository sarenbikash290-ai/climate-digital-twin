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
                st.markdown("**🗺️ Spatial Distribution Map**")
                fig_map = plot_spatial_heatmap(
                    current_map, lats, lons,
                    f"{variable} — {selected_date}", colorscale, unit
                )
                st.plotly_chart(fig_map, use_container_width=True)

        with col_ts:
            with st.container(border=True):
                st.markdown("**📈 30-Day Area Average**")
                s = max(0, day_idx - 15)
                e = min(len(data), day_idx + 15)
                ts_vals  = [float(np.nanmean(denormalize(data[i, var_idx], norm_stats[var_key])))
                            for i in range(s, e)]
                ts_dates = [(START_DATE + timedelta(days=i)).strftime("%b %d")
                            for i in range(s, e)]
                fig_ts = plot_time_series(ts_dates, ts_vals, "", unit, color)
                st.plotly_chart(fig_ts, use_container_width=True)

            with st.container(border=True):
                st.markdown("**📊 Statistics**")
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
            st.markdown("**🗺️ Professional Climate Map — Maharashtra Digital Twin**")
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
            avg_val = float(np.nanmean(pred_map))
            delta   = avg_val - area_avg
            st.metric(
                f"Avg {unit}",
                f"{avg_val:.2f}",
                f"{delta:+.2f} vs today"
            )

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