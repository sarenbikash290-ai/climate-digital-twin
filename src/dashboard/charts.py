import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np


def plot_loss_history(history: dict):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=history["train_loss"], name="Train Loss",
        mode="lines+markers",
        line=dict(color="#1A73E8", width=2),
        marker=dict(size=5)
    ))
    fig.add_trace(go.Scatter(
        y=history["val_loss"], name="Val Loss",
        mode="lines+markers",
        line=dict(color="#E8710A", width=2),
        marker=dict(size=5)
    ))
    fig.update_layout(
        title="Model Training History",
        xaxis_title="Epoch", yaxis_title="MSE Loss",
        template="plotly_white", height=320,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        paper_bgcolor="white", plot_bgcolor="#f8f9fa",
        font=dict(family="Inter, sans-serif", color="#333")
    )
    return fig


def plot_spatial_heatmap(data: np.ndarray, lats: np.ndarray, lons: np.ndarray,
                          title: str, colorscale: str, unit: str):
    """Render climate data as a density overlay on a real OpenStreetMap base layer."""
    import pandas as pd

    # Flatten the grid into (lat, lon, value) rows — skip NaN
    rows = []
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            val = float(data[i, j])
            if not np.isnan(val):
                rows.append({"lat": lat, "lon": lon, "value": val})
    df = pd.DataFrame(rows)

    if df.empty:
        fig = go.Figure()
        fig.update_layout(title=title, height=420)
        return fig

    centre_lat = float(df["lat"].mean())
    centre_lon = float(df["lon"].mean())

    fig = go.Figure(go.Densitymapbox(
        lat=df["lat"],
        lon=df["lon"],
        z=df["value"],
        radius=18,
        colorscale=colorscale,
        colorbar=dict(
            title=dict(text=unit, side="right"),
            thickness=14, len=0.85,
            tickfont=dict(size=11, family="Inter, sans-serif"),
        ),
        hovertemplate=(
            "<b>Lat:</b> %{lat:.2f}°N<br>"
            "<b>Lon:</b> %{lon:.2f}°E<br>"
            f"<b>{unit}:</b> " + "%{z:.2f}<extra></extra>"
        ),
        opacity=0.75,
        showscale=True,
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#1e293b",
                                         family="Inter, sans-serif")),
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=centre_lat, lon=centre_lon),
            zoom=5.5,
        ),
        height=420,
        margin=dict(l=0, r=0, t=45, b=0),
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", color="#333"),
    )
    return fig



def plot_time_series(dates: list, values: list, title: str,
                     ylabel: str, color: str = "#1A73E8"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=values,
        mode="lines+markers",
        line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor=f"rgba(26,115,232,0.1)"
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#333")),
        xaxis_title="Date", yaxis_title=ylabel,
        template="plotly_white", height=280,
        margin=dict(l=40, r=20, t=50, b=60),
        paper_bgcolor="white", plot_bgcolor="#f8f9fa",
        font=dict(family="Inter, sans-serif", color="#333"),
        xaxis=dict(tickangle=-45)
    )
    return fig


def plot_prediction_comparison(actual: np.ndarray, predicted: np.ndarray,
                                variable: str, unit: str,
                                lats=None, lons=None):
    """Actual vs Predicted as side-by-side bar chart + diff metric — map-friendly."""
    import pandas as pd

    colorscale = "Blues" if variable == "rainfall" else "RdYlBu_r"

    # Summary bar chart: area-averaged actual vs predicted
    act_avg  = float(np.nanmean(actual))
    pred_avg = float(np.nanmean(predicted))
    diff_avg = pred_avg - act_avg
    diff_pct = (diff_avg / (abs(act_avg) + 1e-8)) * 100

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Actual",
        x=["Actual", "Predicted"],
        y=[act_avg, pred_avg],
        marker_color=["#1A73E8", "#E8710A"],
        text=[f"{act_avg:.2f} {unit}", f"{pred_avg:.2f} {unit}"],
        textposition="outside",
        textfont=dict(size=12, family="Inter, sans-serif"),
        width=0.4,
    ))

    fig.add_annotation(
        x=0.5, y=1.08, xref="paper", yref="paper",
        text=f"Δ Difference: <b>{diff_avg:+.2f} {unit}</b>  ({diff_pct:+.1f}%)",
        showarrow=False,
        font=dict(size=13, color="#dc2626" if diff_avg < 0 else "#16a34a",
                  family="Inter, sans-serif"),
        bgcolor="#fef2f2" if diff_avg < 0 else "#f0fdf4",
        bordercolor="#fca5a5" if diff_avg < 0 else "#86efac",
        borderwidth=1, borderpad=6
    )

    var_label = variable.replace("_", " ").title()
    fig.update_layout(
        title=dict(
            text=f"{var_label} — Area Average: Actual vs Predicted",
            font=dict(size=14, color="#1e293b", family="Inter, sans-serif")
        ),
        yaxis_title=unit,
        template="plotly_white", height=340,
        margin=dict(l=50, r=30, t=80, b=50),
        paper_bgcolor="white", plot_bgcolor="#f8fafc",
        font=dict(family="Inter, sans-serif", color="#333"),
        showlegend=False,
    )
    return fig


def plot_scenario_impact(baseline: np.ndarray, modified: np.ndarray,
                          lats, lons, variable: str, unit: str):
    """Baseline vs What-If scenario as a grouped bar + delta chart on real map tiles."""
    import pandas as pd

    colorscale = "Blues" if variable == "rainfall" else "RdYlBu_r"
    centre_lat = float(np.mean(lats))
    centre_lon = float(np.mean(lons))

    # Flatten modified grid for the mapbox view
    rows = []
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            val = float(modified[i, j])
            if not np.isnan(val):
                rows.append({"lat": lat, "lon": lon, "value": val})
    df = pd.DataFrame(rows)

    base_avg = float(np.nanmean(baseline))
    mod_avg  = float(np.nanmean(modified))
    diff_pct = ((mod_avg - base_avg) / (abs(base_avg) + 1e-8)) * 100

    fig = go.Figure(go.Densitymapbox(
        lat=df["lat"],
        lon=df["lon"],
        z=df["value"],
        radius=20,
        colorscale=colorscale,
        colorbar=dict(
            title=dict(text=unit, side="right"),
            thickness=14, len=0.85,
            tickfont=dict(size=11, family="Inter, sans-serif"),
        ),
        hovertemplate=(
            "<b>Lat:</b> %{lat:.2f}°N<br>"
            "<b>Lon:</b> %{lon:.2f}°E<br>"
            f"<b>Scenario {unit}:</b> " + "%{z:.2f}<extra></extra>"
        ),
        opacity=0.78,
        showscale=True,
    ))

    var_label = variable.replace("_", " ").title()
    arrow = "▲" if mod_avg >= base_avg else "▼"
    fig.update_layout(
        title=dict(
            text=(
                f"{var_label} — What-If Scenario Map &nbsp;|&nbsp; "
                f"Baseline: {base_avg:.2f} {unit}  {arrow}  "
                f"Scenario: {mod_avg:.2f} {unit}  ({diff_pct:+.1f}%)"
            ),
            font=dict(size=13, color="#1e293b", family="Inter, sans-serif")
        ),
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=centre_lat, lon=centre_lon),
            zoom=5.5,
        ),
        height=400,
        margin=dict(l=0, r=0, t=55, b=0),
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", color="#333"),
    )
    return fig