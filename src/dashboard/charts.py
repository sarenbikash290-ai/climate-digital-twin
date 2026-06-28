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
    fig = go.Figure(data=go.Heatmap(
        z=data, x=lons, y=lats,
        colorscale=colorscale,
        colorbar=dict(
            title=dict(text=unit, side="right"),
            thickness=15, len=0.9
        ),
        hoverongaps=False,
        zsmooth="best",
        hovertemplate="Lat: %{y:.2f}°N<br>Lon: %{x:.2f}°E<br>Value: %{z:.2f}<extra></extra>"
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#333")),
        xaxis_title="Longitude (°E)",
        yaxis_title="Latitude (°N)",
        template="plotly_white",
        height=400,
        margin=dict(l=50, r=20, t=50, b=50),
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", color="#333")
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
                                variable: str, unit: str):
    colorscale = "Blues" if variable == "rainfall" else "RdYlBu_r"
    vmin = min(np.nanmin(actual), np.nanmin(predicted))
    vmax = max(np.nanmax(actual), np.nanmax(predicted))

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Actual", "Predicted", "Difference"),
        horizontal_spacing=0.08
    )

    fig.add_trace(go.Heatmap(
        z=actual, colorscale=colorscale,
        zmin=vmin, zmax=vmax,
        zsmooth="best",
        showscale=False
    ), row=1, col=1)

    fig.add_trace(go.Heatmap(
        z=predicted, colorscale=colorscale,
        zmin=vmin, zmax=vmax,
        zsmooth="best",
        showscale=True,
        colorbar=dict(x=0.63, title=unit, thickness=12)
    ), row=1, col=2)

    diff = predicted - actual
    fig.add_trace(go.Heatmap(
        z=diff, colorscale="RdBu",
        zmid=0,
        zsmooth="best",
        showscale=True,
        colorbar=dict(x=1.0, title=f"Δ{unit}", thickness=12)
    ), row=1, col=3)

    fig.update_layout(
        title=dict(
            text=f"{variable.replace('_', ' ').title()} — Actual vs Predicted vs Difference",
            font=dict(size=14, color="#333")
        ),
        template="plotly_white", height=360,
        margin=dict(l=20, r=80, t=60, b=20),
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", color="#333")
    )
    return fig


def plot_scenario_impact(baseline: np.ndarray, modified: np.ndarray,
                          lats, lons, variable: str, unit: str):
    """Side by side baseline vs what-if scenario."""
    colorscale = "Blues" if variable == "rainfall" else "RdYlBu_r"
    vmin = min(np.nanmin(baseline), np.nanmin(modified))
    vmax = max(np.nanmax(baseline), np.nanmax(modified))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Baseline", "What-If Scenario")
    )

    fig.add_trace(go.Heatmap(
        z=baseline, x=lons, y=lats,
        colorscale=colorscale,
        zmin=vmin, zmax=vmax,
        zsmooth="best", showscale=False
    ), row=1, col=1)

    fig.add_trace(go.Heatmap(
        z=modified, x=lons, y=lats,
        colorscale=colorscale,
        zmin=vmin, zmax=vmax,
        zsmooth="best", showscale=True,
        colorbar=dict(title=unit, thickness=12)
    ), row=1, col=2)

    fig.update_layout(
        template="plotly_white", height=360,
        margin=dict(l=20, r=80, t=60, b=20),
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", color="#333")
    )
    return fig