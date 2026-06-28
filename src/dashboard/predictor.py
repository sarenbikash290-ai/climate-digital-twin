import torch
import numpy as np
import pickle
from src.model.convlstm import ConvLSTM

def load_model(model_path: str = "models/best_model.pth"):
    """Load trained ConvLSTM model."""
    model = ConvLSTM(
        in_channels     = 3,
        hidden_channels = 64,
        kernel_size     = 3,
        num_layers      = 2,
        pred_len        = 3,
        out_channels    = 3
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

def load_meta(variable: str, year: int):
    """Load normalization metadata."""
    path = f"data/processed/{variable}/{variable}_{year}_meta.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)

def denormalize(data: np.ndarray, norm_stats: dict) -> np.ndarray:
    """Reverse normalization to get real values."""
    if norm_stats["method"] == "minmax":
        return data * (norm_stats["max"] - norm_stats["min"]) + norm_stats["min"]
    elif norm_stats["method"] == "zscore":
        return data * norm_stats["std"] + norm_stats["mean"]

def predict(model, input_sequence: np.ndarray) -> np.ndarray:
    """
    Run model prediction.
    input_sequence: (seq_len, 3, H, W) normalized
    returns: (pred_len, 3, H, W) normalized
    """
    x = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, 3, H, W)
    with torch.no_grad():
        pred = model(x)  # (1, pred_len, 3, H, W)
    return pred.squeeze(0).numpy()  # (pred_len, 3, H, W)

def get_prediction_with_real_values(model, data: np.ndarray, day_idx: int, norm_stats: dict):
    """
    Get prediction for a specific day with denormalized values.
    data: (total_days, 3, H, W) normalized
    Returns dict with rainfall, max_temp, min_temp predictions
    """
    seq = data[day_idx:day_idx + 7]        # (7, 3, H, W)
    pred_norm = predict(model, seq)         # (3, 3, H, W)

    results = {}
    variables = ["rainfall", "max_temp", "min_temp"]
    units     = ["mm/day", "°C", "°C"]

    for i, (var, unit) in enumerate(zip(variables, units)):
        pred_channel = pred_norm[:, i, :, :]   # (3, H, W)
        real_vals    = denormalize(pred_channel, norm_stats[var])
        results[var] = {"data": real_vals, "unit": unit}

    return results