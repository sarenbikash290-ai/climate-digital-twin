import numpy as np
import os
import pickle
from src.data_pipeline.parser import (
    parse_rainfall_binary,
    parse_temperature_binary,
    RAINFALL_GRID,
    TEMP_GRID
)

# ─── Maharashtra Bounding Box ──────────────────────────────────────────────────
MAHARASHTRA_BOUNDS = {
    "lat_min": 15.5,
    "lat_max": 22.5,
    "lon_min": 72.5,
    "lon_max": 80.5
}

def crop_to_maharashtra(data_dict: dict):
    """Crop full India grid to Maharashtra bounding box."""
    lats = data_dict["lats"]
    lons = data_dict["lons"]
    data = data_dict["data"]
    
    bounds = MAHARASHTRA_BOUNDS
    
    lat_mask = (lats >= bounds["lat_min"]) & (lats <= bounds["lat_max"])
    lon_mask = (lons >= bounds["lon_min"]) & (lons <= bounds["lon_max"])
    
    cropped_data = data[:, lat_mask, :][:, :, lon_mask]
    cropped_lats = lats[lat_mask]
    cropped_lons = lons[lon_mask]
    
    print(f"✂️  Cropped to Maharashtra: {cropped_data.shape}")
    print(f"   Lat: {cropped_lats[0]:.2f} → {cropped_lats[-1]:.2f}")
    print(f"   Lon: {cropped_lons[0]:.2f} → {cropped_lons[-1]:.2f}")
    
    return {
        **data_dict,
        "data": cropped_data,
        "lats": cropped_lats,
        "lons": cropped_lons
    }

def normalize(data: np.ndarray, method: str = "minmax"):
    """Normalize data array, ignoring NaNs."""
    if method == "minmax":
        min_val = np.nanmin(data)
        max_val = np.nanmax(data)
        normalized = (data - min_val) / (max_val - min_val + 1e-8)
        return normalized, {"min": min_val, "max": max_val, "method": "minmax"}
    
    elif method == "zscore":
        mean_val = np.nanmean(data)
        std_val = np.nanstd(data)
        normalized = (data - mean_val) / (std_val + 1e-8)
        return normalized, {"mean": mean_val, "std": std_val, "method": "zscore"}

def fill_missing_values(data: np.ndarray, method: str = "interpolate"):
    """Fill NaN values using specified method."""
    if method == "zero":
        return np.nan_to_num(data, nan=0.0)
    elif method == "mean":
        mean_val = np.nanmean(data)
        return np.where(np.isnan(data), mean_val, data)
    elif method == "interpolate":
        # Simple temporal interpolation along day axis
        filled = data.copy()
        for i in range(data.shape[1]):
            for j in range(data.shape[2]):
                series = data[:, i, j]
                if np.any(np.isnan(series)):
                    nans = np.isnan(series)
                    indices = np.arange(len(series))
                    filled[:, i, j] = np.interp(
                        indices,
                        indices[~nans],
                        series[~nans]
                    ) if np.sum(~nans) > 1 else np.nan_to_num(series, nan=0.0)
        return filled

def save_processed(data_dict: dict, output_dir: str):
    """Save processed numpy array and metadata."""
    os.makedirs(output_dir, exist_ok=True)
    variable = data_dict["variable"]
    year = data_dict["year"]
    
    # Save numpy array
    np_path = os.path.join(output_dir, f"{variable}_{year}.npy")
    np.save(np_path, data_dict["data"])
    
    # Save metadata
    meta = {k: v for k, v in data_dict.items() if k != "data"}
    meta_path = os.path.join(output_dir, f"{variable}_{year}_meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    
    print(f"💾 Saved: {np_path}")
    print(f"💾 Saved metadata: {meta_path}")

def preprocess_all(years: list):
    """Full preprocessing pipeline for all years."""
    
    for year in years:
        print(f"\n{'='*50}")
        print(f"Processing Year: {year}")
        print(f"{'='*50}")
        
        # ── Rainfall ──
        rf_path = f"data/raw/rainfall/Rainfall_ind{year}_rfp25.grd"
        if os.path.exists(rf_path):
            rf_data = parse_rainfall_binary(rf_path, year)
            rf_mh = crop_to_maharashtra(rf_data)
            rf_mh["data"] = fill_missing_values(rf_mh["data"])
            rf_norm, rf_stats = normalize(rf_mh["data"])
            rf_mh["data"] = rf_norm
            rf_mh["norm_stats"] = rf_stats
            save_processed(rf_mh, "data/processed/rainfall")
        else:
            print(f"⚠️  Rainfall file not found for {year}")
        
        # ── Max Temperature ──
        mt_path = f"data/raw/max_temp/Maxtemp_MaxT_{year}.GRD"
        if os.path.exists(mt_path):
            mt_data = parse_temperature_binary(mt_path, year, "max")
            mt_mh = crop_to_maharashtra(mt_data)
            mt_mh["data"] = fill_missing_values(mt_mh["data"])
            mt_norm, mt_stats = normalize(mt_mh["data"])
            mt_mh["data"] = mt_norm
            mt_mh["norm_stats"] = mt_stats
            save_processed(mt_mh, "data/processed/max_temp")
        else:
            print(f"⚠️  Max temp file not found for {year}")
        
        # ── Min Temperature ──
        mn_path = f"data/raw/min_temp/Mintemp_MinT_{year}.GRD"
        if os.path.exists(mn_path):
            mn_data = parse_temperature_binary(mn_path, year, "min")
            mn_mh = crop_to_maharashtra(mn_data)
            mn_mh["data"] = fill_missing_values(mn_mh["data"])
            mn_norm, mn_stats = normalize(mn_mh["data"])
            mn_mh["data"] = mn_norm
            mn_mh["norm_stats"] = mn_stats
            save_processed(mn_mh, "data/processed/min_temp")
        else:
            print(f"⚠️  Min temp file not found for {year}")

if __name__ == "__main__":
    years = list(range(2018, 2024))
    preprocess_all(years)
    print("\n✅ All preprocessing complete!")