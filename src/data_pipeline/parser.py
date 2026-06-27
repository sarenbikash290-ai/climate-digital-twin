import numpy as np
import os

# ─── IMD Grid Specifications ───────────────────────────────────────────────────

RAINFALL_GRID = {
    "lat_start": 6.5,
    "lat_end": 38.5,
    "lon_start": 66.5,
    "lon_end": 100.0,
    "resolution": 0.25,
    "missing_value": -999.0
}

TEMP_GRID = {
    "lat_start": 7.5,
    "lat_end": 37.5,
    "lon_start": 67.5,
    "lon_end": 97.5,
    "resolution": 1.0,
    "missing_value": 99.9
}

def get_grid_dims(grid_spec):
    """Calculate number of lat/lon points from grid specification."""
    lats = np.arange(
        grid_spec["lat_start"],
        grid_spec["lat_end"] + grid_spec["resolution"],
        grid_spec["resolution"]
    )
    lons = np.arange(
        grid_spec["lon_start"],
        grid_spec["lon_end"] + grid_spec["resolution"],
        grid_spec["resolution"]
    )
    return lats, lons

def parse_rainfall_binary(filepath: str, year: int):
    """
    Parse IMD rainfall binary (.grd) file.
    Returns: dict with keys 'data' (days x lat x lon), 'lats', 'lons'
    """
    lats, lons = get_grid_dims(RAINFALL_GRID)
    n_lat = len(lats)
    n_lon = len(lons)
    
    # Determine number of days (leap year check)
    n_days = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
    
    expected_size = n_days * n_lat * n_lon * 4  # float32 = 4 bytes
    actual_size = os.path.getsize(filepath)
    
    print(f"📂 Parsing: {os.path.basename(filepath)}")
    print(f"   Grid: {n_lat} lats × {n_lon} lons × {n_days} days")
    print(f"   Expected size: {expected_size} bytes | Actual: {actual_size} bytes")
    
    data = np.fromfile(filepath, dtype=np.float32)
    data = data.reshape((n_days, n_lat, n_lon))
    
    # Mask missing values
    data = np.where(data == RAINFALL_GRID["missing_value"], np.nan, data)
    
    return {
        "data": data,
        "lats": lats,
        "lons": lons,
        "year": year,
        "variable": "rainfall",
        "unit": "mm/day"
    }

def parse_temperature_binary(filepath: str, year: int, temp_type: str = "max"):
    """
    Parse IMD temperature binary (.GRD) file.
    Returns: dict with keys 'data' (days x lat x lon), 'lats', 'lons'
    """
    lats, lons = get_grid_dims(TEMP_GRID)
    n_lat = len(lats)
    n_lon = len(lons)
    
    n_days = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
    
    print(f"📂 Parsing: {os.path.basename(filepath)}")
    print(f"   Grid: {n_lat} lats × {n_lon} lons × {n_days} days")
    
    data = np.fromfile(filepath, dtype=np.float32)
    
    # Auto-detect grid size if reshape fails
    total_points = len(data)
    points_per_day = total_points // n_days
    if points_per_day != n_lat * n_lon:
        print(f"⚠️  Grid mismatch! Expected {n_lat}×{n_lon}={n_lat*n_lon}, got {points_per_day} per day")
        # Find correct dimensions
        import math
        sqrt_val = int(math.sqrt(points_per_day))
        for i in range(sqrt_val, 0, -1):
            if points_per_day % i == 0:
                n_lat = i
                n_lon = points_per_day // i
                print(f"   Auto-detected grid: {n_lat} × {n_lon}")
                break
    
    data = data.reshape((n_days, n_lat, n_lon))
    
    # Mask missing values
    data = np.where(data == TEMP_GRID["missing_value"], np.nan, data)
    
    return {
        "data": data,
        "lats": lats,
        "lons": lons,
        "year": year,
        "variable": f"{temp_type}_temperature",
        "unit": "°C"
    }

if __name__ == "__main__":
    # Test parsing a single file
    result = parse_rainfall_binary("data/raw/rainfall/RF25_ind2020_rfp25.grd", 2020)
    print(f"\n✅ Parsed rainfall shape: {result['data'].shape}")
    print(f"   Lat range: {result['lats'][0]} to {result['lats'][-1]}")
    print(f"   Lon range: {result['lons'][0]} to {result['lons'][-1]}")
    print(f"   Sample value (day 0): {result['data'][0].mean():.2f} mm/day")