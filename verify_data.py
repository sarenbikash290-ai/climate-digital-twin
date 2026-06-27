import os
import numpy as np

# ─── Check raw files exist ─────────────────────────────────────────────────────
print("=" * 50)
print("🔍 Checking Raw Files...")
print("=" * 50)

years = list(range(2018, 2024))

rainfall_files = []
max_temp_files = []
min_temp_files = []

for year in years:
    rf = f"data/raw/rainfall/Rainfall_ind{year}_rfp25.grd"
    mt = f"data/raw/max_temp/Maxtemp_MaxT_{year}.GRD"
    mn = f"data/raw/min_temp/Mintemp_MinT_{year}.GRD"

    rf_ok = os.path.exists(rf)
    mt_ok = os.path.exists(mt)
    mn_ok = os.path.exists(mn)

    rainfall_files.append(rf_ok)
    max_temp_files.append(mt_ok)
    min_temp_files.append(mn_ok)

    print(f"\nYear {year}:")
    print(f"  Rainfall  : {'✅ Found' if rf_ok else '❌ Missing'} {f'({os.path.getsize(rf)/1e6:.1f} MB)' if rf_ok else ''}")
    print(f"  Max Temp  : {'✅ Found' if mt_ok else '❌ Missing'} {f'({os.path.getsize(mt)/1e6:.1f} MB)' if mt_ok else ''}")
    print(f"  Min Temp  : {'✅ Found' if mn_ok else '❌ Missing'} {f'({os.path.getsize(mn)/1e6:.1f} MB)' if mn_ok else ''}")

print("\n" + "=" * 50)
print("📊 Summary:")
print(f"  Rainfall files  : {sum(rainfall_files)}/6")
print(f"  Max Temp files  : {sum(max_temp_files)}/6")
print(f"  Min Temp files  : {sum(min_temp_files)}/6")

# ─── Try parsing one file ──────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("🧪 Testing Parser on 2020 Rainfall...")
print("=" * 50)

from src.data_pipeline.parser import parse_rainfall_binary, parse_temperature_binary

try:
    rf_data = parse_rainfall_binary("data/raw/rainfall/Rainfall_ind2020_rfp25.grd", 2020)
    print(f"✅ Rainfall parsed successfully!")
    print(f"   Shape     : {rf_data['data'].shape}  (days x lat x lon)")
    print(f"   Lat range : {rf_data['lats'][0]} → {rf_data['lats'][-1]}")
    print(f"   Lon range : {rf_data['lons'][0]} → {rf_data['lons'][-1]}")
    print(f"   Max value : {np.nanmax(rf_data['data']):.2f} mm/day")
    print(f"   Min value : {np.nanmin(rf_data['data']):.2f} mm/day")
    print(f"   NaN count : {np.sum(np.isnan(rf_data['data']))}")
except Exception as e:
    print(f"❌ Rainfall parsing failed: {e}")

print("\n🧪 Testing Parser on 2020 Max Temperature...")
try:
    mt_data = parse_temperature_binary("data/raw/max_temp/Maxtemp_MaxT_2020.GRD", 2020, "max")
    print(f"✅ Max Temp parsed successfully!")
    print(f"   Shape     : {mt_data['data'].shape}  (days x lat x lon)")
    print(f"   Lat range : {mt_data['lats'][0]} → {mt_data['lats'][-1]}")
    print(f"   Lon range : {mt_data['lons'][0]} → {mt_data['lons'][-1]}")
    print(f"   Max value : {np.nanmax(mt_data['data']):.2f} °C")
    print(f"   Min value : {np.nanmin(mt_data['data']):.2f} °C")
    print(f"   NaN count : {np.sum(np.isnan(mt_data['data']))}")
except Exception as e:
    print(f"❌ Max Temp parsing failed: {e}")

print("\n🧪 Testing Parser on 2020 Min Temperature...")
try:
    mn_data = parse_temperature_binary("data/raw/min_temp/Mintemp_MinT_2020.GRD", 2020, "min")
    print(f"✅ Min Temp parsed successfully!")
    print(f"   Shape     : {mn_data['data'].shape}  (days x lat x lon)")
    print(f"   Max value : {np.nanmax(mn_data['data']):.2f} °C")
    print(f"   Min value : {np.nanmin(mn_data['data']):.2f} °C")
    print(f"   NaN count : {np.sum(np.isnan(mn_data['data']))}")
except Exception as e:
    print(f"❌ Min Temp parsing failed: {e}")

print("\n" + "=" * 50)
print("✅ Verification Complete!")
print("=" * 50)