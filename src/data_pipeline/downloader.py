import os
import requests
from tqdm import tqdm

# IMD Data URLs
DATA_SOURCES = {
    "rainfall": {
        "base_url": "https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_Bin.html",
        "local_dir": "data/raw/rainfall"
    },
    "max_temp": {
        "base_url": "https://imdpune.gov.in/cmpg/Griddata/Max_1_Bin.html",
        "local_dir": "data/raw/max_temp"
    },
    "min_temp": {
        "base_url": "https://www.imdpune.gov.in/cmpg/Griddata/Min_1_Bin.html",
        "local_dir": "data/raw/min_temp"
    }
}

def create_directories():
    """Create all necessary directories."""
    dirs = [
        "data/raw/rainfall",
        "data/raw/max_temp",
        "data/raw/min_temp",
        "data/processed/rainfall",
        "data/processed/max_temp",
        "data/processed/min_temp"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"✅ Directory ready: {d}")

def download_imd_rainfall(years: list, local_dir: str = "data/raw/rainfall"):
    """
    Download IMD gridded rainfall binary files for given years.
    IMD rainfall files are named as: RF25_ind{YYYY}_rfp25.grd
    """
    base = "https://www.imdpune.gov.in/cmpg/Griddata/"
    
    for year in tqdm(years, desc="Downloading Rainfall"):
        filename = f"RF25_ind{year}_rfp25.grd"
        url = base + filename
        local_path = os.path.join(local_dir, filename)
        
        if os.path.exists(local_path):
            print(f"⚠️  Already exists, skipping: {filename}")
            continue
        
        try:
            response = requests.get(url, timeout=60)
            if response.status_code == 200:
                with open(local_path, 'wb') as f:
                    f.write(response.content)
                print(f"✅ Downloaded: {filename}")
            else:
                print(f"❌ Failed ({response.status_code}): {filename}")
        except Exception as e:
            print(f"❌ Error downloading {filename}: {e}")

def download_imd_temperature(years: list, temp_type: str = "max"):
    """
    Download IMD gridded temperature binary files.
    Max temp files: Maxtemp_MaxT_{YYYY}.GRD
    Min temp files: Mintemp_MinT_{YYYY}.GRD
    """
    base = "https://imdpune.gov.in/cmpg/Griddata/"
    
    if temp_type == "max":
        local_dir = "data/raw/max_temp"
        prefix = "Maxtemp_MaxT_"
    else:
        local_dir = "data/raw/min_temp"
        prefix = "Mintemp_MinT_"
    
    for year in tqdm(years, desc=f"Downloading {temp_type.capitalize()} Temp"):
        filename = f"{prefix}{year}.GRD"
        url = base + filename
        local_path = os.path.join(local_dir, filename)
        
        if os.path.exists(local_path):
            print(f"⚠️  Already exists, skipping: {filename}")
            continue
        
        try:
            response = requests.get(url, timeout=60)
            if response.status_code == 200:
                with open(local_path, 'wb') as f:
                    f.write(response.content)
                print(f"✅ Downloaded: {filename}")
            else:
                print(f"❌ Failed ({response.status_code}): {filename}")
        except Exception as e:
            print(f"❌ Error downloading {filename}: {e}")

if __name__ == "__main__":
    create_directories()
    # Download last 5 years as a PoC dataset
    years = list(range(2018, 2024))
    download_imd_rainfall(years)
    download_imd_temperature(years, temp_type="max")
    download_imd_temperature(years, temp_type="min")