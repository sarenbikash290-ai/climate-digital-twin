import numpy as np
import os
import torch
from torch.utils.data import Dataset

class MaharashtraClimateDataset(Dataset):
    """
    Loads processed .npy files and creates sequences for ConvLSTM.
    Input  : past SEQ_LEN days of (rainfall, max_temp, min_temp)
    Target : next PRED_LEN days of (rainfall, max_temp, min_temp)
    """

    def __init__(self, years: list, seq_len: int = 7, pred_len: int = 3):
        self.seq_len = seq_len
        self.pred_len = pred_len

        rainfall_list = []
        max_temp_list = []
        min_temp_list = []

        print("📦 Loading processed datasets...")

        for year in years:
            rf_path  = f"data/processed/rainfall/rainfall_{year}.npy"
            mt_path  = f"data/processed/max_temp/max_temperature_{year}.npy"
            mn_path  = f"data/processed/min_temp/min_temperature_{year}.npy"

            if not all(os.path.exists(p) for p in [rf_path, mt_path, mn_path]):
                print(f"⚠️  Skipping {year} — files missing")
                continue

            rf = np.load(rf_path)   # (days, lat, lon)
            mt = np.load(mt_path)   # (days, lat, lon)
            mn = np.load(mn_path)   # (days, lat, lon)

            # Upsample temperature to match rainfall grid size
            mt = self._upsample(mt, rf.shape[1], rf.shape[2])
            mn = self._upsample(mn, rf.shape[1], rf.shape[2])

            rainfall_list.append(rf)
            max_temp_list.append(mt)
            min_temp_list.append(mn)

            print(f"   ✅ Loaded {year} — shape: {rf.shape}")

        # Concatenate all years along time axis
        self.rainfall = np.concatenate(rainfall_list, axis=0)   # (total_days, lat, lon)
        self.max_temp = np.concatenate(max_temp_list, axis=0)
        self.min_temp = np.concatenate(min_temp_list, axis=0)

        print(f"\n📊 Total days loaded: {self.rainfall.shape[0]}")
        print(f"   Rainfall shape : {self.rainfall.shape}")
        print(f"   Max Temp shape : {self.max_temp.shape}")
        print(f"   Min Temp shape : {self.min_temp.shape}")

        # Stack into single array: (total_days, 3, lat, lon)
        self.data = np.stack([self.rainfall, self.max_temp, self.min_temp], axis=1)
        print(f"   Combined shape : {self.data.shape}  (days, channels, lat, lon)")

        # Build valid sequence indices
        total_days = self.data.shape[0]
        self.indices = []
        for i in range(total_days - seq_len - pred_len + 1):
            self.indices.append(i)

        print(f"   Total sequences: {len(self.indices)}")

    def _upsample(self, data: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        """Upsample temperature grid to match rainfall grid using repeat interpolation."""
        from scipy.ndimage import zoom
        d, h, w = data.shape
        zoom_h = target_h / h
        zoom_w = target_w / w
        upsampled = zoom(data, (1, zoom_h, zoom_w), order=1)  # bilinear
        return upsampled

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        end   = start + self.seq_len

        # Input: (seq_len, channels, lat, lon)
        x = self.data[start:end]

        # Target: (pred_len, channels, lat, lon)
        y = self.data[end:end + self.pred_len]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )


if __name__ == "__main__":
    dataset = MaharashtraClimateDataset(years=list(range(2018, 2024)))
    print(f"\n🔍 Sample batch:")
    x, y = dataset[0]
    print(f"   Input  shape: {x.shape}  (seq_len, channels, lat, lon)")
    print(f"   Target shape: {y.shape}  (pred_len, channels, lat, lon)")