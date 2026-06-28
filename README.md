# 🌏 AI-Powered Digital Twin of India's Climate

A Proof-of-Concept AI-driven Digital Twin of India's Climate System
focused on Maharashtra, built for the ISRO Hackathon.

## 🎯 Objectives
- AI-based short-term prediction of rainfall & temperature
- Interactive geospatial visualization dashboard
- What-if scenario simulation module
- Scalable framework for national deployment

## 📦 Datasets
- IMD Gridded Rainfall (0.25° × 0.25°)
- IMD Max/Min Temperature (1.0° × 1.0°)
- INSAT LST, SST (MOSDAC)

## 🧠 Model
- ConvLSTM — captures both spatial & temporal climate patterns
- Input: 7 days of rainfall + max/min temperature maps
- Output: Next 3 days predictions

## 🚀 Getting Started

### Install dependencies
pip install -r requirements.txt

### Run data pipeline (first time only)
Download IMD data manually from:
- Rainfall: https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_Bin.html
- Max Temp: https://imdpune.gov.in/cmpg/Griddata/Max_1_Bin.html
- Min Temp: https://www.imdpune.gov.in/cmpg/Griddata/Min_1_Bin.html

Place files in data/raw/ folders then run:
python main.py

### Train model only
python -m src.model.train

## 📁 Project Structure
climate_digital_twin/
├── data/
│   ├── raw/          # IMD binary files (not in git)
│   └── processed/    # Processed .npy files (not in git)
├── src/
│   ├── data_pipeline/  # Download, parse, preprocess
│   ├── model/          # ConvLSTM architecture & training
│   ├── dashboard/      # Visualization (coming soon)
│   └── simulation/     # What-if module (coming soon)
├── models/             # Saved checkpoints (not in git)
├── requirements.txt
└── main.py

## 🛠️ Tech Stack
- Python, NumPy, PyTorch
- Streamlit, Plotly, Folium
- IMD & ISRO national datasets

## 👥 Team
ISRO Hackathon 2026
Team Odyssey

Members:
- Bikash Saren
- Ritwik Mridha
- Soumen Bera
- Adhish Sadukhan