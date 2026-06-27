import os
from src.model.train import train_model

if __name__ == "__main__":
    print("🌏 Climate Digital Twin — Full Pipeline")
    print("=" * 50)

    # Phase 1: Already done ✅
    print("\n✅ Phase 1: Data Pipeline already complete!")
    print("   Skipping download & preprocessing...")

    # Phase 2: Model Training
    print("\n🧠 Starting Model Training...")
    model, history = train_model(years=list(range(2018, 2024)), epochs=30)

    print("\n✅ All phases complete!")