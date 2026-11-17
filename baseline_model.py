"""Train and measure the unpruned baseline decision tree."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from utils import load_data, split_data, measure_model, save_json

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
MODELS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    metrics = measure_model(model, X_test, y_test)
    metrics["Model"] = "Baseline"
    joblib.dump(model, MODELS / "baseline_model.joblib")
    save_json(metrics, RESULTS / "baseline_metrics.json")
    print_metrics("BASELINE MODEL PERFORMANCE", metrics)

def print_metrics(title, metrics):
    print("\n" + "=" * 48)
    print(title)
    print("=" * 48)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:22}: {value:.4f}")
        else:
            print(f"{key:22}: {value}")

if __name__ == "__main__":
    main()
