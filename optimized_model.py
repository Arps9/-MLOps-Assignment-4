"""Train and measure a pruned decision tree (maximum depth = 4)."""
from pathlib import Path
import joblib
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
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    metrics = measure_model(model, X_test, y_test)
    metrics["Model"] = "Optimized"
    joblib.dump(model, MODELS / "optimized_model.joblib")
    save_json(metrics, RESULTS / "optimized_metrics.json")
    print("\n" + "=" * 48)
    print("OPTIMIZED MODEL PERFORMANCE")
    print("=" * 48)
    for key, value in metrics.items():
        print(f"{key:22}: {value:.4f}" if isinstance(value, float)
              else f"{key:22}: {value}")

if __name__ == "__main__":
    main()
