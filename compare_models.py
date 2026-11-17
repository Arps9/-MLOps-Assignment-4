"""Compare baseline and optimized model measurements and save results.csv."""
from pathlib import Path
import json
import pandas as pd
from utils import get_model_size_kb

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

def read_metrics(filename):
    with (RESULTS / filename).open(encoding="utf-8") as file:
        return json.load(file)

def main():
    baseline = read_metrics("baseline_metrics.json")
    optimized = read_metrics("optimized_metrics.json")

    baseline["Model_Size_KB"] = get_model_size_kb(MODELS / "baseline_model.joblib")
    optimized["Model_Size_KB"] = get_model_size_kb(MODELS / "optimized_model.joblib")

    columns = [
        "Model", "Accuracy", "Inference_Time_ms", "Model_Size_KB",
        "Tree_Depth", "Number_of_Nodes"
    ]
    df = pd.DataFrame([baseline, optimized])[columns]
    df.to_csv(RESULTS / "results.csv", index=False)

    print("\nMODEL PERFORMANCE OPTIMIZATION COMPARISON")
    print(df.to_string(index=False, formatters={
        "Accuracy": "{:.2f}".format,
        "Inference_Time_ms": "{:.4f}".format,
        "Model_Size_KB": "{:.2f}".format,
    }))

    b, o = df.iloc[0], df.iloc[1]
    print("\nOPTIMIZATION IMPACT")
    print(f"Accuracy change: {o['Accuracy'] - b['Accuracy']:+.2f} percentage points")
    for label, col in [
        ("Inference time reduction", "Inference_Time_ms"),
        ("Model size reduction", "Model_Size_KB"),
        ("Tree depth reduction", "Tree_Depth"),
        ("Node count reduction", "Number_of_Nodes"),
    ]:
        if b[col]:
            change = (b[col] - o[col]) / b[col] * 100
            print(f"{label}: {change:+.2f}%")

    print(f"\nSaved comparison to {RESULTS / 'results.csv'}")

if __name__ == "__main__":
    main()
