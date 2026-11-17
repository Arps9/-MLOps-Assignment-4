"""Create comparison bar charts from results/results.csv."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "results" / "results.csv"
GRAPH_DIR = ROOT / "graphs"
GRAPH_DIR.mkdir(exist_ok=True)

def make_chart(df, column, title, ylabel, filename):
    plt.figure(figsize=(7, 5))
    plt.bar(df["Model"], df[column])
    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / filename, dpi=160)
    plt.close()
    print(f"Saved {GRAPH_DIR / filename}")

def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError("Run baseline_model.py, optimized_model.py, and compare_models.py first.")
    df = pd.read_csv(CSV_PATH)
    make_chart(df, "Accuracy", "Accuracy Comparison", "Accuracy (%)", "accuracy_comparison.png")
    make_chart(df, "Inference_Time_ms", "Inference Time Comparison", "Time (ms per row)", "inference_time_comparison.png")
    make_chart(df, "Model_Size_KB", "Model Size Comparison", "Size (KB)", "model_size_comparison.png")
    make_chart(df, "Tree_Depth", "Tree Depth Comparison", "Depth", "tree_depth_comparison.png")
    make_chart(df, "Number_of_Nodes", "Number of Nodes Comparison", "Nodes", "node_count_comparison.png")

if __name__ == "__main__":
    main()
