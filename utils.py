"""Shared dataset, split, benchmarking and output helpers."""
from pathlib import Path
import json
import time
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_data():
    """Use scikit-learn's built-in Breast Cancer Wisconsin dataset.

    The assignment report does not name a dataset; this reproducible built-in
    dataset is selected for this repository implementation.
    """
    dataset = load_breast_cancer(as_frame=True)
    return dataset.data, dataset.target

def split_data(X, y):
    return train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

def measure_model(model, X_test, y_test, repeats=1000, warmup=50):
    """Measure accuracy, average per-row prediction latency and tree size."""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Warm up before measuring to reduce one-off startup/caching effects.
    sample = X_test.iloc[: min(10, len(X_test))]
    for _ in range(warmup):
        model.predict(sample)

    start = time.perf_counter()
    for _ in range(repeats):
        model.predict(sample)
    elapsed = time.perf_counter() - start
    inference_ms = (elapsed / (repeats * len(sample))) * 1000

    tree = model.tree_
    return {
        "Accuracy": float(accuracy * 100),
        "Inference_Time_ms": float(inference_ms),
        "Model_Size_KB": float("nan"),  # Filled after serialization below.
        "Tree_Depth": int(tree.max_depth),
        "Number_of_Nodes": int(tree.node_count),
    }

def save_json(metrics, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=4), encoding="utf-8")

def get_model_size_kb(path):
    return Path(path).stat().st_size / 1024
