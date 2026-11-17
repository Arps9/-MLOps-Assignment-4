# Assignment 4 – Model Performance Optimization

**Student:** Shreyash Pawar  
**Roll No.:** 4111031  
**PRN:** 22311673  
**Branch:** CSE (Data Science)

## Objective

Compare an unoptimized decision-tree classifier with a pruned decision-tree classifier. Measure predictive accuracy, inference latency, serialized model size, tree depth, and number of nodes.

## Optimization technique

This implementation uses **pruning by limiting maximum tree depth**. The baseline tree has no explicit depth limit; the optimized tree uses `max_depth=4`.

The report discusses quantization, pruning, and knowledge distillation, but selects pruning because it directly applies to decision trees.

## Dataset note

The uploaded assignment report does not identify the dataset used in its experiment. To make this repository runnable without a separate data file, this implementation uses scikit-learn's built-in Breast Cancer Wisconsin dataset and a stratified 80:20 split (`random_state=42`). Therefore, the exact metrics may differ from the report's screenshots and summary table. Do not claim that this implementation reproduces the report's exact numerical results unless you verify them locally.

## Folder structure

```text
Model-Performance-Optimization-Assignment-4/
├── README.md
├── requirements.txt
├── .gitignore
├── utils.py
├── baseline_model.py
├── optimized_model.py
├── compare_models.py
├── performance_graphs.py
├── data/
├── screenshots/
├── graphs/
└── results/             # created when scripts run
    └── results.csv
```

## Setup on Windows

Open Command Prompt in this project folder.

```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the experiment

Run each script in this order:

```cmd
python baseline_model.py
python optimized_model.py
python compare_models.py
python performance_graphs.py
```

The scripts save the serialized models in `models/`, metric JSON files and `results.csv` in `results/`, and comparison charts in `graphs/`. These generated artifacts are ignored by Git by default.

## Metrics

- **Accuracy:** percentage of correct predictions on the held-out test set.
- **Inference time:** repeated, warmed-up prediction latency, reported in milliseconds per row.
- **Model size:** serialized `.joblib` file size in KB.
- **Tree depth:** maximum depth of the fitted decision tree.
- **Number of nodes:** total nodes in the fitted tree.

Inference timing is hardware- and load-dependent. Compare repeated measurements on the same machine; small timing differences should not be over-interpreted.

## GitHub upload

Create an empty GitHub repository, then run from this folder:

```cmd
git init
git add .
git commit -m "Add Assignment 4 model optimization project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/Model-Performance-Optimization-Assignment-4.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username.

## Future work

- Repeat benchmarks and report mean and standard deviation.
- Use k-fold cross-validation.
- Tune `max_depth` and cost-complexity pruning (`ccp_alpha`).
- Compare quantization and knowledge distillation on a neural-network model.
