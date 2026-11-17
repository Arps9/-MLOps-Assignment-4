# Assignment 4 – Source-based summary

The report's objective is to explore model optimization techniques and measure their impact on efficiency.

## Techniques covered
1. Quantization: lower numerical precision, such as float32 to int8.
2. Pruning: remove low-value parts of a model. For decision trees, reduce branches/depth.
3. Knowledge distillation: train a smaller student model to imitate a larger teacher.

## Technique selected
Pruning was selected because the experiment uses a decision tree. Quantization and distillation were left as future work.

## Reported results (from the submitted PDF)
| Metric | Baseline | Optimized | Reported change |
|---|---:|---:|---:|
| Accuracy | 91.23% | 93.86% | +2.63 points |
| Inference time | 0.2912 ms | 0.2161 ms | -25.80% |
| Model size | 4.13 KB | 2.88 KB | -30.27% |
| Tree depth | 7 | 4 | -42.86% |
| Nodes | 37 | 21 | -43.24% |

The report also notes an earlier inconsistent timing run and recommends repeated warm-up benchmarks. The code in this repository performs repeated timing to make measurements less sensitive to one-off overhead.

## Source limitation
The PDF does not name the dataset or provide its original Python source code. This repository therefore makes its dataset choice explicit in README.md and uses a built-in scikit-learn dataset for a reproducible runnable example. Its output values should not be represented as the exact original report results.
