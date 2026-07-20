# Quantum Transfer-Continual Learning (QTCL)

This repository contains the official implementation of the **Quantum Transfer-Continual Learning (QTCL)** framework, a hybrid quantum-classical architecture designed to mitigate catastrophic forgetting in sequential task learning. 

By combining frozen pretrained convolutional neural network backbones (such as ResNet-18, MobileNetV2, or EfficientNet-B0) with a dual-path classification head (concatenating classical MLP features and variational quantum circuit outputs), the QTCL framework achieves learning stability and retention comparable to fully classical networks under non-stationary streams of data.

---

## Project Structure

```text
qtcl-github-code/
├── cl_methods.py          # Implementations of continual learning algorithms (DER++, Replay, EWC, Naive, SI)
├── trainer.py             # Core online training loop, metrics computation, and main training entry point
├── metrics.py             # Evaluation metrics calculations (Average Accuracy, Average Forgetting, BWT, etc.)
├── ewc.py                 # Fisher Information Matrix calculations for Elastic Weight Consolidation
├── runner.py              # Experiment runner orchestrating grid search parameters and job dispatching
├── config.yaml            # Default configuration and hyperparameters (datasets, backbones, heads, seeds)
├── config_smoke.yaml      # Fast verification configuration (smoke test) for debug executions
├── requirements.txt       # Python package dependencies
│
├── data/                  # Continual learning dataset split generators
│   ├── __init__.py
│   ├── download_datasets.py  # Utility to download datasets prior to launching multi-job arrays
│   ├── split_mnist.py        # Split MNIST partitioning loader (10% subset, ImageNet normalization)
│   ├── split_cifar10.py      # Split CIFAR-10 partitioning loader (10% subset, ImageNet normalization)
│   └── split_cifar100.py     # Split CIFAR-100 partitioning loader (20-task and 50-task configurations)
│
└── heads/                 # Definition of classical and hybrid classification heads
    ├── __init__.py
    ├── classical_head.py  # Standard Linear, MLP, and Deep MLP heads
    └── qiskit_head.py     # VQC heads (using Qiskit Aer, including Heron-calibrated noise models and ZNE/PEC/DD)
```

---

## Getting Started

### Prerequisites

All code is written in Python 3.10+. It is highly recommended to run the code in a virtual environment (conda or virtualenv).

1. Clone or copy this repository to your local system.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Downloading Datasets

To prevent network races or multiple instances trying to write to the same folders during parallel execution, download the raw datasets first:
```bash
python data/download_datasets.py
```
By default, the datasets will be saved under `./data/datasets/`.

---

## Running Experiments

### 1. Single Execution
You can train a single configuration using `trainer.py`. For example, to train a hybrid quantum head on Split MNIST with a ResNet-18 backbone using seed 42 under the default parameters:
```bash
python trainer.py --dataset split_mnist --backbone resnet18 --head qk_ideal --seed 42
```

To enable specific continual learning methods (e.g. DER++ or EWC) or adjust hyperparameter weights:
```bash
python trainer.py --dataset split_mnist --backbone resnet18 --head qk_ideal --seed 42 --cl_method derpp --alpha 0.7 --beta 0.7
```

### 2. Multi-Job Grid Search
To launch the full benchmarking suite across multiple datasets, backbones, seeds, and head configurations, run `runner.py`:
```bash
python runner.py --config config.yaml
```

For a fast verification run to check the compatibility of dependencies, run the smoke test:
```bash
python runner.py --config config_smoke.yaml
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
