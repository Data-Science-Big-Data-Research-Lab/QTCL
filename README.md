# Quantum Transfer-Continual Learning (QTCL)

Code of the study *Quantum Transfer-Continual Learning*, which compares a classical multilayer
perceptron head with hybrid heads that add a variational quantum circuit, on top of frozen
ImageNet-pretrained backbones (ResNet-18, MobileNetV2, EfficientNet-B0), in a sequence of binary
tasks (Split MNIST, Split CIFAR-10, Split CIFAR-100) trained with continual learning methods
(DER++ by default).

## Contents

```text
runner.py            builds the list of runs from a configuration and executes them
trainer.py           training and evaluation of one run
cl_methods.py        continual learning methods: naive, L2, EWC, SI, experience replay, DER++
metrics.py           accuracy matrix, average accuracy, forgetting, transfer, F1 and AUC
report.py            tables of the article computed from the run records
config.yaml          configuration of the experiments
config_smoke.yaml    small configuration for a quick check of the installation
data/                split datasets and dataset download
heads/               classical heads and hybrid heads (Qiskit)
```

## Installation

Python 3.10. The versions in `requirements.txt` are those of the environment that ran the
experiments.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On a machine without a GPU, add `--extra-index-url https://download.pytorch.org/whl/cpu` to install
the CPU build of PyTorch.

Download the datasets before launching parallel jobs:

```bash
python data/download_datasets.py --root ./data/datasets
```

## Running

Quick check of the installation (two tasks, 5% of Split MNIST):

```bash
python runner.py --config config_smoke.yaml --dataset split_mnist --backbone resnet18 --head mlp --seed 42
```

One run of the main benchmark, and the whole benchmark:

```bash
python runner.py --config config.yaml --dataset split_cifar10 --backbone resnet18 --head qk_ideal --seed 42
python runner.py --config config.yaml
```

Additional studies are selected with `--study`, for example the circuit ablation:

```bash
python runner.py --config config.yaml --study ablation
```

`--dry-run` lists the runs without executing them, and runs already recorded in `output_dir` are
skipped. Each run writes `runs.csv`, `training_log.csv` and the CodeCarbon `emissions.csv` under
`output_dir`.

Tables from the run records:

```bash
python report.py --results ./results --out ./tables
```

## Protocol details

- All tasks share one head with two outputs and the task identity is not used (domain-incremental
  setting).
- DER++ and experience replay fill their buffer at the end of each task, with the first training
  batches of the task and the logits of the model just trained.
- The encoder that feeds the circuit is not trained by default (`train_encoder: false` in the head
  configuration). Expectation values are estimated with precision 1/64 (`estimator_precision`), and 0
  gives exact values in the ideal backend.
- The noisy backend uses a gate-noise model with thermal relaxation and depolarizing errors. Its
  estimator computes expectation values without simulating measurements, so readout errors and the
  number of shots do not affect the results.
