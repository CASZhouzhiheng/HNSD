# HyperNSD: Hypergraph Neural Stochastic Diffusion

This repository contains a clean, self-contained implementation of HyperNSD for:

- OOD detection under label, feature, and structure shifts;
- misclassification detection on ID test nodes.

The model operates directly on the node-hyperedge incidence domain. Its Euler-Maruyama update combines a learnable drift term with an adaptive stochastic forcing term:

```text
dX(t) = -G^T A_theta(X(t)) G X(t) dt + G^T B_phi(X(t)) G dW(t)
```

For an incidence pair `(e, v)`, the implemented gradient is the paper's
degree-normalized operator:

```text
(nabla f)(e, v) = f(v) / sqrt(d_v) - (1 / |e|) sum_{u in e} f(u) / sqrt(d_u)
```

## Installation

Create an environment with a CUDA-compatible PyTorch build, then install:

```bash
pip install -r requirements.txt
```

## Data

See [data/README.md](data/README.md) for the expected layout. Dataset files are
ignored by Git by default.

## Metrics and model selection

Every run reports AUROC, AUPR positive, AUPR negative, and FPR95. For OOD,
positive means ID; for misclassification, positive means a correct prediction.
The selected checkpoint is always the epoch with the best **validation AUROC**.

## Project layout

```text
HyperNSD/
  hypernsd/        Model, incidence operators, data pipeline, metrics, trainer
  data/        Local datasets, ignored by Git
  results/     Local JSON metrics, ignored by Git
  main.py      Unified experiment entry point
```
