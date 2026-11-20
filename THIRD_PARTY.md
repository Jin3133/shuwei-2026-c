# Third-party references and dependencies

## TCN architecture

- Source: <https://github.com/locuslab/TCN>
- License: MIT
- Usage: the local `TemporalBlock` follows the published causal, dilated residual-convolution pattern and is rewritten for multi-horizon 3D regression.

## FilterPy

- Source: <https://github.com/rlabbe/filterpy>
- License: MIT
- Planned usage: optional linear Kalman baseline and covariance propagation. It is not vendored.

## PyTorch

- Source: <https://github.com/pytorch/pytorch>
- License: BSD-style
- Usage: tensor operations, GRU, convolution layers, optimization and checkpointing.

## Later Q3 dependencies

- LightGBM or scikit-learn for quantile duration regression.
- Google OR-Tools for deterministic/oracle and rolling scheduling experiments.

Third-party repositories must not be copied wholesale. Pin released package versions when the experiment environment is frozen.

## Missingness and calibration references

- GRU-D: <https://github.com/PeterChe1990/GRU-D> (MIT). Design reference for explicit masks and time-since-observation; old implementation is not vendored.
- MAPIE: <https://github.com/scikit-learn-contrib/MAPIE> (BSD-3-Clause). Planned optional split-conformal calibration on validation environments.
- Microsoft ProbTS: <https://github.com/microsoft/ProbTS>. Interface reference for probabilistic forecasting; not added as a runtime dependency.

## Scheduling references

- Google OR-Tools: <https://github.com/google/or-tools> (Apache-2.0). Planned optional deterministic/oracle and rolling-horizon solver.
- PyJobShop: <https://github.com/PyJobShop/PyJobShop>. Constraint-modelling reference; not required by the core simulator.
