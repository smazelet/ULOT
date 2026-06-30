# ULOT

Code and artifacts for **Unsupervised Learning for Optimal Transport plan prediction between unbalanced graphs**.

This repository contains the `ulot` Python package, the paper reproduction scripts, JSON parameter files, pretrained checkpoints, solver losses, and saved figure data used for the SBM experiments.

## Install

From the repository root:

```bash
python -m pip install -U pip
python -m pip install -e .
```

The package metadata installs the runtime dependencies used by the library and scripts. For the exact pinned environment used during development, install `requirements.txt` first and then install the local package without changing those pins:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Check that the package imports:

```bash
python -c "import ulot; print(ulot.__version__)"
```

## Repository Layout

- `ulot/`: importable Python package with the architectures, FUGW solver, losses, dataset utilities, and training helpers.
- `parameter_files/`: JSON configurations for the SBM and IBC experiments.
- `parameter_files/params_SBM.json`: default SBM training configuration, matching `params_SBM_10000.json`.
- `results/ULOT/trained_model/`: pretrained ULOT checkpoints.
- `results/ULOT/test_performances/`: saved test results for the pretrained SBM model.
- `results/solver_losses/`: FUGW solver losses for SBM graph pairs.
- `results/figure_files/`: saved tensors and graph objects used by `visus.py`.
- `run_train.py`, `run_test.py`, `run_ot_solver.py`: main train, evaluation, and solver scripts.
- `effect_rho_alpha.py`, `functional_minimization.py`, `label_propagation.py`, `accuracy_surfaces.py`, `similarity_mass.py`, `visus.py`: scripts used to generate or visualize the paper figures.

## Command Line Usage

Installing the package exposes console commands for every paper script. The original script files are still available, so `python run_test.py` and `ulot-test` run the same code.

| Command | Original script | Purpose |
| --- | --- | --- |
| `ulot-train` | `run_train.py` | Train a ULOT model from a JSON parameter file. |
| `ulot-test` | `run_test.py` | Evaluate a pretrained ULOT model on SBM graph pairs. |
| `ulot-solve` | `run_ot_solver.py` | Compute FUGW losses with the IBPP solver on SBMs. |
| `ulot-visus` | `visus.py` | Recreate the saved SBM figures from `results/figure_files/`. |
| `ulot-effect-rho-alpha` | `effect_rho_alpha.py` | Generate transport plans for varying `rho` and `alpha`. |
| `ulot-functional-minimization` | `functional_minimization.py` | Generate the functional minimization figure data. |
| `ulot-label-propagation` | `label_propagation.py` | Run label propagation tradeoff experiments. |
| `ulot-accuracy-surfaces` | `accuracy_surfaces.py` | Generate label propagation accuracy surfaces. |
| `ulot-similarity-mass` | `similarity_mass.py` | Compute the ULOT plan-mass similarity matrix. |

Most scripts read and write paths such as `parameter_files/...` and `results/...`. Run them from the repository root to use the checked-in artifacts. If the package is installed from a built wheel and those folders are not present in the current directory, the console commands copy the packaged artifacts into the current directory before launching the script.

## Common Workflows

Evaluate the pretrained SBM model without overwriting saved results:

```bash
ulot-test --param_file params_SBM_10000.json --nosave
```

Train the default SBM model:

```bash
ulot-train
```

Use the larger SBM parameter file:

```bash
ulot-train --param_file params_SBM_50000.json
ulot-test --param_file params_SBM_50000.json
```

Recreate the visualization figures from the saved figure files:

```bash
ulot-visus
```

Compute the solver baseline losses:

```bash
ulot-solve
```

## Python Usage

```python
from ulot.utils import get_dataset, get_loss, get_model

params = {
    "dataset": "SBM",
    "n_pairs": 10,
    "model_type": "ULOT",
    "loss_name": "FUGW",
    "hidden_channels": 64,
    "hidden_channels_message": 16,
    "out_channels": 256,
    "num_layers": 5,
    "temperature": 3,
    "alpha_enc": 10,
}

dataset = get_dataset(params)
example = dataset[0]
model = get_model(params, in_channels=example.x_s.shape[1])
loss = get_loss(params)
```

## Notes

- The checked-in artifacts preserve the original paper reproduction layout.
- `requirements.txt` is intentionally pinned for reproducibility. The package metadata uses compatible dependency ranges so the project can be installed in newer Python environments.
- No model architecture, loss, solver, training loop, or experiment computation has been changed for packaging.

## Citation

```bibtex
@article{mazelet2025unsupervised,
  title={Unsupervised Learning for Optimal Transport plan prediction between unbalanced graphs},
  author={Mazelet, Sonia and Flamary, R{\'e}mi and Thirion, Bertrand},
  journal={arXiv preprint arXiv:2506.12025},
  year={2025}
}
```
