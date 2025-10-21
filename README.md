# ULOT : Optimal transport plan prediction between unbalanced graphs
Supplemental material for Unsupervised Learning for Optimal Transport plan prediction between unbalanced graphs

This repository contains all files to replicate figures on SBMs.
It is organized in the following way:

- `results/ULOT/trained_model`: pre-trained model trained for the SBM datasets on respectively $10 000$ and $50 000$ pairs, and pre-trained model on the IBC dataset
- `results/ULOT/figure_files`: files to replicate Figures with `visus.py`
- `results/ULOT/solver_losses`: solver FUGW losses for SBM graph pairs
- `visus.py`: code to replicate the figures of Section 3.1 of the paper
- `accuracy_surfaces.py`: code to generate accuracy surfaces on the label propagation task (Figure 10)
- `functional_minimization.py`: code for minimizing a graph with respect the the FUGW loss (Figure 5).
- `label_propagation.py`: code to optimize the tradeoff parameters on the label propagation task (Figure 4).
- `run_ot_solver.py`: code to compute the FUGW loss with the IBPP solver on SBMs (Figure 9, left).
- `run_test.py`: code to test the model on SBMs (Figure 9, left).
- `run_train.py`: code to train the model.
- `similarity_mass.py`: code to compute the ULOT FUGW similarity matrix (Figure 6).
- `effect_rho_alpha.py`: code to compute the ULOT transport plan between a pair of SBMs for increasing values of tradeoff paramters (Figures 2 and 3)
- `parameter_files`: parameter files in the JSON format for the model trained on SBMs and the model trained on the IBC dataset


# To cite this paper
@article{mazelet2025unsupervised,
  title={Unsupervised Learning for Optimal Transport plan prediction between unbalanced graphs},
  author={Mazelet, Sonia and Flamary, R{\'e}mi and Thirion, Bertrand},
  journal={arXiv preprint arXiv:2506.12025},
  year={2025}
}

