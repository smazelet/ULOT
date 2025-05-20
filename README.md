# ULOT : Optimal transport plan prediction between unbalanced graphs
Supplemental material for Unsupervised Learning for Optimal Transport plan prediction between unbalanced graphs

This repository contains all files to replicate figures on SBMs
It is organized in the following way:

- `results/ULOT/trained_model`: pre-trained model trained for the SBM datasets on respectively $10 000$ and $50 000$ pairs, and pre-trianed model on the IBC dataset
- `results/ULOT/figure_files`: files to replicate Figures with `visus.py`
- `results/ULOT/true_losses`: files to replicate Figures with `visus.py`
- `visus.py`: code to replicate the figures of Section 3.1 of the paper
- `accuracy_surfaces.py`: code to generate the data for Figure 10
- `baselines.py`: code to generate the data for Figure 7 (right)
- `functional_minimization.py`: code to generate the data for Figure 5
- `label_propagation.py`: code to generate the data for Figure 4
- `run_ot_solver.py`: code to generate the data for Figure 9 (left)
- `run_test.py`: code to generate the data for Figure 9 (left)
- `run_train.py`: code to train the model
- `similarity_mass.py`: code to generate the data for Figure 6 
- `effect_rho_alpha.py`: code to generate the data for Figures 2 and 3
- `parameter_files`: parameter files in the JSON format for the model trained on SBMs and the model trained on the IBC dataset





