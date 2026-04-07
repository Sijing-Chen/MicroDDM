# MicroDDM 

**Official code and data for the paper:** *A Dual-Layer Microfluidic Platform for the Characterization of Synthetic Multi-Input Bistable Circuits*

This repository contains the Python scripts and processed datasets used for image processing, parameter fitting, and Dynamic Delay Modeling (DDM) simulations.

### Python Scripts

**1. Data Processing**
* **`Mask Batch.py`**: Extracts quantitative protein expression data from microscopic images (Related: Fig 5b, Fig S5).
* **`Plateau.py`**: Extracts plateau values from dynamic trajectories (Related: Fig S6).
* **`Normalized Data.py`**: Visualizes the normalized fluorescence data (Related: Fig 6a).

**2. Parameter Fitting & Analysis**
* **`Steady-state Fitting.py`**: Generates 1000 optimal steady-state parameter sets using LHS and least-squares optimization.
* **`Steady params screening.py`**: Screens the 1000 steady-state parameter sets via global phase-space simulations to identify attractors and verify bistability.
* **`Dynamic Fitting.py`**: Generates 1000 optimal dynamic parameter sets using LHS and least-squares optimization.
* **`Steady Params Distribution.py`**: Plots the distribution of steady-state parameters (Related: Fig S9a).
* **`Dynamic Params Distribution.py`**: Analyzes dynamic parameters (distributions, correlation matrix, and PCA) (Related: Fig S9b, c, d).

**3. Simulation**
* **`DDM simulation.py`**: Compares the Dynamic Delay Model and standard ODE model against experimental data (Related: Fig 6c, Fig S7).
* **`Noise.py`**: Runs simulations with added stochastic noise across the full induction matrix (Related: Fig 6d, Fig S8).

### Experimental & Parameter Data
* **`all_group_results_630.xlsx`**: Integrated experimental fluorescence trajectories across 16 induction conditions.
* **`best_params_1000_iterations_1110.xlsx`**: The ensemble of 1000 optimal steady-state parameter sets.
* **`best_dynamic_params_bsp166_[1-5].xlsx`**: The ensemble of 1000 optimal dynamic parameter sets, split into 5 files (200 sets each) generated via parallel computing.
