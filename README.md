# Hydrodynamic Forces and Torques on Spherical Particles in Flow

This repository contains a dataset of hydrodynamic forces and torques acting on spherical particles suspended in a stationary flow, along with data-driven models that predict these forces. It serves as the supplementary material for the paper
**[A Data-Driven Approach for Predicting Hydrodynamic Forces on Spherical Particles Using Volume Fraction Representations](https://doi.org/10.1063/5.0279971)**.
If you use this repository, please cite the above publication.

---

## Repository Structure

The repository consists of four main folders:

* **Models**: Data-driven models for predicting hydrodynamic forces.
* **Usage\_example**: Scripts demonstrating how to run the models.
* **Train\_data**: Datasets used for model training.
* **Test\_data**: Datasets used for model evaluation.

---

## Models

This folder contains machine learning models implemented in Python using TensorFlow. The models predict hydrodynamic forces acting on fixed particles based on global flow conditions and the spatial arrangement of neighboring particles. Two model types are included:

* **Local Volume Fraction-Based (LVFB) Models**
* **Particle Position-Based (PPB) Models**

Detailed descriptions of these models can be found in the accompanying [paper](https://doi.org/10.1063/5.0279971).

---

## Usage Example

This folder provides example scripts for running the LVFB and PPB models:

* `1_create_vtk_for_LVFB_model.py` – Generates local volume fraction fields as VTK files.
* `2_LVFB_model_run.py` – Runs the LVFB model and compares results with prDNS simulations.
* `3_PPB_model_run.py` – Runs the PPB model and compares results with prDNS simulations.
* `4_benchmark_LVFB_model.py` – Runs the LVFB model on random data and evaluates the model's run-time.

The following helper scripts are also included:

* `particle_subdomain.py` – Functions for particle domain extraction.
* `Volume_fraction.py` – Functions for volume fraction calculation.

Required dependencies for running the preprocessing and model scripts are listed in the `environment_preprocess` and `environment_models` files.

---

## Train\_data and Test\_data

These folders contain all training and testing simulations used for the development of the LVFB and PPB models. A full description of the simulation setup is available in the [paper](https://doi.org/10.1063/5.0279971).

Each `.h5` file represents a single simulation and includes:

* `Re` – Reynolds number
* `phi` – Particle volume fraction
* `time` – Non-dimensional time
* `xmax`, `xmin`, `ymax`, etc. – Non-dimensional domain boundaries
* `fixed` – Particle-specific data:

  * `F` – Non-dimensional hydrodynamic force
  * `R` – Non-dimensional particle radius
  * `T` – Non-dimensional hydrodynamic torque
  * `X` – Non-dimensional particle positions

---

### Training Simulations

| \$\phi\$/Re | 2   | 5   | 10  | 20  | 40  |
| ----------- | --- | --- | --- | --- | --- |
| 0.1         | 250 | 250 | 250 | 250 | 250 |
| 0.2         | 250 | 250 | 250 | 250 | 250 |
| 0.3         | 160 | 160 | 160 | 160 | 160 |

* **Total number of training simulations:** 3,300
* **Total number of particles:** 854,850

---

### Testing Simulations

| \$\phi\$/Re | 2  | 5  | 10 | 20 | 40 |
| ----------- | -- | -- | -- | -- | -- |
| 0.1         | 50 | 50 | 50 | 50 | 50 |
| 0.2         | 50 | 50 | 50 | 50 | 50 |
| 0.3         | 40 | 40 | 40 | 40 | 40 |

* **Total number of testing simulations:** 700
* **Total number of particles:** 187,650

---

This dataset provides a comprehensive foundation for training and evaluating models that predict hydrodynamic forces on particles in flow environments.
