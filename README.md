# Hydrodynamic Forces and Torques on Spherical Particles in Flow

This repository contains a dataset of hydrodynamic forces and torques acting on spherical particles suspended in a stationary flow together with Data-Driven models that predict the hydrodynamic forces acting on particles. It serves as supplementary material for the paper **XXX**. If you use this repository, please cite **XXX**.

## Repository Structure
The repository consists of four main folders:
- **Models**: Contains data-driven models for predicting hydrodynamic forces.
- **Usage_example**: Includes scripts demonstrating how to run the models.
- **Train_data**: Training datasets used to develop the models.
- **Test_data**: Testing datasets for model evaluation.

## Models
This folder contains machine learning models that predict hydrodynamic forces acting on fixed particles based on global flow conditions and neighboring particle positions. The models are implemented in Python using TensorFlow and include:
- **Local Volume Fraction-Based (LVFB) Models**
- **Particle Position-Based (PPB) Models**

These models are described in detail in the paper **XXX**.

## Usage Example
This folder provides scripts for running the LVFB and PPB models:

- `1_create_vtk_for_LVFB_model.py` – Generates local volume fraction distributions as VTK files.
- `2_LVFB_model_run.py` – Runs the LVFB model and compares results to prDNS simulations.
- `3_PPB_model_run.py` – Runs the PPB model and compares results to prDNS simulations.

Additionally, `particle_subdomain.py` and `Volume_fraction.py` contain supplementary functions used in the above scripts.

The required dependencies for running the preprocessing script (`1_create_vtk_for_LVFB_model.py`) and the models are listed in the `environment_preprocess` and `environment_models`.

## Train_data and Test_data
These folders contain all training and testing simulations used for LVFB and PPB model development. Details of the simulations are described in **XXX**.

Each `.h5` file represents a single simulation and includes:
- `Re` – Reynolds number
- `phi` – Particle volume fraction
- `time` – Non-dimensional time units
- `xmax`, `xmin`, `ymax`, etc. – Non-dimensional domain size coordinates
- `fixed` – Particle-related data:
  - `F` – Non-dimensional hydrodynamic force
  - `R` – Non-dimensional particle radius
  - `T` – Non-dimensional hydrodynamic torque
  - `X` – Non-dimensional particle positions

### Training Simulations
| $\phi$/Re  | 2  | 5  | 10  | 20  | 40  |
|-------------|----|----|----|----|----|
| 0.1         | 250| 250| 250| 250| 250|
| 0.2         | 250| 250| 250| 250| 250|
| 0.3         | 160| 160| 160| 160| 160|

**Total Training Simulations:** 3,300

**Total Particles:** 854,850

### Testing Simulations
| $\phi$/Re  | 2  | 5  | 10  | 20  | 40  |
|-------------|----|----|----|----|----|
| 0.1         | 50 | 50 | 50 | 50 | 50 |
| 0.2         | 50 | 50 | 50 | 50 | 50 |
| 0.3         | 40 | 40 | 40 | 40 | 40 |

**Total Testing Simulations:** 700

**Total Particles:** 187,650

This dataset provides a comprehensive foundation for training and evaluating models predicting hydrodynamic forces on particles in flow.

