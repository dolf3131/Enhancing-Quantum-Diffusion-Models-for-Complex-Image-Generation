# Quantum Diffusion on MNIST

MNIST-only quantum diffusion models with two architectures:
- `PQC`: three stacked parameterized circuits with an ancilla in the middle block.
- `NNCPQC`: MLP-driven activation parameters plus PQC/activation stacks.

## Setup
- Create the environment: `conda env create -f environment.yml && conda activate qdm_venv`
- Data is downloaded automatically to `data/` on first run.

## Training
- Run: `python3 -m src.trainer`
- Hyperparameters live in `src/trainer.py` (`model_type`, learning rates, depths, digits, etc.). MNIST inputs are 16x16 (flattened to 256), so keep `num_qubits=8`.
- Outputs per run go to `results/<timestamp>/run_*/trial_*/`:
  - `Params/` holds `current*.pt` and `best*.pt` checkpoints.
  - `TensorBoard/` contains scalars and sampled image grids (logged every 50 epochs).
- On clusters, `job.sbatch` runs the same entrypoint (`sbatch job.sbatch`).

## Visualizing Training
- TensorBoard: `tensorboard --logdir=results/` (losses, learning rates, diffusion forward pass, generated samples).
- Checkpointed samples: see `Images/` from the sampling script for qualitative outputs.
