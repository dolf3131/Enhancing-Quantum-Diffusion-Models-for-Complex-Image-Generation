## Sampling
- Requires a trained `Params/` directory.
- Example (PQC):  
  `python3 -m src.sampling --checkpoint results/<ts>/run_1/trial_1/Params --model-type PQC --layers 4 4 4`
- Example (NNCPQC):  
  `python3 -m src.sampling --checkpoint results/<ts>/run_1/trial_1/Params --model-type NNCPQC --num-layers 20 --MLP-depth 5 --MLP-width 64 --PQC-depth 4 --ACT-depth 4 --num-ancilla 1 --batch-size 64`
- Images are saved under `Images/` and intermediate denoising trajectories per sample are saved as grids.

## Visualizing Training
- Requires a `implementation` directory
- TensorBoard: `tensorboard --logdir=results/` (losses, learning rates, diffusion forward pass, generated samples).