# src.utils.plot_functions.py

import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import os


from src.models.ansatzes import QuantumUNet
from src.utils.training_functions import assemble_input
from src.utils.schedule import get_default_device

device = get_default_device()


def show_mnist_alphas(mnist_images, alphas_bar, writer, device, height=16, width=16):
    """
    Visualize forward diffusion on a random MNIST sample.
    (This function remains mostly unchanged as it depends on the schedule, not the model architecture)
    """
    mnist_images = mnist_images.to(device)
    # Randomly select one image
    idx = torch.randint(0, mnist_images.shape[0], size=(1,), device=device)
    image = mnist_images[idx].squeeze(0)  # Shape: (D,)

    D = image.numel()
    if height * width != D:
        raise ValueError(f"Cannot reshape image of size {D} into ({height}, {width}).")

    images = [image]
    T = len(alphas_bar)
    
    # Calculate forward diffusion for each step t
    for t in range(T):
        # assemble_input adds noise according to the schedule
        assembled = assemble_input(image.unsqueeze(0), [t], alphas_bar)
        images.append(torch.abs(assembled.squeeze(0)))

    # Plotting
    fig, axs = plt.subplots(2, T + 1, figsize=(20, 6))
    plt.suptitle('Forward diffusion process', fontsize=16)
    bins = np.arange(-0.5, 2.5, 0.2)

    for i in range(T + 1):
        # Image Row
        axs[0, i].imshow(images[i].view(height, width).cpu().detach().numpy(), cmap='gray')
        axs[0, i].axis('off')
        axs[0, i].set_title(f't={i}')

        # Histogram Row
        pixel_values = images[i].cpu().detach().numpy().flatten()
        histogram, _ = np.histogram(pixel_values, bins=bins)
        axs[1, i].bar(bins[:-1], histogram, width=0.2, color='#0504aa', alpha=0.7)
        axs[1, i].set_xlim([-1, 3])
        axs[1, i].set_xlabel('Intensity')
        axs[1, i].set_ylabel('Freq')

    for ax in axs.flatten():
        ax.tick_params(axis='both', which='both', length=0)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    writer.add_figure('Forward diffusion (MNIST)', fig)
    plt.close(fig)


def log_generated_samples(
    directory,
    epoch,
    T,
    num_qubits,
    writer,
    *,
    model_type="unet", # Default to unet
    num_layers=None,   # Unused in Qiskit UNet but kept for compatibility
    init_variance,
    betas,
    pqc_layers=None,
    activation=False,
    bottleneck_qubits=4, # bottlenect
    use_pooling=False,
    num_samples=16,
    # Below are legacy arguments, kept for signature compatibility
    MLP_depth=None, MLP_width=None, PQC_depth=None, ACT_depth=None, num_ancilla=None, batch_size=64
):
    """
    Run reverse diffusion from noise and log a grid of samples to TensorBoard.
    Updated for Qiskit-based QuantumUNet.
    """
    dim = 2 ** num_qubits
    
    side = int(math.isqrt(dim))
    if side * side != dim:
        # Cannot reshape to an image grid (e.g., if num_qubits is odd)
        return

    # Use default layers if not provided
    if pqc_layers is None:
        pqc_layers = [4]

    try:
        # Initialize the Qiskit-based QuantumUNet
        # Note: We create a fresh instance to load the weights into
        circuit_clone = QuantumUNet(
            num_qubits=num_qubits,
            layers=pqc_layers,
            T=T,
            init_variance=init_variance,
            betas=betas,
            activation=activation,
            device=device,
            bottleneck_qubits=bottleneck_qubits, # Must be passed
            use_pooling=use_pooling
        ).to(device)

        # Load parameters for the specific epoch
        # The new load_current_params expects 'epoch' to find the correct file
        circuit_clone.load_current_params(directory, epoch=epoch)
        circuit_clone.eval()
        
    except FileNotFoundError:
        print(f"No checkpoint found for epoch {epoch}, skipping logging.")
        return
    except Exception as e:
        print(f"Error initializing model for logging: {e}")
        return

    # Generate Samples
    with torch.no_grad():
        # 1. Start with pure noise (Complex Gaussian)
        batch = torch.randn(num_samples, dim, 2, device=device)
        batch = torch.view_as_complex(batch).to(torch.complex64)
        
        # Normalize initial noise
        batch = batch / (torch.norm(batch, p=2, dim=1, keepdim=True) + 1e-8)

        # 2. Reverse Diffusion Loop
        for t in range(T, 0, -1):
            # (A) Create time tensor for the entire batch
            # Shape: (num_samples,)
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
            
            # (B) Forward pass (Denoising)
            # Pass (Batch, Dim) and (Batch,) directly
            predicted_batch = circuit_clone(batch, t_tensor)
            
            # (C) Update batch
            batch = predicted_batch
            
            # (D) Renormalize (Crucial for quantum states)
            batch = batch / (torch.norm(batch, p=2, dim=1, keepdim=True) + 1e-8)

        # 3. Process Final Images (Magnitude)
        images = torch.abs(batch).cpu().numpy().reshape(num_samples, side, side)

    # Plotting Grid
    grid_side = int(math.ceil(math.sqrt(num_samples)))
    fig, axes = plt.subplots(grid_side, grid_side, figsize=(2 * grid_side, 2 * grid_side))
    
    # Flatten axes for easy iteration
    if isinstance(axes, np.ndarray):
        axes_flat = axes.flat
    else:
        axes_flat = [axes]

    for idx, ax in enumerate(axes_flat):
        if idx < num_samples:
            ax.imshow(images[idx], cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    
    # Add to TensorBoard
    writer.add_figure('Generated samples', fig, global_step=epoch)
    plt.close(fig)

def plot_denoising_evolution_no_labels(model, num_qubits, T, device, writer, epoch=None):
    """
    Plots the denoising evolution for 10 random samples.
    (Since the model is unconditional, these rows represent random samples, not specific digits 0-9).
    
    Rows: Sample 0-9
    Columns: Time steps (from T to 0)
    """
    model.eval()
    
    # We generate 10 random samples
    num_samples = 10
    dim = 2 ** num_qubits
    side = int(math.isqrt(dim)) # ex: 16 (for 8 qubits)
    
    # Define snapshots to visualize (all T steps or a subset)
    num_snapshots = T
    snapshot_timesteps = torch.linspace(T-1, 0, num_snapshots).int().tolist()
    
    history = [] 

    with torch.no_grad():
        # 1. Initialize Random Noise (Batch=10, Dim, 2)
        current_state = torch.view_as_complex(torch.randn(num_samples, dim, 2, device=device)).to(torch.complex64)
        
        # 2. Backpropagation Loop (T-1 -> 0)
        for t in range(T - 1, -1, -1):
            
            # Save snapshot for plotting
            if t in snapshot_timesteps:
                imgs = torch.abs(current_state).cpu().numpy().reshape(num_samples, side, side)
                history.append(imgs)

            # Prepare Input: (T, Batch, Dim)
            circuit_input = torch.zeros(T, num_samples, dim, device=device, dtype=torch.complex64)
            circuit_input[t] = current_state
            
            # Normalization
            norm = torch.norm(circuit_input, p=2, dim=2, keepdim=True)
            circuit_input = circuit_input / (norm + 1e-8)
            
            # Flatten: (T, B, D) -> (T*B, D)
            circuit_input_flat = circuit_input.view(-1, dim)
            
            # --- MODIFICATION: Removed labels argument ---
            # The model is unconditional, so we just pass the input.
            pred_flat = model(circuit_input_flat)
            # ---------------------------------------------
            
            # Reshape: (T*B, D) -> (T, B, D)
            pred = pred_flat.view(T, num_samples, dim)
            
            # Update state: x_{t-1} = Model(x_t)
            current_state = pred[t]
            
        # Save final state (t=0)
        if 0 not in snapshot_timesteps:
            imgs = torch.abs(current_state).cpu().numpy().reshape(num_samples, side, side)
            history.append(imgs)

    # 3. Plotting
    num_cols = len(history)
    fig, axes = plt.subplots(num_samples, num_cols, figsize=(2 * num_cols, 2 * num_samples))
    
    time_labels = [f"t={t}" for t in snapshot_timesteps]
    if 0 not in snapshot_timesteps: 
        time_labels.append("t=0 (Final)")

    for row in range(num_samples): # Sample 0~9
        for col in range(num_cols): # Time Steps
            ax = axes[row, col]
            
            img = history[col][row] 
            
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            
            # Label changed to "Sample" because generation is random
            if col == 0:
                ax.text(-5, side//2, f"Sample {row}", fontsize=12, va='center', fontweight='bold')
            
            if row == 0:
                ax.set_title(time_labels[col], fontsize=12)

    plt.tight_layout()
    
    # Add to TensorBoard
    if writer is not None:
        writer.add_figure('Denoising samples', fig, global_step=epoch)
        
    plt.close(fig)


