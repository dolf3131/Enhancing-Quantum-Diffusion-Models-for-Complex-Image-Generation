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
    model,            
    epoch,
    T,
    num_qubits,
    writer,
    device,
    target_digits=None, # Conditional
    num_samples=16
):
    """
    Run reverse diffusion from noise and log a grid of samples to TensorBoard.
    Updated for Qiskit-based QuantumUNet.
    """
    model.eval()
    dim = 2 ** num_qubits
    
    side = int(math.isqrt(dim))
    if side * side != dim:
        # Cannot reshape to an image grid (e.g., if num_qubits is odd)
        return

    if target_digits is not None:
        # Conditional: 지정된 숫자를 반복해서 채움
        # 예: [0, 1] -> [0, 1, 0, 1, ...] (num_samples 만큼)
        labels_list = (target_digits * (num_samples // len(target_digits) + 1))[:num_samples]
        labels = torch.tensor(labels_list, device=device).long()
    else:
        # Unconditional
        labels = None

    # Generate Samples
    with torch.no_grad():
        # 1. Start with pure noise (Complex Gaussian)
        batch = torch.randn(num_samples, dim, 2, device=device)
        batch = torch.view_as_complex(batch).to(torch.complex64)
        
        # Normalize initial noise
        batch = batch / (torch.norm(batch, p=2, dim=1, keepdim=True) + 1e-8)

        # 2. Reverse Diffusion Loop
        for t in range(T - 1, -1, -1):
            # (A) Create time tensor for the entire batch
            # Shape: (num_samples,)
            current_t_val = t + 1
            t_tensor = torch.full((num_samples,), current_t_val, device=device, dtype=torch.long)
            
            # (B) Forward pass (Denoising)
            # Pass (Batch, Dim) and (Batch,) directly
            if labels is not None:
                # Conditional
                predicted_batch = model(batch, t=t_tensor, labels=labels)
            else:
                # Unconditional
                predicted_batch = model(batch, t=t_tensor)
            
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

def plot_denoising_evolution(model, num_qubits, T, device, writer, epoch=None, target_digits=None):
    """
    Plots the denoising evolution.
    - If target_digits is provided: Runs in CONDITIONAL mode (generates specific digits).
    - If target_digits is None: Runs in UNCONDITIONAL mode (generates 10 random samples).
    
    Args:
        target_digits (list, optional): List of digits to generate (e.g., [0, 1]). 
                                        If None, generates 10 random samples.
    """
    model.eval()
    
    # --- 1. Determine Mode (Conditional vs Unconditional) ---
    if target_digits is not None:
        # [Conditional Mode]
        num_samples = len(target_digits)
        labels = torch.tensor(target_digits, device=device).long()
        row_prefix = "Digit\n"
        row_values = target_digits
    else:
        # [Unconditional Mode]
        num_samples = 10
        labels = None
        row_prefix = "Sample"
        row_values = range(num_samples) # 0 to 9 (just indices)

    dim = 2 ** num_qubits
    side = int(math.isqrt(dim)) 
    
    # Define snapshots
    snapshot_timesteps = torch.linspace(T, 1, 10).int().unique().tolist()
    # Sort descending just in case
    snapshot_timesteps.sort(reverse=True)
    
    history = [] 

    with torch.no_grad():
        # --- 2. Initialize Noise ---
        current_state = torch.view_as_complex(torch.randn(num_samples, dim, 2, device=device)).to(torch.complex64)
        current_state = current_state / (torch.norm(current_state, p=2, dim=1, keepdim=True) + 1e-8)
        
        # Save Initial Noise (t=T) if requested
        if T in snapshot_timesteps:
            imgs = torch.abs(current_state).cpu().numpy().reshape(num_samples, side, side)
            history.append(imgs)

        # --- 3. Denoising Loop (T -> 1) ---
        for t in range(T, 0, -1):
            
            # (A) Time Tensor
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
            
            # (B) Model Execution
            # Pass only the current_state. 
            # The model retrieves weights for time 't' using t_tensor.
            prediction = model(current_state, t=t_tensor, labels=labels)
            
            # (C) Update State & Renormalize
            current_state = prediction
            current_state = current_state / (torch.norm(current_state, p=2, dim=1, keepdim=True) + 1e-8)
            
            # (D) Save Snapshot (for the NEXT step, which is t-1)
            # If we just computed step t->t-1, the current state effectively represents t-1.
            next_t = t - 1
            if next_t in snapshot_timesteps or next_t == 0: 
                # Note: usually we want to see t=0 (final result)
                imgs = torch.abs(current_state).cpu().numpy().reshape(num_samples, side, side)
                history.append(imgs)

    # --- 4. Plotting ---
    num_cols = len(history)
    fig, axes = plt.subplots(num_samples, num_cols, figsize=(2 * num_cols, 2 * num_samples))
    
    if num_samples == 1: axes = axes.reshape(1, -1)
    if num_cols == 1: axes = axes.reshape(-1, 1)
    
    # Create column titles (Time steps)
    # We collected snapshots: [T, ..., 0]
    # We need to map collected history back to labels
    col_labels = []
    if T in snapshot_timesteps: col_labels.append(f"t={T}")
    for t_val in snapshot_timesteps:
        if t_val != T: # Avoid double adding T
            col_labels.append(f"t={t_val}")
    if 0 not in snapshot_timesteps: # If 0 wasn't in snapshots but we added final state
         # Logic check: loop logic adds t-1. So if last t=1, we added t=0.
         pass
    
    # A simplified label generation for safety
    plot_labels = [f"Step {i}" for i in range(num_cols)]
    # Try to be more specific if lengths match
    # (This part can be adjusted based on exact snapshot logic preference)
    
    for row in range(num_samples): 
        val = row_values[row]
        
        for col in range(num_cols):
            ax = axes[row, col]
            img = history[col][row] 
            
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            
            if col == 0:
                ax.text(-5, side//2, f"{row_prefix} {val}", fontsize=12, va='center', fontweight='bold')
            
            if row == 0:
                ax.set_title(plot_labels[col], fontsize=10)

    plt.tight_layout()
    
    if writer is not None:
        tag_name = 'Conditional Evolution' if labels is not None else 'Unconditional Evolution'
        writer.add_figure(tag_name, fig, global_step=epoch)
        
    plt.close(fig)

