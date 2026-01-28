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
    num_snapshots = T # (Start, ... Intermediate ..., End)
    snapshot_timesteps = torch.linspace(T-1, 0, num_snapshots).int().tolist()
    
    history = [] 

    with torch.no_grad():
        # 2. Initialize Noise
        # Shape: (Batch=num_samples, Dim, 2)
        current_state = torch.view_as_complex(torch.randn(num_samples, dim, 2, device=device)).to(torch.complex64)
        
        # 3. Denoising Loop (T-1 -> 0)
        for t in range(T - 1, -1, -1):
            
            # (A) Save Snapshot
            if t in snapshot_timesteps:
                imgs = torch.abs(current_state).cpu().numpy().reshape(num_samples, side, side)
                history.append(imgs)

            # (B) Prepare Input
            # Shape: (T, Batch, Dim) -> Need to verify model input expectation
            circuit_input = torch.zeros(T, num_samples, dim, device=device, dtype=torch.complex64)
            circuit_input[t] = current_state
            
            # Normalization
            norm = torch.norm(circuit_input, p=2, dim=2, keepdim=True)
            circuit_input = circuit_input / (norm + 1e-8)
            
            # Flatten: (T*B, Dim)
            circuit_input_flat = circuit_input.view(-1, dim)

            current_t_val = t + 1
            t_tensor = torch.full((T * num_samples,), current_t_val, device=device, dtype=torch.long)
            
            # (C) Model Execution
            if labels is not None:
                labels_repeated = labels.repeat(T)
                # Conditional
                pred_flat = model(circuit_input_flat, t=t_tensor, labels=labels_repeated)
            else:
                # Unconditional
                pred_flat = model(circuit_input_flat, t=t_tensor)

            # Reshape Output: (T, Batch, Dim)
            pred = pred_flat.view(T, num_samples, dim)
            
            # Update State
            current_state = pred[t]
            
        # Save Final State (t=0)
        if 0 not in snapshot_timesteps:
            imgs = torch.abs(current_state).cpu().numpy().reshape(num_samples, side, side)
            history.append(imgs)

    # --- 4. Plotting ---
    num_cols = len(history)
    
    # Dynamic Figure Size
    fig, axes = plt.subplots(num_samples, num_cols, figsize=(2 * num_cols, 2 * num_samples))
    
    # Handle 1D axes array if num_samples is 1
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    # Time Labels
    time_labels = [f"t={t}" for t in snapshot_timesteps]
    if 0 not in snapshot_timesteps: 
        time_labels.append("Final")

    for row in range(num_samples): 
        val = row_values[row] # Digit value or Sample index
        
        for col in range(num_cols):
            ax = axes[row, col]
            
            img = history[col][row] 
            
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            
            # Row Label (Left side)
            if col == 0:
                ax.text(-5, side//2, f"{row_prefix} {val}", fontsize=12, va='center', fontweight='bold')
            
            # Time Label (Top side)
            if row == 0:
                ax.set_title(time_labels[col], fontsize=12)

    plt.tight_layout()
    
    # Add to TensorBoard
    if writer is not None:
        tag_name = 'Conditional Evolution' if labels is not None else 'Unconditional Evolution'
        writer.add_figure(tag_name, fig, global_step=epoch)
        
    plt.close(fig)

