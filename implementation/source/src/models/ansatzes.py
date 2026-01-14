import torch
import torch.nn as nn
import pennylane as qml
import numpy as np


# Qiskit Imports
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap

from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

backend_sim = AerSimulator()
pm = generate_preset_pass_manager(target=backend_sim.target, optimization_level=3, seed_transpiler=42)

# PennyLane
import pennylane as qml

class ComplexLeakyReLU(nn.Module):
    """LeakyReLU that acts independently on real and imaginary parts."""
    def __init__(self, negative_slope=0.01):
        super(ComplexLeakyReLU, self).__init__()
        self.negative_slope = negative_slope
        self.real_leaky_relu = nn.LeakyReLU(negative_slope=self.negative_slope)
        self.imag_leaky_relu = nn.LeakyReLU(negative_slope=self.negative_slope)

    def forward(self, input):
        return torch.complex(
            self.real_leaky_relu(input.real),
            self.imag_leaky_relu(input.imag)
        )

class ComplexLinear(nn.Module):
    """Complex-valued Linear Layer."""
    def __init__(self, in_features, out_features, activation=True):
        super().__init__()
        self.fc = nn.Linear(in_features * 2, out_features) # Output is Real for simplicity in next steps
        self.activation = activation
        self.act_fn = nn.ReLU()

    def forward(self, x_complex):
        # Concatenate Real and Imag parts
        x_cat = torch.cat([x_complex.real, x_complex.imag], dim=-1) 
        out = self.fc(x_cat)
        if self.activation:
            out = self.act_fn(out)
        return out


def ConvUnit(params, wires):
    """
    User-defined 2-Qubit Convolution Layer (3 params)
    Args:
        params: Flat tensor containing all parameters
        wires: [control, target]
        param_idx: Current index pointer in params
    Returns:
        Updated param_idx
    """
    param_idx = 0
    
    qml.RZ(-np.pi/2, wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    
    qml.RZ(params[param_idx], wires=wires[0]); param_idx += 1
    qml.RY(params[param_idx], wires=wires[1]); param_idx += 1
    
    qml.CNOT(wires=[wires[0], wires[1]])
    
    qml.RY(params[param_idx], wires=wires[1]); param_idx += 1
    
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RZ(np.pi/2, wires=wires[0])
    
    return param_idx

def MixingBlock(params, wires):
    """
    RX + CRY based variational ansatz
    Input params: 1D Tensor for this block
    """
    n = len(wires)
    param_idx = 0
    
    # 1. RX Layer (n params)
    for i in range(n):
        qml.RX(params[param_idx], wires=wires[i])
        param_idx += 1
    
    # 2. CRY Layer (Pairwise)
    
    # Even pairs: (0,1), (2,3)...
    for i in range(0, n-1, 2):
        qml.CNOT(wires=[wires[i], wires[(i+1)%n]])
        qml.RZ(params[param_idx], wires=wires[(i+1)%n]); param_idx += 1
        qml.CNOT(wires=[wires[i], wires[(i+1)%n]])
        
    # Odd pairs: (1,2), (3,4)...
    for i in range(1, n, 2):
        qml.CNOT(wires=[wires[i], wires[(i+1)%n]])
        qml.RZ(params[param_idx], wires=wires[(i+1)%n]); param_idx += 1
        qml.CNOT(wires=[wires[i], wires[(i+1)%n]])
    
    return param_idx


class PennyLanePQC(nn.Module):
    def __init__(self, num_qubits, reps=2):
        super(PennyLanePQC, self).__init__()
        self.num_qubits = num_qubits
        self.reps = reps
        
        # Block 1 & 3 (N qubits)
        self.params_per_block_n = self._calculate_params(num_qubits)
        self.params_per_block_ancilla = self._calculate_params(num_qubits + 1)
        
        self.total_params = self.params_per_block_n * 2 + self.params_per_block_ancilla

        # --- Devices ---
        self.dev_n = qml.device("default.qubit", wires=num_qubits)
        self.dev_ancilla = qml.device("default.qubit", wires=num_qubits + 1)

        # --- QNodes 정의 ---
        self.qnode1 = self._create_qnode_state(self.dev_n, num_qubits)
        self.qnode2 = self._create_qnode_state(self.dev_ancilla, num_qubits + 1)
        self.qnode3 = self._create_qnode_expval(self.dev_n, num_qubits)

    def _calculate_params(self, n_wires):
        """Calculate total parameters for given number of wires and repetitions."""
        num_conv_even = len(range(0, n_wires - 1, 2))
        num_conv_odd  = len(range(1, n_wires, 2)) 
        total_pairs = num_conv_even + num_conv_odd
        total_conv = total_pairs * 3
        total_mixing = n_wires + total_pairs
        return (total_conv + total_mixing) * self.reps

    def _circuit_ansatz(self, weights, wires):
        """common ansatz structure for the QNodes."""
        if weights.ndim == 2:
            weights = weights.T
        n_wires = len(wires)
        idx = 0
        for d in range(self.reps):
            # Conv Layer
            for i in range(0, n_wires-1, 2):
                idx += ConvUnit(weights[idx:], wires=[wires[i], wires[(i+1)%n_wires]])
            for i in range(1, n_wires, 2):
                idx += ConvUnit(weights[idx:], wires=[wires[i], wires[(i+1)%n_wires]])
            # Mixing Layer
            idx += MixingBlock(weights[idx:], wires=wires)

    def _create_qnode_state(self, dev, n_wires):
        """Returns state vector QNode"""
        @qml.qnode(dev, interface='torch', diff_method='backprop')
        def circuit(inputs, weights):
            qml.StatePrep(inputs, wires=range(n_wires))
            self._circuit_ansatz(weights, range(n_wires))
            return qml.state()
        return circuit

    def _create_qnode_expval(self, dev, n_wires):
        """Returns expectation value QNode"""
        @qml.qnode(dev, interface='torch', diff_method='backprop')
        def circuit(inputs, weights):
            qml.StatePrep(inputs, wires=range(n_wires))
            self._circuit_ansatz(weights, range(n_wires))
            # UNet의 decoder 입력 차원(N)에 맞추기 위해 PauliZ 측정
            return [qml.expval(qml.PauliZ(i)) for i in range(n_wires)]
        return circuit

    def forward(self, input_state_or_latent, all_weights):
        """
        Args:
            input_state_or_latent: 
            all_weights: (Total Params,) 
        """
        
        batch_size = input_state_or_latent.shape[0]

        p1_end = self.params_per_block_n
        p2_end = p1_end + self.params_per_block_ancilla
        
        w1 = all_weights[:, :p1_end]
        w2 = all_weights[:, p1_end:p2_end]
        w3 = all_weights[:, p2_end:]

        # --- 1. Block 1 (Data Qubits) ---
        # input shape: (Batch, 2^N) -> Output: (Batch, 2^N)
        state1 = self.qnode1(input_state_or_latent, w1)
        if isinstance(state1, (list, tuple)):
            state1 = torch.stack(state1, dim=1)
        
        # --- 2. Ancilla Addition ---
        zeros = torch.zeros_like(state1)
        state1_ancilla = torch.cat([state1, zeros], dim=1) # (Batch, 2^{N+1})
        

        # --- 3. Block 2 (Data + Ancilla) ---
        state2 = self.qnode2(state1_ancilla, w2)
        if isinstance(state2, (list, tuple)):
            state2 = torch.stack(state2, dim=1)

        # Renormalize
        state2_sliced = state2[:, :2**self.num_qubits]
        
        epsilon = 1e-8
        norm = torch.norm(state2_sliced, p=2, dim=1, keepdim=True)
        state2_norm = state2_sliced / (norm + epsilon)

        # --- 5. Block 3 (Data Qubits) ---
        final_out = self.qnode3(state2_norm, w3)
        if isinstance(final_out, (list, tuple)):
            final_out = torch.stack(final_out, dim=1)    
        final_out = final_out.float()

        return final_out

# --- [3] Qiskit or PennyLane Hybrid Quantum U-Net ---
class QuantumUNet(nn.Module):
    """
    Hybrid Quantum-Classical U-Net (Optimized for AncillaPennyLanePQC)
    Encoder -> PennyLane PQC (Bottleneck) -> Decoder
    """
    def __init__(self, num_qubits, layers, T, init_variance, betas, activation=False, device='cpu', bottleneck_qubits=4, use_pooling=False):
        super(QuantumUNet, self).__init__()
        
        self.device = torch.device(device)
        self.T = T
        self.betas = betas
        self.init_variance = init_variance
        self.best_loss = float('inf')
        
        # --- Dimensions ---
        # Input: Complex vector of size 2^n
        self.input_dim = 2 ** num_qubits          
        self.input_flat_dim = self.input_dim * 2  # Real + Imag
        
        hidden_dim = (2 ** num_qubits) * 2        # Classical Hidden Layer Size
        self.pqc_input_dim = 2 ** bottleneck_qubits # Dimension for PQC input (Amplitude Embedding etc.)
        
        # --- [1] Encoder ---
        # Complex Input (Real+Imag concatenated) -> Latent Feature
        self.enc1 = nn.Linear(self.input_flat_dim, hidden_dim)
        self.act1 = nn.ReLU() # Activation added for better encoding
        
        self.enc2 = nn.Linear(hidden_dim, self.pqc_input_dim) 
        self.pqc_input_norm = nn.LayerNorm(self.pqc_input_dim)

        # --- [2] Quantum Bottleneck (Ancilla PQC) ---
        # PQC 정의 (AncillaPennyLanePQC 사용)
        self.pqc_layer = PennyLanePQC(num_qubits=bottleneck_qubits, reps=layers)

        # Time-dependent Weights for PQC
        # Shape: (Total Time Steps, Total Parameters in PQC)
        self.pqc_weights = nn.Parameter(
            torch.randn(
                T, self.pqc_layer.total_params, 
                device=self.device
            ) * init_variance
        )

        # --- [3] Decoder ---
        # Input: PQC Output (bottleneck_qubits) + Skip Connection (input_flat_dim)
        # PQC output size is typically 'bottleneck_qubits' (measurements)
        self.dec1 = nn.Linear(bottleneck_qubits + self.input_flat_dim, hidden_dim)
        self.act_dec1 = nn.ReLU()
        
        self.final = nn.Linear(hidden_dim, self.input_flat_dim)

    def forward(self, input, t=None):
        """
        Args:
            input: (Batch, Dim) Complex Tensor (or Real/Imag last dim)
            t: (Batch,) Tensor containing time steps (1 to T)
        """
        # 1. Input Preparation
        # input shape이 (Batch, Dim) 이라고 가정 (Complex Type)
        if input.is_complex():
            x_real = input.real
            x_imag = input.imag
            x_flat = torch.cat([x_real, x_imag], dim=-1) # (Batch, 2*Dim)
        else:
            # 이미 펼쳐져 있거나 실수형인 경우 처리 (상황에 따라 조정)
            x_flat = input 

        batch_size = x_flat.shape[0]

        # --- Encoder Forward ---
        e1 = self.act1(self.enc1(x_flat))    # (Batch, hidden_dim)
        latent = self.enc2(e1)               # (Batch, pqc_input_dim)

        # Normalization for Quantum Embedding
        latent = self.pqc_input_norm(latent)
        # Numerical Stability
        if torch.isnan(latent).any():
            latent = torch.nan_to_num(latent, nan=0.0)
        
        epsilon = 1e-8
        latent = latent / (torch.norm(latent, p=2, dim=-1, keepdim=True) + epsilon)

        # --- Time-dependent Parameter Selection (Vectorized) ---
        if t is None:
            # t가 없으면 첫 번째 시간 단계 가중치 사용 (Batch 전체에 복사)
            selected_weights = self.pqc_weights[0].unsqueeze(0).expand(batch_size, -1)
        else:
            # t: (Batch,) Tensor. 값 범위: 1 ~ T
            # Indexing: t-1 (0 ~ T-1)
            # self.pqc_weights: (T, Params)
            # selected_weights: (Batch, Params) -> 각 샘플에 맞는 가중치를 병렬로 가져옴
            t_idx = (t - 1).long()
            t_idx = torch.clamp(t_idx, 0, self.T - 1) # 안전장치
            selected_weights = self.pqc_weights[t_idx]

        # --- PQC Execution ---
        # latent: (Batch, n_qubits), selected_weights: (Batch, n_params)
        # AncillaPennyLanePQC.forward 내부에서 Batch 처리를 지원해야 함 (이전 턴 수정사항 반영됨)
        pqc_out = self.pqc_layer(latent, selected_weights)

        # [중요] List -> Tensor 및 Double -> Float 변환
        if isinstance(pqc_out, (list, tuple)):
            pqc_out = torch.stack(pqc_out, dim=1)
        
        pqc_out = pqc_out.float() # Double to Float casting

        # --- Decoder Forward ---
        # Skip Connection: PQC Output + Original Input
        d1_input = torch.cat([pqc_out, x_flat.float()], dim=1)
        
        d1 = self.act_dec1(self.dec1(d1_input))
        out_feats = self.final(d1)
        
        # Output Reconstruction
        dim = self.input_dim
        out_real = out_feats[:, :dim]
        out_imag = out_feats[:, dim:]
        
        # Normalize Output
        out = torch.complex(out_real, out_imag)
        out = out / (torch.norm(out, p=2, dim=1, keepdim=True) + epsilon)
        
        return out

    # --- Utils ---
    
    def save_params(self, directory, epoch=None, best=False):
        if best: filename = 'best_model.pt'
        elif epoch is not None: filename = f'epoch{epoch}_model.pt'
        else: filename = 'current_model.pt'
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'epoch': epoch,
            'best_loss': self.best_loss
        }, f'{directory}/{filename}')

    def load_current_params(self, directory, epoch=None):
        filename = f'epoch{epoch}_model.pt' if epoch is not None else 'current_model.pt'
        path = f'{directory}/{filename}'
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {path}")
        except FileNotFoundError:
            print(f"No checkpoint found at {path}")

    def load_best_params(self, directory, noise=None):
        self._load_params_from_file(f'{directory}/best_model.pt')

    def _load_params_from_file(self, path):
        try:
            checkpoint = torch.load(path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.load_state_dict(checkpoint)
            print(f"Loaded model from {path}")
        except FileNotFoundError:
            print(f"No checkpoint found at {path}, skipping...")
        except Exception as e:
            print(f"Error loading model: {e}")

    def update_best_params(self, directory, losses):
        if isinstance(losses, (list, tuple)):
            current_loss = sum(losses)
        elif isinstance(losses, torch.Tensor):
            current_loss = losses.sum().item()
        else:
            current_loss = losses

        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.save_params(directory, best=True)
            
    def get_pqc_params(self):
        return list(self.parameters())

    def get_mlp_params(self):
        return []
    

    