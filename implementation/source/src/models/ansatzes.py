import torch
import torch.nn as nn
import pennylane as qml
import numpy as np

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

class WeightProbe:
    """
    Dummy class to probe maximum index accessed in weights tensor.
    Used for estimating number of parameters needed without actual values.
    """
    def __init__(self, parent=None, offset=0):
        self.max_idx = -1 # Track maximum index accessed
        self.parent = parent  # 원본 Probe를 기억함 (분신인 경우)
        self.offset = offset

    def __getitem__(self, key):
        if isinstance(key, int):
            current_idx = self.offset + key
            if self.parent:
                self.parent.report_idx(current_idx)
            else:
                self.report_idx(current_idx)
            return 0.1 # 
        
        elif isinstance(key, slice):
            start = key.start if key.start is not None else 0
            # [핵심] 새로운 '분신 Probe'를 만들어서 반환!
            # 이 분신은 값을 꺼낼 때마다 원본(self)에게 위치를 보고함
            return WeightProbe(parent=(self.parent if self.parent else self), 
                               offset=self.offset + start)
            
    def report_idx(self, idx):
        """인덱스 갱신 (가장 큰 인덱스 기억)"""
        self.max_idx = max(self.max_idx, idx)

    @property
    def shape(self):
        return (1000,)
    


def ConvUnit(params, wires):
    """
    User-defined 2-Qubit Convolution Layer (3 params)
    Args:
        params: Flat tensor containing all parameters
        wires: [control, target]
    Returns:
        Updated param_idx
    """
    idx = 0
    
    qml.RZ(-np.pi/2, wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    
    qml.RZ(params[idx], wires=wires[0]); idx += 1
    qml.RY(params[idx], wires=wires[1]); idx += 1
    
    qml.CNOT(wires=[wires[0], wires[1]])
    
    qml.RY(params[idx], wires=wires[1]); idx += 1
    
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RZ(np.pi/2, wires=wires[0])
    
    return idx

def MixingBlock_U3(params, wires):
    """
    [Improved] U3 Mixing
    Simulates particle/information exchange between qubits.
    Better for diffusion processes.
    """
    n = len(wires)
    idx = 0
    # 1. Global Rotation (Mix basis)
    for i in range(n):
        qml.U3(params[idx], params[idx+1], params[idx+2], wires=wires[i])
        idx += 3
    # 2. Entangling Layers
    # Even pairs
    for i in range(0, n-1, 2):
        qml.CNOT(wires=[wires[i], wires[(i+1)%n]])

    # Odd pairs
    for i in range(1, n, 2):
        qml.CNOT(wires=[wires[i], wires[(i+1)%n]])
    return idx

def MixingBlock(params, wires):
    """
    [Improved] XY Mixing
    Simulates particle/information exchange between qubits.
    Better for diffusion processes.
    """
    n = len(wires)
    idx = 0
    # 1. Global Rotation (Mix basis)
    for i in range(n):
        qml.RY(params[idx], wires=wires[i]); idx += 1

    # 2. Entangling Layers
    # Even pairs
    for i in range(0, n-1, 2):
        qml.CNOT(wires=[wires[i], wires[(i+1)%n]])
        qml.RZ(params[idx], wires=wires[(i+1)%n])
        qml.CNOT(wires=[wires[i], wires[(i+1)%n]])
        idx += 1

    # Odd pairs
    for i in range(1, n, 2):
        qml.CNOT(wires=[wires[i], wires[(i+1)%n]])
        qml.RZ(params[idx], wires=wires[(i+1)%n])
        qml.CNOT(wires=[wires[i], wires[(i+1)%n]])
        idx += 1
    return idx


class PennyLanePQC(nn.Module):
    def __init__(self, num_qubits, reps=2, activation=True):
        super(PennyLanePQC, self).__init__()
        self.num_qubits = num_qubits
        self.reps = reps
        self.activation = activation
        if self.activation:
            self.act_func = ComplexLeakyReLU()
        
        # Block 1 & 3 (N qubits)
        self.params_per_rep = self._count_params_per_rep(num_qubits)
        self.params_per_block = self.params_per_rep * reps
        self.total_params = self.params_per_block * 3

        print(f"[Auto-Config] Total params to learn: {self.total_params}")

        # --- Devices ---
        self.dev_n = qml.device("default.qubit", wires=num_qubits)
        self.dev_ancilla = qml.device("default.qubit", wires=num_qubits + 1)

        # --- QNodes 정의 ---
        self.qnode1 = self._create_qnode_state(self.dev_n, num_qubits)
        self.qnode2 = self._create_qnode_state(self.dev_ancilla, num_qubits + 1)
        self.qnode3 = self._create_qnode_expval(self.dev_n, num_qubits)

    def _layer_ansatz(self, layer_params, wires):
        """
        Single layer ansatz with weight sharing.
        Args:
            layer_params: 1D Tensor of parameters for this layer
            wires: List of qubit indices
        """
        n_wires = len(wires)
        
        ptr = 0
        
        # 1. Conv Even (3 params -> or 9 params)
        # n_conv = 3 # IsingZZ 등으로 바꾸면 이 숫자만 9로 변경 등
        # for i in range(0, n_wires-1, 2):
        #     ConvUnit(layer_params[ptr:ptr+n_conv], wires=[wires[i], wires[(i+1)%n_wires]])
        #     ptr += n_conv
        
        # # 2. Conv Odd
        # for i in range(1, n_wires, 2):
        #     ConvUnit(layer_params[ptr:ptr+n_conv], wires=[wires[i], wires[(i+1)%n_wires]])
        #     ptr += n_conv

            
        # 3. Mixing
        ptr += MixingBlock_U3(layer_params[ptr:], wires=wires)
        
        

    def _count_params_per_rep(self, n_wires):
        """
        Estimate number of parameters per repetition using WeightProbe.
        Args:
            n_wires: Number of qubits (wires)
        Returns:
            total_params: Estimated number of parameters per repetition
        """
        probe = WeightProbe()
        wires = range(n_wires)
      
        try:
            idx = 0
            self._layer_ansatz(probe, wires)
            
        except Exception as e:
            pass
            
        return probe.max_idx + 1

    def _circuit_ansatz(self, weights, wires):
        """실제 실행용 (파라미터 카운팅 로직과 구조 동일해야 함)"""
        # weights shape: (reps, params_per_rep)
        for d in range(self.reps):
            self._layer_ansatz(weights[d], wires)
            

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
        
        p_len = self.params_per_block
        
        w1 = all_weights[:, :p_len]
        w2 = all_weights[:, p_len:2*p_len]
        w3 = all_weights[:, 2*p_len:]

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
        if self.activation:
            state2_norm = self.act_func(state2_norm)
            state2_norm = state2_norm / torch.norm(state2_norm.abs(), p=2, dim=1, keepdim=True)


        # --- 5. Block 3 (Data Qubits) ---
        final_out = self.qnode3(state2_norm, w3)
        if isinstance(final_out, (list, tuple)):
            final_out = torch.stack(final_out, dim=1)    
        final_out = final_out.float()

        return final_out

# --- PennyLane Hybrid Quantum U-Net ---
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
        self.pqc_layer = PennyLanePQC(num_qubits=bottleneck_qubits, reps=layers, activation=activation)

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
            selected_weights = self.pqc_weights[0].unsqueeze(0).expand(batch_size, -1)
        else:
            # t: (Batch,) Tensor. 값 범위: 1 ~ T
            # Indexing: t-1 (0 ~ T-1)
            # self.pqc_weights: (T, Params)
            t_idx = (t - 1).long()
            t_idx = torch.clamp(t_idx, 0, self.T - 1) # 안전장치
            selected_weights = self.pqc_weights[t_idx]

        # --- PQC Execution ---
        # latent: (Batch, n_qubits), selected_weights: (Batch, n_params)
        pqc_out = self.pqc_layer(latent, selected_weights)

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
   

    