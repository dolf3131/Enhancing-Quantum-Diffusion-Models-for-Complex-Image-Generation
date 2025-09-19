# Model Workflow

```mermaid
graph TD
    subgraph "Forward Process (Noise Addition)"
        A[F-MNIST Image Data] --> B{Quantum State Encoding};
        B --> C[Clean Quantum State <br>|ψ(0)⟩];
        C --> D{Forward Diffusion Circuit <br>(Adding Gaussian Noise)};
        D --> E[Noisy Quantum State <br>|ψ(t)⟩];
    end

    subgraph "Reverse Process (Noise Removal)"
        E --> F{Variational Quantum Circuit <br>(VQC)};
        subgraph "Classical Optimization"
            H[Loss Function <br>(e.g., Fidelity, KL Divergence)] --> I{Classical Optimizer};
            I --> F;
        end
        F --> G[Denoised Quantum State <br>|ψ'(0)⟩];
        G --> J{Quantum State Measurement};
    end

    subgraph "Outcome & Evaluation"
        J --> K[Generated F-MNIST Image];
        K --> L[Comparison with Real Data];
        L --> M[Model Evaluation <br>(FID, IS, etc.)];
    end