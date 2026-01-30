# Enhancing Quantum Diffusion Models for Complex Image Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PennyLane](https://img.shields.io/badge/PennyLane-Powered-orange.svg)](https://pennylane.ai/)
[![Pytorch](https://img.shields.io/badge/PyTorch-Enabled-red.svg)](https://pytorch.org/)

---
> DOI: 10.13140/RG.2.2.22364.04483
---

> **QAMP 2025 Project** > *Enhancing Quantum Diffusion Models for Complex Image Generation*

This repository contains the official implementation of the **Hybrid Quantum-Classical U-Net**, a novel architecture designed to overcome the scalability and expressibility limitations of Quantum Diffusion Models (QDMs). By integrating a **Quantum Bottleneck** with **Adaptive Non-Local Observables (ANO)** into a classical U-Net structure, this model successfully generates multi-class MNIST digits (0-9) with high structural coherence.

---

## Key Features

* **Hybrid Architecture:** Combines the feature extraction power of classical encoders/decoders with the high-dimensional expressivity of a Quantum Bottleneck ($N=4$ qubits).
* **Adaptive Non-Local Observables (ANO):** Utilizes trainable Hermitian observables to extract rich, non-local features from the quantum state, solving the "measurement bottleneck" of standard VQCs.
* **Skip Connections:** Mitigates information loss during the quantum compression phase (256 dims $\to$ 4 qubits), ensuring image sharpness.
* **Multi-Class Generation:** Successfully mitigates mode collapse, generating distinct samples for all 10 digit classes (0-9), a significant improvement over prior binary-class quantum diffusion models.

## Model Architecture

The model follows a U-Net design where the bottleneck layer is replaced by a Parameterized Quantum Circuit (PQC).

1.  **Classical Encoder:** Compresses $16 \times 16$ input images into a latent vector.
2.  **Quantum Bottleneck:** Maps latent vectors to a 4-qubit Hilbert space via Amplitude Encoding and processes them with a Variational Quantum Circuit (VQC).
3.  **ANO Measurement:** Extracts quantum features using trainable measurements.
4.  **Classical Decoder:** Reconstructs the image, aided by Skip Connections from the encoder.

## Code Available

The source codes are in `/implementation/source` directory. Please check `README` in there.