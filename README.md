# [QAMP Project Proposal: Enhancing Quantum Diffusion Models for Complex Image Generation]

## Description
This project aims to explore and enhance the capabilities of Quantum Diffusion Models (QDMs), a novel class of quantum generative models, by building upon the foundational work presented in [arXiv:2311.15444.](https://arxiv.org/pdf/2311.15444) The original paper successfully demonstrated a QDM for generating simple handwritten digits from the MNIST dataset. Our project will begin by replicating these baseline results to gain a deep, practical understanding of the model's architecture and training dynamics.

The core of the project involves systematically improving the original QDM. We will investigate several enhancement strategies, including designing more expressive and hardware-efficient quantum circuit ansatze, implementing advanced data encoding techniques, and exploring alternative optimizers like the Quantum Natural Gradient to mitigate issues such as barren plateaus. A key focus will be on integrating quantum error mitigation techniques to improve the model's performance and robustness on noisy NISQ-era hardware.

Finally, we will validate the enhanced model by applying it to a more complex dataset, Fashion-MNIST (FMNIST). By successfully generating these more intricate images, we aim to demonstrate the improved generative power and scalability of our enhanced QDM, thereby contributing a valuable case study and open-source implementation to the quantum machine learning community.

## Deliverables
*List the expected outcomes and deliverables for this project. Consider including both primary deliverables and a minimal viable project (MVP) in case the full scope cannot be completed within 3 months.*

**Primary Deliverables:**
- [ ] An interactive Jupyter notebook comparing the performance of at least two enhancement strategies (e.g., improved circuit ansatz, alternative optimizer) against the baseline model on the MNIST dataset.
- [ ] A full implementation of the best-performing enhanced QDM trained on the FMNIST dataset, including code for data preprocessing, model training, and image generation.
- [ ] A technical blog post or a final report summarizing the project's methodology, key findings, challenges encountered, and quantitative/qualitative analysis of the generated images.

**Minimal Viable Product (MVP):**
- [ ] A clean, well-documented Jupyter notebook that successfully reproduces the results of the original QDM paper (arXiv:2311.15444) on the MNIST dataset using Qiskit.

## Mentors

### If you are a MENTEE seeking a mentor:
**Type of mentor sought:** I am seeking a mentor with a strong background in Quantum Machine Learning (QML), particularly with experience in variational quantum algorithms (VQAs) and quantum generative models. The ideal mentor would have deep knowledge of Qiskit, hands-on experience running experiments on real quantum hardware, and an understanding of the challenges involved, such as noise and barren plateaus. Experience with quantum error mitigation techniques and classical generative models (like diffusion models or GANs) would be a significant plus.

**Desired mentoring style:** I would appreciate a collaborative and guidance-oriented mentoring style. I am looking for a mentor who can provide high-level direction, help brainstorm solutions to technical challenges, and offer feedback on code and research methodology. Regular weekly or bi-weekly check-ins to discuss progress and plan next steps would be ideal.

## Mentees

### If you are a MENTEE proposing this project:
**Name:** Jeongbin Jo 
**GitHub:** @dolf3131 
**What I do:** I am a student/researcher with a background in physics and computational science. I have hands-on experience in quantum computing, specifically with implementing algorithms such as VQA, QAOA, and various QML models using Qiskit and PennyLane. I am deeply interested in the intersection of state-of-the-art machine learning and quantum computation, with a focus on developing practical quantum advantages.

**Learning goals:** Through this project, I aim to gain in-depth expertise in building, training, and scaling a cutting-edge quantum generative model. I want to move beyond textbook algorithms and tackle a real research problem, learning to navigate the practical challenges of NISQ-era hardware. My goals are to deepen my understanding of advanced circuit design, quantum error mitigation, and optimization techniques, and ultimately, to produce a high-quality open-source project that contributes to the Qiskit community.

---

**Additional Information:**
- **Time commitment (for mentees):** I am able to commit approximately 10+ hours per week to this project.
- **Prerequisites:**
    - Proficiency in Python and experience with scientific computing libraries (NumPy, Matplotlib).
    - Solid hands-on experience with Qiskit for building and simulating quantum circuits.
    - A strong theoretical understanding of quantum computing fundamentals (qubits, gates, entanglement, measurement).
    - Familiarity with the core concepts of variational quantum algorithms and classical machine learning (especially generative models is a plus).
- **Resources:** A standard development environment with Python and Qiskit installed. Access to cloud-based quantum simulators and IBM Quantum hardware will be required for experiments.

---

Checklist after creating this issue:
- [ ] Submit the [2025 Mentor Sign-up Form](https://airtable.com/appjU5TOUDYPwBIqO/pagKT6723KMerYQyc/form) if I am a mentor
OR
- [x] Submit the 2025 QAMP Application Form if I am a mentee (link in Slack and Discord)

---

## Plan

| Phase | Week | Topic | Detailed Objectives |
| :--- | :--- | :--- | :--- |
| **Phase 1: Foundational Research & Data Preprocessing** | **Week 1** | In-depth Paper Analysis & Data Preprocessing | - Thoroughly understand key concepts of the `arXiv:2311.15444` paper. <br> - Devise a preprocessing strategy for the F-MNIST dataset. <br> - Set up `Qiskit` or `PennyLane` environment and test a basic encoding circuit. |
| | **Week 2** | Noise Study & Performance Metrics | - Learn how to apply noise models in quantum simulation. <br> - Select and research key performance metrics (e.g., FID, IS) for the quantum diffusion model. |
| --- | --- | --- | --- |
| **Phase 2: Model Design, Implementation & Optimization** | **Week 3** | Model Architecture Design & Implementation (1) | - Design a **Parameterized Quantum Circuit (PQC)** optimized for F-MNIST. <br> - Implement the **forward diffusion process** circuit. |
| | **Week 4** | Model Architecture Design & Implementation (2) | - Design and implement a **Variational Quantum Circuit (VQC)** to predict the **noise score**. <br> - Conduct initial tests to explore how circuit depth and width impact performance. |
| | **Week 5** | Hybrid Learning Loop Setup & Testing | - Integrate the classical optimizer with the quantum simulator to create a hybrid learning loop. <br> - Implement the chosen loss function and perform training tests on a small dataset. |
| | **Week 6** | F-MNIST Training (1) | - Start initial model training on grayscale F-MNIST images. <br> - Monitor the training process and debug any issues that arise. |
| | **Week 7** | F-MNIST Training (2) | - Optimize model parameters and perform hyperparameter tuning. <br> - Analyze changes in the quality of generated images as the model learns. |
| | **Week 8** | Model Enhancement & Error Mitigation | - Analyze model performance to identify bottlenecks. <br> - Attempt to integrate **error mitigation** techniques into the training process to improve performance in noisy environments. |
| --- | --- | --- | --- |
| **Phase 3: Evaluation & Final Reporting** | **Week 9** | Performance Evaluation & Qualitative Analysis | - Calculate quantitative metrics like **FID** and **IS** for generated images. <br> - Perform qualitative analysis by visualizing the latent space using methods like **t-SNE**. |
| | **Week 10** | Result Compilation & Visualization | - Compile all experimental data (loss curves, metrics, etc.). <br> - Create visual materials like image galleries and graphs for the report. |
| | **Week 11** | Research Report Draft | - Draft the research report including the introduction, methodology, results, and discussion. |
| | **Week 12** | Final Report & Presentation Preparation | - Finalize the research report based on mentor feedback. <br> - Prepare presentation slides for the project's conclusion. |