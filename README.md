# V3S-GIA: A Plug-and-Play VAE-based Three-Stage Gradient Inversion Attack Framework Beyond Task-Specific Priors

**[Note to Reviewers]**
This repository contains the official implementation of the paper **"V3S-GIA: A Plug-and-Play VAE-based Three-Stage Gradient Inversion Attack Framework Beyond Task-Specific Priors"**.
This repository is anonymized for double-blind review. All personal information and metadata have been removed.

---

## 🛠️ Installation

### Environment Setup
We provide automated scripts to set up the environment easily.

**Windows Users:**
Simply double-click the `install.bat` file in the root directory.

**Linux / macOS Users:**
Please run the shell script in your terminal:

```bash
# Grant execution permission first
chmod +x install.sh

# Run installation
./install.sh
```

---

## 📥 Pre-trained Models Setup

To reproduce the results, please download the required pre-trained models and place them in the **root directory** of this project.

### Required Files:

1.  **Stable Diffusion VAE (Fine-tuned)**
    * **Folder Name:** `sd-vae-ft-mse`
    * *Instruction:* Please ensure the folder contains `config.json` and model weights (e.g., `diffusion_pytorch_model.bin` and `diffusion_pytorch_model.safetensors`).

2.  **ResNet50 MoCo v2**
    * **File Name:** `moco_v2_800ep_pretrain.pth.tar`

> **Note:** Due to the file size limits of the anonymous repository, we do not host these large weights directly. Please refer to the supplementary materials for download links.

---

## 🚀 Quick Start

After installation and downloading the models, you can run the demo script to evaluate the attack performance:

```bash
python toy.py
```

