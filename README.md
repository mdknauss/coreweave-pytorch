# CoreWeave PyTorch + Weights & Biases Demo

Clean, modular PyTorch training example built as preparation for the **AI Solutions Engineer (Post-Sales - W&B)** role at CoreWeave.

## Features
- Modular code structure (`src/` package)
- Full experiment tracking with Weights & Biases
- GPU memory & utilization logging
- Model checkpointing + W&B Artifacts
- Docker + NVIDIA container ready
- VS Code Remote Container support

## Quick Start

```bash
# 1. Inside the container
pip install -r requirements.txt
wandb login

# 2. Train the model
python src/train.py

coreweave-pytorch-demo/
├── src/
│   ├── model.py
│   ├── train.py
│   └── utils.py
├── inference.py
├── README.md
├── requirements.txt
├── config.yaml
├── .gitignore
└── best_model.pth

Tech Stack

PyTorch + CUDA (RTX 2080 SUPER)
Weights & Biases for experiment tracking
Docker + NVIDIA PyTorch container
VS Code Remote - Containers

This project demonstrates the type of reproducible, containerized workflows I help customers implement on CoreWeave's GPU cloud.