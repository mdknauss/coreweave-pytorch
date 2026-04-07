# CoreWeave PyTorch + Weights & Biases Demo

**Portfolio Project** — Built as preparation for the **AI Solutions Engineer (Post-Sales - W&B)** role at CoreWeave.

This project demonstrates a clean, production-ready workflow for training PyTorch models on GPU infrastructure using Docker, modular code structure, and full experiment tracking with Weights & Biases.

### Key Features

- **Modular architecture** (`src/` package) – easy to maintain and extend
- Full **Weights & Biases** integration (live metrics, artifacts, GPU monitoring)
- Containerized environment using NVIDIA PyTorch Docker image
- GPU-accelerated training with proper memory management
- Model checkpointing and artifact logging to W&B
- Ready for scaling to multi-GPU / distributed training on CoreWeave

### Tech Stack

- **PyTorch** + CUDA
- **Weights & Biases** (experiment tracking & model artifacts)
- **Docker** + NVIDIA Container Toolkit
- **VS Code** Remote Containers
- Python 3.10+

### Project Structure

```bash
coreweave-pytorch/
├── src/
│   ├── model.py          # SimpleCNN model definition
│   ├── train.py          # Training loop with W&B logging
│   └── utils.py          # GPU logging utilities
├── inference.py
├── config.yaml
├── requirements.txt
├── docker-compose.yml
├── README.md
└── .gitignore

Quick Start:

# 1. Start the environment
docker compose up -d

# 2. Enter the container
docker compose exec pytorch bash

# 3. Install dependencies and login to W&B
pip install -r requirements.txt
wandb login

# 4. Train the model
python src/train.py
