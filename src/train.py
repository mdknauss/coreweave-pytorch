import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import wandb
import yaml
from pathlib import Path

from model import SimpleCNN
from utils import log_gpu_stats

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize W&B
wandb.init(
    project="coreweave-pytorch-demo",
    name="cifar10-cnn-modular",
    config=config
)

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

print(f"🚀 Training on {device} → {torch.cuda.get_device_name(0)}")

# Training loop
best_acc = 0.0
best_model_path = "best_model.pth"

for epoch in range(config['epochs']):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if batch_idx % 50 == 49:
            wandb.log({
                "train/loss": running_loss / 50,
                "train/accuracy": 100. * correct / total,
                "train/epoch": epoch + 1,
            })
            log_gpu_stats()
            running_loss = 0.0
            correct = 0
            total = 0

    epoch_acc = 100. * correct / total if total > 0 else 0
    print(f"Epoch {epoch+1}/{config['epochs']} → Accuracy: {epoch_acc:.2f}%")

    if epoch_acc > best_acc:
        best_acc = epoch_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"   New best accuracy: {best_acc:.2f}% - Model saved")

# Log final model artifact
artifact = wandb.Artifact("cifar10-cnn-model", type="model", description="Best SimpleCNN trained on CIFAR-10")
artifact.add_file(best_model_path)
wandb.log_artifact(artifact)

print("\n✅ Training completed! Best accuracy:", f"{best_acc:.2f}%")
wandb.finish()