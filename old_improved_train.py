import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import wandb
from pathlib import Path

# ========================= W&B Setup =========================
wandb.init(
    project="coreweave-pytorch-demo",
    name="cifar10-cnn-improved",
    config={
        "model": "SimpleCNN",
        "dataset": "CIFAR10",
        "epochs": 8,
        "batch_size": 128,
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "gpu": torch.cuda.get_device_name(0)
    }
)

config = wandb.config

# ========================= Data =========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(
    trainset, 
    batch_size=config.batch_size, 
    shuffle=True, 
    num_workers=4, 
    pin_memory=True
)

# ========================= Model =========================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

print(f"🚀 Training on {device} → {config.gpu} | Batch size: {config.batch_size}")

# ========================= Training Loop =========================
best_acc = 0.0
best_model_path = "best_model.pth"

for epoch in range(config.epochs):
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

        # Log every 50 batches
        if batch_idx % 50 == 49:
            wandb.log({
                "train/loss": running_loss / 50,
                "train/accuracy": 100. * correct / total,
                "train/epoch": epoch + 1,
                "gpu/memory_allocated_gb": torch.cuda.memory_allocated(0) / 1e9,
                "gpu/memory_reserved_gb": torch.cuda.memory_reserved(0) / 1e9,
            })
            running_loss = 0.0
            correct = 0
            total = 0

    # Epoch summary
    epoch_acc = 100. * correct / total if total > 0 else 0
    print(f"Epoch {epoch+1}/{config.epochs} → Accuracy: {epoch_acc:.2f}%")

    # Save best model
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"   New best accuracy: {best_acc:.2f}% - Model saved")

print("\n✅ Training completed successfully!")
print(f"Best accuracy: {best_acc:.2f}%")

# Log model as W&B artifact
artifact = wandb.Artifact("cifar10-cnn-model", type="model", description="Best SimpleCNN on CIFAR-10")
artifact.add_file(best_model_path)
wandb.log_artifact(artifact)

wandb.finish()