import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import wandb
import yaml
from pathlib import Path

from model import SimpleCNN
from utils import log_gpu_stats

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# ====================== W&B INIT ======================
wandb.init(
    project="coreweave-pytorch-demo",
    name="v2-with-augmentation",
    config=config
)

# ====================== DATA AUGMENTATION ======================
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),           # Random crop + padding
    transforms.RandomHorizontalFlip(),              # Randomly flip image
    transforms.RandomRotation(15),                  # Small random rotation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)

# Create train/validation split
train_size = int(0.8 * len(trainset))
val_size = len(trainset) - train_size
trainset, valset = random_split(trainset, [train_size, val_size])

trainloader = DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
valloader = DataLoader(valset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

# ====================== MODEL ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

print(f"🚀 Training v2 with augmentation on {device} → {torch.cuda.get_device_name(0)}")
print(f"Training samples: {len(trainset)} | Validation samples: {len(valset)}")

# ====================== TRAINING LOOP ======================
best_val_acc = 0.0

for epoch in range(config['epochs']):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()

    # Validation phase
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0.0

    with torch.no_grad():
        for inputs, labels in valloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    train_acc = 100. * train_correct / train_total
    val_acc = 100. * val_correct / val_total

    wandb.log({
        "train/loss": train_loss / len(trainloader),
        "train/accuracy": train_acc,
        "val/loss": val_loss / len(valloader),
        "val/accuracy": val_acc,
        "epoch": epoch + 1,
    })

    log_gpu_stats()

    print(f"Epoch {epoch+1}/{config['epochs']} | "
          f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model_v2.pth")
        print(f"   New best validation accuracy: {best_val_acc:.2f}% - Model saved")

print("\n✅ Training v2 completed!")
print(f"Best validation accuracy: {best_val_acc:.2f}%")
wandb.finish()
