import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import wandb
import yaml

from model import ImprovedCNN
from utils import log_gpu_stats

# ====================== LOAD CONFIG ======================
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

wandb.init(
    project="coreweave-pytorch-demo",
    name="v4-improved-cnn",
    config=config
)

# ====================== DATA AUGMENTATION ======================
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(12),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
train_size = int(0.85 * len(trainset))
val_size = len(trainset) - train_size
trainset, valset = random_split(trainset, [train_size, val_size])

trainloader = DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
valloader = DataLoader(valset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

# ====================== MODEL ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImprovedCNN().to(device)

criterion = torch.nn.CrossEntropyLoss()

# ====================== OPTIMIZER FROM CONFIG ======================
optimizer_name = config.get('optimizer', 'Adam').lower()

if optimizer_name == 'adam':
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-4)
    )
elif optimizer_name == 'sgd':
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['learning_rate'],
        momentum=config.get('momentum', 0.9),
        weight_decay=config.get('weight_decay', 1e-4)
    )
else:
    print(f"Warning: Unknown optimizer '{optimizer_name}'. Using Adam instead.")
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='max', 
    factor=0.5, 
    patience=6, 
    min_lr=1e-6
)

print(f"🚀 Training v4 - ImprovedCNN using {optimizer_name.upper()} optimizer on {device}")

best_val_acc = 0.0
patience_counter = 0
max_patience = 12

for epoch in range(config['epochs']):
    # Training phase
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

    scheduler.step(val_acc)
    current_lr = optimizer.param_groups[0]['lr']

    wandb.log({
        "train/loss": train_loss / len(trainloader),
        "train/accuracy": train_acc,
        "val/loss": val_loss / len(valloader),
        "val/accuracy": val_acc,
        "learning_rate": current_lr,
        "epoch": epoch + 1,
    })

    log_gpu_stats()

    print(f"Epoch {epoch+1:2d}/{config['epochs']} | "
          f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | LR: {current_lr:.6f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model_v4.pth")
        print(f"   New best val accuracy: {best_val_acc:.2f}% - Model saved")
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= max_patience:
        print(f"Early stopping triggered at epoch {epoch+1}")
        break

print("\n✅ Training v4 completed!")
print(f"Best validation accuracy: {best_val_acc:.2f}%")
wandb.finish()