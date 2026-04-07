import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
import sys

from src.model import SimpleCNN

# ====================== CONFIG ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model_v2.pth"          # Use the new model from v2 training

# Reliable test image URL (airplane)
TEST_IMAGE_URL = "https://picsum.photos/id/1015/640/480"

# ====================== LOAD MODEL ======================
try:
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"✅ Model loaded successfully on {device} → {MODEL_PATH}")
except FileNotFoundError:
    print(f"❌ ERROR: Model file '{MODEL_PATH}' not found.")
    print("   Please run 'python src/train_v2.py' first to train the model.")
    sys.exit(1)
except Exception as e:
    print(f"❌ ERROR loading model: {e}")
    sys.exit(1)

# ====================== DOWNLOAD IMAGE ======================
print(f"Downloading test image from: {TEST_IMAGE_URL}")
try:
    response = requests.get(TEST_IMAGE_URL, timeout=10)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content)).convert("RGB")
    print("✅ Image downloaded successfully")
except Exception as e:
    print(f"❌ Failed to download image: {e}")
    sys.exit(1)

# ====================== PREPROCESS IMAGE ======================
transform = transforms.Compose([
    transforms.Resize((32, 32)),                    # Match CIFAR-10 size
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

input_tensor = transform(img).unsqueeze(0).to(device)

# ====================== RUN INFERENCE ======================
with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class = output.argmax(dim=1).item()
    confidence = probabilities[predicted_class].item() * 100

# CIFAR-10 classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

print("\n" + "="*65)
print(f"Predicted class     : {classes[predicted_class].upper()}")
print(f"Confidence          : {confidence:.2f}%")
print("="*65)

# Show top 3 predictions
print("\nTop 3 predictions:")
top3_prob, top3_idx = torch.topk(probabilities, 3)
for i in range(3):
    print(f"  {i+1}. {classes[top3_idx[i]]:<12} {top3_prob[i].item()*100:.2f}%")

# Optional: Save this inference result to W&B (nice for portfolio)
try:
    import wandb
    wandb.init(project="coreweave-pytorch-demo", name="inference-test", resume="allow")
    wandb.log({
        "inference/predicted_class": classes[predicted_class],
        "inference/confidence": confidence
    })
    wandb.finish()
except:
    pass