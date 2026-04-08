import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
import sys

from src.model import ImprovedCNN

# ====================== CONFIG ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model_v4.pth"

# Test image URL (airplane)
TEST_IMAGE_URL = "https://picsum.photos/id/1015/640/480"

# Minimum confidence threshold
CONFIDENCE_THRESHOLD = 60.0   # If below this, we'll be cautious

# ====================== LOAD MODEL ======================
try:
    model = ImprovedCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"✅ Model loaded successfully on {device} → {MODEL_PATH}")
except Exception as e:
    print(f"❌ ERROR loading model: {e}")
    sys.exit(1)

# ====================== DOWNLOAD IMAGE ======================
print(f"Downloading test image from: {TEST_IMAGE_URL}")
try:
    response = requests.get(TEST_IMAGE_URL, timeout=10)
    response.raise_for_status()
    original_img = Image.open(BytesIO(response.content)).convert("RGB")
    print(f"✅ Image downloaded successfully (original size: {original_img.size})")
except Exception as e:
    print(f"❌ Failed to download image: {e}")
    sys.exit(1)

# ====================== IMPROVED PREPROCESSING ======================
# Better preprocessing to match CIFAR-10 style more closely
transform = transforms.Compose([
    transforms.Resize(36),                    # Slightly larger first
    transforms.CenterCrop(32),                # Center crop to 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

input_tensor = transform(original_img).unsqueeze(0).to(device)

print(f"Processed image shape: {input_tensor.shape}")

# ====================== RUN INFERENCE ======================
with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class_idx = output.argmax(dim=1).item()
    confidence = probabilities[predicted_class_idx].item() * 100

# CIFAR-10 classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

predicted_class = classes[predicted_class_idx]

print("\n" + "="*70)
print(f"Predicted class     : {predicted_class.upper()}")
print(f"Confidence          : {confidence:.2f}%")
print("="*70)

# Top 3 predictions
print("\nTop 3 predictions:")
top3_prob, top3_idx = torch.topk(probabilities, 3)
for i in range(3):
    print(f"  {i+1}. {classes[top3_idx[i]]:<12} {top3_prob[i].item()*100:.2f}%")

# Confidence-based feedback
if confidence < CONFIDENCE_THRESHOLD:
    print(f"\n⚠️  Low confidence ({confidence:.1f}%). The model may be uncertain or the image is quite different from training data.")
elif predicted_class == "airplane":
    print("\n🎯 Good prediction! The model correctly identified the airplane.")
else:
    print(f"\n⚠️  The model is confident but may be wrong. This image is quite different from the CIFAR-10 training set.")

# Optional: Log to W&B
try:
    import wandb
    wandb.init(project="coreweave-pytorch-demo", name="inference-test-v4", resume="allow")
    wandb.log({
        "inference/predicted_class": predicted_class,
        "inference/confidence": confidence,
        "inference/top1_confidence": confidence
    })
    wandb.finish()
except:
    pass