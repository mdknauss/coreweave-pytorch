import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
import sys

from src.model import SimpleCNN

# ====================== CONFIG ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model.pth"

# List of fallback image URLs in case one fails
TEST_IMAGE_URLS = [
    "https://picsum.photos/id/1015/640/480",   # Airplane
    "https://picsum.photos/id/1016/640/480",   # Another airplane
    "https://picsum.photos/id/870/640/480",    # Plane in sky
    "https://picsum.photos/id/201/640/480",    # Aviation related
]

# ====================== LOAD MODEL ======================
try:
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"✅ Model loaded successfully on {device}")
except FileNotFoundError:
    print(f"❌ ERROR: Model file '{MODEL_PATH}' not found.")
    print("   Please run training first (python src/train.py)")
    sys.exit(1)
except Exception as e:
    print(f"❌ ERROR loading model: {e}")
    sys.exit(1)

# ====================== DOWNLOAD IMAGE WITH ERROR HANDLING ======================
img = None
for url in TEST_IMAGE_URLS:
    try:
        print(f"Downloading test image from: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()        # Raises HTTPError for bad responses (404, 403, etc.)
        
        img = Image.open(BytesIO(response.content)).convert("RGB")
        print("✅ Image downloaded successfully")
        break
        
    except requests.exceptions.HTTPError as e:
        print(f"   HTTP Error ({url}): {e.response.status_code} - Trying next URL...")
    except requests.exceptions.RequestException as e:
        print(f"   Network error ({url}): {e} - Trying next URL...")
    except Exception as e:
        print(f"   Unexpected error with image ({url}): {e} - Trying next URL...")

if img is None:
    print("❌ ERROR: Failed to download any test image.")
    print("   Please check your internet connection or provide a local image path.")
    sys.exit(1)

# ====================== PREPROCESS & INFERENCE ======================
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

input_tensor = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class = output.argmax(dim=1).item()
    confidence = probabilities[predicted_class].item() * 100

# CIFAR-10 class names
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

print("\n" + "="*60)
print(f"Predicted class     : {classes[predicted_class].upper()}")
print(f"Confidence          : {confidence:.2f}%")
print("="*60)

# Top 3 predictions
print("\nTop 3 predictions:")
top3_prob, top3_idx = torch.topk(probabilities, 3)
for i in range(3):
    print(f"  {i+1}. {classes[top3_idx[i]]:<12} {top3_prob[i].item()*100:.2f}%")