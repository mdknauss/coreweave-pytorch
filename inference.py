import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
import wandb

from src.model import SimpleCNN

# Load best model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

print("✅ Model loaded successfully for inference")

# Example: Download a test image and run inference
# (You can replace the URL with any CIFAR-10 like image)
url = "https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane.png"
response = requests.get(url)
img = Image.open(BytesIO(response.content))

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

input_tensor = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(input_tensor)
    predicted_class = output.argmax(dim=1).item()

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']

print(f"Predicted class: {classes[predicted_class]}")