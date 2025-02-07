import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np

# ========== CONFIGURATIONS ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = 224
num_classes = 6

# ========== RESNET50 MODEL ==========
def build_resnet50():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    num_features = model.fc.in_features

    # Fine-tune only last 30 layers
    for param in list(model.parameters())[:-30]:
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes),
        nn.Softmax(dim=1)
    )

    return model.to(device)

# Load the model
model = build_resnet50()
model.load_state_dict(torch.load("best_model_apple.pth", map_location=device, weights_only=True))
model.eval()  # Set the model to evaluation mode

# ========== IMAGE PREPROCESSING ==========
def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply custom preprocessing (CLAHE)
    image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel = image_lab[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    image_lab[:, :, 0] = l_channel
    image = cv2.cvtColor(image_lab, cv2.COLOR_LAB2RGB)

    # Resize and normalize the image
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = Image.fromarray(image)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)

# ========== PREDICTION FUNCTION ==========
def predict_image(image_path, model, class_names):
    # Preprocess the image
    image = preprocess_image(image_path)

    # Perform inference
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = predicted.item()

    # Get the class name
    predicted_class_name = class_names[predicted_class]
    return predicted_class_name

# ========== CLASS NAMES ==========
# Replace with your actual class names
class_names = {
    0: "leaf spot",
    1: "apple scab",
    2: "black rot",
    3: "cedar rust",
    4: "healthy",
    5: "powdery mildew"
}

# ========== TEST ON A REAL-WORLD IMAGE ==========
image_path = r"C:\Users\ASUS\Downloads\black_rot_apple.jpeg"
predicted_class = predict_image(image_path, model, class_names)
print(f"Predicted Class: {predicted_class}")