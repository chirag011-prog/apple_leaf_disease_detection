import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# ========== CONFIGURATIONS ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define dataset directories
train_dir = r"C:\Users\ASUS\Downloads\Apple Leaf Disease\AppleLeafDisease- dataset\split_dataset\train"
val_dir = r"C:\Users\ASUS\Downloads\Apple Leaf Disease\AppleLeafDisease- dataset\split_dataset\val"
test_dir = r"C:\Users\ASUS\Downloads\Apple Leaf Disease\AppleLeafDisease- dataset\split_dataset\test"

img_size = 224
batch_size = 64
num_classes = 6
num_epochs = 15
learning_rate = 5e-5

# ========== CUSTOM DATASET CLASS ==========
class PVLeafDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(os.listdir(root_dir))}

        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.samples.append((img_path, self.class_to_idx[class_name]))

    def custom_preprocessing(self, image):
        # Convert image to LAB color space & apply CLAHE
        image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel = image_lab[:, :, 0]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        image_lab[:, :, 0] = l_channel
        image = cv2.cvtColor(image_lab, cv2.COLOR_LAB2RGB)

        return image

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply custom preprocessing
        image = self.custom_preprocessing(image)
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label

# ========== DATA TRANSFORMS ==========
train_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomRotation(40),
    transforms.RandomResizedCrop(img_size, scale=(0.8, 1.2)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ========== LOAD DATA ==========
train_dataset = PVLeafDataset(train_dir, transform=train_transform)
val_dataset = PVLeafDataset(val_dir, transform=test_transform)
test_dataset = PVLeafDataset(test_dir, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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

model = build_resnet50()
print(model)

# ========== LOSS & OPTIMIZER ==========
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ========== TRAINING FUNCTION ==========
def train(model, train_loader, val_loader, num_epochs, patience=10):
    best_val_loss = float('inf')
    patience_counter = 0
    save_path = "best_model_apple.pth"

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc = correct / total

        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print("Model saved!")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    return model

# Train the model
model = train(model, train_loader, val_loader, num_epochs)

# Load best model
model.load_state_dict(torch.load("best_model_apple.pth"))

# ========== TESTING FUNCTION ==========
def evaluate(model, test_loader):
    model.eval()
    correct, total = 0, 0
    test_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    test_loss /= len(test_loader.dataset)
    test_acc = correct / total

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# Evaluate the model
evaluate(model, test_loader)
