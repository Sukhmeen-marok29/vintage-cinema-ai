import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import CinematicAestheticClassifier

def train_model():
    # Setup hardware acceleration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

   # --- AUTO-GENERATE DUMMY DATA SAFELY ---
    import shutil
    class_names = ['70s_technicolor', '90s_filmic', 'neo_noir']
    
    for class_name in class_names:
        target_dir = os.path.join('dataset', 'train', class_name)
        
        # If it accidentally exists as a file, remove it
        if os.path.exists(target_dir) and not os.path.isdir(target_dir):
            os.remove(target_dir)
            
        # Create the actual directory cleanly
        os.makedirs(target_dir, exist_ok=True)
            
        # Check if the directory has no files inside it, then add a dummy image
        if len(os.listdir(target_dir)) == 0:
            print(f"Generating temporary dummy image for class: {class_name}")
            dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(target_dir, 'dummy_placeholder.jpg'), dummy_img)
    # ----------------------------------------

    # Standard data pipeline transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    train_dataset = datasets.ImageFolder(root='dataset/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    print(f"Detected classes: {train_dataset.classes}")

    # Initialize model
    model = CinematicAestheticClassifier(num_classes=len(train_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Simple training loop
    epochs = 10
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = (correct / total) * 100
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%")

    # Save weights
    torch.save(model.state_dict(), 'aesthetic_classifier.pth')
    print("Classifier weights saved successfully as 'aesthetic_classifier.pth'")

if __name__ == '__main__':
    train_model()