import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add src to path
sys.path.append(os.path.abspath('src'))
from dataset import RAFCEDataset, get_transforms

def main():
    # Parameters
    IMG_DIR = '../aligned'
    LABEL_FILE = '../RAFCE_emolabel.txt'
    PARTITION_FILE = '../RAFCE_partition.txt'
    OUTPUT_DIR = 'outputs/baseline'
    
    BATCH_SIZE = 32
    EPOCHS = 10  # Reduced for initial run
    LR = 0.001
    NUM_CLASSES = 14
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Using device: {DEVICE}")

    # Data Loading
    train_transform = get_transforms(augment=True)
    val_transform = get_transforms(augment=False)
    
    print("Loading datasets...")
    # split=1 for train, split=2 for test
    train_dataset = RAFCEDataset(IMG_DIR, LABEL_FILE, PARTITION_FILE, split=1, transform=train_transform)
    test_dataset = RAFCEDataset(IMG_DIR, LABEL_FILE, PARTITION_FILE, split=2, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Model Definition
    print("Initializing ResNet50...")
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Training Loop
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.4f}")
    
    # Save Model
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'resnet50_ferce_latest.pth'))
    print("Model saved.")

    # Evaluation
    print("Evaluating...")
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print(classification_report(all_labels, all_preds))
    print("Training process completed.")

if __name__ == "__main__":
    main()
