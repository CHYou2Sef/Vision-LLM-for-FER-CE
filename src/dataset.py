import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class RAFCEDataset(Dataset):
    """
    Custom Dataset class for the RAF-CE (Real-world Affective Faces of Compound Emotions) dataset.
    This class handles loading image names, mapping them to aligned JPEG files, and loading 
    numerical emotion labels.
    """
    def __init__(self, root_dir, label_file, partition_file, split=None, transform=None):
        """
        Initialization of the dataset.
        
        Args:
            root_dir (string): Directory with all the images (e.g., './aligned').
            label_file (string): Path to the emotion labels TXT file.
            partition_file (string): Path to the partition info TXT file (Train/Val/Test).
            split (int, optional): 0 for train, 1 for val, 2 for test. If None, returns all.
            transform (callable, optional): Optional transform (Augmentation/Normalization) to be applied.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Load labels from the space-separated text file
        # Format expected: 'imagename.jpg label'
        labels_df = pd.read_csv(label_file, sep=' ', header=None, names=['image_name', 'label'])
        
        # Load partition info to filter by Train/Val/Test
        # Format expected: 'imagename.jpg split_id'
        partition_df = pd.read_csv(partition_file, sep=' ', header=None, names=['image_name', 'partition'])
        
        # Merge both dataframes on index filename to align labels with their respective splits
        self.data = pd.merge(labels_df, partition_df, on='image_name')
        
        # Filter data based on the requested split if provided
        if split is not None:
            self.data = self.data[self.data['partition'] == split].reset_index(drop=True)

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Loads and returns a sample from the dataset at the given index.
        
        Args:
            idx (int): Index of the sample to fetch.
        Returns:
            tuple: (image, label) where image is the processed tensor and label is the integer class.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the original image name (e.g., 0001.jpg)
        img_name = self.data.iloc[idx, 0]
        
        # RAF-CE 'aligned' folders typically name files as '0001_aligned.jpg'
        # We perform string manipulation to match this pattern.
        base_name = img_name.split('.')[0]
        full_img_name = f"{base_name}_aligned.jpg"
        img_path = os.path.join(self.root_dir, full_img_name)
        
        # Load image via PIL and ensure RGB format
        image = Image.open(img_path).convert('RGB')
        label = int(self.data.iloc[idx, 1])

        # Apply transforms (e.g., resizing, normalization, augmentation)
        if self.transform:
            image = self.transform(image)

        return image, label

def get_transforms(img_size=224, augment=False):
    """
    Standard preprocessing pipeline for ResNet and other Vision models.
    
    Args:
        img_size (int): Target size for the square resize.
        augment (bool): If True, applies random flips, rotations, and color variations.
    Returns:
        torchvision.transforms.Compose: The transformation pipeline.
    """
    if augment:
        # Augmentation pipeline for training robustness
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            # Normalization based on ImageNet statistics
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Simple validation/test pipeline
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
