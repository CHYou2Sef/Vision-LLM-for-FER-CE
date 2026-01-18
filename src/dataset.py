import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class RAFCEDataset(Dataset):
    def __init__(self, root_dir, label_file, partition_file, split=None, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images (e.g., './aligned').
            label_file (string): Path to the emotion labels (e.g., 'RAFCE_emolabel.txt').
            partition_file (string): Path to the partition info (e.g., 'RAFCE_partition.txt').
            split (int, optional): 0 for train, 1 for val, 2 for test. If None, returns all.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Load labels
        labels_df = pd.read_csv(label_file, sep=' ', header=None, names=['image_name', 'label'])
        
        # Load partitions
        partition_df = pd.read_csv(partition_file, sep=' ', header=None, names=['image_name', 'partition'])
        
        # Merge
        self.data = pd.merge(labels_df, partition_df, on='image_name')
        
        if split is not None:
            # Researchers often use 0/2 for train/test or similar. 
            # common: 1=test? Let's check the partition file again.
            # Usually RAF-CE has a specific split.
            self.data = self.data[self.data['partition'] == split].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data.iloc[idx, 0]
        # Map 0001.jpg to 0001_aligned.jpg
        base_name = img_name.split('.')[0]
        full_img_name = f"{base_name}_aligned.jpg"
        img_path = os.path.join(self.root_dir, full_img_name)
        
        image = Image.open(img_path).convert('RGB')
        label = int(self.data.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, label

def get_transforms(img_size=224, augment=False):
    if augment:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
