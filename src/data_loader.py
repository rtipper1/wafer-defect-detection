import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np

class WaferDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_data_loaders(data_dir='data/wm811k', batch_size=16, img_size=224):
    """
    Create train/val/test data loaders
    """
    data_dir = Path(data_dir)
    
    # Get all image paths and labels
    image_paths = []
    labels = []
    class_names = []
    
    # Iterate through class folders
    for class_folder in sorted(data_dir.iterdir()):
        if class_folder.is_dir():
            class_name = class_folder.name
            class_names.append(class_name)
            class_idx = len(class_names) - 1
            
            # Get all images in this class folder
            for img_path in class_folder.glob('*'):
                if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
                    image_paths.append(str(img_path))
                    labels.append(class_idx)
    
    print(f"Found {len(image_paths)} images across {len(class_names)} classes")
    print(f"Classes: {class_names}")
    
    # Convert to numpy arrays
    image_paths = np.array(image_paths)
    labels = np.array(labels)
    
    # Split: 70% train, 15% val, 15% test
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(train_paths)} images")
    print(f"  Val: {len(val_paths)} images")
    print(f"  Test: {len(test_paths)} images")
    
    # Define transforms (for ViT)
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = WaferDataset(train_paths, train_labels, train_transform)
    val_dataset = WaferDataset(val_paths, val_labels, val_transform)
    test_dataset = WaferDataset(test_paths, test_labels, val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader, class_names

# Test the data loader
if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_names = get_data_loaders()
    
    # Test one batch
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Label example: {labels[:5]}")