"""Tea leaf disease dataset loader."""
import os
from pathlib import Path
from typing import Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

CLASSES = ["Anthracnose", "AlgalLeafSpot", "BirdEyeSpot", "BrownBlight", "Healthy"]

def get_transforms(train: bool, size: int = 256) -> T.Compose:
    if train:
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.2, 0.2, 0.1),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

class TeaLeafDataset(Dataset):
    def __init__(self, root: str, split: str = "train", image_size: int = 256):
        self.root      = Path(root) / split
        self.transform = get_transforms(split == "train", image_size)
        self.samples   = []
        for label, cls in enumerate(CLASSES):
            cls_dir = self.root / cls
            if not cls_dir.exists(): continue
            for f in cls_dir.iterdir():
                if f.suffix.lower() in (".jpg",".jpeg",".png"):
                    self.samples.append((str(f), label))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label

def build_loaders(root, batch_size=32, num_workers=4):
    tr = DataLoader(TeaLeafDataset(root,"train"), batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    va = DataLoader(TeaLeafDataset(root,"val"),   batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return tr, va
