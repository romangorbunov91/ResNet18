import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from typing import List, Tuple, Optional

class TinyImageNetDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        selected_classes: Optional[List[str]] = None
        ):

        self.root = root
        self.split = split
        self.transform = transform
        
        self.samples: List[Tuple[str, int]] = []  # (img_path, class_id)
        self.class_names: List[str] = []
        self.class_to_idx: dict = {}
        
        with open(os.path.join(self.root, 'wnids.txt'), 'r') as f:
            all_class_names = [line.strip() for line in f.readlines()]
            #self.class_names = [line.strip() for line in f.readlines()]
        #self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}

        # Select only specified classes, or all if None
        if selected_classes is not None:
            # Validate that all selected classes exist
            invalid = set(selected_classes) - set(all_class_names)
            if invalid:
                raise ValueError(f"Selected classes not in dataset: {invalid}")
            self.class_names = selected_classes
        else:
            self.class_names = all_class_names

        # Create class_to_idx mapping for selected classes only
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}
        self.idx_to_class = {i: cls_name for cls_name, i in self.class_to_idx.items()}

        # Load data.
        data_dir = os.path.join(self.root, split)
       
        if split == 'train':
            for class_id in os.listdir(data_dir):
                img_dir = os.path.join(data_dir, class_id, 'images')
                for img_name in os.listdir(img_dir):
                    img_path = os.path.join(img_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[class_id]))

        elif split == 'val':
            val_annotations_path = os.path.join(data_dir, 'val_annotations.txt')
            if not os.path.exists(val_annotations_path):
                raise FileNotFoundError(f"{val_annotations_path} not found.")
            with open(val_annotations_path, 'r') as f:
                for line in f:
                    line_parts = line.strip().split('\t')
                    # Skip malformed lines.
                    if len(line_parts) < 2:
                        continue  
                    img_name, class_id = line_parts[0], line_parts[1]                    
                    
                    if class_id not in self.class_to_idx:
                        raise ValueError(f"Unknown class_id '{class_id}' in {val_annotations_path}")
                    img_path = os.path.join(data_dir, 'images', img_name)
                    self.samples.append((img_path, self.class_to_idx[class_id]))
        
        else:
            raise ValueError("'split' must be 'train' or 'val'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, label