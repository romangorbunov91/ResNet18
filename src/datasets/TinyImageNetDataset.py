import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from typing import List, Tuple, Optional


class TinyImageNetDataset(Dataset):
    def __init__(
        self,
        root: str = "tiny-imagenet-200",
        split: str = "train",  # 'train', 'val'
        transform: Optional[T.Compose] = None,
    ):
        """
        Args:
            root (str): path to tiny-imagenet-200 folder
            split (str): 'train' or 'val'
            transform (callable, optional): torchvision transform
        """
        self.root = root
        self.split = split
        self.transform = transform or T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.samples: List[Tuple[str, int]] = []  # (image_path, class_id)
        self.classes: List[str] = []               # class names (wnids)
        self.class_to_idx: dict = {}

        # Load class names and mapping
        wnids_path = os.path.join(root, "wnids.txt")
        with open(wnids_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        if split == "train":
            self._load_train_data()
        elif split == "val":
            self._load_val_data()
        else:
            raise ValueError("split must be 'train' or 'val'")

    def _load_train_data(self):
        train_dir = os.path.join(self.root, "train")
        for class_id in os.listdir(train_dir):
            class_dir = os.path.join(train_dir, class_id)
            if not os.path.isdir(class_dir):
                continue
            images_dir = os.path.join(class_dir, "images")
            for img_name in os.listdir(images_dir):
                if img_name.endswith(".JPEG"):
                    img_path = os.path.join(images_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[class_id]))

    def _load_val_data(self):
        val_dir = os.path.join(self.root, "val")
        val_annotations = os.path.join(val_dir, "val_annotations.txt")
        with open(val_annotations, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                img_name, class_id = parts[0], parts[1]
                img_path = os.path.join(val_dir, "images", img_name)
                if class_id in self.class_to_idx:
                    self.samples.append((img_path, self.class_to_idx[class_id]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label