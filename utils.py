"""
Author: Dimas Ahmad
Description: This file contains utility functions for the project.
"""

import os
import torch
import torchvision
from PIL import Image
from typing import Optional, Callable, Tuple, Any


class Caltech101Exercise(torchvision.datasets.vision.VisionDataset):
    """
    Custom dataset class for Caltech, the dataset is stored in the following structure:
    root/train/class_x/xxx.jpg or root/test/class_x/xxx.jpg
    There are 10 classes in total and 15 images per class per subset.
    """
    def __init__(
            self,
            data_dir: str = "./Caltech_101/",
            split: str = 'train',
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root=os.path.join(data_dir, split), transform=transform, target_transform=target_transform)
        self.classes = sorted([d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = self._make_dataset()

    def _make_dataset(self) -> list[Tuple[str, int]]:
        """
        Create a list of (image_path, class_idx) tuples
        """
        images = []
        for c in self.classes:
            class_dir = os.path.join(self.root, c)
            class_idx = self.class_to_idx[c]

            img_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            for img_file in img_files:
                img_path = os.path.join(class_dir, img_file)
                item = (img_path, class_idx)
                images.append(item)

        return images

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.samples)


def get_loaders(batch_size=32, shuffle_train=True, resize_size=256, crop_size=224):
    """
    Get train and test loaders for Caltech101 dataset
    """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((resize_size, resize_size)),
        torchvision.transforms.CenterCrop(crop_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = Caltech101Exercise(split='train', transform=transform)
    test_dataset = Caltech101Exercise(split='test', transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def get_model_size(model):
    """
    Get the number of parameters in a model
    """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    memory_mb = num_params * 4 / (1024*1024)
    return num_params, memory_mb
