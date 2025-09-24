import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import albumentations as A

def load_4band_image(rgb_path, nrg_path, size=(256, 256)):
    """
    Load a 4-channel image: [NIR, R, G, B], resize, and normalize to [0,1].
    Args:
        rgb_path (str): Path to RGB image.
        nrg_path (str): Path to NRG image.
        size (tuple): Output size (width, height).
    Returns:
        np.ndarray: [4, H, W], float32
    """
    rgb = cv2.imread(rgb_path)
    nrg = cv2.imread(nrg_path)
    nir = nrg[..., 0:1]
    four_band = np.concatenate([nir, rgb], axis=-1)
    four_band = cv2.resize(four_band, size, interpolation=cv2.INTER_LINEAR)
    four_band = four_band.transpose(2, 0, 1) / 255.0
    return four_band.astype(np.float32)

def load_mask(mask_path, size=(256, 256)):
    """
    Load and binarize the mask, resize to target size.
    Args:
        mask_path (str): Path to mask image.
        size (tuple): Output size (width, height).
    Returns:
        np.ndarray: [1, H, W], float32, values 0 or 1
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    mask = (mask > 0).astype(np.float32)
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return mask[np.newaxis, ...]

class SegmentationDataset(Dataset):
    """
    PyTorch dataset for segmentation. Supports 4-channel input and optional augmentation.
    """
    def __init__(self, pairs, size=(256, 256), augment=None):
        """
        Args:
            pairs (list): List of (rgb_path, nrg_path, mask_path).
            size (tuple): Output size (width, height).
            augment (albumentations.Compose or None): Optional augmentation.
        """
        self.samples = pairs
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, nrg_path, mask_path = self.samples[idx]
        image = load_4band_image(rgb_path, nrg_path, size=self.size)
        mask = load_mask(mask_path, size=self.size)
        if self.augment:
            image_aug = image.transpose(1, 2, 0)  # [4, H, W] -> [H, W, 4]
            mask_aug = mask[0, ...]               # [1, H, W] -> [H, W]
            augmented = self.augment(image=image_aug, mask=mask_aug)
            image = augmented['image'].transpose(2, 0, 1)  # [H, W, 4] -> [4, H, W]
            mask = augmented['mask'][np.newaxis, ...]      # [H, W] -> [1, H, W]
        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)
