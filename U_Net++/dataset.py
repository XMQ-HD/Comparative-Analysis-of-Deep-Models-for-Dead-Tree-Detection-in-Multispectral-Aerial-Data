import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import albumentations as A

def load_4band_image(rgb_path, nrg_path, size=(256, 256)):
    """
    Load a 4-band image by combining NIR (from NRG) and RGB channels,
    resize to target size, normalize to [0, 1], and change to channel-first order.
    Args:
        rgb_path (str): Path to the RGB image.
        nrg_path (str): Path to the NRG image (NIR assumed in channel 0).
        size (tuple): Target size (width, height).
    Returns:
        np.ndarray: 4-channel float32 array with shape [4, H, W].
    """
    rgb = cv2.imread(rgb_path)
    nrg = cv2.imread(nrg_path)
    nir = nrg[..., 0:1]  # Get the first channel as NIR
    four_band = np.concatenate([nir, rgb], axis=-1)  # Combine to [H, W, 4]
    four_band = cv2.resize(four_band, size, interpolation=cv2.INTER_LINEAR)
    four_band = four_band.transpose(2, 0, 1) / 255.0  # [4, H, W], normalize to [0,1]
    return four_band.astype(np.float32)

def load_mask(mask_path, size=(256, 256)):
    """
    Load the segmentation mask, binarize it (foreground:1, background:0),
    and resize to target size. Returns a single-channel mask.
    Args:
        mask_path (str): Path to the mask image.
        size (tuple): Target size (width, height).
    Returns:
        np.ndarray: Single-channel float32 array with shape [1, H, W].
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    mask = (mask > 0).astype(np.float32)  # Binarization: foreground=1, background=0
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return mask[np.newaxis, ...]  # [1, H, W]

class SegmentationDataset(Dataset):
    """
    PyTorch Dataset for segmentation tasks supporting 4-channel input images and optional data augmentation.
    """
    def __init__(self, pairs, size=(256, 256), augment=None):
        """
        Args:
            pairs (list): Each element is a tuple of (rgb_path, nrg_path, mask_path)
            size (tuple): Output image size (width, height)
            augment (albumentations.Compose or None): Optional albumentations augmentation pipeline
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

        # Apply data augmentation if augmentations are provided
        # albumentations expects image: [H, W, C] and mask: [H, W]
        if self.augment:
            image_aug = image.transpose(1, 2, 0)        # [4, H, W] -> [H, W, 4]
            mask_aug = mask[0, ...]                     # [1, H, W] -> [H, W]
            augmented = self.augment(image=image_aug, mask=mask_aug)
            image = augmented['image'].transpose(2, 0, 1)  # [H, W, 4] -> [4, H, W]
            mask = augmented['mask'][np.newaxis, ...]      # [H, W] -> [1, H, W]

        # Return tensors ready for PyTorch model
        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)
