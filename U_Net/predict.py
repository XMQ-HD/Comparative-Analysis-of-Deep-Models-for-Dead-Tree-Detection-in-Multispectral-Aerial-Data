import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from U_Net import UNet
from dataset import SegmentationDataset
import pickle

print("predict.py started")

# ====== Config ======
model_path = r"C:\Users\28600\Desktop\COMP9517\Segementation_image\U_Net\runs\unet_loss_focal+dice_size_256_opt_adam_bs_8_lr_0.0005\best_model.pth"
test_samples_pkl = "val_samples.pkl"   # Pickle file containing validation/test set sample paths
save_dir = "predict_results8"          # Directory to save prediction visualizations
os.makedirs(save_dir, exist_ok=True)

# ====== Device ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_iou(pred, mask, threshold=0.5):
    """
    Compute IoU between prediction and ground truth mask.
    """
    pred = (pred > threshold).float()
    intersection = (pred * mask).sum()
    union = ((pred + mask) > 0).float().sum()
    if union == 0:
        return float('nan')
    return float((intersection / union).item())

# ====== Load Model ======
model = UNet(in_channels=4, out_channels=1).to(device)
state_dict = torch.load(model_path, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.eval()

# ====== Load Dataset ======
with open(test_samples_pkl, 'rb') as f:
    test_samples = pickle.load(f)  # Each element: (rgb_path, nrg_path, mask_path)

test_dataset = SegmentationDataset(test_samples, size=(256, 256))

num_to_save = 5  # Number of visualizations to save
ious = []

for idx in range(len(test_dataset)):
    image, mask = test_dataset[idx]         
    image = image.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)

    # ====== Inference ======
    with torch.no_grad():
        output = model(image)
        pred = torch.sigmoid(output)

    iou = compute_iou(pred, mask)
    ious.append(iou)

    if idx < num_to_save:
        pred_np = pred.squeeze().cpu().numpy()
        mask_np = mask.squeeze().cpu().numpy()
        img_np  = image.squeeze().cpu().numpy()

        rgb_img = np.clip((img_np[1:4].transpose(1, 2, 0) * 255), 0, 255).astype(np.uint8)
        nir_img = np.clip((img_np[0] * 255), 0, 255).astype(np.uint8)

        plt.figure(figsize=(10, 2))
        plt.subplot(1, 4, 1)
        plt.imshow(nir_img, cmap="gray")
        plt.title("NIR")

        plt.subplot(1, 4, 2)
        plt.imshow(rgb_img)
        plt.title("RGB")

        plt.subplot(1, 4, 3)
        plt.imshow(mask_np, cmap="gray")
        plt.title("Ground Truth")

        plt.subplot(1, 4, 4)
        plt.imshow(pred_np > 0.5, cmap="gray")
        plt.title("Prediction")

        plt.tight_layout()
        rgb_name = os.path.basename(test_samples[idx][0]).replace('.png', '')
        out_file = os.path.join(save_dir, f"{idx:03d}_{rgb_name}_result.png")
        plt.savefig(out_file, dpi=200)
        plt.close()

print(f"Top {num_to_save} prediction results saved in {save_dir}")
print(f"Mean IoU on the test set: {np.nanmean(ious):.4f}")
