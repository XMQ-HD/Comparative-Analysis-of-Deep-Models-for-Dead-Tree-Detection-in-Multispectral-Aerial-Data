import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from train import TreeSegmentationDataset, get_deeplabv3_model, compute_iou, compute_precision_recall_f1
from cbam import inject_cbam_into_layer

print("predict.py started")

# ========== Configuration ==========
model_path = "../Deeplabv3/runs/loss_focal+tversky_size_512_opt_adam,batch_size_8,lr-0.0001,epochs100/best_model.pth" # change to your model
paired_samples_pkl = "../paired_samples.pkl"
paired_samples_pkl = "../paired_samples.pkl"  # same dataset as training
save_dir = "predict_results"
os.makedirs(save_dir, exist_ok=True)

# ========== Device ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ========== Load Model ==========
use_cbam = True
model = get_deeplabv3_model(use_cbam=use_cbam).to(device)

if use_cbam:
    print("Injecting CBAM into ResNet layer4 for prediction...")
    inject_cbam_into_layer(model.backbone.layer4, device=device)




# ✅ load weights after CBAM is injected
state_dict = torch.load(model_path, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.eval()


# ========== Load Dataset & Split ==========
with open(paired_samples_pkl, "rb") as f:
    samples = pickle.load(f)

np.random.shuffle(samples)
split = int(0.8 * len(samples))  # 80% train, 20% val
val_samples = samples[split:]    # we only use the 20% here

test_dataset = TreeSegmentationDataset(val_samples, img_size=(256, 256))

# ========== Prediction ==========
# ========== Prediction ==========
num_to_save = 5
ious, precisions, recalls, f1s = [], [], [], []
results = []  # store (idx, iou, image, mask, pred)

for idx in range(len(test_dataset)):
    image, mask = test_dataset[idx]
    image = image.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)["out"]
        pred = torch.sigmoid(output)

    # Compute metrics
    iou = compute_iou(output, mask)
    p, r, f1 = compute_precision_recall_f1(output, mask)

    ious.append(iou)
    precisions.append(p)
    recalls.append(r)
    f1s.append(f1)

    results.append((idx, iou, image.cpu(), mask.cpu(), pred.cpu()))

    # Save first N as usual
    if idx < num_to_save:
        pred_np = pred.squeeze().cpu().numpy()
        mask_np = mask.squeeze().cpu().numpy()
        img_np = image.squeeze().cpu().numpy()
        rgb_img = np.clip((img_np[0:3].transpose(1, 2, 0) * 255), 0, 255).astype(np.uint8)
        nir_img = np.clip((img_np[3] * 255), 0, 255).astype(np.uint8)

        plt.figure(figsize=(10, 2))
        plt.subplot(1, 4, 1); plt.imshow(nir_img, cmap="gray"); plt.title("NIR"); plt.axis("off")
        plt.subplot(1, 4, 2); plt.imshow(rgb_img); plt.title("RGB"); plt.axis("off")
        plt.subplot(1, 4, 3); plt.imshow(mask_np, cmap="gray"); plt.title("Ground Truth"); plt.axis("off")
        plt.subplot(1, 4, 4); plt.imshow(pred_np > 0.5, cmap="gray"); plt.title("Predicted"); plt.axis("off")
        plt.tight_layout()
        out_file = os.path.join(save_dir, f"{idx:03d}_sample.png")
        plt.savefig(out_file, dpi=200); plt.close()

# ========== Find Best & Worst ==========
best_idx, best_iou, best_img, best_mask, best_pred = max(results, key=lambda x: x[1])
worst_idx, worst_iou, worst_img, worst_mask, worst_pred = min(results, key=lambda x: x[1])

def save_case(idx, img, mask, pred, iou, case_name):
    img_np = img.squeeze().numpy()
    mask_np = mask.squeeze().numpy()
    pred_np = pred.squeeze().numpy()
    rgb_img = np.clip((img_np[0:3].transpose(1, 2, 0) * 255), 0, 255).astype(np.uint8)
    nir_img = np.clip((img_np[3] * 255), 0, 255).astype(np.uint8)

    plt.figure(figsize=(10, 2))
    plt.subplot(1, 4, 1); plt.imshow(nir_img, cmap="gray"); plt.title("NIR"); plt.axis("off")
    plt.subplot(1, 4, 2); plt.imshow(rgb_img); plt.title("RGB"); plt.axis("off")
    plt.subplot(1, 4, 3); plt.imshow(mask_np, cmap="gray"); plt.title("Ground Truth"); plt.axis("off")
    plt.subplot(1, 4, 4); plt.imshow(pred_np > 0.5, cmap="gray"); plt.title(f"Predicted\nIoU={iou:.3f}"); plt.axis("off")
    plt.tight_layout()
    out_file = os.path.join(save_dir, f"{case_name}_{idx:03d}_IoU{iou:.3f}.png")
    plt.savefig(out_file, dpi=200); plt.close()

# Save best & worst
save_case(best_idx, best_img, best_mask, best_pred, best_iou, "BEST")
save_case(worst_idx, worst_img, worst_mask, worst_pred, worst_iou, "WORST")

# ========== Print Metrics ==========
print(f"Top {num_to_save} results saved in {save_dir}")
print(f"Mean IoU: {np.nanmean(ious):.4f}")
print(f"Precision: {np.mean(precisions):.4f}")
print(f"Recall: {np.mean(recalls):.4f}")
print(f"F1-score: {np.mean(f1s):.4f}")
print(f"✅ Best IoU: {best_iou:.4f} (index {best_idx}), Worst IoU: {worst_iou:.4f} (index {worst_idx})")
