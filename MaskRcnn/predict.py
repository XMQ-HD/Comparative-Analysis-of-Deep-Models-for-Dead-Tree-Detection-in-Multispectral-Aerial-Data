import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import cv2
from torch.utils.data import DataLoader
from train import TreeInstanceDataset, collate_fn, get_maskrcnn_model, compute_iou, compute_precision_recall_f1

print("predict_maskrcnn.py started")

# ========== Configuration ==========
model_path = "../Rcnn/runs/maskrcnn_loss_focal+tversky_size_512_opt_adamw_bs_8_lr_0.0005/best_model.pth" # change to your model
paired_samples_pkl = "../paired_samples.pkl"
save_dir = "predict_results_maskrcnn"
os.makedirs(save_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ========== Load Model ==========
model = get_maskrcnn_model(num_classes=2, image_size=256).to(device)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict, strict=False)
model.eval()

# ========== Load Dataset & Split ==========
with open(paired_samples_pkl, "rb") as f:
    samples = pickle.load(f)

np.random.shuffle(samples)
split = int(0.8 * len(samples))
val_samples = samples[split:]
val_dataset = TreeInstanceDataset(val_samples, (256, 256), train=False)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

# ========== Prediction ==========
results = []
num_to_save = 5

for idx, (images, targets) in enumerate(val_loader):
    images = [img.to(device) for img in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    with torch.no_grad():
        outputs = model(images)

    o = outputs[0]
    if "masks" not in o or o["masks"].numel() == 0:
        pred_mask = torch.zeros((256, 256), device=device)
    else:
        pred_mask = (o["masks"][:, 0] > 0.5).float().sum(dim=0).clamp(0, 1)

    gt_mask = targets[0]["masks"].sum(dim=0).clamp(0, 1).float()

    iou = compute_iou(pred_mask, gt_mask)
    p, r, f1 = compute_precision_recall_f1(pred_mask, gt_mask)

    results.append((idx, iou, p, r, f1, images[0].cpu(), gt_mask.cpu(), pred_mask.cpu(), val_samples[idx]))


    # Save first N as preview
    # Save first N as preview
    if idx < num_to_save:
        img_np = images[0].cpu().numpy()
        rgb_img = np.clip((img_np[1:4].transpose(1, 2, 0) * 255), 0, 255).astype(np.uint8)
        nir_img = np.clip((img_np[0] * 255), 0, 255).astype(np.uint8)

        gt_np = gt_mask.cpu().numpy()
        pred_np = pred_mask.cpu().numpy()

        plt.figure(figsize=(10, 2))
        plt.subplot(1, 4, 1);
        plt.imshow(nir_img, cmap="gray");
        plt.title("NIR");
        plt.axis("off")
        plt.subplot(1, 4, 2);
        plt.imshow(rgb_img);
        plt.title("RGB");
        plt.axis("off")
        plt.subplot(1, 4, 3);
        plt.imshow(gt_np, cmap="gray");
        plt.title("Ground Truth");
        plt.axis("off")
        plt.subplot(1, 4, 4);
        plt.imshow(pred_np > 0.5, cmap="gray");
        plt.title(f"Predicted\nIoU={iou:.3f}");
        plt.axis("off")
        plt.tight_layout()
        out_file = os.path.join(save_dir, f"{idx:03d}_sample.png")
        plt.savefig(out_file, dpi=200);
        plt.close()

# ========== Find Best & Worst ==========
best_case = max(results, key=lambda x: x[1])
worst_case = min(results, key=lambda x: x[1])

def save_case(case, case_name):
    idx, iou, p, r, f1, img, gt_mask, pred_mask, sample = case
    rgb_path, nrg_path, mask_path = sample  # unpack the dataset triplet
    print(f"🔹 {case_name} case files: RGB={rgb_path}, NRG={nrg_path}, MASK={mask_path}")

    # move everything to CPU
    img_np = img.numpy()
    gt_np = gt_mask.cpu().numpy()
    pred_np = pred_mask.cpu().numpy()

    # extract RGB + NIR
    rgb_img = np.clip((img_np[1:4].transpose(1, 2, 0) * 255), 0, 255).astype(np.uint8)
    nir_img = np.clip((img_np[0] * 255), 0, 255).astype(np.uint8)

    plt.figure(figsize=(10, 2))
    plt.subplot(1, 4, 1); plt.imshow(nir_img, cmap="gray"); plt.title("NIR"); plt.axis("off")
    plt.subplot(1, 4, 2); plt.imshow(rgb_img); plt.title("RGB"); plt.axis("off")
    plt.subplot(1, 4, 3); plt.imshow(gt_np, cmap="gray"); plt.title("Ground Truth"); plt.axis("off")
    plt.subplot(1, 4, 4); plt.imshow(pred_np > 0.5, cmap="gray"); plt.title(f"Predicted\nIoU={iou:.3f}"); plt.axis("off")
    plt.tight_layout()

    out_file = os.path.join(save_dir, f"{case_name}_{idx:03d}_IoU{iou:.3f}.png")
    plt.savefig(out_file, dpi=200); plt.close()
    print(f"✅ {case_name} case saved: {out_file}")




save_case(best_case, "BEST")
save_case(worst_case, "WORST")

# ========== Print Metrics ==========
mean_iou = np.mean([c[1] for c in results])
mean_p = np.mean([c[2] for c in results])
mean_r = np.mean([c[3] for c in results])
mean_f1 = np.mean([c[4] for c in results])

print(f"Mean IoU: {mean_iou:.4f}")
print(f"Precision: {mean_p:.4f}")
print(f"Recall: {mean_r:.4f}")
print(f"F1-score: {mean_f1:.4f}")
print(f"Best IoU: {best_case[1]:.4f} (index {best_case[0]}), Worst IoU: {worst_case[1]:.4f} (index {worst_case[0]})")
