import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
import random
import argparse
from sklearn.metrics import confusion_matrix
import seaborn as sns


# -----------------------------
# Loss functions
# -----------------------------
def dice_loss(pred, target, epsilon=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 1 - (2. * intersection + epsilon) / (union + epsilon)

def focal_loss(pred, target, alpha=0.75, gamma=2):
    bce = nn.BCEWithLogitsLoss(reduction='none')(pred, target)
    pt = torch.exp(-bce)
    return ((alpha * (1 - pt) ** gamma) * bce).mean()

def tversky_loss(pred, target, alpha=0.7, beta=0.3, epsilon=1e-6):
    pred = torch.sigmoid(pred)
    tp = (pred * target).sum()
    fp = ((1 - target) * pred).sum()
    fn = (target * (1 - pred)).sum()
    return 1 - (tp + epsilon) / (tp + alpha * fp + beta * fn + epsilon)

def combo_loss(pred, target):
    return 0.5 * focal_loss(pred, target) + 0.5 * dice_loss(pred, target)

def select_loss(loss_type):
    if loss_type == "bce+dice":
        return lambda pred, target: 0.5 * nn.BCEWithLogitsLoss()(pred, target) + 0.5 * dice_loss(pred, target)
    elif loss_type == "focal":
        return focal_loss
    elif loss_type == "tversky":
        return tversky_loss
    elif loss_type == "focal+dice":
        return combo_loss
    elif loss_type == "focal+tversky":
        return lambda pred, target: 0.5 * focal_loss(pred, target) + 0.5 * tversky_loss(pred, target)
    else:
        raise ValueError(f"Unsupported loss function: {loss_type}")


# -----------------------------
# Dataset utilities
# -----------------------------
def load_4band_image(rgb_path, nrg_path, size=(256, 256)):
    rgb = cv2.imread(rgb_path)
    nrg = cv2.imread(nrg_path)
    nir = nrg[..., 0:1]
    four_band = np.concatenate([nir, rgb], axis=-1)
    four_band = cv2.resize(four_band, size, interpolation=cv2.INTER_LINEAR)
    four_band = four_band.transpose(2, 0, 1) / 255.0
    return four_band.astype(np.float32)

def load_mask(mask_path, size=(256, 256)):
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    mask = (mask > 0).astype(np.float32)
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return mask[np.newaxis, ...]

class TreeInstanceDataset(Dataset):
    def __init__(self, pairs, size=(256, 256), train=True):
        self.samples = pairs
        self.size = size
        self.train = train  # <-- add this flag

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, nrg_path, mask_path = self.samples[idx]
        image = load_4band_image(rgb_path, nrg_path, size=self.size)
        mask = load_mask(mask_path, size=self.size)
        image = torch.as_tensor(image, dtype=torch.float32)
        mask = torch.as_tensor(mask, dtype=torch.uint8)

        # --- Apply augmentations only during training ---
        if self.train:
            import torchvision.transforms.functional as TF
            import random

            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)

            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            if random.random() > 0.5:
                rgb = image[:3, :, :]
                nir = image[3:, :, :]
                rgb = TF.adjust_brightness(rgb, brightness_factor=random.uniform(0.7, 1.3))
                image = torch.cat([rgb, nir], dim=0)

        # (rest of your connectedComponents + target building code)
        instance_mask = (mask[0] > 0).numpy().astype(np.uint8)
        num_objs, labels_map = cv2.connectedComponents(instance_mask)
        masks, boxes = [], []
        for obj_id in range(1, num_objs):
            obj_mask = (labels_map == obj_id)
            if obj_mask.sum() < 10:
                continue
            masks.append(torch.tensor(obj_mask, dtype=torch.uint8))
            pos = np.where(obj_mask)
            xmin, xmax = np.min(pos[1]), np.max(pos[1])
            ymin, ymax = np.min(pos[0]), np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        if len(masks) == 0:
            masks = [torch.zeros(self.size, dtype=torch.uint8)]
            boxes = [[0, 0, 1, 1]]

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.ones((len(masks),), dtype=torch.int64),
            "masks": torch.stack(masks),
            "image_id": torch.tensor([idx]),
        }

        return image, target



def collate_fn(batch):
    return tuple(zip(*batch))


# -----------------------------
# Model
# -----------------------------
def get_maskrcnn_model(num_classes=2, image_size=256):
    backbone = resnet_fpn_backbone(backbone_name='resnet101', weights='IMAGENET1K_V1')
    conv1 = backbone.body.conv1
    new_conv1 = nn.Conv2d(4, conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        new_conv1.weight[:, :3] = conv1.weight
        new_conv1.weight[:, 3] = conv1.weight[:, 0]
    backbone.body.conv1 = new_conv1
    anchor_generator = AnchorGenerator(
        sizes=((2,), (4,), (8,), (16,), (32,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )
    model = MaskRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator)
    model.transform = GeneralizedRCNNTransform(
        min_size=image_size,
        max_size=image_size,
        image_mean=[0.5, 0.5, 0.5, 0.5],
        image_std=[0.25, 0.25, 0.25, 0.25]
    )
    return model


# -----------------------------
# Metrics
# -----------------------------
def compute_iou(pred, mask, threshold=0.5, smooth=1e-6):
    if pred.numel() == 0 or mask.numel() == 0:
        return 0.0
    pred = torch.sigmoid(pred)
    pred_bin = (pred > threshold).float()
    mask = mask.float()
    intersection = (pred_bin * mask).sum()
    union = ((pred_bin + mask) > 0).float().sum()
    if union == 0:
        return 0.0
    return float(((intersection + smooth) / (union + smooth)).item())

def compute_confusion_matrix(all_preds, all_gts, threshold=0.5):
    preds = (all_preds > threshold).cpu().numpy().astype(int).flatten()
    gts = (all_gts > 0.5).cpu().numpy().astype(int).flatten()
    cm = confusion_matrix(gts, preds, labels=[0, 1])
    return cm

def compute_precision_recall_f1(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    target = (target > 0.5).float()
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return precision.item(), recall.item(), f1.item()

def plot_metrics(history, save_dir):
    epochs = range(1, len(history["iou"]) + 1)
    plt.figure()
    plt.plot(epochs, history["iou"], label="IoU")
    plt.plot(epochs, history["precision"], label="Precision")
    plt.plot(epochs, history["recall"], label="Recall")
    plt.plot(epochs, history["f1"], label="F1-score")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Metrics Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "metrics_curve.png"))
    plt.close()

def plot_confusion_matrix(model, val_loader, device, save_dir, threshold=0.5):
    model.eval()
    all_preds, all_gts = [], []
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            batch_preds = []
            for o in outputs:
                if "masks" not in o or o['masks'].numel() == 0:
                    merged = torch.zeros((images[0].shape[1], images[0].shape[2]), device=device)
                else:
                    merged = (o['masks'][:, 0] > 0.5).float().sum(dim=0).clamp(0, 1)
                batch_preds.append(merged)
            preds_tensor = torch.stack(batch_preds, dim=0)
            gts_tensor = torch.stack([t['masks'].sum(dim=0).clamp(0, 1) for t in targets], dim=0).float()
            all_preds.append(preds_tensor)
            all_gts.append(gts_tensor)

    all_preds = torch.cat(all_preds, dim=0)
    all_gts = torch.cat(all_gts, dim=0)
    preds = (all_preds > threshold).cpu().numpy().astype(int).flatten()
    gts = (all_gts > 0.5).cpu().numpy().astype(int).flatten()
    cm = confusion_matrix(gts, preds, labels=[0, 1])

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Background", "Tree"],
                yticklabels=["Background", "Tree"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()
    print(f"✅ Confusion matrix saved to {os.path.join(save_dir, 'confusion_matrix.png')}")


# -----------------------------
# Args
# -----------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Train Mask R-CNN on Tree Segmentation Dataset")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--loss', type=str, default='bce+dice',
                        choices=['bce+dice', 'focal', 'tversky', 'focal+dice', 'focal+tversky'])
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adamw', 'sgd'])
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='runs')
    return parser.parse_args()


# -----------------------------
# Training
# -----------------------------
def main():
    args = get_args()

    with open("../paired_samples.pkl", "rb") as f:
        pairs = pickle.load(f)

    random.shuffle(pairs)
    split = int(0.8 * len(pairs))
    train_pairs, val_pairs = pairs[:split], pairs[split:]

    folder_name = f"maskrcnn_loss_{args.loss}_size_{args.img_size}_opt_{args.optimizer}_bs_{args.batch_size}_lr_{args.lr}"
    save_folder = os.path.join(args.save_dir, folder_name)
    os.makedirs(save_folder, exist_ok=True)

    train_dataset = TreeInstanceDataset(train_pairs, (args.img_size, args.img_size), train=True)
    val_dataset = TreeInstanceDataset(val_pairs, (args.img_size, args.img_size), train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_maskrcnn_model(num_classes=2, image_size=args.img_size).to(device)

    # Optimizer
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    mask_loss_fn = select_loss(args.loss)

    history = {"iou": [], "precision": [], "recall": [], "f1": []}
    best_iou, best_metrics = 0, {}

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            det_loss = sum(loss for k, loss in loss_dict.items() if k != "loss_mask")
            seg_loss = loss_dict.get("loss_mask", torch.tensor(0.0, device=device))
            total_loss_batch = det_loss + seg_loss
            total_loss_batch.backward()
            optimizer.step()
            total_loss += total_loss_batch.item()

        print(f"[Train] Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

        # Validation
        model.eval()
        ious, precisions, recalls, f1s = [], [], [], []
        all_preds, all_gts = [], []
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                outputs = model(images)
                batch_preds = []
                for o in outputs:
                    if "masks" not in o or o['masks'].numel() == 0:
                        merged = torch.zeros((args.img_size, args.img_size), device=device)
                    else:
                        merged = (o['masks'][:, 0] > 0.5).float().sum(dim=0).clamp(0, 1)
                    batch_preds.append(merged)

                preds_tensor = torch.stack(batch_preds, dim=0)
                gt_masks = torch.stack([t['masks'].sum(dim=0).clamp(0, 1) for t in targets], dim=0).float()

                ious.append(compute_iou(preds_tensor, gt_masks))
                p, r, f1 = compute_precision_recall_f1(preds_tensor, gt_masks)
                precisions.append(p)
                recalls.append(r)
                f1s.append(f1)

                all_preds.append(preds_tensor)
                all_gts.append(gt_masks)

        all_preds = torch.cat(all_preds, dim=0)
        all_gts = torch.cat(all_gts, dim=0)
        miou = np.nanmean(ious)
        p = np.mean(precisions)
        r = np.mean(recalls)
        f1 = np.mean(f1s)

        print(f"[Val] Epoch {epoch+1} | mIoU: {miou:.4f} | Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}")

        history["iou"].append(miou)
        history["precision"].append(p)
        history["recall"].append(r)
        history["f1"].append(f1)

        # Save best model + confusion matrix at best IoU
        if miou > best_iou:
            best_iou = miou
            best_metrics = {
                "iou": miou,
                "precision": p,
                "recall": r,
                "f1": f1,
                "epoch": epoch + 1
            }
            torch.save(model.state_dict(), os.path.join(save_folder, "best_model.pth"))
            print(f"✅ Best model updated at epoch {epoch + 1} with IoU: {miou:.4f}")

            cm = compute_confusion_matrix(all_preds, all_gts)
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Background", "Tree"],
                        yticklabels=["Background", "Tree"])
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"Confusion Matrix (Best IoU Epoch {epoch + 1})")
            plt.savefig(os.path.join(save_folder, "confusion_matrix_best.png"))
            plt.close()

    # Save curves
    plot_metrics(history, save_folder)

    # Save best metrics
    with open(os.path.join(save_folder, "best_metrics.txt"), "w") as f:
        for k, v in best_metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    # Save final model
    model_path = os.path.join(save_folder, "maskrcnn_tree_seg.pth")
    torch.save(model.state_dict(), model_path)
    print(f"✅ Training complete. Model saved to: {model_path}")
    print(f"Settings → Epochs: {args.epochs}, Batch Size: {args.batch_size}, "
          f"Loss: {args.loss}, Optimizer: {args.optimizer}")


    plot_confusion_matrix(model, val_loader, device, save_folder)

    # Save best metrics
    with open(os.path.join(save_folder, "best_metrics.txt"), "w") as f:
        for k, v in best_metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    # Save final model
    model_path = os.path.join(save_folder, "maskrcnn_tree_seg.pth")
    torch.save(model.state_dict(), model_path)
    print(f"✅ Training complete. Model saved to: {model_path}")
    print(f"Settings → Epochs: {args.epochs}, Batch Size: {args.batch_size}, Loss: {args.loss}, Optimizer: {args.optimizer}")


if __name__ == '__main__':
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()
    main()