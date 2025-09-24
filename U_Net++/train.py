import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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
from U_Net import UNetPP
from dataset import SegmentationDataset  # Use your updated dataset.py

import albumentations as A

# ========== Loss Functions ==========
def dice_loss(pred, target, epsilon=1e-6):
    """
    Compute Dice loss for binary segmentation.
    """
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 1 - (2. * intersection + epsilon) / (union + epsilon)

def focal_loss(pred, target, alpha=0.75, gamma=2):
    """
    Compute Focal loss for imbalanced binary segmentation.
    """
    bce = nn.BCEWithLogitsLoss(reduction='none')(pred, target)
    pt = torch.exp(-bce)
    return ((alpha * (1 - pt) ** gamma) * bce).mean()

def tversky_loss(pred, target, alpha=0.7, beta=0.3, epsilon=1e-6):
    """
    Compute Tversky loss for segmentation tasks.
    """
    pred = torch.sigmoid(pred)
    tp = (pred * target).sum()
    fp = ((1 - target) * pred).sum()
    fn = (target * (1 - pred)).sum()
    return 1 - (tp + epsilon) / (tp + alpha * fp + beta * fn + epsilon)

def combo_loss(pred, target):
    """
    Combo loss: combination of focal and dice losses.
    """
    return 0.5 * focal_loss(pred, target) + 0.5 * dice_loss(pred, target)

def select_loss(loss_type):
    """
    Select the loss function according to the argument.
    """
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

# ========== Metrics & Plotting ==========
def compute_iou(pred, mask, threshold=0.5, smooth=1e-6):
    """
    Compute Intersection over Union (IoU) metric.
    """
    pred = torch.sigmoid(pred)
    pred_bin = (pred > threshold).float()
    mask = mask.float()
    intersection = (pred_bin * mask).sum()
    union = ((pred_bin + mask) > 0).float().sum()
    if union == 0:
        return 0.0
    return float(((intersection + smooth) / (union + smooth)).item())

def compute_precision_recall_f1(pred, target, threshold=0.5):
    """
    Compute Precision, Recall, and F1-score for binary segmentation.
    """
    pred = torch.sigmoid(pred)
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
    """
    Plot validation metrics (IoU, Precision, Recall, F1) over epochs.
    """
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
    """
    Plot and save the confusion matrix on the validation set.
    """
    model.eval()
    all_preds, all_gts = [], []
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            all_preds.append(outputs.cpu())
            all_gts.append(masks.cpu())
    all_preds = torch.cat(all_preds, dim=0)
    all_gts = torch.cat(all_gts, dim=0)
    preds = (torch.sigmoid(all_preds) > threshold).numpy().astype(int).flatten()
    gts = (all_gts > 0.5).numpy().astype(int).flatten()
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

# ========== Argument Parser ==========
def get_args():
    """
    Parse command-line arguments for training.
    """
    parser = argparse.ArgumentParser(description="Train U-Net++ on Tree Segmentation Dataset")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--loss', type=str, default='focal+dice',
                        choices=['bce+dice', 'focal', 'tversky', 'focal+dice', 'focal+tversky'])
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adamw', 'sgd'])
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='runs')
    return parser.parse_args()

# ========== Main Training ==========
def main():
    args = get_args()
    # Load all sample pairs from pickle file
    with open("paired_samples.pkl", "rb") as f:
        pairs = pickle.load(f)
    random.shuffle(pairs)
    split = int(0.8 * len(pairs))
    train_pairs, val_pairs = pairs[:split], pairs[split:]

    # Save the validation set indices for reproducibility or further evaluation
    with open("val_samples.pkl", "wb") as f:
        pickle.dump(val_pairs, f)
    print(f"✅ Validation set saved to val_samples.pkl, sample count: {len(val_pairs)}")

    folder_name = f"unetpp_loss_{args.loss}_size_{args.img_size}_opt_{args.optimizer}_bs_{args.batch_size}_lr_{args.lr}"
    save_folder = os.path.join(args.save_dir, folder_name)
    os.makedirs(save_folder, exist_ok=True)

    # ----------- Data Augmentation for Training -----------
    train_aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
    ])
    # ----------- Build Datasets and DataLoaders -----------
    train_dataset = SegmentationDataset(train_pairs, (args.img_size, args.img_size), augment=train_aug)
    val_dataset   = SegmentationDataset(val_pairs, (args.img_size, args.img_size), augment=None)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNetPP(in_channels=4, out_channels=1).to(device)

    # Select optimizer
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    mask_loss_fn = select_loss(args.loss)
    history = {"iou": [], "precision": [], "recall": [], "f1": []}
    best_iou, best_metrics = 0, {}

    # --------------- Training Loop ---------------
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = mask_loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Train] Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

        # --------------- Validation ---------------
        model.eval()
        ious, precisions, recalls, f1s = [], [], [], []
        all_preds, all_gts = [], []
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                ious.append(compute_iou(outputs, masks))
                p, r, f1 = compute_precision_recall_f1(outputs, masks)
                precisions.append(p)
                recalls.append(r)
                f1s.append(f1)
                all_preds.append(outputs.cpu())
                all_gts.append(masks.cpu())
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
            print(f"✅ Best model updated at epoch {epoch+1} with IoU: {miou:.4f}")

    plot_metrics(history, save_folder)
    plot_confusion_matrix(model, val_loader, device, save_folder)

    # Save the best validation metrics
    with open(os.path.join(save_folder, "best_metrics.txt"), "w") as f:
        for k, v in best_metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    print(f"✅ Training complete. Model saved to: {model_path}")
    print(f"Settings → Epochs: {args.epochs}, Batch Size: {args.batch_size}, Loss: {args.loss}, Optimizer: {args.optimizer}")

if __name__ == '__main__':
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()
    main()

