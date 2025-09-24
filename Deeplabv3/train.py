import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
import cv2
import numpy as np
import pickle
from tqdm import tqdm
import argparse
import matplotlib
matplotlib.use('Agg')  # <- Must be set before importing pyplot
import matplotlib.pyplot as plt
from cbam import CBAM
from cbam import inject_cbam_into_layer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
class TreeSegmentationDataset(Dataset):
    def __init__(self, samples, img_size=(256, 256)):
        self.samples = samples
        self.img_size = img_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, nir_path, mask_path = self.samples[idx]

        if not (os.path.exists(rgb_path) and os.path.exists(nir_path) and os.path.exists(mask_path)):
            raise FileNotFoundError(f"Missing file in sample {idx}")

        # load and return your image, nir, mask...

        rgb = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
        nir = cv2.imread(nir_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        if rgb is None or nir is None or mask is None:
            return self.__getitem__((idx + 1) % len(self.samples))

        rgb = cv2.resize(rgb, self.img_size).astype(np.float32) / 255.
        nir = cv2.resize(nir, self.img_size).astype(np.float32) / 255.
        nir = nir[..., 0:1]

        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.float32)

        image = np.concatenate([rgb, nir], axis=-1)
        image = torch.from_numpy(image.transpose(2, 0, 1))
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask

def dice_loss(pred, target, epsilon=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 1 - (2. * intersection + epsilon) / (union + epsilon)

def focal_loss(pred, target, alpha=0.8, gamma=2):
    bce = nn.BCEWithLogitsLoss(reduction='none')(pred, target)
    pt = torch.exp(-bce)
    return ((alpha * (1 - pt) ** gamma) * bce).mean()

def combo_loss(pred, target):
    return 0.5 * focal_loss(pred, target) + 0.5 * dice_loss(pred, target)

def compute_iou(pred, mask, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * mask).sum()
    union = ((pred + mask) > 0).float().sum()
    if union == 0:
        return float('nan')
    return float((intersection / union).item())
def compute_precision_recall_f1(pred, target, threshold=0.3):
    pred = (torch.sigmoid(pred) > threshold).float()
    target = (target > 0.5).float()

    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return precision.item(), recall.item(), f1.item()

def get_deeplabv3_model(use_cbam=False):
    model = deeplabv3_resnet101(weights=None)  # Use pretrained if needed

    # ✅ Modify first conv for 4-channel input
    old_conv = model.backbone.conv1
    new_conv = nn.Conv2d(
        4, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    with torch.no_grad():
        new_conv.weight[:, :3] = old_conv.weight
        new_conv.weight[:, 3] = old_conv.weight[:, 0]  # NIR = copy red channel
    model.backbone.conv1 = new_conv

    # ✅ Update classifier for 1-class binary mask
    model.classifier[-1] = nn.Conv2d(256, 1, kernel_size=1)
    return model


def get_args():
    parser = argparse.ArgumentParser(description='Train DeepLabV3+ on Tree Segmentation Dataset')

    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--img_size', type=int, default=256, help='Image size (H and W)')
    parser.add_argument('--loss', type=str, default='bce+dice',
                            choices=['bce+dice', 'focal', 'tversky', 'focal+dice', 'focal+tversky'],
                            help='Loss function')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adamw', 'sgd'], help='Optimizer')
    parser.add_argument('--num_workers', type=int, default=0,
                            help='Dataloader num_workers (set 0 for low memory systems)')
    parser.add_argument('--use_cbam', action='store_true', help='Enable CBAM attention in ResNet backbone')
    parser.add_argument('--save_dir', type=str, default='runs', help='Directory to save checkpoints and results')

    return parser.parse_args()


def compute_confusion_matrix(model, val_loader, device, threshold=0.5):
    all_preds = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)['out']
            preds = (torch.sigmoid(outputs) > threshold).float()

            all_preds.extend(preds.cpu().numpy().astype(int).reshape(-1))
            all_targets.extend(masks.cpu().numpy().astype(int).reshape(-1))

    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])
    return cm


def plot_confusion_matrix(cm, save_dir):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Background", "Dead Tree"])
    disp.plot(cmap=plt.cm.Blues, values_format="d")
    plt.title("Confusion Matrix (Best Epoch)")

    # Build correct save path
    save_path = os.path.join(save_dir, "best_confusion_matrix.png")
    os.makedirs(save_dir, exist_ok=True)  # ensure directory exists
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Confusion matrix saved to {save_path}")


def plot_metrics_and_save_best(history, best_metrics, save_dir):
    # Save best metrics
    with open(os.path.join(save_dir, "best_metrics.txt"), "w") as f:
        for k, v in best_metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    # 📊 Plot curves and save
    plt.figure()
    plt.plot(history["iou"], label="IoU")
    plt.plot(history["precision"], label="Precision")
    plt.plot(history["recall"], label="Recall")
    plt.plot(history["f1"], label="F1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Metrics Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "metrics_curve.png"))
    plt.close()

def train_model(model, train_loader, val_loader, device, epochs, lr,
                optimizer_type="adam", use_cbam=False, save_dir="results"):
    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    history = {"iou": [], "precision": [], "recall": [], "f1": []}
    best_iou = 0.0
    best_metrics = {}
    best_epoch = 0
    best_model_path = os.path.join(save_dir, "best_model.pth")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = combo_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Train] Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        ious, precisions, recalls, f1s = [], [], [], []
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)['out']

                ious.append(compute_iou(outputs, masks))
                p, r, f1 = compute_precision_recall_f1(outputs, masks)
                precisions.append(p)
                recalls.append(r)
                f1s.append(f1)

        miou = np.nanmean(ious)
        p, r, f1 = np.mean(precisions), np.mean(recalls), np.mean(f1s)

        print(f"[Val] Epoch {epoch+1} | mIoU: {miou:.4f} | Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}")

        history["iou"].append(miou)
        history["precision"].append(p)
        history["recall"].append(r)
        history["f1"].append(f1)

        if miou > best_iou:
            best_iou = miou
            best_epoch = epoch + 1
            best_metrics = {"iou": miou, "precision": p, "recall": r, "f1": f1, "epoch": best_epoch}
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Best model updated at epoch {best_epoch} with IoU: {miou:.4f}")

    # 🔹 After training: compute confusion matrix for best epoch
    print(f"🔹 Loading best model from epoch {best_epoch} for confusion matrix...")
    model.load_state_dict(torch.load(best_model_path))
    cm = compute_confusion_matrix(model, val_loader, device)
    plot_confusion_matrix(cm, save_dir)


    plot_metrics_and_save_best(history, best_metrics, save_dir)
    return model, history





def main():
    args = get_args()

    # Generate a folder name based on args
    folder_name = f"loss_{args.loss}_size_{args.img_size}_opt_{args.optimizer},batch_size_{args.batch_size},lr-{args.lr},epochs{args.epochs}"
    save_folder = os.path.join(args.save_dir, folder_name)
    os.makedirs(save_folder, exist_ok=True)

    # Load and split dataset
    with open("../paired_samples.pkl", "rb") as f:
        samples = pickle.load(f)

    np.random.shuffle(samples)
    split = int(0.8 * len(samples))
    train_samples = samples[:split]
    val_samples = samples[split:]

    # Prepare datasets and dataloaders
    train_set = TreeSegmentationDataset(train_samples, img_size=(args.img_size, args.img_size))
    val_set = TreeSegmentationDataset(val_samples, img_size=(args.img_size, args.img_size))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ Create model before injecting CBAM
    model = get_deeplabv3_model(use_cbam=args.use_cbam).to(device)

    if args.use_cbam:
        print("Injecting CBAM into ResNet layer4...")
        inject_cbam_into_layer(model.backbone.layer4, name="layer4", device=device)

    # Train model
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        save_dir=save_folder
    )

    # Save model inside the subfolder
    model_path = os.path.join(save_folder, "deeplabv3_tree_seg.pth")
    torch.save(model.state_dict(), model_path)

    # Plot metrics
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

    plot_metrics(history, save_folder)

    # Final message
    print(f"✅ Training complete. Model saved to: {model_path}")
    print(f"Settings → Epochs: {args.epochs}, Batch Size: {args.batch_size}, Loss: {args.loss}, Optimizer: {args.optimizer}, CBAM: {args.use_cbam}")


if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()
    main()

