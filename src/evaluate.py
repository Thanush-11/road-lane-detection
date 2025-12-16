from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.models import UNetSmall


def pixel_accuracy(pred, gt):
    correct = (pred == gt).sum()
    total = gt.size
    return correct / total


def precision_score(pred, gt, eps=1e-6):
    tp = np.logical_and(pred == 1, gt == 1).sum()
    fp = np.logical_and(pred == 1, gt == 0).sum()
    return tp / (tp + fp + eps)


def recall_score(pred, gt, eps=1e-6):
    tp = np.logical_and(pred == 1, gt == 1).sum()
    fn = np.logical_and(pred == 0, gt == 1).sum()
    return tp / (tp + fn + eps)


def f1_score(pred, gt, eps=1e-6):
    prec = precision_score(pred, gt, eps)
    rec = recall_score(pred, gt, eps)
    return 2 * prec * rec / (prec + rec + eps)


def iou(pred, gt, eps=1e-6):
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return (inter + eps) / (union + eps)


def dice(pred, gt, eps=1e-6):
    inter = (pred & gt).sum()
    return (2 * inter + eps) / (pred.sum() + gt.sum() + eps)


def main():
    root = Path("data/processed/tusimple")
    img_dir = root / "images"
    mask_dir = root / "masks"
    val_list = (root / "splits/val.txt").read_text().splitlines()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path("models/checkpoints/unet_tusimple_best.pt")

    model = UNetSmall().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    img_h, img_w = 360, 640  # must match training

    # Lists to collect metrics across the whole validation set
    ious, dices = [], []
    accs, precs, recs, f1s = [], [], [], []

    with torch.no_grad():
        for name in tqdm(val_list, desc="Evaluating"):
            img = cv2.imread(str(img_dir / name))
            gt = cv2.imread(str(mask_dir / name), cv2.IMREAD_GRAYSCALE)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
            gt = cv2.resize(gt, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

            x = (img.astype(np.float32) / 255.0)
            x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)

            logits = model(x)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            pred = (prob > 0.5).astype(np.uint8)       # 0/1 prediction
            gt_bin = (gt > 127).astype(np.uint8)       # 0/1 ground truth

            # IoU/Dice (boolean form)
            ious.append(iou(pred.astype(bool), gt_bin.astype(bool)))
            dices.append(dice(pred.astype(bool), gt_bin.astype(bool)))

            # Accuracy/Precision/Recall/F1 (0/1 arrays)
            accs.append(pixel_accuracy(pred, gt_bin))
            precs.append(precision_score(pred, gt_bin))
            recs.append(recall_score(pred, gt_bin))
            f1s.append(f1_score(pred, gt_bin))

    print(f"Val IoU      : {float(np.mean(ious)):.4f}")
    print(f"Val Dice     : {float(np.mean(dices)):.4f}")
    print(f"Val Accuracy : {float(np.mean(accs)):.4f}")
    print(f"Val Precision: {float(np.mean(precs)):.4f}")
    print(f"Val Recall   : {float(np.mean(recs)):.4f}")
    print(f"Val F1 Score : {float(np.mean(f1s)):.4f}")


if __name__ == "__main__":
    main()
