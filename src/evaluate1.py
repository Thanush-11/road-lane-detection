from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.models import UNetSmall

def pixel_accuracy(pred, gt):
    """
    pred, gt: binary numpy arrays (0 or 1)
    """
    correct = (pred == gt).sum()
    total = pred.size
    return correct / total


def precision(pred, gt, eps=1e-6):
    """
    Precision = TP / (TP + FP)
    """
    tp = np.logical_and(pred == 1, gt == 1).sum()
    fp = np.logical_and(pred == 1, gt == 0).sum()
    return tp / (tp + fp + eps)


def recall(pred, gt, eps=1e-6):
    """
    Recall = TP / (TP + FN)
    """
    tp = np.logical_and(pred == 1, gt == 1).sum()
    fn = np.logical_and(pred == 0, gt == 1).sum()
    return tp / (tp + fn + eps)










acc = pixel_accuracy(pred, gt)
prec = precision(pred, gt)
rec = recall(pred, gt)

print(f"Pixel Accuracy : {acc:.4f}")
print(f"Precision      : {prec:.4f}")
print(f"Recall         : {rec:.4f}")
