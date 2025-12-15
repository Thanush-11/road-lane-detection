import os
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def _read_json_lines(json_path: Path):
    items = []
    with json_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _draw_lane_mask(img_w: int, img_h: int, lanes, h_samples, thickness=8):
    """
    lanes: list of list of x coords (same length as h_samples), -2 for missing.
    h_samples: list of y coords
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    for lane in lanes:
        pts = []
        for x, y in zip(lane, h_samples):
            if x is None or x < 0:
                continue
            pts.append((int(x), int(y)))

        # draw as polyline if enough points
        if len(pts) >= 2:
            cv2.polylines(mask, [np.array(pts, dtype=np.int32)], isClosed=False, color=255, thickness=thickness)

    return mask


def prepare_tusimple(
    raw_root="data/raw/tusimple",
    out_root="data/processed/tusimple",
    val_ratio=0.1,
    seed=42,
    thickness=8
):
    raw_root = Path(raw_root)
    out_root = Path(out_root)
    clips_dir = raw_root / "clips"

    json_files = [
        raw_root / "label_data_0313.json",
        raw_root / "label_data_0531.json",
        raw_root / "label_data_0601.json",
    ]

    for jf in json_files:
        if not jf.exists():
            raise FileNotFoundError(f"Missing: {jf}")

    out_images = out_root / "images"
    out_masks = out_root / "masks"
    out_splits = out_root / "splits"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)
    out_splits.mkdir(parents=True, exist_ok=True)

    # Collect all items
    items = []
    for jf in json_files:
        items.extend(_read_json_lines(jf))

    # Deterministic shuffle
    rng = np.random.default_rng(seed)
    idx = np.arange(len(items))
    rng.shuffle(idx)
    items = [items[i] for i in idx]

    n_val = int(len(items) * val_ratio)
    val_items = items[:n_val]
    train_items = items[n_val:]

    def process_split(split_name, split_items):
        list_path = out_splits / f"{split_name}.txt"
        with list_path.open("w", encoding="utf-8") as lf:
            for it in tqdm(split_items, desc=f"Preparing {split_name}"):
                # Example: "clips/0313-1/6040/20.jpg"
                rel_img = it["raw_file"]
                src_img_path = raw_root / rel_img
                if not src_img_path.exists():
                    # Some releases store raw_file relative to clips folder
                    src_img_path = clips_dir / rel_img.replace("clips/", "")
                if not src_img_path.exists():
                    raise FileNotFoundError(f"Image not found: {rel_img}")

                img = cv2.imread(str(src_img_path))
                if img is None:
                    raise RuntimeError(f"Failed to read: {src_img_path}")
                h, w = img.shape[:2]

                mask = _draw_lane_mask(
                    img_w=w,
                    img_h=h,
                    lanes=it["lanes"],
                    h_samples=it["h_samples"],
                    thickness=thickness
                )

                # Create stable filename (replace slashes)
                safe_name = rel_img.replace("/", "__").replace("\\", "__")
                dst_img_path = out_images / safe_name
                dst_mask_path = out_masks / safe_name

                # Copy image, write mask
                shutil.copyfile(src_img_path, dst_img_path)
                cv2.imwrite(str(dst_mask_path), mask)

                lf.write(f"{safe_name}\n")

    process_split("train", train_items)
    process_split("val", val_items)

    print("Done.")
    print(f"Train list: {out_splits/'train.txt'}")
    print(f"Val list:   {out_splits/'val.txt'}")
    print(f"Images:     {out_images}")
    print(f"Masks:      {out_masks}")


if __name__ == "__main__":
    prepare_tusimple()
