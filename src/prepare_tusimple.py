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
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    for lane in lanes:
        pts = []
        for x, y in zip(lane, h_samples):
            if x is None or x < 0:
                continue
            pts.append((int(x), int(y)))

        if len(pts) >= 2:
            cv2.polylines(
                mask,
                [np.array(pts, dtype=np.int32)],
                isClosed=False,
                color=255,
                thickness=thickness,
            )

    return mask


def prepare_tusimple(
    raw_root="data/raw/tusimple",
    out_root="data/processed/tusimple",
    val_ratio=0.1,
    seed=42,
    thickness=8,
):
    raw_root = Path(raw_root)

    # Handle official TuSimple structure: data/raw/tusimple/train_set/...
    if (raw_root / "train_set").exists():
        raw_root = raw_root / "train_set"

    # Find label files dynamically
    json_files = sorted(raw_root.glob("label_data_*.json"))
    if not json_files:
        raise FileNotFoundError(
            f"No label_data_*.json found under: {raw_root}\n"
            f"Expected e.g. {raw_root / 'label_data_0313.json'}"
        )

    clips_dir = raw_root / "clips"
    if not clips_dir.exists():
        raise FileNotFoundError(
            f"Missing clips directory: {clips_dir}\n"
            f"Expected TuSimple images under .../clips/..."
        )

    out_root = Path(out_root)
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

    def resolve_image_path(raw_file: str) -> Path:
        # raw_file is often like: "clips/0313-1/6040/20.jpg"
        p1 = raw_root / raw_file
        if p1.exists():
            return p1

        # If raw_file starts with "clips/", try relative to clips_dir
        if raw_file.startswith("clips/") or raw_file.startswith("clips\\"):
            p2 = clips_dir / raw_file.split("clips", 1)[1].lstrip("/\\")
            if p2.exists():
                return p2

        # Last resort: treat as relative inside clips_dir
        p3 = clips_dir / raw_file
        if p3.exists():
            return p3

        raise FileNotFoundError(f"Image not found for raw_file='{raw_file}' (checked under {raw_root})")

    def process_split(split_name, split_items):
        list_path = out_splits / f"{split_name}.txt"
        with list_path.open("w", encoding="utf-8") as lf:
            for it in tqdm(split_items, desc=f"Preparing {split_name}"):
                rel_img = it["raw_file"]
                src_img_path = resolve_image_path(rel_img)

                img = cv2.imread(str(src_img_path))
                if img is None:
                    raise RuntimeError(f"Failed to read: {src_img_path}")
                h, w = img.shape[:2]

                mask = _draw_lane_mask(
                    img_w=w,
                    img_h=h,
                    lanes=it["lanes"],
                    h_samples=it["h_samples"],
                    thickness=thickness,
                )

                safe_name = rel_img.replace("/", "__").replace("\\", "__")
                dst_img_path = out_images / safe_name
                dst_mask_path = out_masks / safe_name

                shutil.copyfile(src_img_path, dst_img_path)
                cv2.imwrite(str(dst_mask_path), mask)

                lf.write(f"{safe_name}\n")

    process_split("train", train_items)
    process_split("val", val_items)

    print("Done.")
    print(f"Raw root used: {raw_root}")
    print(f"Train list:    {out_splits/'train.txt'}")
    print(f"Val list:      {out_splits/'val.txt'}")
    print(f"Images:        {out_images}")
    print(f"Masks:         {out_masks}")


if __name__ == "__main__":
    prepare_tusimple()
