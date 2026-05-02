# -*- coding: utf-8 -*-
"""
扩展图片下载脚本
从items.csv中选取有image_url且尚未下载图片的商品，下载到5万张
优先下载交互频率高的商品
"""

import argparse
import os
from urllib.parse import urlparse
from urllib.request import urlretrieve

import pandas as pd
from tqdm import tqdm


def choose_extension(image_url):
    path = urlparse(str(image_url)).path.lower()
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        if path.endswith(ext):
            return ext
    return ".jpg"


def main():
    parser = argparse.ArgumentParser(description="Expand product images to ~50k")
    parser.add_argument("--items-path", default="datasets/amazon_reviews23/processed/items.csv")
    parser.add_argument("--output-dir", default="datasets/amazon_reviews23/images")
    parser.add_argument("--history-path", default="datasets/amazon_reviews23/processed/user_history.csv")
    parser.add_argument("--target-count", type=int, default=50000, help="Target total image count")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    existing = set(
        f.replace(".jpg", "").replace(".jpeg", "").replace(".png", "").replace(".webp", "")
        for f in os.listdir(args.output_dir)
        if f.endswith((".jpg", ".jpeg", ".png", ".webp"))
    )
    print(f"Existing images: {len(existing)}")

    needed = max(0, args.target_count - len(existing))
    if needed == 0:
        print(f"Already have {len(existing)} images, target {args.target_count} reached.")
        return

    print(f"Need to download {needed} more images.")

    items = pd.read_csv(args.items_path)
    items["item_id"] = items["item_id"].astype(str)

    if "image_url" not in items.columns:
        raise ValueError("items.csv must contain an image_url column")

    items = items[items["image_url"].notna() & (items["image_url"].astype(str).str.len() > 5)].copy()
    print(f"Items with image_url: {len(items)}")

    items = items[~items["item_id"].isin(existing)].copy()
    print(f"Items not yet downloaded: {len(items)}")

    if args.history_path and os.path.exists(args.history_path):
        history = pd.read_csv(args.history_path)
        history["item_id"] = history["item_id"].astype(str)
        item_freq = history["item_id"].value_counts().to_dict()
        items["freq"] = items["item_id"].map(item_freq).fillna(0).astype(int)
        items = items.sort_values("freq", ascending=False)
        print(f"Sorted by interaction frequency (top freq: {items['freq'].iloc[0] if len(items) > 0 else 0})")

    items = items.head(needed).copy()
    print(f"Will attempt to download: {len(items)} images")

    downloaded = 0
    failed = 0

    for _, row in tqdm(items.iterrows(), total=len(items)):
        item_id = str(row["item_id"])
        image_url = str(row["image_url"])
        ext = choose_extension(image_url)
        output_path = os.path.join(args.output_dir, f"{item_id}{ext}")

        if os.path.exists(output_path):
            downloaded += 1
            continue

        try:
            urlretrieve(image_url, output_path)
            if os.path.getsize(output_path) < 500:
                os.remove(output_path)
                failed += 1
            else:
                downloaded += 1
        except Exception:
            failed += 1

    final_count = len([
        f for f in os.listdir(args.output_dir)
        if f.endswith((".jpg", ".jpeg", ".png", ".webp"))
    ])
    print(f"\nDownloaded: {downloaded}")
    print(f"Failed: {failed}")
    print(f"Total images now: {final_count}")
    print(f"Images dir: {args.output_dir}")


if __name__ == "__main__":
    main()
