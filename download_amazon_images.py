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
    parser = argparse.ArgumentParser(description="Download product images from an items.csv file.")
    parser.add_argument("--items-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--history-path", default="")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    items = pd.read_csv(args.items_path)
    if "image_url" not in items.columns:
        raise ValueError("items.csv must contain an image_url column")

    items = items[items["image_url"].notna() & (items["image_url"].astype(str).str.len() > 0)].copy()
    items["item_id"] = items["item_id"].astype(str)

    if args.history_path:
        history = pd.read_csv(args.history_path)
        history["item_id"] = history["item_id"].astype(str)
        target_item_ids = history["item_id"].value_counts().index.tolist()
        items = items.set_index("item_id")
        items = items.loc[items.index.intersection(target_item_ids)].reset_index()
        order_map = {item_id: idx for idx, item_id in enumerate(target_item_ids)}
        items["history_rank"] = items["item_id"].map(order_map)
        items = items.sort_values("history_rank").drop(columns=["history_rank"])

    if args.limit > 0:
        items = items.head(args.limit).copy()

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
            downloaded += 1
        except Exception:
            failed += 1

    print(f"Downloaded: {downloaded}")
    print(f"Failed: {failed}")
    print(f"Images dir: {args.output_dir}")


if __name__ == "__main__":
    main()
