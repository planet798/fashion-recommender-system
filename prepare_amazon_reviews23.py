import argparse
import gzip
import json
from pathlib import Path

import pandas as pd

from amazon_dataset_paths import AMAZON_IMAGES, AMAZON_PROCESSED, AMAZON_RAW, ensure_amazon_dirs


def _open_text(path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _iter_jsonl(path):
    with _open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _pick_title(record):
    for key in ["title", "parent_title", "product_title"]:
        value = record.get(key)
        if value:
            return str(value)
    return ""


def _pick_image_url(record):
    image = record.get("images")
    if isinstance(image, list) and image:
        first = image[0]
        if isinstance(first, dict):
            for key in ["hi_res", "large", "thumb"]:
                value = first.get(key)
                if isinstance(value, str) and value:
                    return value
        if isinstance(first, str) and first:
            return first

    if isinstance(image, dict):
        for key in ["large", "hi_res", "thumb", "variant"]:
            value = image.get(key)
            if isinstance(value, list) and value:
                return str(value[0])
            if isinstance(value, str) and value:
                return value

    for key in ["image_url", "imageURLHighRes", "imageURL"]:
        value = record.get(key)
        if isinstance(value, list) and value:
            return str(value[0])
        if isinstance(value, str) and value:
            return value

    return ""


def convert_metadata(meta_path, output_items_path):
    rows = []

    for record in _iter_jsonl(meta_path):
        asin = record.get("parent_asin") or record.get("asin")
        title = _pick_title(record)
        if not asin or not title:
            continue

        rows.append({
            "item_id": str(asin),
            "title": title,
            "brand": str(record.get("store", "")),
            "categories": " > ".join(record.get("categories", [])) if isinstance(record.get("categories"), list) else str(record.get("categories", "")),
            "price": str(record.get("price", "")),
            "description": " ".join(record.get("description", [])) if isinstance(record.get("description"), list) else str(record.get("description", "")),
            "image_url": _pick_image_url(record)
        })

    items_df = pd.DataFrame(rows).drop_duplicates(subset=["item_id"])
    items_df.to_csv(output_items_path, index=False)
    return items_df


def convert_reviews(review_path, output_history_path, max_users=None, min_interactions=5, max_interactions=30):
    rows = []

    for record in _iter_jsonl(review_path):
        user_id = record.get("user_id") or record.get("reviewerID")
        item_id = record.get("parent_asin") or record.get("asin")
        timestamp = record.get("timestamp") or record.get("unixReviewTime")
        if not user_id or not item_id or timestamp is None:
            continue

        rows.append({
            "user_id": str(user_id),
            "item_id": str(item_id),
            "timestamp": int(timestamp)
        })

    history = pd.DataFrame(rows)
    if history.empty:
        raise ValueError("review file produced no usable interactions")

    history = history.sort_values(["user_id", "timestamp"])
    user_sizes = history.groupby("user_id").size()
    valid_users = user_sizes[(user_sizes >= min_interactions) & (user_sizes <= max_interactions)].index
    history = history[history["user_id"].isin(valid_users)].copy()

    if max_users is not None:
        keep_users = sorted(history["user_id"].unique())[:max_users]
        history = history[history["user_id"].isin(keep_users)].copy()

    history["timestamp"] = history.groupby("user_id").cumcount() + 1
    history.to_csv(output_history_path, index=False)
    return history


def main():
    parser = argparse.ArgumentParser(description="Prepare Amazon Reviews 2023 fashion data without touching the existing toy data pipeline.")
    parser.add_argument("--meta-path", required=True, help="Path to the raw metadata jsonl/jsonl.gz file.")
    parser.add_argument("--review-path", required=True, help="Path to the raw reviews jsonl/jsonl.gz file.")
    parser.add_argument("--max-users", type=int, default=2000)
    parser.add_argument("--min-interactions", type=int, default=5)
    parser.add_argument("--max-interactions", type=int, default=30)
    args = parser.parse_args()

    ensure_amazon_dirs()
    items_path = AMAZON_PROCESSED / "items.csv"
    history_path = AMAZON_PROCESSED / "user_history.csv"

    items_df = convert_metadata(args.meta_path, items_path)
    history_df = convert_reviews(
        args.review_path,
        history_path,
        max_users=args.max_users,
        min_interactions=args.min_interactions,
        max_interactions=args.max_interactions
    )

    print(f"Saved metadata to: {items_path}")
    print(f"Saved interactions to: {history_path}")
    print(f"Items: {len(items_df)}")
    print(f"Interactions: {len(history_df)}")
    print(f"Users: {history_df['user_id'].nunique()}")
    print(f"Images directory (download separately): {AMAZON_IMAGES}")
    print(f"Raw input directory: {AMAZON_RAW}")


if __name__ == "__main__":
    main()
