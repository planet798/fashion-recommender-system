import argparse
from pathlib import Path
from urllib.request import urlretrieve

from amazon_dataset_paths import AMAZON_RAW, ensure_amazon_dirs


REVIEW_BASE = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories"
META_BASE = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories"


def download_file(url, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        print(f"Skip existing file: {output_path}")
        return

    print(f"Downloading: {url}")
    urlretrieve(url, output_path)
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Download Amazon Reviews 2023 category files from McAuley Lab.")
    parser.add_argument(
        "--category",
        default="Clothing_Shoes_and_Jewelry",
        help="Category name used in the official Amazon Reviews 2023 release, for example Clothing_Shoes_and_Jewelry or Amazon_Fashion."
    )
    args = parser.parse_args()

    ensure_amazon_dirs()
    category = args.category
    review_url = f"{REVIEW_BASE}/{category}.jsonl.gz"
    meta_url = f"{META_BASE}/meta_{category}.jsonl.gz"

    review_path = AMAZON_RAW / f"{category}.jsonl.gz"
    meta_path = AMAZON_RAW / f"meta_{category}.jsonl.gz"

    download_file(review_url, review_path)
    download_file(meta_url, meta_path)

    print("Download complete.")
    print(f"Review file: {review_path}")
    print(f"Metadata file: {meta_path}")


if __name__ == "__main__":
    main()
