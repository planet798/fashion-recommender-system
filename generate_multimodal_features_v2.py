import argparse
import json
import os
import tempfile

os.environ["TEMP"] = "D:\\TEMP"
os.environ["TMP"] = "D:\\TEMP"
os.environ["TMPDIR"] = "D:\\TEMP"
tempfile.tempdir = "D:\\TEMP"

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from fusion_utils import load_fusion_config, normalize, weighted_fusion


def find_image_path(image_dir, item_id):
    item_id = str(item_id)
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        path = os.path.join(image_dir, f"{item_id}{ext}")
        if os.path.exists(path):
            return path
    return None


def main():
    parser = argparse.ArgumentParser(description="Generate multimodal features for any dataset path.")
    parser.add_argument("--items-path", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--clip-model-path", default="models/clip-vit-base-patch32")
    parser.add_argument("--text-model-path", default="models/paraphrase-MiniLM-L3-v2")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only-with-images", action="store_true",
                        help="Only process items that have images in image_dir")
    parser.add_argument("--history-path", default="")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--text-weight", type=float, default=None)
    parser.add_argument("--image-weight", type=float, default=None)
    parser.add_argument("--normalize-final", action="store_true")
    parser.add_argument("--fusion-config-path", default="")
    args = parser.parse_args()

    fusion_defaults = load_fusion_config(config_path=args.fusion_config_path or None)
    alpha = float(args.text_weight) if args.text_weight is not None else float(
        fusion_defaults["alpha"] if args.alpha is None else args.alpha
    )
    beta = float(args.image_weight) if args.image_weight is not None else float(
        fusion_defaults["beta"] if args.beta is None else args.beta
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading models...")
    clip_model = CLIPModel.from_pretrained(args.clip_model_path).to(device)
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model_path)
    text_model = SentenceTransformer(args.text_model_path)
    print("Models loaded")

    items = pd.read_csv(args.items_path)
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

    if args.only_with_images:
        valid_ids = set()
        for f in os.listdir(args.image_dir):
            if f.endswith((".jpg", ".jpeg", ".png", ".webp")):
                valid_ids.add(f.rsplit(".", 1)[0])
        before = len(items)
        items = items[items["item_id"].isin(valid_ids)].copy()
        print(f"Filtered by images: {before} -> {len(items)} items")

    os.makedirs(args.output_dir, exist_ok=True)
    text_features = {}
    image_features = {}
    multimodal_features = {}

    print(
        f"Generating multimodal features... alpha={alpha} "
        f"beta={beta} normalize_final={args.normalize_final}",
        flush=True
    )

    for idx, (_, row) in enumerate(tqdm(items.iterrows(), total=len(items))):
        item_id = str(row["item_id"])
        title = str(row.get("title", ""))
        brand = str(row.get("brand", ""))
        categories = str(row.get("categories", ""))
        description = str(row.get("description", ""))
        text_input = " ".join([title, brand, categories, description]).strip()

        text_emb = normalize(text_model.encode(text_input))

        image_path = find_image_path(args.image_dir, item_id)
        if image_path is not None:
            try:
                image = Image.open(image_path).convert("RGB")
                inputs = clip_processor(images=image, return_tensors="pt").to(device)
                with torch.no_grad():
                    img_emb = clip_model.get_image_features(**inputs)
                img_emb = normalize(img_emb.cpu().numpy()[0])
            except Exception:
                img_emb = np.zeros(512, dtype=np.float32)
        else:
            img_emb = np.zeros(512, dtype=np.float32)

        multimodal_emb = np.concatenate([text_emb, img_emb]).astype(np.float32)

        if args.normalize_final:
            multimodal_emb = normalize(multimodal_emb)

        text_features[item_id] = text_emb
        image_features[item_id] = img_emb
        multimodal_features[item_id] = multimodal_emb

        if (idx + 1) % 5000 == 0:
            np.save(os.path.join(args.output_dir, "text_features.npy"), text_features)
            np.save(os.path.join(args.output_dir, "image_features.npy"), image_features)
            np.save(os.path.join(args.output_dir, "multimodal_features.npy"), multimodal_features)
            print(f"  [Checkpoint] Saved {len(multimodal_features)} items at step {idx + 1}", flush=True)

    np.save(os.path.join(args.output_dir, "text_features.npy"), text_features)
    np.save(os.path.join(args.output_dir, "image_features.npy"), image_features)
    np.save(os.path.join(args.output_dir, "multimodal_features.npy"), multimodal_features)
    with open(os.path.join(args.output_dir, "fusion_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "alpha": alpha,
                "beta": beta,
                "text_weight": alpha,
                "image_weight": beta,
                "normalize_final": args.normalize_final,
                "items_processed": len(multimodal_features),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("Saved features to:", args.output_dir)
    print("Items processed:", len(multimodal_features))


if __name__ == "__main__":
    main()