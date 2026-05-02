# -*- coding: utf-8 -*-
"""
BERT特征预计算脚本
使用all-MiniLM-L6-v2模型为所有商品生成语义嵌入，并构建FAISS索引
"""

import argparse
import os
import tempfile

os.environ["TEMP"] = "D:\\TEMP"
os.environ["TMP"] = "D:\\TEMP"
os.environ["TMPDIR"] = "D:\\TEMP"
os.environ["HF_HOME"] = "D:\\HF_HOME"
os.environ["TRANSFORMERS_CACHE"] = "D:\\HF_HOME\\transformers"
tempfile.tempdir = "D:\\TEMP"

import numpy as np
import pandas as pd
import faiss
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Generate BERT features for baseline retrieval")
    parser.add_argument("--items-path", required=True, help="Path to items.csv")
    parser.add_argument("--output-dir", required=True, help="Output directory for features and index")
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2",
                        help="SentenceTransformer model name or local path")
    parser.add_argument("--save-model-path", default="",
                        help="If set, save model to this local path after download")
    parser.add_argument("--batch-size", type=int, default=256, help="Encoding batch size")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of items (0=all)")
    parser.add_argument("--images-dir", default="",
                        help="If set, only process items that have images in this directory")
    parser.add_argument("--device", default="", help="Device (cuda/cpu, auto-detect if empty)")
    args = parser.parse_args()

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading model: {args.model_name}")
    model = SentenceTransformer(args.model_name, device=device)
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"Model loaded. Embedding dim: {embedding_dim}")

    if args.save_model_path:
        print(f"Saving model to: {args.save_model_path}")
        model.save(args.save_model_path)
        print("Model saved.")

    items = pd.read_csv(args.items_path)
    items["item_id"] = items["item_id"].astype(str)

    if args.images_dir and os.path.isdir(args.images_dir):
        valid_ids = set(
            f.replace(".jpg", "") for f in os.listdir(args.images_dir) if f.endswith(".jpg")
        )
        before = len(items)
        items = items[items["item_id"].isin(valid_ids)].copy()
        print(f"Filtered by images: {before} -> {len(items)} items")

    if args.limit > 0:
        items = items.head(args.limit).copy()

    print(f"Processing {len(items)} items...")

    item_ids = []
    texts = []
    for _, row in items.iterrows():
        item_id = str(row["item_id"])
        parts = [str(row.get(col, "")) for col in ["title", "brand", "categories", "description"]]
        text = " ".join(parts).strip()
        item_ids.append(item_id)
        texts.append(text)

    all_embeddings = []
    print("Encoding items...")
    for start in tqdm(range(0, len(texts), args.batch_size)):
        end = min(start + args.batch_size, len(texts))
        batch = texts[start:end]
        embs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.append(embs)
        
        if ((start // args.batch_size) + 1) % 20 == 0:
            checkpoint_path = os.path.join(args.output_dir, "bert_features_checkpoint.npy")
            partial_embs = np.vstack(all_embeddings).astype(np.float32)
            partial_ids = item_ids[:len(partial_embs)]
            partial_dict = {iid: emb for iid, emb in zip(partial_ids, partial_embs)}
            np.save(checkpoint_path, partial_dict)
            processed = len(partial_embs)
            print(f"  [Checkpoint] Encoded {processed}/{len(texts)} items ({processed*100/len(texts):.1f}%)", flush=True)

    embeddings = np.vstack(all_embeddings).astype(np.float32)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    embeddings = embeddings / norms

    os.makedirs(args.output_dir, exist_ok=True)

    features_dict = {iid: emb for iid, emb in zip(item_ids, embeddings)}
    features_path = os.path.join(args.output_dir, "bert_features.npy")
    np.save(features_path, features_dict)
    print(f"Saved features to: {features_path}")

    print("Building FAISS HNSW index...")
    index = faiss.IndexHNSWFlat(embedding_dim, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 40
    index.hnsw.efSearch = 64
    index.add(embeddings)

    index_path = os.path.join(args.output_dir, "bert_faiss.index")
    faiss.write_index(index, index_path)
    print(f"Saved FAISS index to: {index_path}")

    ids_path = os.path.join(args.output_dir, "bert_faiss_ids.npy")
    np.save(ids_path, np.array(item_ids))
    print(f"Saved item IDs to: {ids_path}")

    print(f"\nDone! Processed {len(item_ids)} items")
    print(f"Feature dimension: {embedding_dim}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
