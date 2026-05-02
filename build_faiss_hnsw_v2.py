import argparse
import os

import faiss
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Build a Faiss HNSW index from feature vectors.")
    parser.add_argument("--feature-path", required=True, help="Path to the .npy file containing feature vectors.")
    parser.add_argument("--index-path", required=True, help="Path to save the Faiss index.")
    parser.add_argument("--ids-path", required=True, help="Path to save the item IDs mapping.")
    parser.add_argument("--feature-type", type=str, default="multimodal", choices=["text", "image", "multimodal"],
                        help="Type of features to index.")
    args = parser.parse_args()

    # 根据特征类型调整输入/输出路径
    if args.feature_type != "multimodal":
        base_dir = os.path.dirname(args.feature_path)
        args.feature_path = os.path.join(base_dir, f"{args.feature_type}_features.npy")
        args.index_path = os.path.join(base_dir, f"faiss_{args.feature_type}.index")
        args.ids_path = os.path.join(base_dir, f"faiss_{args.feature_type}_ids.npy")

    print(f"Building index for '{args.feature_type}' features...")
    print(f"Loading features from: {args.feature_path}")

    features = np.load(args.feature_path, allow_pickle=True).item()
    item_ids = list(features.keys())
    vectors = np.array([features[item_id] for item_id in item_ids], dtype=np.float32)

    # 确保向量是C-contiguous的
    vectors = np.ascontiguousarray(vectors)

    # 归一化向量
    faiss.normalize_L2(vectors)

    dim = vectors.shape[1]
    print(f"Vector dimension: {dim}")

    # 使用内积作为度量
    index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 40
    index.hnsw.efSearch = 64

    print("Training index...")
    index.add(vectors)
    print(f"Index built successfully. Total vectors: {index.ntotal}")

    # 保存索引和ID
    output_dir = os.path.dirname(args.index_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Saving index to: {args.index_path}")
    faiss.write_index(index, args.index_path)

    print(f"Saving item IDs to: {args.ids_path}")
    np.save(args.ids_path, np.array(item_ids))

    print("Index and IDs saved successfully.")


if __name__ == "__main__":
    main()