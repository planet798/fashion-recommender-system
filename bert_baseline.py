# -*- coding: utf-8 -*-
"""
BERT语义检索基线
用于替代BM25作为Text Search模块的基线模型
使用all-MiniLM-L6-v2进行语义文本检索，解决BM25关键词匹配与多模态模型语义理解不匹配的问题
"""

import os
import heapq
import tempfile

os.environ["TEMP"] = "D:\\TEMP"
os.environ["TMP"] = "D:\\TEMP"
os.environ["TMPDIR"] = "D:\\TEMP"
tempfile.tempdir = "D:\\TEMP"

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


class BERTRetriever:

    def __init__(self, items_path=None, model_name_or_path=None,
                 bert_features_path=None, bert_index_path=None, bert_ids_path=None,
                 valid_ids=None, device=None):
        from data_config import config
        items_path = items_path or config.items_csv
        model_name_or_path = model_name_or_path or config.bert_model_path
        bert_features_path = bert_features_path or config.bert_features
        bert_index_path = bert_index_path or config.bert_index
        bert_ids_path = bert_ids_path or config.bert_ids

        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.items = pd.read_csv(items_path)
        if valid_ids is not None:
            valid_set = set(str(v) for v in valid_ids)
            self.items = self.items[self.items["item_id"].astype(str).isin(valid_set)]

        self.item_titles = {
            str(row["item_id"]): str(row["title"])
            for _, row in self.items.iterrows()
        }

        self.item_texts = {}
        for _, row in self.items.iterrows():
            item_id = str(row["item_id"])
            parts = [str(row.get(col, "")) for col in ["title", "brand", "categories", "description"]]
            text = " ".join(parts).strip()
            self.item_texts[item_id] = text

        print(f"[BERTRetriever] Loading model from {model_name_or_path}...")
        self.model = SentenceTransformer(model_name_or_path, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"[BERTRetriever] Model loaded. dim={self.embedding_dim}")

        self.item_ids_list = []
        self.item_embeddings = None
        self.faiss_index = None

        if os.path.exists(bert_features_path) and os.path.exists(bert_index_path) and os.path.exists(bert_ids_path):
            print(f"[BERTRetriever] Loading pre-computed features...")
            features_dict = np.load(bert_features_path, allow_pickle=True).item()
            self.item_ids_list = list(features_dict.keys())
            self.item_embeddings = np.array(
                [features_dict[k] for k in self.item_ids_list], dtype=np.float32
            )
            self._normalize_embeddings()
            self.faiss_index = faiss.read_index(bert_index_path)
            print(f"[BERTRetriever] Loaded {len(self.item_ids_list)} items with FAISS index")
        else:
            print(f"[BERTRetriever] Pre-computed features not found, building from items...")
            self._build_features_and_index(bert_features_path, bert_index_path, bert_ids_path)

    def _normalize_embeddings(self):
        norms = np.linalg.norm(self.item_embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        self.item_embeddings = self.item_embeddings / norms

    def _build_features_and_index(self, features_path, index_path, ids_path):
        print(f"[BERTRetriever] Encoding {len(self.item_texts)} items...")
        item_ids = list(self.item_texts.keys())
        texts = [self.item_texts[iid] for iid in item_ids]

        batch_size = 256
        all_embeddings = []
        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            batch = texts[start:end]
            embs = self.model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
            all_embeddings.append(embs)
            if end % 1000 == 0 or end == len(texts):
                print(f"  Encoded {end}/{len(texts)} items")

        self.item_embeddings = np.vstack(all_embeddings).astype(np.float32)
        self.item_ids_list = item_ids

        features_dict = {iid: emb for iid, emb in zip(item_ids, self.item_embeddings)}
        os.makedirs(os.path.dirname(features_path), exist_ok=True)
        np.save(features_path, features_dict)
        print(f"[BERTRetriever] Saved features to {features_path}")

        self._build_faiss_index(index_path, ids_path)

    def _build_faiss_index(self, index_path, ids_path):
        dim = self.embedding_dim
        index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 40
        index.hnsw.efSearch = 64

        self._normalize_embeddings()
        index.add(self.item_embeddings)

        self.faiss_index = index

        faiss.write_index(index, index_path)
        np.save(ids_path, np.array(self.item_ids_list))
        print(f"[BERTRetriever] Built FAISS index with {len(self.item_ids_list)} items, saved to {index_path}")

    def _encode_query(self, query):
        vec = self.model.encode(query, normalize_embeddings=True)
        return np.array(vec, dtype=np.float32)

    def search(self, query, topk=10, exclude_item_ids=None):
        exclude_item_ids = set(exclude_item_ids or [])
        query_vec = self._encode_query(query).reshape(1, -1)

        search_k = min(topk + len(exclude_item_ids), len(self.item_ids_list))
        search_k = max(search_k, topk)
        scores, indices = self.faiss_index.search(query_vec, search_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            item_id = str(self.item_ids_list[idx])
            if item_id in exclude_item_ids:
                continue
            results.append((item_id, float(score)))
            if len(results) >= topk:
                break

        return results

    def score_item(self, query, item_id):
        item_id = str(item_id)
        if item_id not in self.item_texts:
            return 0.0

        query_vec = self._encode_query(query)

        try:
            idx = self.item_ids_list.index(item_id)
            item_vec = self.item_embeddings[idx]
        except ValueError:
            item_vec = self.model.encode(
                self.item_texts[item_id], normalize_embeddings=True
            )
            item_vec = np.array(item_vec, dtype=np.float32)

        similarity = float(np.dot(query_vec, item_vec))
        return similarity

    def build_user_query(self, history_df, user_id, max_len=5):
        user_df = history_df[history_df["user_id"] == user_id].sort_values("timestamp", ascending=False)
        item_ids = [str(item_id) for item_id in user_df["item_id"].tolist()[:max_len]]
        titles = [self.item_titles.get(iid, "") for iid in item_ids]
        query = " ".join(titles).strip()
        return query, item_ids

    def recommend_for_user(self, history_df, user_id, topk=10, max_history=5):
        query, history_items = self.build_user_query(history_df, user_id, max_len=max_history)
        if not query:
            return []
        return self.search(query, topk=topk, exclude_item_ids=history_items)
