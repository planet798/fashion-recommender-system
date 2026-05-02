# -*- coding: utf-8 -*-
"""
DIN (Deep Interest Network) 排序模型
参考阿里DIN论文，引入注意力机制对用户历史行为加权
核心思想: 针对候选商品，动态计算用户历史行为的注意力权重
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

from data_config import config


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(4 * hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, 1)

    def forward(self, query, keys, values, mask=None):
        query_expanded = query.unsqueeze(1).expand_as(keys)
        combined = keys * query_expanded
        diff = keys - query_expanded
        input_feat = torch.cat([keys, query_expanded, combined, diff], dim=-1)
        h = torch.relu(self.fc1(input_feat))
        scores = self.fc2(h).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        weighted = torch.sum(values * weights.unsqueeze(-1), dim=1)
        return weighted, weights


class DINRanker:
    def __init__(self, text_feature_path=None, image_feature_path=None,
                 multimodal_feature_path=None, history_path=None):
        from data_config import config as cfg
        self.text_features = np.load(
            text_feature_path or cfg.text_features, allow_pickle=True
        ).item()
        self.image_features = np.load(
            image_feature_path or cfg.image_features, allow_pickle=True
        ).item()
        self.multimodal_features = np.load(
            multimodal_feature_path or cfg.multimodal_features, allow_pickle=True
        ).item()
        if history_path:
            self.history = pd.read_csv(history_path)
            self.history["item_id"] = self.history["item_id"].astype(str)

        self.text_dim = 384
        self.image_dim = 512
        self.fusion_dim = 896
        self.attention = AttentionLayer(self.fusion_dim)

    def _normalize(self, vec):
        vec = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def _get_user_history_items(self, user_id, max_len=20):
        if not hasattr(self, 'history'):
            return []
        user_df = self.history[self.history["user_id"] == user_id]
        user_df = user_df.sort_values("timestamp", ascending=False)
        return [str(iid) for iid in user_df["item_id"].tolist()[:max_len]]

    def compute_attention_scores(self, query_vec, candidate_vecs):
        """
        DIN注意力: 候选商品对用户历史行为的注意力权重
        query_vec: 候选商品向量 (fusion_dim,)
        candidate_vecs: 用户历史商品向量列表 [(fusion_dim,), ...]
        返回: 加权后的用户表示向量
        """
        if len(candidate_vecs) == 0:
            return np.zeros(self.fusion_dim, dtype=np.float32)

        history_matrix = np.array(candidate_vecs, dtype=np.float32)
        query_tensor = torch.tensor(query_vec, dtype=torch.float32).unsqueeze(0)
        keys_tensor = torch.tensor(history_matrix, dtype=torch.float32).unsqueeze(0)
        values_tensor = keys_tensor.clone()

        with torch.no_grad():
            weighted, weights = self.attention(query_tensor, keys_tensor, values_tensor)

        return weighted.numpy()[0], weights.numpy()[0]

    def rerank_text_query(self, query_text, candidates, topk=8,
                          text_model=None, clip_model=None, clip_processor=None,
                          device="cpu"):
        """
        Text Search场景的DIN重排序
        对每个候选商品，用DIN注意力机制计算其与查询的匹配度
        """
        if text_model is not None:
            enhanced_query = (
                f"This is a fashion product search query about {query_text}. "
                f"Please focus on product category, style, use case, and fine-grained visual semantics."
            )
            query_text_vec = self._normalize(text_model.encode(enhanced_query))
        else:
            query_text_vec = None

        rows = []
        for candidate in candidates:
            if len(candidate) == 3:
                item_id, recall_score, bm25_score = candidate
            else:
                item_id, recall_score = candidate
                bm25_score = 0.0
            item_id = str(item_id)

            item_multi = self.multimodal_features.get(item_id)
            if item_multi is None:
                continue
            item_multi_vec = self._normalize(item_multi)

            if query_text_vec is not None:
                item_text = self.text_features.get(item_id)
                if item_text is not None:
                    item_text_vec = self._normalize(item_text)
                    text_sim = float(cosine_similarity(
                        [query_text_vec], [item_text_vec]
                    )[0][0])
                else:
                    text_sim = 0.0
            else:
                text_sim = 0.0

            item_img = self.image_features.get(item_id)
            clip_sim = 0.0
            if item_img is not None and clip_model is not None:
                item_img_vec = self._normalize(item_img)
                if query_text_vec is not None:
                    from fusion_utils import build_text_query_fusion
                    query_fusion = self._normalize(
                        build_text_query_fusion(query_text_vec)
                    )
                    clip_sim = float(cosine_similarity(
                        [query_fusion], [item_multi_vec]
                    )[0][0])

            din_score = 0.0
            if query_text_vec is not None:
                query_fusion = self._normalize(
                    np.concatenate([
                        query_text_vec,
                        np.zeros(self.image_dim, dtype=np.float32)
                    ])
                )
                din_weighted, _ = self.compute_attention_scores(
                    query_fusion, [item_multi_vec]
                )
                din_score = float(np.dot(self._normalize(din_weighted), item_multi_vec))

            rank_score = (
                0.35 * text_sim +
                0.20 * clip_sim +
                0.25 * max(din_score, 0.0) +
                0.20 * float(recall_score if isinstance(recall_score, (int, float)) else 0.0)
            )

            rows.append({
                "item_id": item_id,
                "rank_score": float(rank_score),
                "text_score": float(text_sim),
                "clip_score": float(clip_sim),
                "din_score": float(din_score),
                "recall_score": float(recall_score),
            })

        rows.sort(key=lambda x: x["rank_score"], reverse=True)
        return rows[:topk]

    def rerank_user_candidates(self, user_id, candidates, topk=8):
        """
        User Recommendation场景的DIN重排序
        对每个候选商品，用DIN注意力机制计算用户历史对候选的注意力加权表示
        """
        history_items = self._get_user_history_items(user_id, max_len=20)
        history_vecs = []
        for iid in history_items:
            vec = self.multimodal_features.get(iid)
            if vec is not None:
                history_vecs.append(self._normalize(vec))

        if not history_vecs:
            avg_vec = np.zeros(self.fusion_dim, dtype=np.float32)
        else:
            avg_vec = self._normalize(np.mean(history_vecs, axis=0))

        rows = []
        for candidate in candidates:
            if isinstance(candidate, dict):
                item_id = str(candidate["item_id"])
                recall_score = candidate.get("final_score", 0.0)
            elif isinstance(candidate, (tuple, list)):
                item_id = str(candidate[0])
                recall_score = candidate[1] if len(candidate) > 1 else 0.0
            else:
                item_id = str(candidate)
                recall_score = 0.0

            item_multi = self.multimodal_features.get(item_id)
            if item_multi is None:
                continue
            item_multi_vec = self._normalize(item_multi)

            if history_vecs:
                din_weighted, attn_weights = self.compute_attention_scores(
                    item_multi_vec, history_vecs
                )
                din_score = float(np.dot(self._normalize(din_weighted), item_multi_vec))
            else:
                din_score = 0.0

            avg_sim = float(cosine_similarity([avg_vec], [item_multi_vec])[0][0])

            rank_score = 0.6 * max(din_score, 0.0) + 0.4 * avg_sim

            rows.append({
                "item_id": item_id,
                "final_score": float(rank_score),
                "din_score": float(din_score),
                "avg_sim": float(avg_sim),
                "recall_score": float(recall_score),
            })

        rows.sort(key=lambda x: x["final_score"], reverse=True)
        return rows[:topk]
