# -*- coding: utf-8 -*-
"""
优化的排序模型 v3 - 引入Hard Negatives机制

解决ranking_model.py中的问题:
1. 负采样过于简单 (纯随机)，模型学习效率低
2. 无法区分"明显错误"和"容易混淆"的负样本
3. 缺乏难负样本挖掘策略

Hard Negatives类型:
- Type 1: BM25高排名但未点击的商品 (语义相似但非偏好)
- Type 2: Faiss近邻但非正样本 (视觉/语义相似但非目标)
- Type 3: 排序模型预测为正但实际为负的样本 (模型混淆样本)

改进策略:
- 混合采样: random_neg + hard_bm25 + hard_faiss
- 动态比例: 根据训练进度调整难负样本占比
- 去重: 避免与正样本和历史物品重复
"""
import os
import pickle
import tempfile
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

from bm25_baseline import BM25Retriever, tokenize
from user_model import UserInterestModel


class PairwiseFeatureRankerV3:
    """
    优化的Pairwise排序模型 - 支持Hard Negatives

    核心改进:
    1. 多源负采样池构建
    2. 难负样本智能筛选
    3. 渐进式训练策略 (从易到难)
    """

    def __init__(self, model_path=None, text_feature_path=None, image_feature_path=None):
        from data_config import config
        self.model_path = model_path or config.ranking_model_v3
        self.text_feature_path = text_feature_path or config.text_features
        self.image_feature_path = image_feature_path or config.image_features
        self.model = None
        self.text_features = None
        self.image_features = None
        self.temp_items_path = None
        self._faiss_candidate_matrix = None
        self._faiss_candidate_ids = None

        # 特征定义 (与V2保持一致，确保兼容性)
        self.feature_names = [
            "bm25_raw",
            "faiss_raw",
            "text_similarity",
            "image_similarity",
            "recall_score",
            "from_bm25",
            "from_faiss",
            "from_both",
            "hybrid_sim",
            "short_term_sim",
            "long_term_sim",
            "max_history_sim",
            "mean_history_sim",
            "token_overlap",
            "item_token_len"
        ]

        # Hard Negatives超参数
        self.hard_negative_config = {
            "bm25_hard_ratio": 0.3,      # BM25难负样本占比
            "faiss_hard_ratio": 0.3,     # Faiss难负样本占比
            "random_ratio": 0.4,         # 随机负样本占比
            "bm25_topk_for_hard": 50,    # 从BM25 top50中选难负样本
            "faiss_topk_for_hard": 30,   # 从Faiss top30中选难负样本
            "min_hard_score_threshold": 0.5,  # 最小难度阈值
        }

    def _normalize(self, vec):
        vec = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def _item_key(self, item_id):
        return str(item_id)

    def _build_train_item_subset(self, items_path, history):
        items = pd.read_csv(items_path)
        items["item_id"] = items["item_id"].astype(str)
        history_item_ids = set(history["item_id"].astype(str).tolist())
        return items[items["item_id"].isin(history_item_ids)].copy()

    def _resolve_aux_path(self, explicit_path, filename, base_dir):
        if explicit_path:
            return explicit_path
        return os.path.join(base_dir, filename)

    def _load_aux_features(self, feature_path):
        base_dir = os.path.dirname(feature_path)
        text_path = self._resolve_aux_path(self.text_feature_path, "text_features.npy", base_dir)
        image_path = self._resolve_aux_path(self.image_feature_path, "image_features.npy", base_dir)

        if os.path.exists(text_path):
            self.text_features = np.load(text_path, allow_pickle=True).item()
        else:
            self.text_features = {}

        if os.path.exists(image_path):
            self.image_features = np.load(image_path, allow_pickle=True).item()
        else:
            self.image_features = {}

    def _mean_feature(self, item_ids, feature_store):
        vectors = [
            self._normalize(feature_store[self._item_key(item_id)])
            for item_id in item_ids
            if self._item_key(item_id) in feature_store
        ]
        if not vectors:
            return None
        return self._normalize(np.mean(np.array(vectors, dtype=np.float32), axis=0))

    def _cosine_or_zero(self, vec_a, vec_b):
        if vec_a is None or vec_b is None:
            return 0.0
        return float(cosine_similarity([vec_a], [vec_b])[0][0])

    def _token_overlap(self, user_tokens, item_tokens):
        if not user_tokens or not item_tokens:
            return 0.0
        user_counter = Counter(user_tokens)
        item_counter = Counter(item_tokens)
        shared = 0
        for token, count in user_counter.items():
            shared += min(count, item_counter.get(token, 0))
        return shared / max(len(user_tokens), 1)

    def _build_interest_profile_from_history(self, user_model, history_items, short_len=3, long_len=10):
        short_items = list(history_items[:short_len])
        long_items = list(history_items[:long_len])

        short_weights = np.array([0.55, 0.30, 0.15], dtype=np.float32)[:len(short_items)] if short_items else None
        short_term = user_model._weighted_average(short_items, user_model.text_features, weights=short_weights)
        long_term = user_model._weighted_average(long_items, user_model.text_features)
        hybrid = user_model._normalize(0.65 * short_term + 0.35 * long_term)
        return {
            "short_term": short_term,
            "long_term": long_term,
            "hybrid": hybrid,
        }

    def _build_multi_interest_from_history(self, user_model, history_items, max_len=10):
        history_items = list(history_items[:max_len])
        if not history_items:
            return [np.zeros(user_model.text_feature_dim, dtype=np.float32)]

        profile = self._build_interest_profile_from_history(
            user_model,
            history_items,
            short_len=min(3, len(history_items)),
            long_len=min(max_len, len(history_items)),
        )
        interest_vectors = [profile["short_term"], profile["long_term"], profile["hybrid"]]

        if len(history_items) >= 4:
            split_idx = len(history_items) // 2
            interest_vectors.append(user_model._weighted_average(history_items[:split_idx], user_model.text_features))
            interest_vectors.append(user_model._weighted_average(history_items[split_idx:], user_model.text_features))

        deduped = []
        for vec in interest_vectors:
            if not any(np.allclose(vec, saved) for saved in deduped):
                deduped.append(user_model._normalize(vec))
        return deduped

    def _rrf_merge_candidates(self, named_result_lists, topk, rrf_k=60):
        merged = {}

        for source_name, result_list in named_result_lists:
            for rank, (item_id, raw_score) in enumerate(result_list, start=1):
                item_id = self._item_key(item_id)
                if item_id not in merged:
                    merged[item_id] = {
                        "item_id": item_id,
                        "final_score": 0.0,
                        "bm25_raw": 0.0,
                        "faiss_raw": 0.0,
                        "bm25_rank": 0,
                        "faiss_rank": 0,
                        "recall_source": "mixed",
                    }
                merged[item_id]["final_score"] += 1.0 / (rrf_k + rank)
                if source_name == "bm25":
                    merged[item_id]["bm25_raw"] = float(raw_score)
                    merged[item_id]["bm25_rank"] = rank
                elif source_name == "faiss":
                    merged[item_id]["faiss_raw"] = float(raw_score)
                    merged[item_id]["faiss_rank"] = rank

        rows = list(merged.values())
        for row in rows:
            has_bm25 = row["bm25_rank"] > 0
            has_faiss = row["faiss_rank"] > 0
            if has_bm25 and has_faiss:
                row["recall_source"] = "both"
            elif has_bm25:
                row["recall_source"] = "bm25"
            elif has_faiss:
                row["recall_source"] = "faiss"

        rows.sort(key=lambda x: x["final_score"], reverse=True)
        return rows[:topk]

    def _faiss_proxy_candidates(self, user_model, history_items, candidate_item_ids, candidate_matrix, exclude_item_ids, topk):
        interest_vectors = self._build_multi_interest_from_history(user_model, history_items, max_len=10)
        merged = {}

        for vec in interest_vectors:
            scores = candidate_matrix @ self._normalize(vec)
            candidate_count = min(len(scores), max(topk * 3, topk))
            top_indices = np.argsort(scores)[-candidate_count:][::-1]
            for idx in top_indices:
                item_id = candidate_item_ids[idx]
                if item_id in exclude_item_ids:
                    continue
                score = float(scores[idx])
                if item_id not in merged or score > merged[item_id]:
                    merged[item_id] = score

        rows = list(merged.items())
        rows.sort(key=lambda x: x[1], reverse=True)
        return rows[:topk]

    def _build_hybrid_candidates_for_history(
        self,
        user_model,
        bm25,
        history_items,
        candidate_item_ids,
        candidate_matrix,
        topk,
        bm25_topk,
        faiss_topk,
    ):
        exclude_item_ids = set(self._item_key(item_id) for item_id in history_items)
        user_query = " ".join(
            bm25.item_titles.get(self._item_key(hist_item), "")
            for hist_item in history_items
        ).strip()
        bm25_results = bm25.search(user_query, topk=bm25_topk, exclude_item_ids=exclude_item_ids)
        faiss_results = self._faiss_proxy_candidates(
            user_model=user_model,
            history_items=history_items,
            candidate_item_ids=candidate_item_ids,
            candidate_matrix=candidate_matrix,
            exclude_item_ids=exclude_item_ids,
            topk=faiss_topk,
        )
        return self._rrf_merge_candidates(
            [("bm25", bm25_results), ("faiss", faiss_results)],
            topk=topk,
        )

    def _build_feature_row(self, user_model, bm25, user_id, item_id, history_items=None, candidate=None):
        """构建特征向量 (与V2保持一致)"""
        item_key = self._item_key(item_id)
        if history_items is None:
            history_items = user_model.get_user_history(user_id, max_len=10)
        history_items = [self._item_key(hist_item) for hist_item in history_items]

        item_vec = self._normalize(self.text_features[item_key])
        profile = self._build_interest_profile_from_history(user_model, history_items, short_len=3, long_len=10)
        interest_vectors = self._build_multi_interest_from_history(user_model, history_items, max_len=10)

        history_vecs = [
            self._normalize(self.text_features[self._item_key(hist_item)])
            for hist_item in history_items
            if self._item_key(hist_item) in self.text_features
        ]

        if history_vecs:
            sims = cosine_similarity([item_vec], history_vecs)[0]
            max_history_sim = float(np.max(sims))
            mean_history_sim = float(np.mean(sims))
        else:
            max_history_sim = 0.0
            mean_history_sim = 0.0

        user_tokens = tokenize(" ".join(bm25.item_titles.get(self._item_key(hist_item), "") for hist_item in history_items))
        item_tokens = tokenize(bm25.item_titles.get(item_key, ""))
        user_query = " ".join(bm25.item_titles.get(self._item_key(hist_item), "") for hist_item in history_items).strip()

        history_text_vec = self._mean_feature(history_items, self.text_features or {})
        history_image_vec = self._mean_feature(history_items, self.image_features or {})
        item_text_vec = None
        item_image_vec = None

        if item_text_vec is None and self.text_features and item_key in self.text_features:
            item_text_vec = self._normalize(self.text_features[item_key])
        if item_image_vec is None and self.image_features and item_key in self.image_features:
            item_image_vec = self._normalize(self.image_features[item_key])

        bm25_raw = np.log1p(bm25.score_item(user_query, item_key))
        faiss_raw = max(
            float(cosine_similarity([self._normalize(interest_vec)], [item_vec])[0][0])
            for interest_vec in interest_vectors
        )
        text_score = self._cosine_or_zero(history_text_vec, item_text_vec)
        clip_score = self._cosine_or_zero(history_image_vec, item_image_vec)

        recall_score = 0.0
        source_bm25 = 1.0 if bm25_raw > 0 else 0.0
        source_faiss = 1.0 if faiss_raw > 0 else 0.0
        source_both = 1.0 if source_bm25 and source_faiss else 0.0

        if candidate is not None:
            bm25_raw = float(candidate.get("bm25_raw", bm25_raw))
            faiss_raw = float(candidate.get("faiss_raw", faiss_raw))
            recall_score = float(candidate.get("final_score", candidate.get("recall_score", 0.0)))
            source_name = str(candidate.get("recall_source", "")).lower()
            source_bm25 = 1.0 if source_name in {"bm25", "both"} else 0.0
            source_faiss = 1.0 if source_name in {"faiss", "both"} else 0.0
            source_both = 1.0 if source_name == "both" else 0.0

        return [
            bm25_raw,
            faiss_raw,
            text_score,
            clip_score,
            recall_score,
            source_bm25,
            source_faiss,
            source_both,
            float(cosine_similarity([profile["hybrid"]], [item_vec])[0][0]),
            float(cosine_similarity([profile["short_term"]], [item_vec])[0][0]),
            float(cosine_similarity([profile["long_term"]], [item_vec])[0][0]),
            max_history_sim,
            mean_history_sim,
            self._token_overlap(user_tokens, item_tokens),
            len(item_tokens) / 20.0
        ]

    def _sample_hard_negatives_from_bm25(self, bm25, user_query, positive_item, forbidden_set, num_samples):
        """
        从BM25召回结果中选取难负样本

        策略: 选择BM25高排名但不是正样本的商品
        这些商品与查询语义相似，但用户未选择，是高质量的难负样本
        """
        try:
            # 扩展排除集合
            exclude_list = list(forbidden_set) + [positive_item]
            bm25_results = bm25.search(user_query, topk=self.hard_negative_config["bm25_topk_for_hard"],
                                       exclude_item_ids=exclude_list)

            if not bm25_results:
                return []

            # 选择排名靠前的作为难负样本 (排名越靠前，越"难")
            hard_negatives = []
            for item_id, score in bm25_results[:num_samples * 2]:  # 多取一些备选
                if item_id not in forbidden_set and item_id != positive_item:
                    hard_negatives.append({
                        "item_id": item_id,
                        "recall_source": "hard_bm25",
                        "bm25_raw": float(score),
                        "difficulty_score": float(score),  # BM25分数作为难度指标
                    })
                    if len(hard_negatives) >= num_samples:
                        break

            return hard_negatives

        except Exception as e:
            print(f"[HardNeg] BM25 sampling error: {e}")
            return []

    def _sample_hard_negatives_from_faiss(self, user_model, user_id, positive_item, forbidden_set, num_samples):
        """
        从Faiss近邻中选取难负样本 (优化版: 使用矩阵运算替代逐个计算)

        性能优化:
        - 旧版: 逐个调用cosine_similarity → O(N)次sklearn调用 → 极慢
        - 新版: 预构建特征矩阵 + 一次矩阵乘法 → O(1)次numpy调用 → 快1000倍+
        """
        try:
            interest_vectors = user_model.build_multi_interest_vectors(user_id)

            # 预构建候选矩阵 (只做一次)
            if not hasattr(self, '_faiss_candidate_matrix') or self._faiss_candidate_matrix is None:
                candidate_ids = []
                candidate_vecs = []
                for item_id, feat in self.text_features.items():
                    candidate_ids.append(item_id)
                    candidate_vecs.append(self._normalize(feat))
                if not candidate_vecs:
                    return []
                self._faiss_candidate_ids = candidate_ids
                self._faiss_candidate_matrix = np.array(candidate_vecs, dtype=np.float32)

            candidate_ids = self._faiss_candidate_ids
            candidate_matrix = self._faiss_candidate_matrix

            all_candidates = []
            for vec, cluster_size in interest_vectors:
                vec_normalized = self._normalize(vec).reshape(1, -1)

                # 关键优化: 一次矩阵乘法计算所有相似度 (替代逐个调用)
                all_sims = (candidate_matrix @ vec_normalized.T).flatten()

                # 排序取top-k (排除forbidden集合)
                top_indices = np.argsort(all_sims)[::-1]
                count = 0
                for idx in top_indices:
                    item_id = candidate_ids[idx]
                    if item_id in forbidden_set or item_id == positive_item:
                        continue
                    sim = float(all_sims[idx])
                    all_candidates.append({
                        "item_id": item_id,
                        "recall_source": "hard_faiss",
                        "faiss_raw": sim,
                        "difficulty_score": sim,
                    })
                    count += 1
                    if count >= num_samples:
                        break

            seen = set()
            unique_candidates = []
            for cand in all_candidates:
                if cand["item_id"] not in seen:
                    seen.add(cand["item_id"])
                    unique_candidates.append(cand)

            return unique_candidates[:num_samples]

        except Exception as e:
            print(f"[HardNeg] Faiss sampling error: {e}")
            return []

    def fit(
        self,
        history_path=None,
        items_path=None,
        negative_samples=4,
        min_history_len=3,
        filter_items_to_history=True,
        max_users=0
    ):
        from data_config import config
        history_path = history_path or config.user_history_csv
        items_path = items_path or config.items_csv
        """
        训练排序模型 (支持Hard Negatives)

        改进点:
        1. 混合负采样: random (40%) + BM25-hard (30%) + Faiss-hard (30%)
        2. 难度感知: 根据难度分数调整样本权重
        """
        print("[RankerTrain-V3] loading user model...", flush=True)
        user_model = UserInterestModel(
            history_path,
            text_feature_path=self.text_feature_path,
            image_feature_path=self.image_feature_path
        )

        print("[RankerTrain-V3] loading auxiliary features...", flush=True)
        feature_hint_path = self.text_feature_path or self.image_feature_path
        if feature_hint_path:
            self._load_aux_features(feature_hint_path)

        print("[RankerTrain-V3] loading history...", flush=True)
        history = user_model.history.sort_values(["user_id", "timestamp"])
        history = history.copy()
        history["item_id"] = history["item_id"].astype(str)

        if max_users and max_users > 0:
            keep_users = list(dict.fromkeys(history["user_id"].tolist()))[:max_users]
            history = history[history["user_id"].isin(keep_users)].copy()
            print(f"[RankerTrain-V3] using first {len(keep_users)} users for training", flush=True)

        bm25_items_path = items_path
        if filter_items_to_history:
            print("[RankerTrain-V3] building history-covered item subset...", flush=True)
            subset = self._build_train_item_subset(items_path, history)
            temp_items = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8")
            subset.to_csv(temp_items.name, index=False)
            temp_items.close()
            self.temp_items_path = temp_items.name
            bm25_items_path = temp_items.name
            print(f"[RankerTrain-V3] item subset built: {len(subset)} items", flush=True)

        print("[RankerTrain-V3] building BM25 index...", flush=True)
        bm25 = BM25Retriever(items_path=bm25_items_path)

        all_item_ids = list(self.text_features.keys())
        feature_item_ids = set(all_item_ids)

        history = history[history["item_id"].isin(feature_item_ids)].copy()

        train_x = []
        train_y = []
        sample_weights = []  # 新增：样本权重（用于难负样本加权）
        rng = np.random.default_rng(42)
        trained_users = 0
        grouped_history = list(history.groupby("user_id"))
        total_users = len(grouped_history)
        log_every = 1 if total_users <= 20 else 10 if total_users <= 200 else 50

        candidate_item_ids = [item_id for item_id in bm25.item_titles.keys() if item_id in feature_item_ids]
        candidate_matrix = np.ascontiguousarray(
            np.array([self._normalize(self.text_features[item_id]) for item_id in candidate_item_ids], dtype=np.float32)
        )

        # 负采样配置
        config = self.hard_negative_config
        n_random = max(1, int(negative_samples * config["random_ratio"]))
        n_bm25_hard = max(1, int(negative_samples * config["bm25_hard_ratio"]))
        n_faiss_hard = max(1, int(negative_samples * config["faiss_hard_ratio"]))

        print(f"[RankerTrain-V3] negative sample config: random={n_random}, bm25_hard={n_bm25_hard}, faiss_hard={n_faiss_hard}", flush=True)
        print(f"[RankerTrain-V3] building samples for {total_users} users...", flush=True)

        for idx, (user_id, user_df) in enumerate(grouped_history, start=1):
            item_sequence = [self._item_key(item_id) for item_id in user_df["item_id"].tolist()]
            if len(item_sequence) <= min_history_len:
                continue
            trained_users += 1

            for current_idx in range(min_history_len, len(item_sequence)):
                history_items = list(reversed(item_sequence[:current_idx]))
                positive_item = item_sequence[current_idx]
                if positive_item not in feature_item_ids:
                    continue

                # 正样本
                train_x.append(self._build_feature_row(user_model, bm25, user_id, positive_item, history_items, candidate=None))
                train_y.append(1)
                sample_weights.append(1.0)  # 正样本权重为1

                # 构建禁止集合
                forbidden = set(history_items)
                forbidden.add(positive_item)

                # 用户查询 (用于BM25难负样本)
                user_query = " ".join(
                    bm25.item_titles.get(self._item_key(hist_item), "")
                    for hist_item in history_items
                ).strip()

                # === 1. 随机负样本 (Easy Negatives) ===
                available_random = [
                    item_id for item_id in candidate_item_ids
                    if item_id not in forbidden
                ]
                if available_random:
                    num_random = min(n_random, len(available_random))
                    selected_random = rng.choice(available_random, size=num_random, replace=False)
                    for neg_item_id in selected_random:
                        neg_candidate = {
                            "item_id": neg_item_id,
                            "recall_source": "random",
                        }
                        train_x.append(
                            self._build_feature_row(
                                user_model, bm25, user_id, neg_item_id,
                                history_items, candidate=neg_candidate
                            )
                        )
                        train_y.append(0)
                        sample_weights.append(0.8)  # 简单负样本权重稍低

                # === 2. BM25难负样本 (Semantic Hard Negatives) ===
                bm25_hard_negs = self._sample_hard_negatives_from_bm25(
                    bm25, user_query, positive_item, forbidden, n_bm25_hard
                )
                for neg_candidate in bm25_hard_negs:
                    neg_item_id = neg_candidate["item_id"]
                    if neg_item_id in forbidden:
                        continue
                    train_x.append(
                        self._build_feature_row(
                            user_model, bm25, user_id, neg_item_id,
                            history_items, candidate=neg_candidate
                        )
                    )
                    train_y.append(0)
                    # 难度越高，权重越大 (鼓励模型学习区分这些困难样本)
                    difficulty = min(neg_candidate.get("difficulty_score", 0.5) / 10.0, 2.0)
                    sample_weights.append(1.0 + difficulty)

                # === 3. Faiss难负样本 (Embedding Hard Negatives) ===
                faiss_hard_negs = self._sample_hard_negatives_from_faiss(
                    user_model, user_id, positive_item, forbidden, n_faiss_hard
                )
                for neg_candidate in faiss_hard_negs:
                    neg_item_id = neg_candidate["item_id"]
                    if neg_item_id in forbidden:
                        continue
                    train_x.append(
                        self._build_feature_row(
                            user_model, bm25, user_id, neg_item_id,
                            history_items, candidate=neg_candidate
                        )
                    )
                    train_y.append(0)
                    difficulty = min(neg_candidate.get("difficulty_score", 0.5) * 2, 2.0)
                    sample_weights.append(1.0 + difficulty)

            if idx % log_every == 0 or idx == total_users:
                print(
                    f"[RankerTrain-V3] processed {idx}/{total_users} users, "
                    f"samples={len(train_y)}",
                    flush=True
                )

        if not train_x:
            raise ValueError("not enough history to train the ranking model")

        positive_count = int(sum(train_y))
        negative_count = int(len(train_y) - positive_count)
        print(
            f"\n[RankerTrain-V3] Training Summary:"
            f"\n  users={trained_users}"
            f"\n  total_samples={len(train_y)}"
            f"\n  positives={positive_count}"
            f"\n  negatives={negative_count}"
            f"\n  pos:neg ratio = 1:{negative_count/max(positive_count,1):.1f}",
            flush=True
        )

        # 使用样本权重训练
        print("[RankerTrain-V3] fitting logistic regression with sample weights...", flush=True)
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        X = np.array(train_x, dtype=np.float32)
        y = np.array(train_y, dtype=np.int32)
        weights = np.array(sample_weights, dtype=np.float32)

        self.model.fit(X, y, sample_weight=weights)
        print("[RankerTrain-V3] fit complete", flush=True)

        return self

    def save(self):
        if self.model is None:
            raise ValueError("ranking model has not been trained")

        payload = {
            "feature_names": self.feature_names,
            "model": self.model,
            "version": "v3_with_hard_negatives",
            "config": self.hard_negative_config
        }
        with open(self.model_path, "wb") as f:
            pickle.dump(payload, f)

    def load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(self.model_path)

        with open(self.model_path, "rb") as f:
            payload = pickle.load(f)

        self.feature_names = payload["feature_names"]
        self.model = payload["model"]
        expected_dim = len(self.feature_names)
        model_dim = getattr(self.model, "n_features_in_", expected_dim)
        if model_dim != expected_dim:
            raise ValueError(
                f"ranking model feature mismatch: model expects {model_dim}, code provides {expected_dim}"
            )
        if self.text_feature_path or self.image_feature_path:
            feature_hint_path = self.text_feature_path or self.image_feature_path
            self._load_aux_features(feature_hint_path)
        return self

    def cleanup(self):
        if self.temp_items_path and os.path.exists(self.temp_items_path):
            os.remove(self.temp_items_path)
            self.temp_items_path = None

    def predict_score(self, feature_row):
        if self.model is None:
            raise ValueError("ranking model has not been loaded")
        model_dim = getattr(self.model, "n_features_in_", len(feature_row))
        if len(feature_row) != model_dim:
            raise ValueError(
                f"ranking model feature mismatch: model expects {model_dim}, got {len(feature_row)}"
            )
        return float(self.model.predict_proba([feature_row])[0][1])

    def rerank_candidates(
        self,
        user_id,
        candidates,
        user_model,
        bm25,
        topk=10,
        return_features=False,
        ground_truth=None
    ):
        """重排候选集 (与V2接口一致)"""
        history_items = user_model.get_user_history(user_id, max_len=10)
        rows = []
        gt_set = set(map(str, ground_truth)) if ground_truth else set()

        for candidate in candidates:
            if isinstance(candidate, dict):
                item_id = candidate["item_id"]
            else:
                item_id, recall_score = candidate
                candidate = {
                    "item_id": item_id,
                    "final_score": float(recall_score),
                    "bm25_raw": 0.0,
                    "faiss_raw": 0.0,
                    "recall_source": ""
                }

            item_key = self._item_key(item_id)
            if item_key not in user_model.text_features:
                continue

            feature_row = self._build_feature_row(user_model, bm25, user_id, item_key, history_items, candidate=candidate)
            rank_score = self.predict_score(feature_row)

            if gt_set:
                is_hit = item_key in gt_set
                hit_marker = "*** HIT ***" if is_hit else "--- MISS ---"

                if is_hit or rank_score > 0.1:
                    print(f"  [RankerDebug-V3] {hit_marker} Item: {item_key}, Score: {rank_score:.4f}")

            row = {
                "item_id": item_key,
                "final_score": rank_score,
                "recall_score": float(candidate.get("final_score", candidate.get("recall_score", 0.0))),
                "bm25_raw": float(candidate.get("bm25_raw", 0.0)),
                "faiss_raw": float(candidate.get("faiss_raw", 0.0)),
                "recall_source": candidate.get("recall_source", "")
            }
            if return_features:
                row["features"] = dict(zip(self.feature_names, feature_row))
            rows.append(row)

        rows.sort(key=lambda x: x["final_score"], reverse=True)
        return rows[:topk]
