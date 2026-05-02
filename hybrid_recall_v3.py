# -*- coding: utf-8 -*-
"""
优化的混合召回模块 v3
解决hybrid_recall_v2.py中的问题:
1. RRF参数k=60过大，导致高质量信号被稀释
2. BM25与Faiss分数量纲差异巨大(2000倍)，RRF仅用排名丢失信息
3. 缺乏动态权重调整机制

优化策略:
- 自适应RRF参数 (根据两路质量动态调整)
- 分数归一化 + 加权混合评分
- 多种融合模式支持
"""
import numpy as np
from bm25_baseline import BM25Retriever
from recall_faiss_v2 import FaissRecallV2


class HybridRecallV3:
    """
    优化的混合召回器 - 解决Hybrid V2的核心问题

    核心改进:
    1. 自适应RRF: k值从固定60改为动态计算
    2. 分数归一化: 将BM25/Faiss分数归一化到[0,1]
    3. 加权融合: alpha*norm_score + beta*rrf_score + gamma*reciprocal_rank
    4. 多模式支持: 可切换不同融合策略
    """

    # 融合模式枚举
    MODE_ADAPTIVE_RRF = "adaptive_rrf"      # 自适应RRF (推荐)
    MODE_WEIGHTED_SCORE = "weighted_score"   # 加权分数融合
    MODE_HYBRID = "hybrid"                   # 混合模式 (归一化分数 + RRF)

    def __init__(self, faiss_recall, bm25=None, mode="faiss_only", text_encoder=None):
        """
        初始化混合召回器

        Args:
            faiss_recall: FaissRecallV2实例
            bm25: BM25Retriever实例 (可选，为None时使用纯FAISS模式)
            mode: 融合模式 ("adaptive_rrf", "weighted_score", "hybrid", "faiss_only")
            text_encoder: SentenceTransformer模型实例 (可选，用于文本编码)
        """
        self.faiss = faiss_recall
        self.bm25 = bm25
        self.mode = mode
        self.text_encoder = text_encoder or getattr(faiss_recall, 'text_encoder', None)

        if bm25 is None:
            self.mode = "faiss_only"

        # 超参数 (可通过调优进一步优化)
        self.params = {
            "adaptive_rrf": {
                "base_k": 30,              # 基础k值 (原60太大)
                "bm25_weight": 0.6,        # BM25权重 (BM25更强)
                "faiss_weight": 0.4,       # Faiss权重
            },
            "weighted_score": {
                "alpha": 0.5,              # 归一化分数权重
                "beta": 0.3,               # RRF分数权重
                "gamma": 0.2,              # 倒数排名权重
            },
            "hybrid": {
                "alpha": 0.4,              # BM25归一化分数
                "beta": 0.3,               # Faiss归一化分数
                "gamma": 0.3,              # RRF融合分数
            }
        }

    def _normalize_scores(self, scores_list):
        """
        Min-Max归一化到[0,1]

        Args:
            scores_list: [(item_id, raw_score), ...]
        Returns:
            [(item_id, normalized_score), ...]
        """
        if not scores_list:
            return []

        raw_scores = [score for _, score in scores_list]
        min_score = min(raw_scores)
        max_score = max(raw_scores)

        if max_score - min_score < 1e-9:
            # 所有分数相同，返回均匀分布
            n = len(scores_list)
            return [(item_id, 1.0 / n) for item_id, _ in scores_list]

        normalized = []
        for item_id, score in scores_list:
            norm_score = (score - min_score) / (max_score - min_score)
            normalized.append((item_id, norm_score))

        return normalized

    def _compute_adaptive_rrf_k(self, bm25_results, faiss_results):
        """
        自适应计算RRF参数k

        策略:
        - 如果某一路质量明显更好，降低其k值以增强其影响力
        - 基于分数方差评估质量 (方差越大，区分度越高，质量越好)

        Args:
            bm25_results: BM25召回结果
            faiss_results: Faiss召回结果
        Returns:
            (bm25_k, faiss_k): 各自的RRF k参数
        """
        base_k = self.params["adaptive_rrf"]["base_k"]

        # 计算各路的分数方差 (作为质量指标)
        def compute_variance(results):
            if len(results) < 2:
                return 0.0
            scores = [score for _, score in results[:10]]  # 取top10计算
            return np.var(scores)

        bm25_var = compute_variance(bm25_results)
        faiss_var = compute_variance(faiss_results)

        # 归一化方差
        total_var = bm25_var + faiss_var + 1e-9
        bm25_quality = bm25_var / total_var
        faiss_quality = faiss_var / total_var

        # 质量越高，k越小 (增强其排名影响力)
        # k范围: [15, 60]
        bm25_k = int(base_k * (1.5 - bm25_quality))
        faiss_k = int(base_k * (1.5 - faiss_quality))

        # 确保k在合理范围内
        bm25_k = max(15, min(60, bm25_k))
        faiss_k = max(15, min(60, faiss_k))

        return bm25_k, faiss_k

    def _rrf_merge_v1(self, named_result_lists, topk, rrf_k=60, return_details=False):
        """
        原始RRF融合 (保留用于对比)
        """
        merged = {}

        for source_name, result_list in named_result_lists:
            for rank, (item_id, raw_score) in enumerate(result_list, start=1):
                item_id = str(item_id)
                if item_id not in merged:
                    merged[item_id] = {
                        "item_id": item_id,
                        "final_score": 0.0,
                        "bm25_raw": 0.0,
                        "faiss_raw": 0.0,
                        "bm25_rank": 0,
                        "faiss_rank": 0,
                        "recall_source": "mixed"
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
        rows = rows[:topk]
        if return_details:
            return rows
        return [(row["item_id"], row["final_score"]) for row in rows]

    def _rrf_merge_adaptive(self, named_result_lists, topk, return_details=False):
        """
        自适应RRF融合 (核心优化1)

        改进点:
        1. 为BM25和Faiss使用不同的k值 (根据各自质量动态调整)
        2. 加入源权重 (BM25通常更可靠，给予更高权重)
        """
        params = self.params["adaptive_rrf"]
        bm25_weight = params["bm25_weight"]
        faiss_weight = params["faiss_weight"]

        # 提取各路结果
        bm25_results = None
        faiss_results = None
        for source_name, result_list in named_result_lists:
            if source_name == "bm25":
                bm25_results = result_list
            elif source_name == "faiss":
                faiss_results = result_list

        # 自适应计算k值
        if bm25_results and faiss_results:
            bm25_k, faiss_k = self._compute_adaptive_rrf_k(bm25_results, faiss_results)
        else:
            bm25_k = faiss_k = params["base_k"]

        # 执行加权RRF
        merged = {}
        for source_name, result_list in named_result_lists:
            # 选择该路的k值和权重
            if source_name == "bm25":
                current_k = bm25_k
                weight = bm25_weight
            else:  # faiss
                current_k = faiss_k
                weight = faiss_weight

            for rank, (item_id, raw_score) in enumerate(result_list, start=1):
                item_id = str(item_id)
                if item_id not in merged:
                    merged[item_id] = {
                        "item_id": item_id,
                        "final_score": 0.0,
                        "bm25_raw": 0.0,
                        "faiss_raw": 0.0,
                        "bm25_rank": 0,
                        "faiss_rank": 0,
                        "recall_source": "mixed"
                    }

                # 加权RRF分数
                rrf_score = weight * (1.0 / (current_k + rank))
                merged[item_id]["final_score"] += rrf_score

                if source_name == "bm25":
                    merged[item_id]["bm25_raw"] = float(raw_score)
                    merged[item_id]["bm25_rank"] = rank
                elif source_name == "faiss":
                    merged[item_id]["faiss_raw"] = float(raw_score)
                    merged[item_id]["faiss_rank"] = rank

        # 后处理
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
        rows = rows[:topk]
        if return_details:
            return rows
        return [(row["item_id"], row["final_score"]) for row in rows]

    def _rrf_merge_weighted_score(self, named_result_lists, topk, return_details=False):
        """
        加权分数融合 (核心优化2)

        改进点:
        1. 对BM25和Faiss分数分别进行Min-Max归一化
        2. 结合归一化分数、RRF分数、倒数排名进行加权融合
        公式: final = α*norm(score) + β*rrf_score + γ*(1/rank)
        """
        params = self.params["weighted_score"]
        alpha = params["alpha"]   # 归一化分数权重
        beta = params["beta"]     # RRF分数权重
        gamma = params["gamma"]   # 倒数排名权重

        # 分别归一化各路分数
        normalized_lists = {}
        for source_name, result_list in named_result_lists:
            normalized_lists[source_name] = self._normalize_scores(result_list)

        # 融合
        merged = {}
        for source_name, result_list in named_result_lists:
            norm_list = normalized_lists[source_name]

            for rank, ((item_id, raw_score), (_, norm_score)) in enumerate(
                zip(result_list, norm_list), start=1
            ):
                item_id = str(item_id)
                if item_id not in merged:
                    merged[item_id] = {
                        "item_id": item_id,
                        "final_score": 0.0,
                        "bm25_raw": 0.0,
                        "faiss_raw": 0.0,
                        "bm25_rank": 0,
                        "faiss_rank": 0,
                        "recall_source": "mixed"
                    }

                # 加权融合
                reciprocal_rank = 1.0 / rank
                weighted_score = (
                    alpha * norm_score +
                    beta * (1.0 / (60 + rank)) +  # RRF分量
                    gamma * reciprocal_rank         # 倒数排名分量
                )
                merged[item_id]["final_score"] += weighted_score

                if source_name == "bm25":
                    merged[item_id]["bm25_raw"] = float(raw_score)
                    merged[item_id]["bm25_rank"] = rank
                elif source_name == "faiss":
                    merged[item_id]["faiss_raw"] = float(raw_score)
                    merged[item_id]["faiss_rank"] = rank

        # 后处理
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
        rows = rows[:topk]
        if return_details:
            return rows
        return [(row["item_id"], row["final_score"]) for row in rows]

    def _rrf_merge_hybrid(self, named_result_lists, topk, return_details=False):
        """
        混合融合模式 (推荐用于生产环境)

        结合优势:
        - 使用归一化分数保留原始信息量
        - 使用RRF保证排名敏感性
        - 动态平衡两路贡献

        公式: final = α*norm(bm25) + β*norm(faiss) + γ*rrf_merged
        """
        params = self.params["hybrid"]
        alpha = params["alpha"]  # BM25归一化权重
        beta = params["beta"]   # Faiss归一化权重
        gamma = params["gamma"] # RRF权重

        # 归一化各路分数
        normalized_lists = {}
        for source_name, result_list in named_result_lists:
            normalized_lists[source_name] = self._normalize_scores(result_list)

        # 先计算传统RRF作为基准
        rrf_baseline = {}
        for source_name, result_list in named_result_lists:
            for rank, (item_id, _) in enumerate(result_list, start=1):
                item_id = str(item_id)
                if item_id not in rrf_baseline:
                    rrf_baseline[item_id] = 0.0
                rrf_baseline[item_id] += 1.0 / (60 + rank)

        # 归一化RRF分数到[0,1]
        if rrf_baseline:
            rrf_values = list(rrf_baseline.values())
            rrf_min, rrf_max = min(rrf_values), max(rrf_values)
            rrf_range = rrf_max - rrf_min
            if rrf_range > 1e-9:
                rrf_norm = {k: (v - rrf_min) / rrf_range for k, v in rrf_baseline.items()}
            else:
                n = len(rrf_norm) if rrf_norm else 1
                rrf_norm = {k: 1.0 / n for k in rrf_baseline}
        else:
            rrf_norm = {}

        # 最终融合
        merged = {}
        for source_name, result_list in named_result_lists:
            norm_list = normalized_lists[source_name]

            for rank, ((item_id, raw_score), (_, norm_score)) in enumerate(
                zip(result_list, norm_list), start=1
            ):
                item_id = str(item_id)
                if item_id not in merged:
                    merged[item_id] = {
                        "item_id": item_id,
                        "final_score": 0.0,
                        "bm25_raw": 0.0,
                        "faiss_raw": 0.0,
                        "bm25_rank": 0,
                        "faiss_rank": 0,
                        "recall_source": "mixed"
                    }

                # 混合评分
                rrf_component = rrf_norm.get(item_id, 0.0)

                if source_name == "bm25":
                    merged[item_id]["final_score"] += alpha * norm_score + gamma * rrf_component
                    merged[item_id]["bm25_raw"] = float(raw_score)
                    merged[item_id]["bm25_rank"] = rank
                elif source_name == "faiss":
                    merged[item_id]["final_score"] += beta * norm_score + gamma * rrf_component
                    merged[item_id]["faiss_raw"] = float(raw_score)
                    merged[item_id]["faiss_rank"] = rank

        # 后处理
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
        rows = rows[:topk]
        if return_details:
            return rows
        return [(row["item_id"], row["final_score"]) for row in rows]

    def _rrf_merge(self, named_result_lists, topk, rrf_k=60, return_details=False):
        """
        统一入口: 根据mode选择融合策略
        """
        if self.mode == self.MODE_ADAPTIVE_RRF:
            return self._rrf_merge_adaptive(named_result_lists, topk, return_details)
        elif self.mode == self.MODE_WEIGHTED_SCORE:
            return self._rrf_merge_weighted_score(named_result_lists, topk, return_details)
        elif self.mode == self.MODE_HYBRID:
            return self._rrf_merge_hybrid(named_result_lists, topk, return_details)
        else:
            # 默认使用原始RRF (向后兼容)
            return self._rrf_merge_v1(named_result_lists, topk, rrf_k, return_details)

    def recall_by_text(self, query, topk=50, bm25_topk=50, faiss_topk=50, return_details=False):
        """文本查询召回"""
        if self.mode == "faiss_only":
            if self.faiss.text_encoder is None and self.text_encoder is None:
                raise ValueError("text_encoder is required for recall_by_text. Please provide a SentenceTransformer model.")
            encoder = self.text_encoder or self.faiss.text_encoder
            text_vec = encoder.encode(query, normalize_embeddings=True)
            text_vec = text_vec.reshape(1, -1).astype(np.float32)
            faiss_results = self.faiss._search(self.faiss.text_index, self.faiss.text_item_ids, text_vec, topk)
            return faiss_results

        bm25_results = self.bm25.search(query, topk=bm25_topk)
        faiss_results = self.faiss.recall_by_text(query, topk=faiss_topk)
        return self._rrf_merge(
            [("bm25", bm25_results), ("faiss", faiss_results)],
            topk=topk,
            return_details=return_details
        )

    def recall_by_user(self, user_id, topk=50, bm25_topk=50, faiss_topk=50, max_history=5, return_details=False):
        """用户推荐召回"""
        if self.mode == "faiss_only":
            faiss_results = self.faiss.recall_by_user(
                user_id=user_id,
                topk=topk,
            )
            return faiss_results

        history_df = self.faiss.user_model.history
        _, history_items = self.bm25.build_user_query(history_df, user_id, max_len=max_history)

        bm25_results = self.bm25.recommend_for_user(
            history_df,
            user_id,
            topk=bm25_topk,
            max_history=max_history
        )
        faiss_results = self.faiss.recall_by_user(
            user_id=user_id,
            topk=faiss_topk,
        )

        return self._rrf_merge(
            [("bm25", bm25_results), ("faiss", faiss_results)],
            topk=topk,
            return_details=return_details
        )
