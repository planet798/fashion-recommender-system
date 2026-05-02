# -*- coding: utf-8 -*-
"""
LambdaMART特征工程模块
为Learning to Rank模型提取排序特征
"""

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
from sklearn.metrics.pairwise import cosine_similarity


class LambdaMARTFeatureExtractor:
    """
    LambdaMART排序特征提取器
    提取以下类别特征:
    1. 基础相似度特征 (与用户历史的多模态相似度)
    2. DIN注意力特征 (历史行为对候选商品的注意力权重)
    3. 统计特征 (历史长度、类目覆盖等)
    4. 召回得分特征 (FAISS召回阶段的得分)
    """

    def __init__(self, text_features_path, image_features_path, multimodal_features_path,
                 history_path, user_model=None):
        self.text_features = np.load(text_features_path, allow_pickle=True).item()
        self.image_features = np.load(image_features_path, allow_pickle=True).item()
        self.multimodal_features = np.load(multimodal_features_path, allow_pickle=True).item()
        self.history = pd.read_csv(history_path)
        self.history["item_id"] = self.history["item_id"].astype(str)

        self.text_dim = 384
        self.image_dim = 512
        self.fusion_dim = 896

        if user_model:
            self.user_model = user_model
        else:
            from user_model import UserInterestModel
            self.user_model = UserInterestModel(
                history_path=history_path,
                text_feature_path=text_features_path,
                image_feature_path=image_features_path
            )

    def _normalize(self, vec):
        vec = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def _get_user_history_items(self, user_id, max_len=50):
        user_df = self.history[self.history["user_id"] == user_id]
        user_df = user_df.sort_values("timestamp", ascending=False)
        return [str(iid) for iid in user_df["item_id"].tolist()[:max_len]]

    def _compute_din_attention(self, query_vec, history_vecs):
        """
        简化版DIN注意力计算
        返回: (注意力加权得分, 注意力权重列表)
        """
        if not history_vecs:
            return 0.0, []

        query_vec = self._normalize(query_vec)
        history_vecs = [self._normalize(v) for v in history_vecs]

        attention_scores = []
        for hist_vec in history_vecs:
            combined = np.concatenate([hist_vec, query_vec, hist_vec * query_vec, hist_vec - query_vec])
            score = np.dot(hist_vec, query_vec)
            attention_scores.append(max(0, score))

        if sum(attention_scores) == 0:
            weights = [1.0 / len(attention_scores)] * len(attention_scores)
        else:
            weights = [s / sum(attention_scores) for s in attention_scores]

        weighted_sum = sum(w * v for w, v in zip(weights, history_vecs))
        attention_score = float(np.dot(self._normalize(weighted_sum), query_vec))

        return attention_score, weights

    def extract_features_for_candidate(self, user_id, candidate_item_id, recall_score=0.0):
        """
        为单个(user, candidate)对提取LambdaMART特征

        Args:
            user_id: 用户ID
            candidate_item_id: 候选商品ID
            recall_score: FAISS召回阶段的得分

        Returns:
            dict: 特征字典
        """
        history_items = self._get_user_history_items(user_id)
        history_len = len(history_items)

        candidate_text = self.text_features.get(candidate_item_id)
        candidate_image = self.image_features.get(candidate_item_id)
        candidate_multi = self.multimodal_features.get(candidate_item_id)

        features = {
            "history_length": history_len,
            "candidate_has_text": 1.0 if candidate_text is not None else 0.0,
            "candidate_has_image": 1.0 if candidate_image is not None else 0.0,
            "candidate_has_multimodal": 1.0 if candidate_multi is not None else 0.0,
            "recall_score": recall_score,
        }

        if candidate_multi is None:
            candidate_multi = np.zeros(self.fusion_dim, dtype=np.float32)
        else:
            candidate_multi = self._normalize(candidate_multi)

        history_text_vecs = []
        history_image_vecs = []
        history_multi_vecs = []

        for item_id in history_items:
            if item_id in self.text_features:
                history_text_vecs.append(self._normalize(self.text_features[item_id]))
            if item_id in self.image_features:
                history_image_vecs.append(self._normalize(self.image_features[item_id]))
            if item_id in self.multimodal_features:
                history_multi_vecs.append(self._normalize(self.multimodal_features[item_id]))

        if history_text_vecs:
            text_sims = [float(np.dot(self._normalize(v), candidate_text)) for v in history_text_vecs if candidate_text is not None]
            features["text_sim_max"] = max(text_sims) if text_sims else 0.0
            features["text_sim_mean"] = np.mean(text_sims) if text_sims else 0.0
            features["text_sim_min"] = min(text_sims) if text_sims else 0.0

            din_score, _ = self._compute_din_attention(candidate_text, history_text_vecs)
            features["text_din_score"] = din_score
        else:
            features["text_sim_max"] = 0.0
            features["text_sim_mean"] = 0.0
            features["text_sim_min"] = 0.0
            features["text_din_score"] = 0.0

        if history_image_vecs:
            image_sims = [float(np.dot(self._normalize(v), candidate_image)) for v in history_image_vecs if candidate_image is not None]
            features["image_sim_max"] = max(image_sims) if image_sims else 0.0
            features["image_sim_mean"] = np.mean(image_sims) if image_sims else 0.0
            features["image_sim_min"] = min(image_sims) if image_sims else 0.0

            din_score, _ = self._compute_din_attention(candidate_image, history_image_vecs)
            features["image_din_score"] = din_score
        else:
            features["image_sim_max"] = 0.0
            features["image_sim_mean"] = 0.0
            features["image_sim_min"] = 0.0
            features["image_din_score"] = 0.0

        if history_multi_vecs:
            multi_sims = [float(np.dot(v, candidate_multi)) for v in history_multi_vecs]
            features["multi_sim_max"] = max(multi_sims)
            features["multi_sim_mean"] = np.mean(multi_sims)
            features["multi_sim_min"] = min(multi_sims)
            features["multi_sim_std"] = np.std(multi_sims)

            din_score, attn_weights = self._compute_din_attention(candidate_multi, history_multi_vecs)
            features["multi_din_score"] = din_score
            features["multi_attention_spread"] = np.std(attn_weights) if len(attn_weights) > 1 else 0.0
            features["multi_attention_max"] = max(attn_weights) if attn_weights else 0.0
        else:
            features["multi_sim_max"] = 0.0
            features["multi_sim_mean"] = 0.0
            features["multi_sim_min"] = 0.0
            features["multi_sim_std"] = 0.0
            features["multi_din_score"] = 0.0
            features["multi_attention_spread"] = 0.0
            features["multi_attention_max"] = 0.0

        if history_multi_vecs and candidate_multi is not None:
            try:
                user_multi_avg = self._normalize(np.mean(history_multi_vecs, axis=0))
                avg_sim = float(np.dot(user_multi_avg, candidate_multi))
                features["user_avg_sim"] = avg_sim
            except:
                features["user_avg_sim"] = 0.0
        else:
            features["user_avg_sim"] = 0.0

        features["text_image_sim_diff"] = features["text_sim_mean"] - features["image_sim_mean"]
        features["text_image_sim_ratio"] = features["text_sim_mean"] / (features["image_sim_mean"] + 1e-6)

        features["rank_score_combined"] = (
            0.3 * features["recall_score"] +
            0.4 * features["multi_din_score"] +
            0.3 * features["user_avg_sim"]
        )

        return features

    def extract_training_data(self, train_history, ground_truth, candidates_dict, topk=100):
        """
        从训练数据中提取LambdaMART训练样本

        Args:
            train_history: 训练历史DataFrame
            ground_truth: {user_id: [positive_item_ids]} 真实标签
            candidates_dict: {user_id: [(item_id, recall_score), ...]} 召回候选
            topk: 每个用户最多采样的候选数量

        Returns:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签向量 (n_samples,) - 1表示点击，0表示未点击
            groups: 每个用户的样本数 (用于LambdaMART的group参数)
            feature_names: 特征名称列表
        """
        X_list = []
        y_list = []
        groups = []

        sample_count = 0
        for user_id in ground_truth:
            if user_id not in candidates_dict:
                continue

            positive_items = set(ground_truth[user_id])
            candidates = candidates_dict[user_id][:topk]

            user_features = []
            user_labels = []

            for item_id, recall_score in candidates:
                label = 1.0 if item_id in positive_items else 0.0

                features = self.extract_features_for_candidate(user_id, item_id, recall_score)
                user_features.append(features)
                user_labels.append(label)

            if user_features:
                X_list.extend(user_features)
                y_list.extend(user_labels)
                groups.append(len(user_features))
                sample_count += len(user_features)

            if sample_count >= 70000:
                break

        feature_names = list(X_list[0].keys()) if X_list else []
        X_df = pd.DataFrame(X_list)
        X = X_df.values.astype(np.float32)
        y = np.array(y_list, dtype=np.float32)

        return X, y, groups, feature_names


if __name__ == "__main__":
    from data_config import config

    print("LambdaMART特征提取器测试")
    print("=" * 50)

    extractor = LambdaMARTFeatureExtractor(
        text_features_path=config.text_features,
        image_features_path=config.image_features,
        multimodal_features_path=config.multimodal_features,
        history_path=config.user_history_csv
    )

    sample_user = "AE22H7WQ4SONBVPSXEDPNPUFT73A"
    sample_item = "B01ICC378E"

    features = extractor.extract_features_for_candidate(sample_user, sample_item, recall_score=0.8)

    print(f"\n用户: {sample_user}")
    print(f"商品: {sample_item}")
    print(f"\n提取的特征数量: {len(features)}")
    print("\n特征列表:")
    for k, v in sorted(features.items()):
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
