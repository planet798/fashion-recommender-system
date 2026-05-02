# -*- coding: utf-8 -*-
"""
LambdaMART排序推理模块
加载训练好的LambdaMART模型进行排序
"""

import os
import pickle
import tempfile

os.environ["TEMP"] = "D:\\TEMP"
os.environ["TMP"] = "D:\\TEMP"
os.environ["TMPDIR"] = "D:\\TEMP"
os.environ["HF_HOME"] = "D:\\HF_HOME"
os.environ["TRANSFORMERS_CACHE"] = "D:\\HF_HOME\\transformers"
tempfile.tempdir = "D:\\TEMP"

import numpy as np
import pandas as pd

from data_config import config


class LambdaMARTRanker:
    """
    LambdaMART排序器
    用于对召回候选进行重排序
    """

    def __init__(self, model_path=None, feature_extractor=None):
        if model_path is None:
            model_path = os.path.join("models", "lambdamart_ranker.pkl")

        if os.path.exists(model_path):
            print(f"[LambdaMARTRanker] Loading model from {model_path}")
            with open(model_path, "rb") as f:
                data = pickle.load(f)
                self.model = data["model"]
                self.feature_names = data["feature_names"]
                self.feature_extractor = data.get("extractor")
                print(f"[LambdaMARTRanker] Model loaded successfully!")
                print(f"[LambdaMARTRanker] Features: {len(self.feature_names)}")
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")

        if self.feature_extractor is None:
            from lambdamart_features import LambdaMARTFeatureExtractor
            self.feature_extractor = LambdaMARTFeatureExtractor(
                text_features_path=config.text_features,
                image_features_path=config.image_features,
                multimodal_features_path=config.multimodal_features,
                history_path=config.user_history_csv
            )

    def rerank_user_candidates(self, user_id, candidates, topk=10):
        """
        对用户的候选商品进行LambdaMART重排序

        Args:
            user_id: 用户ID
            candidates: 候选商品列表 [(item_id, recall_score), ...] 或 [item_id, ...]
            topk: 返回前topk个

        Returns:
            list: 重排序后的商品列表 [(item_id, final_score), ...]
        """
        if not candidates:
            return []

        features_list = []
        item_ids = []

        for candidate in candidates:
            if isinstance(candidate, dict):
                item_id = str(candidate["item_id"])
                recall_score = candidate.get("recall_score", 0.0)
            elif isinstance(candidate, (tuple, list)):
                item_id = str(candidate[0])
                recall_score = candidate[1] if len(candidate) > 1 else 0.0
            else:
                item_id = str(candidate)
                recall_score = 0.0

            item_ids.append(item_id)

            try:
                features = self.feature_extractor.extract_features_for_candidate(
                    user_id, item_id, recall_score
                )
                features_list.append(features)
            except Exception as e:
                print(f"[LambdaMARTRanker] Error extracting features for {item_id}: {e}")
                features_list.append({})

        if not features_list:
            return []

        feature_df = pd.DataFrame(features_list)

        for fname in self.feature_names:
            if fname not in feature_df.columns:
                feature_df[fname] = 0.0

        X = feature_df[self.feature_names].values.astype(np.float32)

        scores = self.model.predict(X)

        ranked_items = sorted(
            zip(item_ids, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked_items[:topk]

    def get_feature_importance(self):
        """
        获取特征重要性
        """
        if self.model is None:
            return []

        importance = self.model.feature_importance(importance_type="gain")
        feature_importance = sorted(
            zip(self.feature_names, importance),
            key=lambda x: x[1],
            reverse=True
        )
        return feature_importance


def load_lambdamart_ranker(model_path=None):
    """
    加载LambdaMART排序器的便捷函数
    """
    return LambdaMARTRanker(model_path=model_path)


if __name__ == "__main__":
    print("=" * 50)
    print("LambdaMART排序器测试")
    print("=" * 50)

    try:
        ranker = load_lambdamart_ranker()
        print(f"\n特征重要性:")
        importance = ranker.get_feature_importance()
        for name, imp in importance[:10]:
            print(f"  {name}: {imp:.2f}")

        sample_user = "AE22H7WQ4SONBVPSXEDPNPUFT73A"
        sample_candidates = [
            ("B01ICC378E", 0.85),
            ("B01NBJRRO0", 0.78),
            ("B071GZ6W9R", 0.72),
            ("B07MQ7YHCY", 0.68),
            ("B015WZO9VS", 0.55),
        ]

        print(f"\n测试用户: {sample_user}")
        print(f"原始候选: {sample_candidates}")

        reranked = ranker.rerank_user_candidates(sample_user, sample_candidates, topk=5)

        print(f"\nLambdaMART重排序结果:")
        for i, (item_id, score) in enumerate(reranked, 1):
            print(f"  {i}. {item_id}: {score:.4f}")

    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请先运行 train_lambdamart.py 训练模型")
