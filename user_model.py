import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


class UserInterestModel:

    def __init__(self, history_path, text_feature_path, image_feature_path):
        self.history = pd.read_csv(history_path)
        self.history["item_id"] = self.history["item_id"].astype(str)
        
        # 分别加载文本和图像特征
        text_raw_features = np.load(text_feature_path, allow_pickle=True).item()
        image_raw_features = np.load(image_feature_path, allow_pickle=True).item()
        
        self.text_features = {str(k): np.array(v, dtype=np.float32) for k, v in text_raw_features.items()}
        self.image_features = {str(k): np.array(v, dtype=np.float32) for k, v in image_raw_features.items()}

        # 获取维度信息
        self.text_feature_dim = next(iter(self.text_features.values())).shape[0]
        self.image_feature_dim = next(iter(self.image_features.values())).shape[0]

        # 注意: 不再预初始化KMeans对象（避免Windows+MKL内存泄漏）
        # 改为在build_multi_interest_vectors中按需创建

    def _normalize(self, vec):
        vec = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def _item_key(self, item_id):
        return str(item_id)

    def _weighted_average(self, item_ids, feature_dict, weights=None):
        vecs = []
        filtered_weights = []

        for index, item_id in enumerate(item_ids):
            item_id = self._item_key(item_id)
            if item_id not in feature_dict:
                continue

            vecs.append(feature_dict[item_id])
            if weights is not None:
                filtered_weights.append(float(weights[index]))

        if not vecs:
            # 根据提供的特征字典确定维度
            dim = next(iter(feature_dict.values())).shape[0]
            return np.zeros(dim, dtype=np.float32)

        vecs = np.array(vecs, dtype=np.float32)

        if weights is None:
            merged = np.mean(vecs, axis=0)
        else:
            weight_array = np.array(filtered_weights, dtype=np.float32)
            weight_sum = weight_array.sum()
            if weight_sum <= 0:
                weight_array = np.ones(len(vecs), dtype=np.float32) / len(vecs)
            else:
                weight_array = weight_array / weight_sum
            merged = np.sum(vecs * weight_array[:, None], axis=0)

        return self._normalize(merged)

    def get_user_history(self, user_id, max_len=5):
        user_df = self.history[self.history["user_id"] == user_id]
        user_df = user_df.sort_values("timestamp", ascending=False)
        return user_df["item_id"].tolist()[:max_len]

    def build_short_term_vector(self, user_id, max_len=3):
        history_items = self.get_user_history(user_id, max_len=max_len)
        if not history_items:
            raise ValueError(f"user {user_id} has no history")

        base_weights = np.array([0.55, 0.30, 0.15], dtype=np.float32)
        weights = base_weights[:len(history_items)]
        # 注意：这里仅为示例，实际应决定使用何种特征
        return self._weighted_average(history_items, self.text_features, weights=weights)

    def build_long_term_vector(self, user_id, max_len=10):
        history_items = self.get_user_history(user_id, max_len=max_len)
        if not history_items:
            raise ValueError(f"user {user_id} has no history")

        return self._weighted_average(history_items, self.text_features)

    def build_interest_profile(self, user_id, short_len=3, long_len=10):
        short_term = self.build_short_term_vector(user_id, max_len=short_len)
        long_term = self.build_long_term_vector(user_id, max_len=long_len)
        hybrid = self._normalize(0.65 * short_term + 0.35 * long_term)
        return {
            "short_term": short_term,
            "long_term": long_term,
            "hybrid": hybrid,
        }

    def build_user_vector(self, user_id, max_len=5):
        # 此函数现在主要用于文本向量，或需重新定义其用途
        profile = self.build_interest_profile(
            user_id,
            short_len=min(3, max_len),
            long_len=max_len,
        )
        return profile["hybrid"]

    def build_average_image_vector(self, user_id, max_len=20):
        """
        为用户的历史记录构建一个平均的图像向量。
        """
        history_items = self.get_user_history(user_id, max_len=max_len)
        if not history_items:
            return self._normalize(np.random.rand(self.image_feature_dim))
        
        return self._weighted_average(history_items, self.image_features)


    def build_multi_interest_vectors(self, user_id, max_history=50, max_clusters=5):
        """
        使用K-Means在文本特征上聚类，从用户历史中提取多兴趣文本向量。
        返回: 一个元组列表 [(cluster_center, cluster_size), ...]

        修复: 避免Windows+MKL内存泄漏 + 聚类数量不足问题
        """
        history_items = self.get_user_history(user_id, max_len=max_history)
        if not history_items:
            return [(self._normalize(np.random.rand(self.text_feature_dim)), 1)]

        history_vectors = [self.text_features[item] for item in history_items if item in self.text_features]
        if not history_vectors:
            return [(self._normalize(np.random.rand(self.text_feature_dim)), 1)]

        # 关键修复: 去重！避免相同向量导致KMeans收敛失败
        unique_vectors = []
        seen = set()
        for vec in history_vectors:
            vec_key = tuple(vec.flatten()) if hasattr(vec, 'flatten') else tuple(vec)
            if vec_key not in seen:
                seen.add(vec_key)
                unique_vectors.append(vec)

        n_samples = len(unique_vectors)

        # 如果去重后样本数太少，直接返回每个唯一向量作为独立兴趣
        if n_samples <= max_clusters:
            return [(self._normalize(vec), 1) for vec in unique_vectors]

        # 动态计算聚类数 (确保 n_clusters < 唯一样本数)
        n_clusters = min(max_clusters, int(np.ceil(np.sqrt(n_samples / 2.0))))
        n_clusters = max(2, min(n_clusters, n_samples - 1))

        try:
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            labels = kmeans.fit_predict(unique_vectors)

            results = []
            for i in range(n_clusters):
                cluster_size = np.sum(labels == i)
                if cluster_size > 0:
                    center = kmeans.cluster_centers_[i]
                    results.append((self._normalize(center), int(cluster_size)))

            if not results:
                return [(self._normalize(vec), 1) for vec in unique_vectors[:max_clusters]]

            return results

        except Exception as e:
            print(f"[WARN] KMeans failed for user {user_id}: {e}, falling back to average vector")
            mean_vec = np.mean(unique_vectors, axis=0)
            return [(self._normalize(mean_vec), n_samples)]