import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from fusion_utils import build_text_query_fusion, load_fusion_config
from user_model import UserInterestModel


class FaissRecallV2:
    def __init__(self, user_model, text_index_path, text_ids_path, image_index_path, image_ids_path, text_encoder=None):
        self.user_model = user_model
        self.text_encoder = text_encoder

        print("Loading Text FAISS index...")
        self.text_index = faiss.read_index(text_index_path)
        self.text_item_ids = np.load(text_ids_path, allow_pickle=True)

        print("Loading Image FAISS index...")
        self.image_index = faiss.read_index(image_index_path)
        self.image_item_ids = np.load(image_ids_path, allow_pickle=True)

    def _normalize(self, vec):
        vec = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            return vec / norm
        return vec

    def _search(self, index, item_ids, query_vec, topk):
        query_vec = self._normalize(query_vec).reshape(1, -1)
        scores, idx = index.search(query_vec, topk)
        
        results = []
        for i, s in zip(idx[0], scores[0]):
            if i >= 0:
                results.append((str(item_ids[i]), float(s)))
        return results

    def _rrf_merge(self, ranked_lists, rrf_k=60):
        merged_results = {}
        for _, results in ranked_lists:
            for rank, (item_id, score) in enumerate(results):
                rrf_score = 1.0 / (rrf_k + rank + 1)
                if item_id not in merged_results:
                    merged_results[item_id] = 0.0
                merged_results[item_id] += rrf_score
        
        return sorted(merged_results.items(), key=lambda x: x[1], reverse=True)

    def recall_by_user(self, user_id, topk=20, rrf_k=60):
        """
        双塔召回：文本塔进行多兴趣召回，图像塔进行单一风格召回，最后进行RRF融合。
        """
        # 1. 文本塔召回 (多兴趣)
        text_interest_vectors = self.user_model.build_multi_interest_vectors(user_id)
        text_recall_results = []
        for vec, _ in text_interest_vectors:
            text_recall_results.extend(self._search(self.text_index, self.text_item_ids, vec, topk))
        
        # 对文本塔内部结果进行RRF初步融合
        text_final_results = self._rrf_merge([("text", text_recall_results)], rrf_k)

        # 2. 图像塔召回 (单一平均风格)
        image_style_vector = self.user_model.build_average_image_vector(user_id)
        image_recall_results = self._search(self.image_index, self.image_item_ids, image_style_vector, topk)

        # 3. 最终融合：将文本塔和图像塔的结果进行RRF融合
        final_merged = self._rrf_merge([
            ("text_tower", text_final_results),
            ("image_tower", image_recall_results)
        ], rrf_k)

        return final_merged[:topk]

    def recall_by_text(self, query, topk=50):
        """
        文本查询召回：使用文本编码器将查询转换为向量，直接在FAISS文本索引中搜索。
        """
        if self.text_encoder is None:
            raise ValueError("text_encoder is required for recall_by_text. Please provide a SentenceTransformer model.")

        text_vec = self.text_encoder.encode(query, normalize_embeddings=True)
        text_vec = text_vec.reshape(1, -1).astype(np.float32)
        results = self._search(self.text_index, self.text_item_ids, text_vec, topk)
        return results