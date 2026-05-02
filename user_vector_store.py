# -*- coding: utf-8 -*-
"""
用户向量持久化存储模块
使用JSON文件存储用户向量和历史记录，避免冷启动
"""

import os
import json
import pickle
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple


class UserVectorStore:
    """
    用户向量存储器
    将用户向量和历史记录持久化到文件系统
    """

    def __init__(self, storage_path=None):
        if storage_path is None:
            storage_path = os.path.join("data", "user_vectors.json")

        self.storage_path = storage_path
        self.vectors_path = storage_path.replace(".json", "_vectors.pkl")
        self.metadata_path = storage_path.replace(".json", "_meta.json")

        os.makedirs(os.path.dirname(self.storage_path) if os.path.dirname(self.storage_path) else "data", exist_ok=True)

        self.vectors = {}
        self.metadata = {}
        self._load()

    def _load(self):
        """加载已有数据"""
        if os.path.exists(self.vectors_path):
            try:
                with open(self.vectors_path, "rb") as f:
                    self.vectors = pickle.load(f)
                print(f"[UserVectorStore] Loaded {len(self.vectors)} user vectors from {self.vectors_path}")
            except Exception as e:
                print(f"[UserVectorStore] Error loading vectors: {e}")
                self.vectors = {}

        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except Exception as e:
                print(f"[UserVectorStore] Error loading metadata: {e}")
                self.metadata = {}

    def _save(self):
        """保存数据"""
        try:
            with open(self.vectors_path, "wb") as f:
                pickle.dump(self.vectors, f)

            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)

            print(f"[UserVectorStore] Saved {len(self.vectors)} user vectors")
        except Exception as e:
            print(f"[UserVectorStore] Error saving: {e}")

    def save_user(self, user_id: str, vector: np.ndarray, history_items: List[str], short_term_vector=None, long_term_vector=None):
        """
        保存单个用户的向量和历史

        Args:
            user_id: 用户ID
            vector: 用户向量 (numpy array)
            history_items: 用户历史商品列表
            short_term_vector: 短期兴趣向量 (可选)
            long_term_vector: 长期兴趣向量 (可选)
        """
        user_id = str(user_id)

        self.vectors[user_id] = {
            "vector": vector.astype(np.float32).tobytes(),
            "vector_shape": vector.shape,
            "short_term": short_term_vector.astype(np.float32).tobytes() if short_term_vector is not None else None,
            "long_term": long_term_vector.astype(np.float32).tobytes() if long_term_vector is not None else None,
            "history": [str(item) for item in history_items],
            "updated_at": datetime.now().isoformat()
        }

        self.metadata[user_id] = {
            "vector_dim": vector.shape[-1] if hasattr(vector, "shape") else len(vector),
            "history_len": len(history_items),
            "updated_at": datetime.now().isoformat()
        }

    def get_user(self, user_id: str) -> Optional[Dict]:
        """
        获取用户向量和历史

        Returns:
            Dict包含: vector, short_term, long_term, history, updated_at
            如果用户不存在返回None
        """
        user_id = str(user_id)
        if user_id not in self.vectors:
            return None

        data = self.vectors[user_id]
        return {
            "vector": np.frombuffer(data["vector"], dtype=np.float32).reshape(data["vector_shape"]),
            "short_term": np.frombuffer(data["short_term"], dtype=np.float32) if data["short_term"] else None,
            "long_term": np.frombuffer(data["long_term"], dtype=np.float32) if data["long_term"] else None,
            "history": data["history"],
            "updated_at": data["updated_at"]
        }

    def has_user(self, user_id: str) -> bool:
        """检查用户是否存在"""
        return str(user_id) in self.vectors

    def get_all_user_ids(self) -> List[str]:
        """获取所有用户ID"""
        return list(self.vectors.keys())

    def delete_user(self, user_id: str):
        """删除用户"""
        user_id = str(user_id)
        if user_id in self.vectors:
            del self.vectors[user_id]
        if user_id in self.metadata:
            del self.metadata[user_id]

    def bulk_save(self, user_data: Dict[str, Dict]):
        """
        批量保存用户数据

        Args:
            user_data: {
                "user_id": {
                    "vector": np.ndarray,
                    "history": [...],
                    "short_term": np.ndarray (可选),
                    "long_term": np.ndarray (可选)
                }
            }
        """
        for user_id, data in user_data.items():
            self.save_user(
                user_id=str(user_id),
                vector=data["vector"],
                history_items=data.get("history", []),
                short_term_vector=data.get("short_term"),
                long_term_vector=data.get("long_term")
            )
        self._save()

    def get_stats(self) -> Dict:
        """获取存储统计信息"""
        return {
            "total_users": len(self.vectors),
            "storage_path": self.storage_path,
            "vectors_file_size_mb": os.path.getsize(self.vectors_path) / 1024 / 1024 if os.path.exists(self.vectors_path) else 0,
            "users": list(self.vectors.keys())[:5]
        }

    def close(self):
        """关闭存储（保存数据）"""
        self._save()


def init_user_vector_store(storage_path=None):
    """
    初始化用户向量存储的便捷函数
    """
    return UserVectorStore(storage_path=storage_path)


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = os.path.join(tmpdir, "test_users.json")

        print("=" * 50)
        print("用户向量存储测试")
        print("=" * 50)

        store = UserVectorStore(storage_path=store_path)

        test_user = "user_001"
        test_vector = np.random.rand(384).astype(np.float32)
        test_history = ["item_A", "item_B", "item_C"]

        store.save_user(test_user, test_vector, test_history)

        retrieved = store.get_user(test_user)
        print(f"\n用户: {test_user}")
        print(f"向量维度: {retrieved['vector'].shape}")
        print(f"历史记录: {retrieved['history']}")
        print(f"更新时间: {retrieved['updated_at']}")

        print(f"\n存储统计: {store.get_stats()}")

        store.close()

        store2 = UserVectorStore(storage_path=store_path)
        print(f"\n重新加载后用户数: {len(store2.get_all_user_ids())}")

        print("\n测试完成!")
