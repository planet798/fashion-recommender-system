# -*- coding: utf-8 -*-
"""
数据路径配置模块
统一管理所有数据源路径，方便在 data/ 和 datasets/ 之间切换
"""

import os


class DataConfig:
    """数据路径配置类"""

    # ===========================================
    # 当前激活的数据源 (修改此处即可切换)
    # ===========================================
    ACTIVE_SOURCE = "amazon"  # 可选: "sample" (data/) 或 "amazon" (datasets/)

    # ===========================================
    # LLM 配置
    # ===========================================
    LLM_BACKEND = "zhipu"  # 可选: "zhipu" (智谱API) 或 "local_qwen" (本地Qwen)
    ZHIPU_API_KEY = "fc237c4f0d2b4d07b16b44e6512b7b02.KMAKdQ2uMfGEWFA4"  # 智谱API Key, 也可通过环境变量 ZHIPU_API_KEY 设置
    ZHIPU_MODEL = "glm-4-flash"  # 智谱模型名称
    LOCAL_LLM_MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct"  # 本地模型路径

    # ===========================================
    # BERT基线模型配置
    # ===========================================
    BERT_MODEL_PATH = "all-MiniLM-L6-v2"
    BERT_FINETUNED_MODEL_PATH = "models/all-MiniLM-L6-v2-finetuned"

    # ===========================================
    # 数据源定义
    # ===========================================
    SOURCES = {
        "sample": {
            "name": "小规模样本数据",
            "root": "data",
            "items_csv": "data/items.csv",
            "user_history_csv": "data/user_history.csv",
            "text_features": "data/text_features.npy",
            "image_features": "data/image_features.npy",
            "multimodal_features": "data/multimodal_features.npy",
            "text_index": "data/faiss_text.index",
            "text_ids": "data/faiss_text_ids.npy",
            "image_index": "data/faiss_image.index",
            "image_ids": "data/faiss_image_ids.npy",
            "ranking_model": "data/ranking_model.pkl",
            "ranking_model_v3": "data/ranking_model_v3.pkl",
            "images_dir": "data/images",
            "fusion_config": "data/fusion_config.json",
            "bert_features": "data/bert_features.npy",
            "bert_index": "data/bert_faiss.index",
            "bert_ids": "data/bert_faiss_ids.npy",
        },
        "amazon": {
            "name": "Amazon Fashion完整数据",
            "root": "datasets/amazon_reviews23/processed",
            "items_csv": "datasets/amazon_reviews23/processed/items.csv",
            "user_history_csv": "datasets/amazon_reviews23/processed/user_history.csv",
            "text_features": "datasets/amazon_reviews23/processed/text_features.npy",
            "image_features": "datasets/amazon_reviews23/processed/image_features.npy",
            "multimodal_features": "datasets/amazon_reviews23/processed/multimodal_features.npy",
            "text_index": "datasets/amazon_reviews23/processed/faiss_text.index",
            "text_ids": "datasets/amazon_reviews23/processed/faiss_text_ids.npy",
            "image_index": "datasets/amazon_reviews23/processed/faiss_image.index",
            "image_ids": "datasets/amazon_reviews23/processed/faiss_image_ids.npy",
            "ranking_model": "datasets/amazon_reviews23/processed/ranking_model.pkl",
            "ranking_model_v3": "datasets/amazon_reviews23/processed/ranking_model_v3.pkl",
            "images_dir": "datasets/amazon_reviews23/images",
            "fusion_config": "datasets/amazon_reviews23/processed/fusion_config.json",
            "bert_features": "datasets/amazon_reviews23/processed/bert_features.npy",
            "bert_index": "datasets/amazon_reviews23/processed/bert_faiss.index",
            "bert_ids": "datasets/amazon_reviews23/processed/bert_faiss_ids.npy",
        }
    }

    def __init__(self, source=None):
        """
        初始化配置

        Args:
            source: 数据源名称，默认使用ACTIVE_SOURCE
        """
        self.source = source or self.ACTIVE_SOURCE
        if self.source not in self.SOURCES:
            raise ValueError(f"未知的数据源: {self.source}, 可选: {list(self.SOURCES.keys())}")
        self._config = self.SOURCES[self.source]

    @property
    def name(self):
        return self._config["name"]

    @property
    def root(self):
        return self._config["root"]

    @property
    def items_csv(self):
        return self._config["items_csv"]

    @property
    def user_history_csv(self):
        return self._config["user_history_csv"]

    @property
    def text_features(self):
        return self._config["text_features"]

    @property
    def image_features(self):
        return self._config["image_features"]

    @property
    def multimodal_features(self):
        return self._config["multimodal_features"]

    @property
    def text_index(self):
        return self._config["text_index"]

    @property
    def text_ids(self):
        return self._config["text_ids"]

    @property
    def image_index(self):
        return self._config["image_index"]

    @property
    def image_ids(self):
        return self._config["image_ids"]

    @property
    def ranking_model(self):
        return self._config["ranking_model"]

    @property
    def ranking_model_v3(self):
        return self._config["ranking_model_v3"]

    @property
    def images_dir(self):
        return self._config["images_dir"]

    @property
    def fusion_config(self):
        path = self._config.get("fusion_config")
        return path if path and os.path.exists(path) else None

    @property
    def bert_model_path(self):
        return self.BERT_MODEL_PATH

    @property
    def bert_finetuned_model_path(self):
        return self.BERT_FINETUNED_MODEL_PATH

    @property
    def bert_features(self):
        return self._config.get("bert_features", "")

    @property
    def bert_index(self):
        return self._config.get("bert_index", "")

    @property
    def bert_ids(self):
        return self._config.get("bert_ids", "")

    def validate(self):
        """验证所有必需文件是否存在"""
        missing = []
        required = [
            self.items_csv,
            self.user_history_csv,
            self.text_features,
            self.image_features,
        ]

        for path in required:
            if not os.path.exists(path):
                missing.append(path)

        if missing:
            print(f"[WARN] 数据源 '{self.name}' 缺少以下文件:")
            for f in missing:
                print(f"  - {f}")
            return False

        print(f"[OK] 数据源 '{self.name}' 验证通过")
        return True

    def info(self):
        """打印当前配置信息"""
        print("=" * 60)
        print(f"[DataConfig] 当前数据源: {self.name} ({self.source})")
        print(f"  根目录: {self.root}")
        print(f"  商品数据: {self.items_csv}")
        print(f"  用户历史: {self.user_history_csv}")
        print(f"  图片目录: {self.images_dir}")
        print("=" * 60)


# 全局默认实例
config = DataConfig()


if __name__ == "__main__":
    config.info()
    config.validate()
