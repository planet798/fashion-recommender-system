import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
import os
import re
import sys


TAG_KEYWORDS = {
    "T恤": ["t-shirt", "tshirt", "tee", "t shirt"],
    "衬衫": ["shirt", "blouse", "button"],
    "卫衣": ["hoodie", "sweatshirt", "crewneck"],
    "毛衣": ["sweater", "knit", "cardigan"],
    "针织衫": ["cardigan", "knit", "pullover"],
    "POLO衫": ["polo"],
    "背心": ["tank", "camisole", "undershirt"],
    "吊带": ["tank", "cami", "strap"],
    "牛仔裤": ["jeans", "denim"],
    "休闲裤": ["casual pants", "slacks"],
    "运动裤": ["sports pants", "athletic pants", "workout pants", "leggings"],
    "短裤": ["shorts"],
    "裙子": ["skirt", "dress"],
    "长裤": ["pants", "trousers"],
    "夹克": ["jacket"],
    "外套": ["outerwear", "coat"],
    "大衣": ["coat", "overcoat"],
    "风衣": ["windbreaker", "wind coat"],
    "羽绒服": ["down jacket", "puffer", "parka"],
    "棉服": ["quilted jacket", "padding jacket"],
    "西装": ["suit", "blazer"],
    "运动鞋": ["sneakers", "sports shoes", "athletic shoes", "running shoes"],
    "帆布鞋": ["canvas shoes", "plimsolls"],
    "靴子": ["boots"],
    "凉鞋": ["sandals"],
    "拖鞋": ["slippers"],
    "皮鞋": ["leather shoes", "dress shoes"],
    "高跟鞋": ["heels", "high heels"],
    "平底鞋": ["flats", "ballet flats"],
    "帽子": ["hat", "cap"],
    "围巾": ["scarf", "shawl"],
    "手套": ["gloves"],
    "皮带": ["belt"],
    "领带": ["tie"],
    "袜子": ["socks", "stockings"],
    "背包": ["backpack", "bag"],
    "手提包": ["handbag", "tote bag"],
    "钱包": ["wallet"],
    "单肩包": ["shoulder bag", "messenger bag"],
    "旅行包": ["travel bag", "luggage"],
    "手表": ["watch"],
    "项链": ["necklace"],
    "耳环": ["earrings"],
    "手链": ["bracelet"],
    "戒指": ["ring"],
    "太阳镜": ["sunglasses", "eyewear"],
    "手机壳": ["phone case", "phone cover"],
    "充电器": ["charger"],
    "耳机": ["headphones", "earbuds", "earphones"],
    "音箱": ["speaker"],
    "键盘": ["keyboard"],
    "鼠标": ["mouse"],
    "相机": ["camera"],
    "化妆品": ["makeup", "cosmetics"],
    "护肤品": ["skincare", "skin care"],
    "香水": ["perfume", "fragrance"],
    "美容工具": ["beauty tools", "makeup tools"],
    "运动服": ["sportswear", "athletic wear"],
    "瑜伽服": ["yoga pants", "yoga wear"],
    "健身器材": ["fitness equipment", "gym equipment"],
    "户外装备": ["outdoor gear", "camping gear"],
    "旅行用品": ["travel accessories", "travel gear"],
    "厨房用品": ["kitchenware", "kitchen accessories"],
    "卧室用品": ["bedroom accessories"],
    "浴室用品": ["bathroom accessories"],
    "装饰品": ["decor", "decoration", "home decor"],
    "家具": ["furniture"],
    "Casual": ["casual", "relaxed", "everyday", "informal"],
    "Formal": ["formal", "dress", "elegant", "business"],
    "Sports": ["sports", "athletic", "workout", "fitness", "gym"],
    "Streetwear": ["streetwear", "urban", "skate", "hip hop", "casual"],
    "Vintage": ["vintage", "retro", "classic", "old school"],
    "Minimalist": ["minimalist", "simple", "clean", "minimal"],
    "Bohemian": ["bohemian", "boho", "flowy", "artistic"],
    "Preppy": ["preppy", "classic", "traditional", "ivy"],
    "Gadgets": ["gadget", "tech", "electronic", "device"],
    "Audio": ["audio", "sound", "music", "headphone", "speaker"],
    "Photography": ["photo", "camera", "photography", "lens"],
    "Gaming": ["gaming", "game", "video game", "console"],
    "Smart Home": ["smart home", "home automation", "iot", "connected"],
    "Wearables": ["wearable", "smartwatch", "fitness tracker", "watch"],
    "Home Decor": ["home decor", "decoration", "interior", "decorate"],
    "Kitchen": ["kitchen", "cooking", "cookware", "utensil"],
    "Books": ["book", "reading", "read", "literature"],
    "Fitness": ["fitness", "exercise", "workout", "gym", "health"],
    "Beauty": ["beauty", "makeup", "cosmetics", "skincare"],
    "Outdoor": ["outdoor", "outside", "camping", "hiking"],
    "Travel": ["travel", "trip", "vacation", "tourism", "luggage"],
    # Chinese interest tags (style/theme level) with bilingual keywords
    "休闲风格": ["casual", "relaxed", "everyday", "informal", "休闲", "日常", "宽松", "舒适"],
    "正式风格": ["formal", "dress", "elegant", "business", "正式", "优雅", "商务", "礼服"],
    "运动风格": ["sports", "athletic", "workout", "fitness", "运动", "健身", "跑步"],
    "街头风格": ["streetwear", "urban", "hip hop", "街头", "潮牌", "嘻哈"],
    "复古风格": ["vintage", "retro", "classic", "old school", "复古", "经典", "怀旧"],
    "简约风格": ["minimalist", "simple", "clean", "minimal", "简约", "极简", "素色", "基础款"],
    "波西米亚风格": ["bohemian", "boho", "flowy", "artistic", "波西米亚", "民族风", "流苏"],
    "学院风格": ["preppy", "classic", "traditional", "ivy", "学院", "英伦", "制服"],
    "数码产品": ["gadget", "tech", "electronic", "device", "数码", "电子产品", "科技"],
    "音频设备": ["audio", "sound", "music", "headphone", "speaker", "音频", "耳机", "音箱"],
    "摄影器材": ["photo", "camera", "photography", "lens", "摄影", "相机", "镜头"],
    "游戏设备": ["gaming", "game", "video game", "console", "游戏", "电竞", "主机"],
    "智能家居": ["smart home", "home automation", "iot", "智能家居", "智能", "自动化"],
    "可穿戴设备": ["wearable", "smartwatch", "fitness tracker", "可穿戴", "智能手表", "手环"],
    "家居装饰": ["home decor", "decoration", "interior", "decorate", "家居", "装饰", "摆件"],
    "厨房用品": ["kitchen", "cooking", "cookware", "utensil", "厨房", "烹饪", "锅具"],
    "图书阅读": ["book", "reading", "literature", "图书", "阅读", "书籍"],
    "健身运动": ["fitness", "exercise", "workout", "gym", "健身", "运动", "锻炼"],
    "美妆护肤": ["beauty", "makeup", "cosmetics", "skincare", "美妆", "护肤", "化妆", "香水"],
    "户外用品": ["outdoor", "camping", "hiking", "户外", "露营", "登山", "野营"],
    "旅行用品": ["travel", "trip", "vacation", "tourism", "行李", "旅行", "出游"],
}


class ColdStartRecommender:
    def __init__(
        self,
        items_df,
        text_features: Dict[str, np.ndarray],
        interest_tags: Dict[str, List[str]] = None,
        text_model_name: str = "models/paraphrase-MiniLM-L3-v2"
    ):
        self.items_df = items_df
        self.text_features = text_features
        self.interest_tags = interest_tags or {}
        self.text_model_name = text_model_name
        self.text_model = None
        self.vector_dim = 384

        self._item_ids = None
        self._normalized_vectors = None
        self._item_texts = None
        self._tag_score_matrix = None
        self._tag_keywords = None
        self._initialized = False

    def _lazy_init(self):
        if self._initialized:
            return

        print(f"[DEBUG _lazy_init] text_features count: {len(self.text_features)}")
        print(f"[DEBUG _lazy_init] items_df shape: {self.items_df.shape}")

        valid_item_ids = set(str(k) for k in self.text_features.keys())
        print(f"[DEBUG _lazy_init] valid_item_ids count: {len(valid_item_ids)}")

        item_id_str_set = set()
        item_id_to_text = {}

        if "description" in self.items_df.columns:
            df_subset = self.items_df[self.items_df["item_id"].astype(str).isin(valid_item_ids)]
            print(f"[DEBUG _lazy_init] df_subset shape (with desc): {df_subset.shape}")
            for _, row in df_subset.iterrows():
                item_id = str(row["item_id"])
                item_id_str_set.add(item_id)
                title = str(row.get("title", "")).lower()
                desc = str(row.get("description", "")).lower()
                item_id_to_text[item_id] = (title, desc)
        else:
            df_subset = self.items_df[self.items_df["item_id"].astype(str).isin(valid_item_ids)]
            print(f"[DEBUG _lazy_init] df_subset shape (no desc): {df_subset.shape}")
            for _, row in df_subset.iterrows():
                item_id = str(row["item_id"])
                item_id_str_set.add(item_id)
                title = str(row.get("title", "")).lower()
                item_id_to_text[item_id] = (title, "")

        print(f"[DEBUG _lazy_init] item_id_str_set count: {len(item_id_str_set)}")
        print(f"[DEBUG _lazy_init] item_id_to_text count: {len(item_id_to_text)}")

        self._item_ids = []
        self._normalized_vectors = []
        self._item_texts = []

        for item_id, vec in self.text_features.items():
            item_id_str = str(item_id)
            if item_id_str in item_id_str_set:
                self._item_ids.append(item_id_str)
                self._normalized_vectors.append(vec / (np.linalg.norm(vec) + 1e-8))
                self._item_texts.append(item_id_to_text.get(item_id_str, ("", "")))

        print(f"[DEBUG _lazy_init] _item_ids count: {len(self._item_ids)}")
        print(f"[DEBUG _lazy_init] _normalized_vectors count: {len(self._normalized_vectors)}")
        print(f"[DEBUG _lazy_init] _item_texts count: {len(self._item_texts)}")

        assert len(self._item_ids) == len(self._item_texts), f"Mismatch: {len(self._item_ids)} vs {len(self._item_texts)}"
        assert len(self._item_ids) == len(self._normalized_vectors), f"Mismatch: {len(self._item_ids)} vs {len(self._normalized_vectors)}"

        self._item_ids = np.array(self._item_ids)
        self._normalized_vectors = np.array(self._normalized_vectors, dtype=np.float32)

        self._tag_keywords = {}
        for interest in TAG_KEYWORDS.keys():
            keywords = TAG_KEYWORDS.get(interest, [interest.lower()])
            patterns = [kw.lower() for kw in keywords]
            self._tag_keywords[interest] = patterns

        self._precompute_tag_scores()
        print(f"[DEBUG _lazy_init] _precompute_tag_scores done, matrix shape: {self._tag_score_matrix.shape}")

        self._initialized = True
        print(f"[DEBUG _lazy_init] Initialization complete!")

    def _precompute_tag_scores(self):
        if self._item_ids is None or len(self._item_ids) == 0:
            return

        n_items = len(self._item_ids)
        interests = list(self._tag_keywords.keys())
        n_interests = len(interests)

        self._tag_score_matrix = np.zeros((n_items, n_interests), dtype=np.float32)

        for i, (title, desc) in enumerate(self._item_texts):
            for j, interest in enumerate(interests):
                score = 0.0
                for kw in self._tag_keywords[interest]:
                    if kw in title:
                        score += 2.0
                    if kw in desc:
                        score += 1.0
                self._tag_score_matrix[i, j] = score

        self._interest_to_idx = {interest: i for i, interest in enumerate(interests)}
        self._idx_to_interest = {i: interest for i, interest in enumerate(interests)}

    def _load_text_model(self):
        if self.text_model is None:
            self.text_model = SentenceTransformer(self.text_model_name)
        return self.text_model

    def _get_keywords_for_interest(self, interest: str) -> List[str]:
        return TAG_KEYWORDS.get(interest, [interest.lower()])

    def _calculate_item_tag_score(self, item_id: str, interest: str) -> float:
        if self._item_ids is None:
            self._lazy_init()

        try:
            idx = np.where(self._item_ids == item_id)[0][0]
        except IndexError:
            return 0.0

        interest_idx = self._interest_to_idx.get(interest)
        if interest_idx is None:
            return 0.0

        return float(self._tag_score_matrix[idx, interest_idx])

    def score_item_against_interests(
        self,
        item_id: str,
        interest_weights: Dict[str, float]
    ) -> float:
        if not interest_weights:
            return 0.0

        if self._item_ids is None:
            self._lazy_init()

        try:
            idx = np.where(self._item_ids == str(item_id))[0][0]
        except IndexError:
            return 0.0

        score = 0.0
        for interest, weight in interest_weights.items():
            interest_idx = self._interest_to_idx.get(interest)
            if interest_idx is None:
                continue
            score += float(self._tag_score_matrix[idx, interest_idx]) * float(weight)
        return score

    def get_item_interest_matches(
        self,
        item_id: str,
        interest_weights: Dict[str, float] = None,
        topn: int = 3,
        min_score: float = 0.05
    ) -> List[Tuple[str, float]]:
        if self._item_ids is None:
            self._lazy_init()

        try:
            idx = np.where(self._item_ids == str(item_id))[0][0]
        except IndexError:
            return []

        matches = []
        for interest, interest_idx in self._interest_to_idx.items():
            raw_score = float(self._tag_score_matrix[idx, interest_idx])
            if raw_score <= 0:
                continue

            weighted_score = raw_score
            if interest_weights:
                weighted_score *= float(interest_weights.get(interest, 0.0))

            if weighted_score >= min_score:
                matches.append((interest, weighted_score))

        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:topn]

    def generate_from_interest_weights(
        self,
        interest_weights: Dict[str, float],
        topk: int = 20
    ) -> List[Tuple[str, float]]:
        print(f"[DEBUG] =================== generate_from_interest_weights START ===================")
        print(f"[DEBUG] interest_weights: {interest_weights}, topk: {topk}")
        try:
            if not interest_weights:
                print("[DEBUG] interest_weights is empty, returning []")
                return []

            self._lazy_init()
            print(f"[DEBUG] After _lazy_init: _item_ids count = {len(self._item_ids)}")

            text_model = self._load_text_model()
            print(f"[DEBUG] text_model loaded")

            query_parts = []
            for interest, weight in interest_weights.items():
                keywords = self._get_keywords_for_interest(interest)
                for _ in range(int(weight * 3)):
                    query_parts.extend(keywords)

            if not query_parts:
                query_parts = list(interest_weights.keys())

            query_text = " ".join(query_parts[:50])
            print(f"[DEBUG] query_text: {query_text[:100]}")

            profile_vec = text_model.encode(query_text, normalize_embeddings=True)
            print(f"[DEBUG] profile_vec shape: {profile_vec.shape}")

            all_sims = np.dot(self._normalized_vectors, profile_vec).flatten()
            print(f"[DEBUG] all_sims stats: min={all_sims.min():.4f}, max={all_sims.max():.4f}, mean={all_sims.mean():.4f}")

            requested_interests = list(interest_weights.keys())
            valid_interests = [i for i in requested_interests if i in self._interest_to_idx]
            print(f"[DEBUG] requested_interests: {requested_interests}, valid_interests: {valid_interests}")

            tag_scores = np.zeros(len(self._item_ids), dtype=np.float32)
            for interest in valid_interests:
                interest_idx = self._interest_to_idx[interest]
                weight = interest_weights[interest]
                tag_scores += self._tag_score_matrix[:, interest_idx] * weight
            print(f"[DEBUG] tag_scores computed, max={tag_scores.max():.4f}")

            if len(valid_interests) > 0:
                max_tag = tag_scores.max()
                if max_tag > 0:
                    tag_scores_normalized = tag_scores / (len(valid_interests) * 6)
                else:
                    tag_scores_normalized = tag_scores
            else:
                tag_scores_normalized = tag_scores

            final_scores = 0.7 * all_sims + 0.3 * tag_scores_normalized
            print(f"[DEBUG] final_scores stats: min={final_scores.min():.4f}, max={final_scores.max():.4f}, mean={final_scores.mean():.4f}")

            top_indices = np.argsort(final_scores)[::-1][:topk]
            print(f"[DEBUG] top_indices count: {len(top_indices)}, first 5: {top_indices[:5]}")

            results = []
            for idx in top_indices:
                item_id = str(self._item_ids[idx])
                score = float(final_scores[idx])
                results.append((item_id, score))

            print(f"[DEBUG] Final return: results has {len(results)} items, first 3: {results[:3]}")
            sys.stdout.flush()

            print(f"[DEBUG] =================== generate_from_interest_weights END ===================")
            return results

        except Exception as e:
            print(f"[DEBUG] EXCEPTION in generate_from_interest_weights: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def generate_initial_recommendations(
        self,
        user_interests: List[str],
        topk: int = 20
    ) -> List[Tuple[str, float]]:
        interest_weights = {interest: 0.5 for interest in user_interests}
        return self.generate_from_interest_weights(interest_weights, topk)

    def get_diverse_recommendations(
        self,
        user_interests: List[str],
        topk: int = 20,
        diversity_weight: float = 0.3
    ) -> List[Dict]:
        try:
            recommendations = self.generate_initial_recommendations(
                user_interests,
                topk=topk * 2
            )

            if not recommendations:
                return []

            text_model = self._load_text_model()

            item_ids = [item_id for item_id, _ in recommendations]
            vecs = []
            valid_items = []

            for item_id in item_ids:
                vec = self.text_features.get(item_id)
                if vec is not None:
                    vecs.append(vec)
                    valid_items.append(item_id)

            if not vecs:
                return [{"item_id": i, "score": s} for i, s in recommendations[:topk]]

            vecs = np.array(vecs)

            scores = [score for _, score in recommendations]
            max_score = max(scores) if scores else 1
            norm_scores = [s / max_score for s in scores]

            diversity_scores = []
            for i, vec_i in enumerate(vecs):
                min_sim = 1.0
                for j, vec_j in enumerate(vecs):
                    if i != j:
                        sim = np.dot(vec_i, vec_j) / (np.linalg.norm(vec_i) * np.linalg.norm(vec_j) + 1e-8)
                        min_sim = min(min_sim, sim)
                diversity_scores.append(1 - min_sim)

            final_scores = []
            for i in range(len(valid_items)):
                combined = (1 - diversity_weight) * norm_scores[i] + diversity_weight * diversity_scores[i]
                final_scores.append((valid_items[i], combined, recommendations[i][1]))

            final_scores.sort(key=lambda x: x[1], reverse=True)

            results = []
            for item_id, combined_score, original_score in final_scores[:topk]:
                results.append({
                    "item_id": item_id,
                    "combined_score": float(combined_score),
                    "relevance_score": float(original_score)
                })

            return results

        except Exception as e:
            print(f"Error getting diverse recommendations: {str(e)}")
            return [{"item_id": i, "score": s} for i, s in recommendations[:topk]]


if __name__ == "__main__":
    import pandas as pd
    from data_config import config

    print("=== 测试冷启动推荐模块 ===")

    print("\n1. 加载数据...")
    items_df = pd.read_csv(config.items_csv)
    items_df["item_id"] = items_df["item_id"].astype(str)

    text_features = {
        str(item_id): np.array(vec, dtype=np.float32)
        for item_id, vec in np.load(config.text_features, allow_pickle=True).item().items()
    }

    print(f"   商品数量: {len(items_df)}")
    print(f"   特征数量: {len(text_features)}")

    print("\n2. 初始化冷启动推荐器...")
    recommender = ColdStartRecommender(
        items_df=items_df,
        text_features=text_features,
        interest_tags={}
    )

    print("\n3. 测试基于兴趣推荐...")
    interest_weights = {
        "T恤": 0.9,
        "运动鞋": 0.8,
        "运动裤": 0.7
    }
    recommendations = recommender.generate_from_interest_weights(
        interest_weights=interest_weights,
        topk=10
    )

    print(f"   兴趣权重: {interest_weights}")
    print(f"   推荐数量: {len(recommendations)}")
    for item_id, score in recommendations[:5]:
        title = items_df.loc[items_df["item_id"] == item_id, "title"].values[0] if len(items_df.loc[items_df["item_id"] == item_id]) > 0 else "Unknown"
        print(f"   - {item_id}: {str(title)[:50]}... (score: {score:.4f})")

    print("\n=== 测试完成 ===")
