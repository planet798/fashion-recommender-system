import sqlite3
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import pickle
import re

# Fallback keyword mapping for when categories table doesn't exist
# Maps item title/description keywords to interest tags
INTEREST_KEYWORD_MAP = {
    "休闲风格": ["casual", "relaxed", "everyday", "informal", "休闲", "日常"],
    "正式风格": ["formal", "dress", "elegant", "business", "正式", "商务", "礼服"],
    "运动风格": ["sports", "athletic", "workout", "fitness", "运动", "健身"],
    "街头风格": ["streetwear", "urban", "hip hop", "街头", "潮牌"],
    "复古风格": ["vintage", "retro", "classic", "复古", "怀旧"],
    "简约风格": ["minimalist", "simple", "clean", "minimal", "简约", "极简"],
    "波西米亚风格": ["bohemian", "boho", "flowy", "artistic", "波西米亚", "民族风"],
    "学院风格": ["preppy", "classic", "traditional", "ivy", "学院", "英伦"],
    "数码产品": ["gadget", "tech", "electronic", "device", "数码", "电子产品"],
    "音频设备": ["audio", "sound", "music", "headphone", "speaker", "耳机", "音箱"],
    "摄影器材": ["camera", "photography", "lens", "photo", "摄影", "相机"],
    "游戏设备": ["gaming", "game", "console", "video game", "游戏", "电竞"],
    "智能家居": ["smart home", "home automation", "iot", "智能家居", "自动化"],
    "可穿戴设备": ["wearable", "smartwatch", "fitness tracker", "可穿戴", "智能手表"],
    "家居装饰": ["home decor", "decoration", "interior", "家居", "装饰", "摆件"],
    "厨房用品": ["kitchen", "cooking", "cookware", "kitchenware", "厨房", "烹饪"],
    "图书阅读": ["book", "reading", "literature", "图书", "阅读", "书籍"],
    "健身运动": ["fitness", "exercise", "workout", "gym", "健身", "运动"],
    "美妆护肤": ["beauty", "makeup", "cosmetics", "skincare", "美妆", "护肤", "化妆"],
    "户外用品": ["outdoor", "camping", "hiking", "户外", "露营", "登山"],
    "旅行用品": ["travel", "trip", "vacation", "tourism", "旅行", "行李", "出游"],
    "Casual": ["casual", "relaxed", "everyday"],
    "Formal": ["formal", "dress", "elegant", "business"],
    "Sports": ["sports", "athletic", "workout"],
    "Streetwear": ["streetwear", "urban", "hip hop"],
    "Vintage": ["vintage", "retro", "classic"],
    "Minimalist": ["minimalist", "simple", "clean", "minimal"],
    "Bohemian": ["bohemian", "boho", "flowy", "artistic"],
    "Preppy": ["preppy", "classic", "traditional", "ivy"],
    "Gadgets": ["gadget", "tech", "electronic", "device"],
    "Audio": ["audio", "sound", "music", "headphone", "speaker"],
    "Photography": ["camera", "photography", "lens", "photo"],
    "Gaming": ["gaming", "game", "console"],
    "Smart Home": ["smart home", "home automation", "iot"],
    "Wearables": ["wearable", "smartwatch", "fitness tracker"],
    "Home Decor": ["home decor", "decoration", "interior"],
    "Kitchen": ["kitchen", "cooking", "cookware"],
    "Books": ["book", "reading", "literature"],
    "Fitness": ["fitness", "exercise", "workout"],
    "Beauty": ["beauty", "makeup", "cosmetics"],
    "Outdoor": ["outdoor", "camping", "hiking"],
    "Travel": ["travel", "trip", "vacation"],
}


class InterestUpdater:
    def __init__(self, db_path: str = "data/user_recommendation.db"):
        self.db_path = db_path
        self.vector_dim = 768

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def update_interest_weights(
        self,
        user_id: str,
        feedback_data: List[Dict]
    ) -> bool:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT interest_tag, weight
                FROM user_interests
                WHERE user_id = ?
            """, (user_id,))
            current_weights = {row[0]: row[1] for row in cursor.fetchall()}

            positive_scores = {}
            negative_scores = {}

            for feedback in feedback_data:
                item_id = feedback.get('item_id')
                liked = feedback.get('liked', False)
                rating = feedback.get('rating')
                comment = feedback.get('comment_text')
                item_tags = self._get_item_interest_tags(item_id)
                if not item_tags:
                    continue

                bucket = None
                score = 0.0
                if liked or (rating and rating >= 4):
                    bucket = positive_scores
                    score = 1.0
                    if rating:
                        score += (rating - 3) * 0.3
                    if comment:
                        score += 0.5
                elif liked is False or (rating and rating <= 2):
                    bucket = negative_scores
                    score = 0.9
                    if rating:
                        score += (3 - rating) * 0.25

                if bucket is None:
                    continue

                for tag in item_tags:
                    cursor.execute("""
                        SELECT interest_category FROM interest_tags
                        WHERE tag = ?
                    """, (tag,))
                    result = cursor.fetchone()
                    if not result:
                        continue

                    category = result[0]
                    if category not in bucket:
                        bucket[category] = {}
                    if tag not in bucket[category]:
                        bucket[category][tag] = 0.0
                    bucket[category][tag] += score

            touched_categories = set(positive_scores.keys()) | set(negative_scores.keys())
            for category in touched_categories:
                pos_tags = positive_scores.get(category, {})
                neg_tags = negative_scores.get(category, {})
                pos_max = max(pos_tags.values()) if pos_tags else 1.0
                neg_max = max(neg_tags.values()) if neg_tags else 1.0
                touched_tags = set(pos_tags.keys()) | set(neg_tags.keys())

                for tag in touched_tags:
                    positive_norm = min(pos_tags.get(tag, 0.0) / (pos_max * 2), 1.0) if pos_tags else 0.0
                    negative_norm = min(neg_tags.get(tag, 0.0) / (neg_max * 2), 1.0) if neg_tags else 0.0
                    current_weight = current_weights.get(tag, 0.35)

                    new_weight = current_weight * 0.72 + positive_norm * 0.38 - negative_norm * 0.52
                    new_weight = max(0.05, min(1.0, new_weight))

                    cursor.execute("""
                        INSERT OR REPLACE INTO user_interests
                        (user_id, interest_category, interest_tag, weight, updated_at)
                        VALUES (
                            ?,
                            ?,
                            ?,
                            ?,
                            CURRENT_TIMESTAMP
                        )
                    """, (user_id, category, tag, new_weight))

            conn.commit()
            conn.close()

            return True

        except Exception as e:
            print(f"Error updating interest weights: {str(e)}")
            return False

    def _get_item_interest_tags(self, item_id: str) -> List[str]:
        """Get interest tags for an item by keyword-matching its title."""
        try:
            # Lazy-load items CSV once (only id + title columns)
            if not hasattr(self, '_items_title_map'):
                self._items_title_map = {}
                import os as _os
                csv_path = _os.path.join(_os.path.dirname(__file__),
                    'datasets', 'amazon_reviews23', 'processed', 'items.csv')
                if _os.path.exists(csv_path):
                    import pandas as pd
                    df = pd.read_csv(csv_path, usecols=['item_id', 'title'])
                    df['item_id'] = df['item_id'].astype(str)
                    self._items_title_map = {
                        str(r['item_id']): str(r['title']).lower()
                        for r in df.to_dict('records')
                    }

            title = self._items_title_map.get(str(item_id), '')
            if not title:
                return []

            matched = []
            for tag, keywords in INTEREST_KEYWORD_MAP.items():
                for kw in keywords:
                    if kw.lower() in title:
                        matched.append(tag)
                        break
            return matched

        except Exception:
            return []

    def generate_preference_vector(
        self,
        user_id: str,
        interest_weights: Dict[str, float] = None
    ) -> np.ndarray:
        try:
            if interest_weights is None:
                conn = self._get_connection()
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT interest_tag, weight
                    FROM user_interests
                    WHERE user_id = ?
                """, (user_id,))

                interest_weights = {row[0]: row[1] for row in cursor.fetchall()}
                conn.close()

            if not interest_weights:
                return np.zeros(self.vector_dim)

            conn = self._get_connection()
            cursor = conn.cursor()

            placeholders = ','.join('?' * len(interest_weights))
            cursor.execute(f"""
                SELECT t.tag, t.embedding
                FROM tag_embeddings t
                WHERE t.tag IN ({placeholders})
            """, list(interest_weights.keys()))

            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return np.zeros(self.vector_dim)

            weighted_sum = np.zeros(self.vector_dim)
            total_weight = 0

            for tag, embedding_bytes in rows:
                embedding = pickle.loads(embedding_bytes)
                weight = interest_weights.get(tag, 0.5)
                weighted_sum += embedding * weight
                total_weight += weight

            if total_weight > 0:
                return weighted_sum / total_weight
            else:
                return np.zeros(self.vector_dim)

        except Exception as e:
            print(f"Error generating preference vector: {str(e)}")
            return np.zeros(self.vector_dim)

    def update_preference_vector(
        self,
        user_id: str,
        short_term: np.ndarray = None,
        long_term: np.ndarray = None
    ) -> bool:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT 1 FROM user_preference_vectors WHERE user_id = ?", (user_id,))
            exists = cursor.fetchone() is not None

            if short_term is not None:
                short_bytes = pickle.dumps(short_term) if isinstance(short_term, np.ndarray) else short_term
            else:
                short_bytes = None

            if long_term is not None:
                long_bytes = pickle.dumps(long_term) if isinstance(long_term, np.ndarray) else long_term
            else:
                long_bytes = None

            if exists:
                set_clauses = ["updated_at = CURRENT_TIMESTAMP"]
                params = []

                if short_bytes is not None:
                    set_clauses.append("short_term_vector = ?")
                    params.append(short_bytes)

                if long_bytes is not None:
                    set_clauses.append("long_term_vector = ?")
                    params.append(long_bytes)

                params.append(user_id)

                cursor.execute(f"""
                    UPDATE user_preference_vectors
                    SET {', '.join(set_clauses)}
                    WHERE user_id = ?
                """, params)
            else:
                cursor.execute("""
                    INSERT INTO user_preference_vectors
                    (user_id, preference_vector, vector_dim, updated_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """, (user_id, pickle.dumps(np.zeros(self.vector_dim)), self.vector_dim))

                if short_bytes is not None:
                    cursor.execute("""
                        UPDATE user_preference_vectors
                        SET short_term_vector = ?
                        WHERE user_id = ?
                    """, (short_bytes, user_id))

                if long_bytes is not None:
                    cursor.execute("""
                        UPDATE user_preference_vectors
                        SET long_term_vector = ?
                        WHERE user_id = ?
                    """, (long_bytes, user_id))

            conn.commit()
            conn.close()

            return True

        except Exception as e:
            print(f"Error updating preference vector: {str(e)}")
            return False

    def get_preference_vector(self, user_id: str) -> Tuple[np.ndarray, np.ndarray]:
        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
                SELECT short_term_vector, long_term_vector
                FROM user_preference_vectors
                WHERE user_id = ?
            """, (user_id,))

            row = cursor.fetchone()
            conn.close()

            if row:
                short_term = pickle.loads(row['short_term_vector']) if row['short_term_vector'] else None
                long_term = pickle.loads(row['long_term_vector']) if row['long_term_vector'] else None
                return short_term, long_term
            else:
                return None, None

        except Exception as e:
            print(f"Error getting preference vector: {str(e)}")
            return None, None

    def combine_short_long_term(
        self,
        short_term: np.ndarray,
        long_term: np.ndarray,
        short_weight: float = 0.6
    ) -> np.ndarray:
        if short_term is None and long_term is None:
            return np.zeros(self.vector_dim)

        if short_term is None:
            return long_term
        if long_term is None:
            return short_term

        return short_weight * short_term + (1 - short_weight) * long_term

    def decay_short_term_interest(
        self,
        user_id: str,
        decay_factor: float = 0.95,
        min_weight: float = 0.1
    ) -> bool:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE user_interests
                SET weight = MAX(?, weight * ?)
                WHERE user_id = ?
            """, (min_weight, decay_factor, user_id))

            conn.commit()
            conn.close()

            return True

        except Exception as e:
            print(f"Error decaying short term interest: {str(e)}")
            return False

    def get_user_interest_profile(self, user_id: str) -> Dict:
        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
                SELECT interest_category, interest_tag, weight, updated_at
                FROM user_interests
                WHERE user_id = ?
                ORDER BY weight DESC
            """, (user_id,))

            rows = cursor.fetchall()
            conn.close()

            profile = {
                "user_id": user_id,
                "interests": [],
                "top_categories": {},
                "last_updated": None
            }

            for row in rows:
                profile["interests"].append({
                    "category": row["interest_category"],
                    "tag": row["interest_tag"],
                    "weight": row["weight"],
                    "updated_at": row["updated_at"]
                })

                cat = row["interest_category"]
                if cat not in profile["top_categories"]:
                    profile["top_categories"][cat] = 0
                profile["top_categories"][cat] += row["weight"]

                if profile["last_updated"] is None or row["updated_at"] > profile["last_updated"]:
                    profile["last_updated"] = row["updated_at"]

            sorted_categories = sorted(
                profile["top_categories"].items(),
                key=lambda x: x[1],
                reverse=True
            )
            profile["top_categories"] = dict(sorted_categories[:5])

            return profile

        except Exception as e:
            print(f"Error getting interest profile: {str(e)}")
            return {"user_id": user_id, "interests": [], "top_categories": {}}

    def recalculate_from_feedback(self, user_id: str) -> bool:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT item_id, liked, rating, comment_text, interacted_at
                FROM user_feedback
                WHERE user_id = ?
                ORDER BY interacted_at DESC
                LIMIT 100
            """, (user_id,))

            feedback_rows = cursor.fetchall()
            conn.close()

            if not feedback_rows:
                return False

            feedback_data = []
            for row in feedback_rows:
                feedback_data.append({
                    'item_id': row[0],
                    'liked': bool(row[1]),
                    'rating': row[2],
                    'comment_text': row[3],
                    'interacted_at': row[4]
                })

            self.update_interest_weights(user_id, feedback_data)

            short_term = self.generate_preference_vector(user_id)
            self.update_preference_vector(user_id, short_term=short_term)

            return True

        except Exception as e:
            print(f"Error recalculating from feedback: {str(e)}")
            return False


if __name__ == "__main__":
    updater = InterestUpdater()

    print("=== 测试兴趣权重更新模块 ===")

    test_user = "test_user_001"

    print("\n1. 获取用户兴趣画像...")
    profile = updater.get_user_interest_profile(test_user)
    print(f"   用户: {profile['user_id']}")
    print(f"   兴趣数量: {len(profile['interests'])}")
    print(f"   热门类别: {profile['top_categories']}")

    print("\n2. 模拟更新兴趣权重...")
    mock_feedback = [
        {'item_id': 'item_001', 'liked': True, 'rating': 5, 'comment_text': '很喜欢'},
        {'item_id': 'item_002', 'liked': True, 'rating': 4, 'comment_text': None},
        {'item_id': 'item_003', 'liked': False, 'rating': 2, 'comment_text': '一般'},
    ]
    result = updater.update_interest_weights(test_user, mock_feedback)
    print(f"   更新结果: {'成功' if result else '失败'}")

    print("\n3. 生成偏好向量...")
    pref_vector = updater.generate_preference_vector(test_user)
    print(f"   向量维度: {pref_vector.shape}")
    print(f"   向量范数: {np.linalg.norm(pref_vector):.4f}")

    print("\n4. 更新偏好向量...")
    result = updater.update_preference_vector(test_user, short_term=pref_vector)
    print(f"   更新结果: {'成功' if result else '失败'}")

    print("\n5. 获取更新后的兴趣画像...")
    updated_profile = updater.get_user_interest_profile(test_user)
    print(f"   热门类别: {updated_profile['top_categories']}")

    print("\n=== 测试完成 ===")
