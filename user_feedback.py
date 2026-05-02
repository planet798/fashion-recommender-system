import sqlite3
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import json


class UserFeedback:
    def __init__(self, db_path: str = "data/user_recommendation.db"):
        self.db_path = db_path

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def add_feedback(
        self,
        user_id: str,
        item_id: str,
        liked: bool = False,
        comment_text: str = None,
        rating: int = None
    ) -> Tuple[bool, str]:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT feedback_id FROM user_feedback
                WHERE user_id = ? AND item_id = ?
            """, (user_id, item_id))
            existing = cursor.fetchone()

            if existing:
                cursor.execute("""
                    UPDATE user_feedback
                    SET liked = COALESCE(?, liked),
                        comment_text = COALESCE(?, comment_text),
                        rating = COALESCE(?, rating),
                        interacted_at = CURRENT_TIMESTAMP
                    WHERE user_id = ? AND item_id = ?
                """, (
                    liked if liked is not None else None,
                    comment_text,
                    rating,
                    user_id,
                    item_id
                ))
                feedback_id = existing[0]
                action = "updated"
            else:
                cursor.execute("""
                    INSERT INTO user_feedback
                    (user_id, item_id, liked, comment_text, rating)
                    VALUES (?, ?, ?, ?, ?)
                """, (user_id, item_id, liked, comment_text, rating))
                feedback_id = cursor.lastrowid
                action = "added"

            conn.commit()
            conn.close()

            return True, f"Feedback {action} successfully (ID: {feedback_id})"

        except Exception as e:
            return False, f"Error adding feedback: {str(e)}"

    def get_user_feedback(self, user_id: str, limit: int = 100) -> List[Dict]:
        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
                SELECT feedback_id, item_id, liked, comment_text, rating,
                       shown_at, interacted_at
                FROM user_feedback
                WHERE user_id = ?
                ORDER BY interacted_at DESC
                LIMIT ?
            """, (user_id, limit))

            rows = cursor.fetchall()
            conn.close()

            return [dict(row) for row in rows]

        except Exception as e:
            print(f"Error getting user feedback: {str(e)}")
            return []

    def get_liked_items(self, user_id: str) -> List[str]:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT item_id FROM user_feedback
                WHERE user_id = ? AND liked = 1
                ORDER BY interacted_at DESC
            """, (user_id,))

            items = [row[0] for row in cursor.fetchall()]
            conn.close()

            return items

        except Exception as e:
            print(f"Error getting liked items: {str(e)}")
            return []

    def get_rated_items(
        self,
        user_id: str,
        min_rating: int = 1
    ) -> List[Tuple[str, int]]:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT item_id, rating FROM user_feedback
                WHERE user_id = ? AND rating >= ?
                ORDER BY interacted_at DESC
            """, (user_id, min_rating))

            items = [(row[0], row[1]) for row in cursor.fetchall()]
            conn.close()

            return items

        except Exception as e:
            print(f"Error getting rated items: {str(e)}")
            return []

    def record_exposure(
        self,
        user_id: str,
        item_ids: List[str]
    ) -> bool:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            for item_id in item_ids:
                cursor.execute("""
                    INSERT INTO exposure_history (user_id, item_id)
                    VALUES (?, ?)
                """, (user_id, item_id))

            conn.commit()
            conn.close()

            return True

        except Exception as e:
            print(f"Error recording exposure: {str(e)}")
            return False

    def get_exposure_history(
        self,
        user_id: str,
        limit: int = 100
    ) -> List[Dict]:
        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
                SELECT e.exposure_id, e.item_id, e.shown_at, e.interacted,
                       CASE WHEN f.feedback_id IS NOT NULL THEN 1 ELSE 0 END as has_interaction
                FROM exposure_history e
                LEFT JOIN user_feedback f ON e.user_id = f.user_id AND e.item_id = f.item_id
                WHERE e.user_id = ?
                ORDER BY e.shown_at DESC
                LIMIT ?
            """, (user_id, limit))

            rows = cursor.fetchall()
            conn.close()

            return [dict(row) for row in rows]

        except Exception as e:
            print(f"Error getting exposure history: {str(e)}")
            return []

    def get_interaction_stats(self, user_id: str) -> Dict:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    COUNT(*) as total_exposures,
                    SUM(CASE WHEN interacted = 1 THEN 1 ELSE 0 END) as interactions,
                    SUM(CASE WHEN liked = 1 THEN 1 ELSE 0 END) as likes,
                    AVG(CAST(rating AS FLOAT)) as avg_rating
                FROM exposure_history e
                LEFT JOIN user_feedback f ON e.user_id = f.user_id AND e.item_id = f.item_id
                WHERE e.user_id = ?
            """, (user_id,))

            row = cursor.fetchone()
            conn.close()

            return {
                "total_exposures": row[0] or 0,
                "interactions": row[1] or 0,
                "likes": row[2] or 0,
                "avg_rating": round(row[3], 2) if row[3] else 0
            }

        except Exception as e:
            print(f"Error getting interaction stats: {str(e)}")
            return {
                "total_exposures": 0,
                "interactions": 0,
                "likes": 0,
                "avg_rating": 0
            }

    def get_user_item_matrix(
        self,
        user_id: str,
        min_rating: int = 1
    ) -> List[Tuple[str, str, int]]:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT user_id, item_id, rating
                FROM user_feedback
                WHERE user_id = ? AND rating >= ?
            """, (user_id, min_rating))

            matrix = [(row[0], row[1], row[2]) for row in cursor.fetchall()]
            conn.close()

            return matrix

        except Exception as e:
            print(f"Error getting user item matrix: {str(e)}")
            return []

    def delete_feedback(self, feedback_id: int) -> Tuple[bool, str]:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("DELETE FROM user_feedback WHERE feedback_id = ?", (feedback_id,))
            deleted = cursor.rowcount > 0

            conn.commit()
            conn.close()

            if deleted:
                return True, "Feedback deleted successfully"
            else:
                return False, "Feedback not found"

        except Exception as e:
            return False, f"Error deleting feedback: {str(e)}"


REAL_INTEREST_TAGS = {
    "上衣": ["T恤", "衬衫", "卫衣", "毛衣", "针织衫", "POLO衫", "背心", "吊带"],
    "裤装": ["牛仔裤", "休闲裤", "运动裤", "短裤", "裙子", "长裤"],
    "外套": ["夹克", "外套", "大衣", "风衣", "羽绒服", "棉服", "西装"],
    "鞋履": ["运动鞋", "帆布鞋", "靴子", "凉鞋", "拖鞋", "皮鞋", "高跟鞋", "平底鞋"],
    "配饰": ["帽子", "围巾", "手套", "皮带", "领带", "袜子"],
    "包包": ["背包", "手提包", "钱包", "单肩包", "旅行包"],
    "腕表珠宝": ["手表", "项链", "耳环", "手链", "戒指", "太阳镜"],
    "数码电子": ["手机壳", "充电器", "耳机", "音箱", "键盘", "鼠标", "相机"],
    "美妆护肤": ["化妆品", "护肤品", "香水", "美容工具"],
    "运动户外": ["运动服", "瑜伽服", "健身器材", "户外装备", "旅行用品"],
    "家居生活": ["厨房用品", "卧室用品", "浴室用品", "装饰品", "家具"]
}

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
    "家具": ["furniture"]
}

ENGLISH_TAG_TO_CHINESE = {}
for chn_tag, keywords in TAG_KEYWORDS.items():
    for kw in keywords:
        ENGLISH_TAG_TO_CHINESE[kw] = chn_tag

# Comprehensive English tag → Chinese display name for the old interest_tags system
TAG_DISPLAY_MAP = {
    "Casual": "休闲风格",
    "Formal": "正式风格",
    "Sports": "运动风格",
    "Streetwear": "街头风格",
    "Vintage": "复古风格",
    "Minimalist": "简约风格",
    "Bohemian": "波西米亚风格",
    "Preppy": "学院风格",
    "Gadgets": "数码产品",
    "Audio": "音频设备",
    "Photography": "摄影器材",
    "Gaming": "游戏设备",
    "Smart Home": "智能家居",
    "Wearables": "可穿戴设备",
    "Home Decor": "家居装饰",
    "Kitchen": "厨房用品",
    "Books": "图书阅读",
    "Fitness": "健身运动",
    "Beauty": "美妆护肤",
    "Outdoor": "户外用品",
    "Travel": "旅行用品",
    "Fashion": "时尚",
    "Electronics": "数码电子",
    "Lifestyle": "生活方式",
}


def tag_to_display(tag: str) -> str:
    """Convert any tag (English or Chinese) to Chinese display name."""
    if tag in TAG_DISPLAY_MAP:
        return TAG_DISPLAY_MAP[tag]
    # Check if tag is a keyword mapped to a Chinese product tag
    if tag.lower() in ENGLISH_TAG_TO_CHINESE:
        return ENGLISH_TAG_TO_CHINESE[tag.lower()]
    return tag  # Already Chinese or unknown, return as-is


def get_all_interests(chinese: bool = True) -> Dict[str, List[str]]:
    """获取兴趣标签，默认为中文版本"""
    if chinese:
        return REAL_INTEREST_TAGS.copy()
    result = {}
    for cat, tags in REAL_INTEREST_TAGS.items():
        result[cat] = list(tags)
    return result


def chinese_to_english_tag(chinese_tag: str) -> str:
    """将中文标签转换为英文关键词"""
    return chinese_tag


if __name__ == "__main__":
    fb = UserFeedback()

    print("=== 测试反馈收集模块 ===")

    test_user = "test_user_001"
    test_items = ["item_001", "item_002", "item_003"]

    print("\n1. 记录商品展示...")
    fb.record_exposure(test_user, test_items)
    print(f"   已记录展示: {test_items}")

    print("\n2. 添加点赞反馈...")
    fb.add_feedback(test_user, "item_001", liked=True)

    print("\n3. 添加带评分的反馈...")
    fb.add_feedback(test_user, "item_002", liked=True, rating=5)

    print("\n4. 添加评论反馈...")
    fb.add_feedback(test_user, "item_003", liked=False, comment_text="这个商品很不错！", rating=4)

    print("\n5. 获取用户所有反馈...")
    feedback_list = fb.get_user_feedback(test_user)
    for fb_item in feedback_list:
        print(f"   商品: {fb_item['item_id']}, 点赞: {fb_item['liked']}, 评分: {fb_item['rating']}")

    print("\n6. 获取用户互动统计...")
    stats = fb.get_interaction_stats(test_user)
    print(f"   总展示: {stats['total_exposures']}, 互动: {stats['interactions']}, "
          f"点赞: {stats['likes']}, 平均分: {stats['avg_rating']}")

    print("\n7. 获取用户喜欢的商品...")
    liked_items = fb.get_liked_items(test_user)
    print(f"   喜欢的商品: {liked_items}")

    print("\n=== 测试完成 ===")
