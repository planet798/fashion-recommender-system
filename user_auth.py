# -*- coding: utf-8 -*-
"""
User Authentication Module
Handles user registration, login, and logout"""

import os
import sys
import uuid
import hashlib
from datetime import datetime
from typing import Optional, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(__file__))
from db.init_db import get_connection

DB_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DB_DIR, exist_ok=True)


def generate_user_id() -> str:
    """生成唯一用户ID"""
    return str(uuid.uuid4())


def hash_password(password: str) -> str:
    """哈希密码存储到数据库"""
    return hashlib.sha256(password.encode()).hexdigest()


class UserAuth:
    """用户认证模块"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(DB_DIR, "user_recommendation.db")
        self.db_path = db_path

    def _get_conn(self):
        return get_connection(self.db_path)

    def _get_interest_category(self, cursor, tag: str) -> str:
        """Return the configured category for a tag, or a fallback."""
        cursor.execute(
            "SELECT category FROM interest_tags WHERE tag = ? LIMIT 1",
            (tag,)
        )
        row = cursor.fetchone()
        if row and row[0]:
            return row[0]
        return "Custom"

    def _insert_user_interest(self, cursor, user_id: str, tag: str, weight: float = 0.5) -> None:
        """Persist a user interest even when the tag is not in interest_tags."""
        tag = (tag or "").strip()
        if not tag:
            return

        cursor.execute(
            """INSERT OR IGNORE INTO user_interests
               (user_id, interest_category, interest_tag, weight, updated_at)
               VALUES (?, ?, ?, ?, ?)""",
            (user_id, self._get_interest_category(cursor, tag), tag, weight, datetime.now())
        )

    def register(self, nickname: str, password: str = None, interests: List[str] = None) -> Tuple[bool, str, str]:
        """
        用户注册
        Args:
            nickname: 用户昵称
            password: 用户密码
            interests: 用户兴趣标签列表

        Returns:
            (success, message, user_id)
        """
        nickname = nickname.strip()

        if not nickname:
            return False, "Nickname cannot be empty", ""

        if len(nickname) < 3:
            return False, "Nickname must be at least 3 characters", ""

        if not nickname.replace("_", "").replace("-", "").isalnum():
            return False, "Nickname can only contain letters, numbers, _ and -", ""

        conn = self._get_conn()
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT user_id FROM users WHERE nickname = ?", (nickname,))
            if cursor.fetchone():
                return False, "Nickname already exists", ""

            user_id = generate_user_id()
            password_hash = hash_password(password) if password else None

            cursor.execute(
                """INSERT INTO users (user_id, nickname, password_hash, created_at, last_active)
                   VALUES (?, ?, ?, ?, ?)""",
                (user_id, nickname, password_hash, datetime.now(), datetime.now())
            )
            if interests:
                for tag in interests:
                    self._insert_user_interest(cursor, user_id, tag, weight=0.5)

            conn.commit()

            return True, "Registration successful", user_id

        except Exception as e:
            conn.rollback()
            return False, f"Registration failed: {str(e)}", ""
        finally:
            conn.close()

    def login(self, nickname: str, password: str = None) -> Tuple[bool, str, str]:
        """
        用户登录

        Args:
            nickname: 用户昵称
            password: 用户密码

        Returns:
            (success, message, user_id)
        """
        nickname = nickname.strip()

        conn = self._get_conn()
        cursor = conn.cursor()

        try:
            cursor.execute(
                "SELECT user_id, password_hash FROM users WHERE nickname = ? AND status = 'active'",
                (nickname,)
            )
            row = cursor.fetchone()

            if not row:
                return False, "User not found", ""

            user_id, stored_hash = row

            if password and stored_hash and stored_hash != hash_password(password):
                return False, "Incorrect password", ""

            cursor.execute(
                "UPDATE users SET last_active = ? WHERE user_id = ?",
                (datetime.now(), user_id)
            )
            conn.commit()

            return True, "Login successful", user_id

        except Exception as e:
            return False, f"Login failed: {str(e)}", ""
        finally:
            conn.close()

    def get_user_info(self, user_id: str) -> Optional[Dict]:
        """
        获取用户信息

        Returns:
            Dict或None
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        try:
            cursor.execute(
                """SELECT user_id, nickname, avatar_url, created_at, last_active, status
                   FROM users WHERE user_id = ?""",
                (user_id,)
            )
            row = cursor.fetchone()

            if not row:
                return None

            return {
                "user_id": row[0],
                "nickname": row[1],
                "avatar_url": row[2],
                "created_at": row[3],
                "last_active": row[4],
                "status": row[5]
            }
        finally:
            conn.close()

    def get_user_interests(self, user_id: str) -> List[Dict]:
        """获取用户兴趣标签列表"""
        conn = self._get_conn()
        cursor = conn.cursor()

        try:
            cursor.execute(
                """SELECT interest_category, interest_tag, weight, updated_at
                   FROM user_interests WHERE user_id = ? ORDER BY weight DESC""",
                (user_id,)
            )
            rows = cursor.fetchall()

            return [
                {
                    "category": r[0],
                    "tag": r[1],
                    "weight": r[2],
                    "updated_at": r[3]
                }
                for r in rows
            ]
        finally:
            conn.close()

    def update_interests(self, user_id: str, interests: List[str]) -> bool:
        """更新用户兴趣标签列表"""
        conn = self._get_conn()
        cursor = conn.cursor()

        try:
            cursor.execute("DELETE FROM user_interests WHERE user_id = ?", (user_id,))
            for tag in interests:
                self._insert_user_interest(cursor, user_id, tag, weight=0.5)

            conn.commit()
            return True
        except Exception as e:
            conn.rollback()
            print(f"Error updating interests: {e}")
            return False
        finally:
            conn.close()

    def delete_user(self, user_id: str) -> bool:
        """删除用户"""
        conn = self._get_conn()
        cursor = conn.cursor()

        try:
            cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            conn.rollback()
            return False
        finally:
            conn.close()


def get_all_interests() -> Dict[str, List[str]]:
    """获取所有兴趣标签"""
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT category, tag FROM interest_tags ORDER BY category, tag")
        rows = cursor.fetchall()

        interests = {}
        for cat, tag in rows:
            if cat not in interests:
                interests[cat] = []
            interests[cat].append(tag)

        conn.close()

        if not interests:
            interests = {
                "时尚": ["休闲风格", "正式风格", "运动风格", "街头风格", "复古风格", "简约风格", "波西米亚风格", "学院风格"],
                "数码电子": ["数码产品", "音频设备", "摄影器材", "游戏设备", "智能家居", "可穿戴设备"],
                "生活方式": ["家居装饰", "厨房用品", "图书阅读", "健身运动", "美妆护肤", "户外用品", "旅行用品"]
            }

        return interests
    except Exception as e:
        print(f"Error getting interests: {str(e)}")
        return {
            "时尚": ["休闲风格", "正式风格", "运动风格", "街头风格", "复古风格", "简约风格", "波西米亚风格", "学院风格"],
            "数码电子": ["数码产品", "音频设备", "摄影器材", "游戏设备", "智能家居", "可穿戴设备"],
            "生活方式": ["家居装饰", "厨房用品", "图书阅读", "健身运动", "美妆护肤", "户外用品", "旅行用品"]
        }


if __name__ == "__main__":
    print("=" * 50)
    print("用户认证系统")
    print("=" * 50)

    auth = UserAuth()

    print("\n1. 用户注册...")
    success, msg, user_id = auth.register("testuser", password="123456", interests=["Casual", "Sports"])
    print(f"注册: {success}, {msg}, user_id: {user_id[:8]}...")

    if success:
        print("\n2. 用户登录...")
        success, msg, uid = auth.login("testuser", "123456")
        print(f"登录: {success}, {msg}, user_id: {uid[:8] if uid else None}...")

        print("\n3. 用户信息查询...")
        info = auth.get_user_info(user_id)
        print(f"用户信息: {info}")

        print("\n4. 用户兴趣标签查询...")
        interests = auth.get_user_interests(user_id)
        print(f"用户兴趣标签: {interests}")

        print("\n5. 所有兴趣标签...")
        all_interests = get_all_interests()
        for cat, tags in all_interests.items():
            print(f"  [{cat}]: {', '.join(tags)}")

    print("\n用户认证系统测试完成!")

