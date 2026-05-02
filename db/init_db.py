# -*- coding: utf-8 -*-
"""
数据库初始化脚本
创建SQLite数据库并初始化表结构
"""

import os
import sqlite3
import tempfile

DB_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DB_PATH = os.path.join(DB_DIR, "user_recommendation.db")
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")


def init_database(db_path=None):
    """初始化数据库"""
    if db_path is None:
        os.makedirs(DB_DIR, exist_ok=True)
        db_path = DB_PATH

    print(f"[InitDB] Initializing database at: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema_sql = f.read()

    cursor.executescript(schema_sql)
    conn.commit()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f"[InitDB] Created tables: {[t[0] for t in tables]}")

    cursor.execute("SELECT COUNT(*) FROM interest_tags")
    tag_count = cursor.fetchone()[0]
    print(f"[InitDB] Interest tags loaded: {tag_count}")

    conn.close()
    print(f"[InitDB] Database initialization complete!")

    return db_path


def get_connection(db_path=None):
    """获取数据库连接"""
    if db_path is None:
        db_path = DB_PATH
    return sqlite3.connect(db_path)


def reset_database():
    """重置数据库（删除并重新创建）"""
    global DB_PATH

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"[ResetDB] Removed old database: {DB_PATH}")

    return init_database()


if __name__ == "__main__":
    print("=" * 50)
    print("数据库初始化")
    print("=" * 50)

    db_path = init_database()
    print(f"\n数据库路径: {db_path}")

    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT DISTINCT category FROM interest_tags ORDER BY category")
    categories = cursor.fetchall()
    print("\n可用兴趣类别:")
    for cat, in categories:
        cursor.execute("SELECT tag FROM interest_tags WHERE category = ?", (cat,))
        tags = [t[0] for t in cursor.fetchall()]
        print(f"  [{cat}]: {', '.join(tags)}")

    conn.close()
    print("\n初始化完成!")
