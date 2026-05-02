-- 用户个性化推荐系统数据库表结构
-- SQLite数据库: user_recommendation.db

-- ============================================
-- 1. 用户信息表
-- ============================================
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,              -- 用户唯一ID (UUID)
    nickname TEXT NOT NULL UNIQUE,          -- 英文昵称，唯一
    password_hash TEXT,                      -- 密码哈希 (可选)
    avatar_url TEXT,                        -- 头像URL
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- 注册时间
    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- 最后活跃时间
    status TEXT DEFAULT 'active'            -- 状态: active/inactive
);

-- ============================================
-- 2. 用户兴趣表
-- ============================================
CREATE TABLE IF NOT EXISTS user_interests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    interest_category TEXT NOT NULL,         -- 兴趣大类 (如: Fashion, Tech)
    interest_tag TEXT NOT NULL,            -- 具体标签 (如: Casual, Sports)
    weight FLOAT DEFAULT 0.5,              -- 兴趣权重 (0-1)
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    UNIQUE(user_id, interest_tag)           -- 同一用户不能有重复标签
);

-- ============================================
-- 3. 用户反馈表 (点赞/评论)
-- ============================================
CREATE TABLE IF NOT EXISTS user_feedback (
    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    liked BOOLEAN DEFAULT FALSE,            -- 是否点赞
    comment_text TEXT,                     -- 评论内容
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),  -- 评分1-5
    shown_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    interacted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- ============================================
-- 4. 商品展示记录表
-- ============================================
CREATE TABLE IF NOT EXISTS exposure_history (
    exposure_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    shown_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    interacted BOOLEAN DEFAULT FALSE,        -- 是否产生互动
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- ============================================
-- 5. 用户偏好向量表 (用于快速检索)
-- ============================================
CREATE TABLE IF NOT EXISTS user_preference_vectors (
    user_id TEXT PRIMARY KEY,
    preference_vector BLOB,                 -- 序列化的偏好向量
    vector_dim INTEGER,                    -- 向量维度
    short_term_vector BLOB,                -- 短期兴趣向量
    long_term_vector BLOB,                 -- 长期兴趣向量
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- ============================================
-- 6. 索引优化
-- ============================================
CREATE INDEX IF NOT EXISTS idx_user_feedback_user ON user_feedback(user_id);
CREATE INDEX IF NOT EXISTS idx_user_feedback_item ON user_feedback(item_id);
CREATE INDEX IF NOT EXISTS idx_exposure_user ON exposure_history(user_id);
CREATE INDEX IF NOT EXISTS idx_interests_user ON user_interests(user_id);
CREATE INDEX IF NOT EXISTS idx_interests_tag ON user_interests(interest_tag);

-- ============================================
-- 兴趣标签定义 (静态数据)
-- ============================================
CREATE TABLE IF NOT EXISTS interest_tags (
    category TEXT NOT NULL,
    tag TEXT NOT NULL,
    description TEXT,
    PRIMARY KEY (category, tag)
);

-- 插入预设兴趣标签
INSERT OR IGNORE INTO interest_tags (category, tag, description) VALUES
-- Fashion
('Fashion', 'Casual', '休闲风格'),
('Fashion', 'Formal', '正式场合'),
('Fashion', 'Sports', '运动风格'),
('Fashion', 'Streetwear', '街头风格'),
('Fashion', 'Vintage', '复古风格'),
('Fashion', 'Minimalist', '简约风格'),
('Fashion', 'Bohemian', '波西米亚'),
('Fashion', 'Preppy', '学院风格'),

-- Electronics
('Electronics', 'Gadgets', '数码配件'),
('Electronics', 'Audio', '音频设备'),
('Electronics', 'Photography', '摄影器材'),
('Electronics', 'Gaming', '游戏设备'),
('Electronics', 'Smart Home', '智能家居'),
('Electronics', 'Wearables', '可穿戴设备'),

-- Lifestyle
('Lifestyle', 'Home Decor', '家居装饰'),
('Lifestyle', 'Kitchen', '厨房用品'),
('Lifestyle', 'Books', '图书'),
('Lifestyle', 'Fitness', '健身器材'),
('Lifestyle', 'Beauty', '美妆护肤'),
('Lifestyle', 'Outdoor', '户外用品'),
('Lifestyle', 'Travel', '旅行用品');
