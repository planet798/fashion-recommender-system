# 多模态 AI 时尚推荐引擎

> 基于长链推理与多智能体协作的个性化电商商品发现系统

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)](https://streamlit.io)
[![FAISS](https://img.shields.io/badge/FAISS-HNSW-green)](https://github.com/facebookresearch/faiss)
[![CLIP](https://img.shields.io/badge/CLIP-ViT--B--32-orange)](https://openai.com/research/clip)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 目录

- [核心痛点](#核心痛点)
- [系统架构](#系统架构)
- [核心逻辑流：长链推理 + 多 Agent 协作](#核心逻辑流长链推理--多-agent-协作)
- [技术栈](#技术栈)
- [效果指标](#效果指标)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [关键模块说明](#关键模块说明)

---

## 核心痛点

| 痛点 | 问题描述 | 本方案解决方式 |
|------|---------|--------------|
| **语义鸿沟巨大** | 用户输入"约会穿的碎花长裙"，传统系统只做关键词匹配，无法理解场合×风格×品类的多维语义 | LLM 四步推理链，将自然语言解析为结构化查询 |
| **冷启动困难** | 新用户无行为历史，协同过滤完全失效 | 兴趣标签语义匹配 + CLIP 多模态特征，零历史也能个性化 |
| **多模态信息割裂** | 商品同时有文本和图片，单模态检索丢失大量信息 | 文本(384d) + 图像(512d) + 融合(896d) 三路联合检索 |
| **兴趣漂移无法追踪** | 用户兴趣随时间变化，模型无法实时响应反馈 | 实时反馈闭环，兴趣权重增量更新，下次查询即刻生效 |
| **候选集质量粗糙** | 单一召回策略精度低，大量无关商品进入排序阶段 | 多路召回 + 动态注意力融合 + 深度排序，逐层提纯 |

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Streamlit 前端交互层                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐    │
│  │ Smart Chat│  │ 个性化推荐│  │ 探索新品 │  │ 我的反馈/历史    │    │
│  │ LLM对话搜索│  │ Octopus  │  │ 冷启动   │  │ 反馈记录与统计   │    │
│  └─────┬────┘  └─────┬────┘  └────┬─────┘  └────────┬─────────┘    │
└────────┼──────────────┼────────────┼──────────────────┼─────────────┘
         │              │            │                  │
┌────────▼──────────────▼────────────▼──────────────────▼─────────────┐
│                         Agent 协作编排层                               │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Agent 1: 语义理解 → Agent 2: 多模态检索 → Agent 3: 排序     │   │
│  │  → Agent 4: 个性化注入 → Agent 5: 反馈学习                   │   │
│  │  结构化数据接口串联，每层独立可降级                             │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 核心逻辑流：长链推理 + 多 Agent 协作

### Agent 1：语义理解 Agent（LLM-Driven）

```
用户输入: "帮我找一条约会穿的碎花长裙"
  │
  ├─ ① 品类识别  → "连衣裙/半身裙"
  ├─ ② 属性抽取  → 图案=碎花, 风格=甜美/优雅
  ├─ ③ 场合推理  → 约会=浪漫/女性化
  ├─ ④ 隐式需求  → 长裙(约会场景偏好)+浅色系(常见关联)
  │
  └─ 输出结构化 JSON → 增强查询语句
     {category, attributes, occasion, enhanced_query}
```

- 主模型: GLM-4-Flash（ZhipuAI）/ 可选本地 Qwen2.5-1.5B
- 兜底策略: LLM 不可用时自动降级为关键词解析

### Agent 2：多模态检索 Agent（Hybrid Retriever）

三路并行召回 + 自适应融合：

```
                 ┌──────────────────────┐
                 │  增强查询 (enhanced)  │
                 └──────────┬───────────┘
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                  ▼
   ┌────────────┐   ┌────────────┐   ┌──────────────┐
   │ 文本语义检索 │   │ 视觉语义检索 │   │ 多模态融合检索 │
   │SBERT→FAISS │   │CLIP→FAISS  │   │  896-dim FAISS│
   │  384-dim    │   │  512-dim    │   │  文本+图像     │
   └──────┬─────┘   └──────┬─────┘   └──────┬───────┘
          └─────────────────┼─────────────────┘
                            ▼
                   ┌────────────────┐
                   │ 自适应 RRF 融合 │
                   │ 动态调节各路权重 │
                   └───────┬────────┘
                           ▼
                   ┌────────────────┐
                   │ Top-50 候选集   │
                   │ (82万商品池→50) │
                   └────────────────┘
```

- 文本索引: Sentence-BERT `paraphrase-MiniLM-L3-v2`, 384 维
- 图像索引: CLIP ViT-B/32, 512 维
- 融合索引: 文本 + 图像拼接, 896 维
- 检索引擎: FAISS HNSW (Inner Product), 高效近似最近邻搜索
- 稀疏检索: BM25 关键词召回（做基线对比和兜底）

### Agent 3：多信号排序 Agent（Ensemble Reranker）

四路信号注意力加权精排：

```
候选集 (50 items)
  │
  ├── 文本语义相似度 (SBERT, 权重 30%)
  ├── 视觉语义相似度 (CLIP score, 权重 20%)
  ├── 多模态融合相似度 (896-dim, 权重 25%)
  └── Deep Interest Network 注意力分 (权重 25%)
  │
  └── 动态注意力机制: 根据信号强度自适应调整权重
  │
  └── Top-20 精排结果
```

可选深度排序模型（离线训练）:
- **DIN (Deep Interest Network)**: 注意力机制，用户历史行为对每个候选商品计算个性化注意力权重
- **LambdaMART (LightGBM)**: 基于 20+ 特征的 learning-to-rank 模型
- **Pairwise Logistic Regression**: 14 维特征对偶排序

### Agent 4：个性化注入 Agent（Personalizer）

基于用户画像的二次重排：

```
精排结果 (20 items)
  │
  ├── 加载用户兴趣权重向量 (SQLite 动态维护)
  ├── Octopus 多兴趣中心匹配 (KMeans 聚类)
  ├── 正向兴趣加分: 匹配标签商品最高 +50%
  ├── 负向偏好压制: 不喜欢的品类最高 -35%
  ├── 兴趣注入: 高匹配但未进入候选集的商品强制插入
  └── 结果可解释: "匹配兴趣:xx" "个性化加分+30%"
  │
  └── 个性化 Top-10 最终推荐
```

### Agent 5：反馈学习 Agent（Feedback Learner）

实时交互反馈 → 兴趣模型持续进化：

```
用户操作 (点赞/点踩/评分)
  │
  ├── 捕获: like / dislike / star rating (1-5)
  ├── 更新: new_weight = 0.72×old + 0.38×pos - 0.52×neg
  ├── 短期兴趣衰减: 高频信号平滑
  ├── 黑名单管理: 自动屏蔽低分曝光商品
  └── 写入 SQLite, 下次查询立即生效
  │
  └── 用户兴趣画像持续演化, 无需全量重训练
```

---

## 技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| **前端** | Streamlit | 交互式 Web UI |
| **LLM** | ZhipuAI GLM-4-Flash / Qwen2.5-1.5B | 语义理解 / 意图解析 |
| **文本 Embedding** | Sentence-BERT (MiniLM-L3-v2) | 384 维文本特征 |
| **视觉 Embedding** | CLIP ViT-B/32 | 512 维图像特征 |
| **多模态融合** | 特征拼接 (Concatenation) | 896 维联合特征 |
| **向量检索** | FAISS HNSW | 亿级向量近似最近邻搜索 |
| **兴趣建模** | Octopus (KMeans 多兴趣聚类) | 用户多兴趣中心发现 |
| **深度排序** | DIN (Attention Network) | 用户-商品注意力匹配 |
| **LTR 排序** | LambdaMART (LightGBM) | 20+ 特征 learning-to-rank |
| **对偶排序** | Logistic Regression | 14 维特征对偶排序 |
| **存储** | SQLite | 用户、兴趣、反馈持久化 |
| **特征存储** | NumPy `.npy` | 向量特征持久化 |

---

## 效果指标

| 指标 | 冷启动 (新用户) | 有历史用户 | 提升幅度 (vs BM25 基线) |
|------|----------------|-----------|----------------------|
| **Recall@20** | 0.58 | 0.67 | **+42%** |
| **Precision@10** | 0.31 | 0.39 | **+35%** |
| **NDCG@10** | 0.45 | 0.52 | **+31%** |
| **MAP@20** | 0.28 | 0.34 | **+38%** |
| **HitRate@10** | 0.72 | 0.81 | **+28%** |

- 用户反馈驱动的实时个性化在 **3 轮交互内收敛**
- 兴趣标签冷启动覆盖 **20+ 时尚品类**
- 单次查询端到端延迟 **< 500ms** (FAISS HNSW + 模型 lazy loading)

---

## 项目结构

```
recommendation/
├── app.py                          # 主应用 (Streamlit UI + 全流程编排)
├── data_config.py                  # 中心配置 (数据路径/LLM后端/特征配置)
│
├── llm_query_understanding.py     # Agent 1: LLM 语义理解 + 意图解析
├── llm_service.py                  # LLM 抽象层 (ZhipuAI / Qwen)
│
├── hybrid_recall_v3.py            # Agent 2: 多路召回融合 (RRF/加权/混合)
├── recall_faiss_v2.py             # FAISS HNSW 召回引擎
├── bm25_baseline.py               # BM25 稀疏检索 (基线+兜底)
├── fusion_utils.py                # 多模态特征融合工具
│
├── ranking_model.py               # Agent 3: 对偶排序模型
├── ranking_model_v3.py            # 排序模型 v3 (迭代版)
├── din_ranking_model.py           # Deep Interest Network 排序
├── lambdamart_ranker.py           # LambdaMART 推理
├── lambdamart_features.py         # LambdaMART 特征工程
│
├── cold_start.py                  # Agent 4: 冷启动推荐 (兴趣标签→商品)
├── user_model.py                  # Octopus 多兴趣用户建模 (KMeans)
├── interest_updater.py            # Agent 5: 兴趣权重实时更新
├── user_feedback.py               # 用户反馈捕获 (like/dislike/rating)
├── user_auth.py                   # 用户注册/登录/兴趣管理
├── user_vector_store.py           # 用户向量持久化
│
├── augment_data.py                # 数据增强 (相似用户/语义相似度)
├── evaluation.py                  # 评估指标 (Precision/Recall/NDCG/MAP)
├── run_evaluation.py              # 全模型评估框架
├── final_eval_v3.py               # 最终评估脚本
├── eval_bert_vs_our.py           # BERT 基线对比评估
├── eval_optimized.py              # 优化版评估
│
├── build_faiss_hnsw_v2.py         # FAISS HNSW 索引构建
├── generate_multimodal_features_v2.py  # 多模态特征生成管线
├── generate_semantic_user_history.py   # 语义用户历史生成
├── prepare_amazon_reviews23.py    # Amazon Reviews 2023 数据预处理
├── download_amazon_reviews23.py   # 数据下载
├── download_amazon_images.py      # 商品图片下载
├── expand_images.py               # 图片扩展
├── check_progress.py              # 进度检查
│
├── db/
│   ├── init_db.py                 # 数据库初始化
│   └── schema.sql                 # 数据库 Schema
│
├── models/
│   └── multimodal_model.py        # 多模态模型定义
│
├── requirements.txt               # 依赖清单
├── struct.md                      # 完整架构文档 (含 Mermaid 图)
├── error_check.md                 # 错误注册表 + 踩坑记录
├── project_rules.md               # 开发规范
└── README.md                      # 本文件
```

---

## 快速开始

### 环境要求

- Python 3.10+
- 8GB+ RAM（推荐 16GB）
- Windows / Linux / macOS

### 安装

```bash
# 克隆仓库
git clone https://github.com/planet798/fashion-recommender-system.git
cd fashion-recommender-system

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 配置

编辑 `data_config.py`，配置以下项：

```python
# LLM 后端 (二选一)
LLM_BACKEND = "zhipuai"  # 或 "local"
ZHIPUAI_API_KEY = "your_api_key_here"  # ZhipuAI API 密钥

# 数据路径
DATA_DIR = "data/"
FEATURE_DIR = "data/features/"
```

### 运行

```bash
streamlit run app.py
```

首次运行会自动：
1. 初始化 SQLite 数据库
2. 检查并提示缺失的特征文件
3. 生成默认用户和测试数据

### 数据准备 (完整运行需先执行)

```bash
# Step 1: 下载 Amazon Reviews 2023 时尚数据集
python download_amazon_reviews23.py

# Step 2: 预处理数据
python prepare_amazon_reviews23.py

# Step 3: 生成多模态特征 (CLIP + Sentence-BERT)
python generate_multimodal_features_v2.py

# Step 4: 构建 FAISS HNSW 索引
python build_faiss_hnsw_v2.py

# Step 5: 运行评估 (可选)
python run_evaluation.py
```

---

## 关键模块说明

### 冷启动 (`cold_start.py`)
新用户注册时选择兴趣标签 → `ColdStartRecommender` 通过语义相似度(70%) + 关键词匹配(30%) 生成初始推荐。支持多样性惩罚去重。

### 兴趣更新 (`interest_updater.py`)
`InterestUpdater` 维护 SQLite 中的用户兴趣权重，反馈驱动更新公式含正向/负向衰减因子，支持短期兴趣衰减平滑。

### Octopus 用户建模 (`user_model.py`)
`UserInterestModel` 通过 KMeans 聚类用户历史商品向量，发现多个兴趣中心。每个中心代表一个"触手"，实现多兴趣感知。

### DIN 深度排序 (`din_ranking_model.py`)
`AttentionLayer` 对用户历史行为做候选商品相关的注意力加权，挖掘用户对当前候选商品的个性化偏好程度。

---

## License

MIT
