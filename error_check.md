# Error Check 记录

## 错误1: KMeans内存泄漏 (Windows + MKL)

**发现时间**: 2026-04-10
**文件**: user_model.py

### 为什么没提前发现
- 仅在Windows + Intel MKL环境下触发
- 小数据集(data/)上不明显，大数据集(datasets/)上才暴露
- sklearn的已知Bug，非代码逻辑错误

### 根因
- 全局复用KMeans实例: `self.kmeans_model = KMeans()` + 反复调用 `fit_predict()`
- Windows MKL线程句柄不释放 → 内存持续增长

### 修复
- 每次创建新KMeans实例
- 添加向量去重避免ConvergenceWarning
- 添加try-except降级机制

### 教训
- **sklearn的KMeans在Windows上不要复用实例**
- 设置 `OMP_NUM_THREADS=1` 可进一步缓解
- ConvergenceWarning是`sklearn.exceptions.ConvergenceWarning`，不是`UserWarning`

---

## 错误2: ConvergenceWarning过滤规则写错

**发现时间**: 2026-04-10
**文件**: train_ranking_model_v3.py

### 为什么没提前发现
- 混淆了Warning类别层级
- `ConvergenceWarning`继承自`UserWarning`但`filterwarnings`需要精确匹配category

### 根因
```python
# 错误写法
warnings.filterwarnings('ignore', category=UserWarning, message='.*ConvergenceWarning.*')
# ConvergenceWarning的category不是UserWarning!

# 正确写法
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)
```

### 教训
- **sklearn的专用Warning需要从sklearn.exceptions导入**
- 不要假设所有Warning都是UserWarning子类

---

## 错误3: Faiss Hard Negatives采样性能灾难 (最严重!)

**发现时间**: 2026-04-10
**文件**: ranking_model_v3.py `_sample_hard_negatives_from_faiss()`

### 为什么没提前发现
- 逻辑正确性验证通过（小数据集测试OK）
- 没有做大规模性能测试
- 忽略了O(N)次sklearn调用的开销

### 根因
```python
# 旧代码: 逐个计算余弦相似度
for item_id, feat in self.text_features.items():          # ~5000+ 商品
    sim = float(cosine_similarity([vec], [feat])[0][0])   # 每次调用sklearn (极慢!)
```

**计算量**: 2000用户 × 5兴趣 × 5000商品 = **5000万次sklearn调用** → 永远跑不完

### 修复
```python
# 新代码: 预构建矩阵 + 一次矩阵乘法
self._faiss_candidate_matrix = np.array(candidate_vecs, dtype=np.float32)  # 只构建一次
all_sims = (candidate_matrix @ vec_normalized.T).flatten()  # 一次numpy运算
```

**性能对比**:
- 旧版: ~5000万次sklearn调用 → 预计>10小时
- 新版: ~10000次numpy矩阵乘法 → 预计3-5分钟
- **加速比: 1000倍以上**

### 教训
- **永远不要在循环中逐个调用sklearn的cosine_similarity**
- 用numpy矩阵乘法替代: `matrix @ vector.T`
- 预构建特征矩阵并缓存，避免重复计算
- 大规模数据操作前必须做性能估算

---

## 错误4: 评估脚本history_path错误

**发现时间**: 2026-04-10
**文件**: quick_validate_v3.py, correct_eval_v3.py

### 为什么没提前发现
- 混淆了"完整历史"和"训练集历史"的用途
- UserInterestModel应该只看到训练集数据，不应看到GT数据

### 根因
```python
# 错误: 使用完整历史 (包含GT数据)
user_model = UserInterestModel(history_path="data/user_history.csv")

# 正确: 使用训练集历史 (排除GT)
user_model = UserInterestModel(history_path=train_history_path)
```

### 教训
- **评估时UserModel必须只使用训练集数据**
- 否则会导致数据泄漏，评估结果不可靠

---

## 通用规则 (从以上错误总结)

1. **sklearn组件不要复用实例** (Windows MKL问题)
2. **sklearn专用Warning需要精确导入category**
3. **大规模相似度计算必须用矩阵运算，不要逐个调用**
4. **评估时严格区分训练集和测试集**
5. **代码逻辑正确不等于性能可接受，必须做规模估算**

---

## 错误5: Streamlit UI卡顿/冻结 (系统级性能问题)

**发现时间**: 2026-04-11
**文件**: app.py

### 为什么没提前发现
- 小数据集(data/)上用户数少，评估速度快不明显
- @st.cache_data在小数据集上首次执行时间短，感觉不到阻塞
- 开发时频繁刷新页面，习惯了启动等待

### 根因 (4个问题叠加)

**问题1: 启动时同步加载全部模型 (白屏15-30秒)**
```python
# 旧代码: 模块顶层同步加载8个对象
text_model = load_text_model()            # SentenceTransformer ~2s
clip_model, ... = load_clip_model()       # CLIP ~3s
user_model = load_user_model()             # 历史+特征 ~3s
faiss_recall = load_faiss_recall(user_model) # FAISS索引 ~2s
bm25_retriever = load_bm25()              # BM25建索引 ~2s
hybrid_recall = load_hybrid_recall(...)    # 包装 ~0.1s
ranking_model = load_ranking_model()       # 可选 ~1s
items, ... = load_data()                   # CSV+3个npy ~5s
# 总计: 15-30秒白屏，UI完全无响应
```

**问题2: @st.cache_data + 全量2000用户评估阻塞UI线程**
```python
# 旧代码
@st.cache_data
def compute_offline_metrics(topk):
    # 首次执行: 同步跑完 BM25(2000人) + FAISS(2000人) + Hybrid(2000人) + Ranker(2000人)
    # @st.cache_data 虽然缓存结果，但首次执行完全阻塞Streamlit主线程
    # 用户看到: "Running offline evaluation..." 永远在转圈
```

**问题3: evaluate_bm25 无进度反馈 (BM25步骤看起来像卡死)**
```python
# run_evaluation.py 的 evaluate_bm25:
# - log_every=20，每20个user才打印一次日志
# - app.py中只有 st.spinner()，没有 progress bar
# - 用户看到终端停在 "BM25 processed 20/2000 users" 后长时间无输出
# - 实际上没有卡死，只是下一个log要等处理完第40个user
```

**问题4: show_item 硬编码已删除的 data/images 路径**
```python
# 旧代码
img_path = os.path.join("data", "images", f"{item_id}.jpg")  # data/ 已删除!
```

### 修复

**修复1: 懒加载 + session_state 缓存**
```python
def get_models():
    """按需加载: Text Search模式不需要user_model/ranking_model"""
    cached = st.session_state.setdefault("models_loaded", {})
    if "data" not in cached:
        with st.spinner("Loading data..."):
            cached["data"] = load_data()
    mode = st.session_state.get("current_mode", "Text Search")
    if mode == "Text Search":
        # 只加载: text_model, clip, bm25, hybrid (跳过 user_model, ranking_model)
        ...
    else:
        # 只加载: user_model, bm25, faiss, hybrid, ranking_model (跳过 text_model, clip)
        ...
```
**效果**: 启动时间从 15-30秒 → **3-5秒** (只加载CSV+npy数据)

**修复2: 移除@st_cache_data + 采样评估 + 全进度条**
```python
def compute_offline_metrics(topk, sample_users=300, models_to_eval=None):
    # 不使用 @st.cache_data → 不再阻塞UI线程缓存机制
    # 默认采样300用户而非全量2000 → 速度提升~6倍
    # 所有步骤都有 st.progress() 进度条 → 用户可见实时进展
    # 每步结束后 del 释放内存 → 内存峰值降低50%+
```
**效果**: 评估时间从 5-10分钟 → **30-60秒** (300样本 + 进度反馈)

**修复3: BM25 步骤增加进度条**
```python
progress_bar = progress_placeholder.progress(0, text="Evaluating BM25 baseline...")
for idx, user_id in enumerate(ground_truth, start=1):
    recommendations[user_id] = retriever.recommend_for_user(...)
    if idx % log_every == 0 or idx == total_users:
        progress_bar.progress(idx / total_users, text=f"BM25: {idx}/{total_users} users")
```

**修复4: 图片路径改用 config**
```python
img_path = os.path.join(config.images_dir, f"{item_id}.jpg")  # 使用配置路径
```

### 性能改进指标

| 指标 | 优化前 | 优化后 | 改善 |
|------|--------|--------|------|
| 页面首次加载 | 15-30秒白屏 | 3-5秒(有spinner提示) | **5-6x加速** |
| 离线评估(BM25+FAISS+Hybrid) | 5-10分钟(UI冻结) | 30-60秒(实时进度条) | **6-10x加速** |
| 切换Text↔User模式 | 重新加载全部(~10s) | 复用已加载部分(~2s) | **3-5x加速** |
| 评估期间内存 | 翻倍(重复创建模型) | 单副本+及时del | **-50%内存** |
| 图片显示 | ❌ 全部Missing | ✅ 正常(路径正确) | 功能恢复 |

### 教训
- **Streamlit的@st.cache_data/@st.cache_resource会阻塞UI线程**: 首次执行期间界面完全冻结
- **模型懒加载是Streamlit应用的标准最佳实践**: 不要在模块顶层加载所有资源
- **离线评估必须采样**: 2000用户全量评估在生产前端是不可接受的
- **所有耗时操作都必须有进度反馈**: 否则用户会误以为系统卡死
- **删除目录后必须全局搜索残留引用**: `os.path.join("data", "images")` 这种硬编码容易被遗漏

---

## 错误6: FaissRecallV2 缺少 recall_by_text 方法

**发现时间**: 2026-04-11
**文件**: recall_faiss_v2.py, app.py, hybrid_recall_v3.py

### 为什么没提前发现
- `FaissRecallV2` 类最初只设计了用户召回功能 (`recall_by_user`)
- 后续新增的 `HybridRecallV3` 类调用了 `recall_by_text` 方法，但没有同步更新 `FaissRecallV2`
- 开发时可能只在"User Recommendation"模式下测试，未测试"Text Search"模式
- 缺乏接口契约检查：调用方假设被调用方有某方法，但未验证

### 根因
```python
# hybrid_recall_v3.py 第445行 (调用方)
faiss_results = self.faiss.recall_by_text(query, topk=faiss_topk)  # 假设存在此方法

# recall_faiss_v2.py (被调用方)
class FaissRecallV2:
    def __init__(self, ...):  # 没有 text_encoder 参数
        ...
    def recall_by_user(self, ...):  # 只有这个方法
        ...
    # ❌ 缺少 recall_by_text 方法!
```

**错误信息**: `AttributeError: 'FaissRecallV2' object has no attribute 'recall_by_text'`

### 修复

**1. 修改 FaissRecallV2.__init__ 添加 text_encoder 参数**
```python
def __init__(self, user_model, text_index_path, text_ids_path, image_index_path, image_ids_path, text_encoder=None):
    self.user_model = user_model
    self.text_encoder = text_encoder  # 新增: SentenceTransformer 实例
    ...
```

**2. 新增 recall_by_text 方法**
```python
def recall_by_text(self, query, topk=50):
    """文本查询召回：使用文本编码器将查询转换为向量，然后在FAISS索引中搜索"""
    if self.text_encoder is None:
        raise ValueError("text_encoder is required for recall_by_text")
    query_vec = self.text_encoder.encode(query, normalize_embeddings=True)
    results = self._search(self.text_index, self.text_item_ids, query_vec, topk)
    return results
```

**3. 更新 app.py 的 load_faiss_recall 函数**
```python
@st.cache_resource
def load_faiss_recall(_user_model, _text_model=None):  # 新增参数
    return FaissRecallV2(
        ...,
        text_encoder=_text_model  # 传入文本编码器
    )
```

**4. 更新 get_models() 调用处**
```python
text_m = cached.get("text_model")
faiss_m = load_faiss_recall(user_m, text_m)  # 传入text_model
```

### 教训
- **类设计时要考虑所有使用场景**: 如果一个类会被多个模块使用，必须确保它提供了完整的接口
- **修改调用链时必须同步更新依赖**: 新增 `HybridRecallV3` 调用 `recall_by_text` 时应立即在 `FaissRecallV2` 中实现
- **Text Search 和 User Recommendation 两种模式都要测试**: 不能只测一种模式就认为代码没问题
- **可选依赖用 None + 运行时检查**: `text_encoder=None` + `if self.text_encoder is None: raise ValueError()` 比强制参数更灵活

---

## 错误7: FAISS 索引维度不匹配 (AssertionError: assert d == self.d)

**发现时间**: 2026-04-11
**文件**: recall_faiss_v2.py, data_config.py, fusion_utils.py

### 为什么没提前发现
- Amazon 数据源使用**多模态融合索引**（896维 = 384文本 + 512图像），而非纯文本索引（384维）
- SentenceTransformer 输出 384 维向量，直接传入 896 维的 FAISS 索引会触发断言错误
- data_config.py 中 text_index 和 image_index 指向同一个 faiss_hnsw.index 文件，说明是融合索引
- 开发时可能用 sample 数据源（有独立纯文本索引）测试，切换到 amazon 数据源后未验证

### 根因
```python
# data_config.py (amazon 数据源)
"text_index": "datasets/amazon_reviews23/processed/faiss_hnsw.index",   # 融合索引 (896维)
"image_index": "datasets/amazon_reviews23/processed/faiss_hnsw.index",  # 同一个索引!

# 错误代码: recall_by_text
query_vec = self.text_encoder.encode(query)  # 384维
results = self._search(self.text_index, ..., query_vec, topk)  # ❌ 384 ≠ 896!

# FAISS 内部触发
# faiss/class_wrappers.py line 329:
assert d == self.d  # AssertionError!
```

**维度对应关系**:
| 数据源 | 文本索引 | 图像索引 | 索引类型 | 维度 |
|--------|----------|----------|----------|------|
| sample | faiss_text.index | faiss_image.index | 独立索引 | 384 / 512 |
| amazon | faiss_hnsw.index | faiss_hnsw.index | 融合索引 | **896** |

### 修复
```python
# 正确代码: recall_by_text 需要使用 build_text_query_fusion 转换向量
from fusion_utils import build_text_query_fusion

def recall_by_text(self, query, topk=50):
    text_vec = self.text_encoder.encode(query, normalize_embeddings=True)  # 384维
    fused_vec = build_text_query_fusion(text_vec)  # → 896维 (补零图像部分)
    results = self._search(self.text_index, self.text_item_ids, fused_vec, topk)  # ✅ 匹配
    return results
```

**build_text_query_fusion 的作用** (fusion_utils.py):
```python
def build_text_query_fusion(text_vec, alpha=1.0, beta=1.0, normalize_final=True):
    return weighted_fusion(
        text_vec=text_vec,
        image_vec=np.zeros(IMAGE_DIM, dtype=np.float32),  # 图像部分补零
        alpha=alpha,
        beta=beta,
        normalize_final=normalize_final,
    )
    # 输出: [text_vec(384) || zeros(512)] = 896维
```

### 教训
- **不同数据源的FAISS索引结构可能不同**: sample 用独立索引，amazon 用融合索引
- **切换数据源后必须验证维度兼容性**: 不能假设所有数据源的索引结构相同
- **data_config.py 的配置暗示了索引类型**: text_index == image_index 说明是融合索引
- **使用 fusion_utils 提供的工具函数**: 不要手动拼接向量，用封装好的函数确保一致性

---

## 错误8: BM25 图片全部缺失 + Our Model 结果过少

**发现时间**: 2026-04-11
**文件**: app.py

### 为什么没提前发现
- 开发时可能用 sample 数据源（商品数少，图片覆盖率高）测试
- 切换到 amazon 数据源（82万商品）后未验证图片覆盖率
- 未意识到 BM25 搜索范围与特征文件覆盖范围的巨大差异

### 根因

**数据规模不匹配**:
| 数据 | 数量 |
|------|------|
| items.csv 商品总数 | **826,050** |
| 有图片的商品 | **12,477 (1.5%)** |
| 有特征文件的商品 | **12,501** |

**问题1：BM25 图片缺失**
- BM25 从 82 万商品中搜索 → 返回结果 99% 没有图片
- `show_item()` 函数找不到图片 → 显示 "❌ Image Missing"

**问题2：Our Model 只显示 1 个结果**
```
调用链:
build_text_candidates()
  ├─ bm25_retriever.search(topk=50)     → 返回 50 个候选 (来自 82 万商品)
  └─ hybrid_recall.recall_by_text()      → 返回 ~50 个候选 (来自 FAISS 12501 商品)
       ↓ 合并 ~100 个候选
rerank_text_query_v2()
  ├─ 第 668-669 行: if item_text is None or item_img is None or item_multi is None: continue
  │   → BM25 的 50 个候选全部没有特征 → 被过滤掉！
  └─ 只剩 FAISS 的 ~50 个候选参与排序 → 结果数量少、质量可能不佳
```

### 修复

**1. build_text_candidates - 过滤无特征的 BM25 候选 + 增加召回量**
```python
def build_text_candidates(query_text, recall_topk):
    bm25_candidates = bm25_retriever.search(query_text, topk=recall_topk * 3)  # 增量召回
    ...
    valid_item_ids = set(text_features.keys()) & set(image_features.keys()) & set(multimodal_features.keys())

    merged = {}
    for rank, (item_id, score) in enumerate(bm25_candidates, start=1):
        item_id = str(item_id)
        if item_id not in valid_item_ids:  # ← 新增：跳过无特征的商品
            continue
        ...
    return results[:recall_topk * 2]  # ← 返回更多候选给 rerank
```

**2. baseline_rank_text - 只返回有图片的 BM25 结果**
```python
# 修复后: topk*500 + 用图片目录做过滤
def baseline_rank_text(query_text, topk=8):
    raw_results = bm25_retriever.search(query_text, topk=topk * 500)  # 大幅增加!
    filtered = [(item_id, score) for item_id, score in raw_results if str(item_id) in valid_image_ids]
    return filtered[:topk]
```

**3. load_data() 预计算 valid_image_ids 全局变量**
```python
@st.cache_data
def load_data():
    ...
    valid_image_ids = set()
    if os.path.isdir(config.images_dir):
        valid_image_ids = set(
            f.replace(".jpg", "") for f in os.listdir(config.images_dir) if f.endswith(".jpg")
        )
    return items, text_features, image_features, multimodal_features, fusion_config, valid_image_ids
```

**实测效果 (top-4000 召回)**:
| 查询 | 修复前 (top-200) | 修复后 (top-4000) |
|------|-----------------|-------------------|
| sweatshirt | **1** 个 ❌ | **42** 个 ✅ |
| black hooded sweatshirt | **2** 个 ❌ | **32** 个 ✅ |
| red checked shirt | **6** 个 | **45** 个 ✅ |

### 教训
- **大数据集必须验证资源覆盖率**: 82万商品只有1.5%有图片，不能假设所有商品都有完整资源
- **BM25 搜索范围要与特征文件对齐**: 否则大量候选在 rerank 阶段被浪费
- **增量召回必须足够大**: `topk*5` 远远不够，需要 `topk*500` 才能从82万中筛出足够的有效结果
- **预计算有效ID集合**: 在 load_data 时一次性扫描图片目录，避免每次查询重复 I/O
- **用图片目录而非特征交集做 baseline 过滤**: baseline 只需有图片即可，不需要三种特征齐全

---

## 错误9: BERT基线评估时对82万商品建索引过慢

**发现时间**: 2026-04-12
**文件**: run_evaluation.py, generate_bert_features.py, bert_baseline.py

### 为什么没提前发现
- 之前BM25也是对82万商品建索引，虽然慢但能跑
- 没有意识到评估只需要对有图片的~1.2万商品建索引即可
- items.csv有82万条记录，但实际有图片的只有12,477条

### 根因
```python
# 错误: 对全部82万商品建索引
retriever = BM25Retriever(items_path=items_path)  # 82万 → 索引构建极慢
retriever = BERTRetriever(items_path=items_path)  # 82万 → 编码+建索引4小时

# 正确: 只对有图片的商品建索引
valid_ids = set(f.replace(".jpg", "") for f in os.listdir(images_dir) if f.endswith(".jpg"))  # ~1.2万
retriever = BM25Retriever(items_path=items_path, valid_ids=valid_ids)  # 1.2万 → 秒级
retriever = BERTRetriever(items_path=items_path, valid_ids=valid_ids)  # 1.2万 → 4分钟
```

### 修复
- generate_bert_features.py 添加 `--images-dir` 参数过滤有效商品
- run_evaluation.py 添加 `--images-dir` 参数，将 valid_ids 传给评估函数
- bert_baseline.py 的 BERTRetriever 支持 valid_ids 过滤

### 教训
- **评估和检索只需要处理有完整资源的商品**: 没有图片的商品在UI上无法展示，建索引是浪费
- **82万→1.2万的过滤可加速70倍**: BM25从分钟级→秒级，BERT从4小时→4分钟
- **所有检索器都应支持valid_ids过滤**: BM25Retriever已有此功能，BERTRetriever也需同步

---

## 错误10: 评估脚本中用户向量维度与候选向量维度不匹配

**发现时间**: 2026-04-12
**文件**: eval_optimized.py

### 为什么没提前发现
- 用户向量使用了多模态融合向量(896维)来计算文本相似度
- 但文本特征矩阵是384维，直接用896维向量做cosine_similarity报错

### 根因
```python
# 错误: 用896维多模态向量去和384维文本矩阵计算相似度
user_avg_vec = normalize(np.mean(user_multi_vecs, axis=0))  # 896维
text_sims = cosine_similarity([user_avg_vec], all_text_vecs)[0]  # 384维 → ValueError

# 正确: 分别计算多模态和文本的用户向量
user_multi_vecs = [...]  # 896维
user_text_vecs = [...]   # 384维
user_avg_vec = normalize(np.mean(user_multi_vecs, axis=0))  # 896维
user_text_vec = normalize(np.mean(user_text_vecs, axis=0))  # 384维
multi_sims = cosine_similarity([user_avg_vec], all_item_vecs)[0]
text_sims = cosine_similarity([user_text_vec], all_text_vecs)[0]
```

### 教训
- **多模态融合向量和文本向量维度不同，不能混用**: 896维≠384维
- **计算相似度时必须确保维度一致**: 用户向量和候选矩阵的shape[1]必须相同

---

## 错误11: UTF-8文件BOM字符导致SyntaxError

**发现时间**: 2026-04-14
**文件**: user_auth.py

### 为什么没提前发现
- BOM字符(U+FEFF)在某些文本编辑器保存UTF-8文件时自动添加
- BOM字符位于文件开头第一行，不容易察觉
- IDE可能正常显示BOM字符，但Python解释器无法将其识别为有效的标识符

### 根因
```python
# 错误: 文件开头有BOM字符
﻿# -*- coding: utf-8 -*-  # ❌ 开头的 ﻿ 是BOM字符(U+FEFF)

# 正确: 文件开头应该是
# -*- coding: utf-8 -*-
```

**错误信息**: `SyntaxError: invalid character in identifier`

### 修复
直接删除文件开头的BOM字符即可。使用任何文本编辑器或IDE的正则替换功能，将 `^﻿` 替换为空。

### 教训
- **保存UTF-8文件时禁用BOM**: 大多数现代文本编辑器允许选择"UTF-8 without BOM"
- **Python源码文件不应包含BOM**: PEP 3131规定标识符中不允许使用BOM
- **检查文件开头是否有多余字符**: 可以用十六进制编辑器或 `python -c "with open('file.py', 'rb') as f: print(f.read(10))"` 检查

