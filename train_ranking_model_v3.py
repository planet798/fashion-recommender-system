# -*- coding: utf-8 -*-
"""
训练脚本: 使用优化后的V3模型进行训练
- 使用HybridRecallV3 (自适应RRF融合)
- 使用PairwiseFeatureRankerV3 (Hard Negatives)
- 已优化: 彻底抑制所有sklearn警告
"""
import sys
import io
import warnings
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ===========================================
# 关键优化: 彻底抑制所有已知警告
# ===========================================
# 1. 抑制KMeans内存泄漏警告 (UserWarning)
warnings.filterwarnings('ignore', category=UserWarning, message='.*memory leak.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*OMP_NUM_THREADS.*')

# 2. 抑制ConvergenceWarning (这是sklearn.exceptions.ConvergenceWarning，不是UserWarning!)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# 3. 抑制FutureWarning
warnings.filterwarnings('ignore', category=FutureWarning)

from data_config import config
from ranking_model_v3 import PairwiseFeatureRankerV3


def main():
    print("=" * 80)
    print("[TRAIN-V3] Training Optimized Ranking Model with Hard Negatives")
    print(f"[TRAIN-V3] Data Source: {config.name} ({config.source})")
    print("=" * 80)

    # 设置OMP_NUM_THREADS环境变量 (进一步抑制MKL线程泄漏)
    os.environ.setdefault('OMP_NUM_THREADS', '1')

    ranker = PairwiseFeatureRankerV3(
        model_path=config.ranking_model_v3,
        text_feature_path=config.text_features,
        image_feature_path=config.image_features
    )

    # 训练配置
    ranker.fit(
        history_path=config.user_history_csv,
        items_path=config.items_csv,
        negative_samples=6,
        min_history_len=3,
        filter_items_to_history=True,
        max_users=0  # 使用全部用户
    )

    # 保存模型
    print("\n[TRAIN-V3] Saving model...", flush=True)
    ranker.save()
    print(f"[TRAIN-V3] Model saved to {config.ranking_model_v3}", flush=True)

    # 清理临时文件
    ranker.cleanup()

    print("\n[TRAIN-V3] Training completed successfully! ✅", flush=True)


if __name__ == "__main__":
    main()
