# Collaborative Baselines

Dataset: Amazon Reviews 2023 - Clothing, Shoes and Jewelry

Protocol:
- Positive feedback: rating >= 4
- Iterative 5-core collaborative dataset
- Leave-Last-Out split
- Fixed warm validation sample: 10,000 users
- Seed: 42
- Metrics: HR@10, NDCG@10, MRR@10

## Popularity

| Metric | Score |
|---|---:|
| HR@10 | 0.015500 |
| NDCG@10 | 0.012202 |
| MRR@10 | 0.011182 |

## LightGCN

Configuration:
- embedding_dim = 32
- num_layers = 2
- learning_rate = 0.001
- BPR loss
- best_epoch = 150

| Metric | Score |
|---|---:|
| HR@10 | 0.015500 |
| NDCG@10 | 0.012355 |
| MRR@10 | 0.011379 |

## LightGCN Ablation

| Embedding | Layers | Best Epoch | NDCG@10 |
|---:|---:|---:|---:|
| 32 | 1 | 220 | 0.012223 |
| 32 | 2 | 150 | 0.012355 |
| 32 | 3 | 150 | 0.012355 |
| 64 | 2 | 130 | 0.012363 |

Selected configuration: embedding_dim=32, num_layers=2.

Reason: dim=64 provides negligible validation improvement while requiring substantially more GPU memory and training time.
