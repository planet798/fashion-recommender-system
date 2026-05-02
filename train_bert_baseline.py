# -*- coding: utf-8 -*-
"""
BERT基线模型微调脚本
使用对比学习(MultipleNegativesRankingLoss)在Amazon Fashion交互数据上微调BERT模型
使基线模型适应时尚领域语义，提升检索效果
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader


def build_training_pairs(items_path, history_path, min_interactions=3):
    """
    从用户交互历史构建训练对
    策略: 对于每个用户，将其交互商品标题拼接为query，每个交互商品作为positive
    """
    items = pd.read_csv(items_path)
    items["item_id"] = items["item_id"].astype(str)

    item_texts = {}
    for _, row in items.iterrows():
        item_id = str(row["item_id"])
        parts = [str(row.get(col, "")) for col in ["title", "brand", "categories", "description"]]
        text = " ".join(parts).strip()
        item_texts[item_id] = text

    history = pd.read_csv(history_path)
    history["item_id"] = history["item_id"].astype(str)

    user_groups = history.groupby("user_id")
    train_examples = []

    for user_id, group in user_groups:
        user_items = group.sort_values("timestamp")["item_id"].tolist()
        if len(user_items) < min_interactions:
            continue

        for i, target_item_id in enumerate(user_items):
            if target_item_id not in item_texts:
                continue

            other_items = [iid for j, iid in enumerate(user_items) if j != i and iid in item_texts]
            if not other_items:
                continue

            query_parts = [item_texts[iid] for iid in other_items[:5]]
            query_text = " ".join(query_parts).strip()

            if not query_text:
                continue

            positive_text = item_texts[target_item_id]
            train_examples.append(InputExample(texts=[query_text, positive_text]))

    return train_examples, item_texts


def build_eval_data(items_path, history_path, holdout_len=2, max_users=None):
    """
    构建评估数据: InformationRetrievalEvaluator格式
    queries: {query_id: query_text}
    corpus: {doc_id: doc_text}
    relevant_docs: {query_id: {doc_id1, doc_id2, ...}}
    """
    items = pd.read_csv(items_path)
    items["item_id"] = items["item_id"].astype(str)

    corpus = {}
    for _, row in items.iterrows():
        item_id = str(row["item_id"])
        parts = [str(row.get(col, "")) for col in ["title", "brand", "categories", "description"]]
        text = " ".join(parts).strip()
        corpus[item_id] = text

    history = pd.read_csv(history_path)
    history["item_id"] = history["item_id"].astype(str)

    if max_users is not None and max_users > 0:
        keep_users = list(dict.fromkeys(history["user_id"].tolist()))[:max_users]
        history = history[history["user_id"].isin(keep_users)].copy()

    queries = {}
    relevant_docs = {}

    for user_id, group in history.groupby("user_id"):
        user_items = group.sort_values("timestamp")["item_id"].tolist()
        if len(user_items) <= holdout_len:
            continue

        train_items = user_items[:-holdout_len]
        test_items = user_items[-holdout_len:]

        query_parts = [corpus[iid] for iid in train_items if iid in corpus]
        if not query_parts:
            continue

        query_text = " ".join(query_parts[:5]).strip()
        if not query_text:
            continue

        query_id = f"user_{user_id}"
        queries[query_id] = query_text
        relevant_docs[query_id] = set(iid for iid in test_items if iid in corpus)

    return queries, corpus, relevant_docs


def main():
    parser = argparse.ArgumentParser(description="Fine-tune BERT baseline model with contrastive learning")
    parser.add_argument("--items-path", required=True, help="Path to items.csv")
    parser.add_argument("--history-path", required=True, help="Path to user_history.csv")
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2",
                        help="Base model name or local path")
    parser.add_argument("--output-path", default="models/all-MiniLM-L6-v2-finetuned",
                        help="Output path for fine-tuned model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="Warmup steps (0=auto 10%%)")
    parser.add_argument("--max-users", type=int, default=0,
                        help="Max users for training (0=all)")
    parser.add_argument("--eval-users", type=int, default=200,
                        help="Max users for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-every-epoch", action="store_true",
                        help="Save model after each epoch")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading base model: {args.model_name}")
    model = SentenceTransformer(args.model_name, device=device)
    print(f"Model loaded. Embedding dim: {model.get_sentence_embedding_dimension()}")

    print("Building training pairs...")
    train_examples, item_texts = build_training_pairs(
        args.items_path, args.history_path
    )
    print(f"Training examples: {len(train_examples)}")

    if not train_examples:
        print("No training examples found. Exiting.")
        return

    print("Building evaluation data...")
    queries, corpus, relevant_docs = build_eval_data(
        args.items_path, args.history_path,
        holdout_len=2, max_users=args.eval_users
    )
    print(f"Evaluation: {len(queries)} queries, {len(corpus)} docs")

    evaluator = None
    if queries and relevant_docs:
        evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name="amazon-fashion"
        )

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=args.batch_size
    )

    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    warmup_steps = args.warmup_steps
    if warmup_steps == 0:
        warmup_steps = int(len(train_dataloader) * args.epochs * 0.1)

    print(f"Training config:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Steps per epoch: {len(train_dataloader)}")

    output_dir = args.output_path

    if args.save_every_epoch:
        for epoch in range(1, args.epochs + 1):
            print(f"\n--- Epoch {epoch}/{args.epochs} ---")
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=1,
                warmup_steps=warmup_steps // args.epochs,
                optimizer_params={"lr": args.lr},
                show_progress_bar=True
            )

            if evaluator:
                print("Running evaluation...")
                eval_score = evaluator(model, output_path=os.path.join(output_dir, f"eval_epoch{epoch}"))
                print(f"Eval score: {eval_score}")

            epoch_path = f"{output_dir}_epoch{epoch}"
            model.save(epoch_path)
            print(f"Saved model to: {epoch_path}")
    else:
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=args.epochs,
            warmup_steps=warmup_steps,
            optimizer_params={"lr": args.lr},
            evaluator=evaluator,
            evaluation_steps=min(500, len(train_dataloader)),
            output_path=output_dir,
            show_progress_bar=True
        )

    model.save(output_dir)
    print(f"\nFine-tuned model saved to: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
