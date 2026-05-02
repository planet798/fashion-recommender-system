import math
import re
import heapq
from collections import Counter, defaultdict

import pandas as pd


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text):
    return TOKEN_PATTERN.findall(str(text).lower())


class BM25Retriever:

    def __init__(self, items_path=None, k1=1.5, b=0.75, valid_ids=None):
        from data_config import config
        items_path = items_path or config.items_csv
        self.items = pd.read_csv(items_path)
        self.k1 = k1
        self.b = b
        if valid_ids is not None:
            valid_set = set(str(v) for v in valid_ids)
            self.items = self.items[self.items["item_id"].astype(str).isin(valid_set)]
        self.item_titles = {
            str(row["item_id"]): str(row["title"])
            for _, row in self.items.iterrows()
        }
        self.doc_tokens = {}
        self.doc_tf = {}
        self.doc_lengths = {}
        self.doc_freq = defaultdict(int)
        self.inverted_index = defaultdict(list)
        self.avg_doc_len = 0.0
        self._build_index()

    def _build_index(self):
        total_length = 0
        for item_id, title in self.item_titles.items():
            tokens = tokenize(title)
            token_counter = Counter(tokens)
            self.doc_tokens[item_id] = tokens
            self.doc_tf[item_id] = token_counter
            self.doc_lengths[item_id] = len(tokens)
            total_length += len(tokens)

            for token in token_counter.keys():
                self.doc_freq[token] += 1
                self.inverted_index[token].append(item_id)

        doc_count = max(len(self.doc_tokens), 1)
        self.avg_doc_len = total_length / doc_count

    def _idf(self, token):
        n = self.doc_freq.get(token, 0)
        doc_count = max(len(self.doc_tokens), 1)
        return math.log(1 + (doc_count - n + 0.5) / (n + 0.5))

    def search(self, query, topk=10, exclude_item_ids=None):
        exclude_item_ids = set(exclude_item_ids or [])
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        query_tf = Counter(query_tokens)
        scores = defaultdict(float)
        candidate_ids = set()

        for token in query_tf:
            candidate_ids.update(self.inverted_index.get(token, []))

        if not candidate_ids:
            return []

        for item_id in candidate_ids:
            if item_id in exclude_item_ids:
                continue

            doc_tf = self.doc_tf[item_id]
            doc_len = self.doc_lengths[item_id]

            for token, qf in query_tf.items():
                tf = doc_tf.get(token, 0)
                if tf == 0:
                    continue

                idf = self._idf(token)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avg_doc_len, 1e-6))
                scores[item_id] += qf * idf * numerator / denominator

        return heapq.nlargest(
            topk,
            ((item_id, float(score)) for item_id, score in scores.items() if score > 0),
            key=lambda x: x[1]
        )

    def score_item(self, query, item_id):
        item_id = str(item_id)
        if item_id not in self.doc_tokens:
            return 0.0

        query_tokens = tokenize(query)
        if not query_tokens:
            return 0.0

        query_tf = Counter(query_tokens)
        doc_tf = self.doc_tf[item_id]
        doc_len = self.doc_lengths[item_id]
        score = 0.0

        for token, qf in query_tf.items():
            tf = doc_tf.get(token, 0)
            if tf == 0:
                continue

            idf = self._idf(token)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avg_doc_len, 1e-6))
            score += qf * idf * numerator / denominator

        return float(score)

    def build_user_query(self, history_df, user_id, max_len=5):
        user_df = history_df[history_df["user_id"] == user_id].sort_values("timestamp", ascending=False)
        item_ids = [str(item_id) for item_id in user_df["item_id"].tolist()[:max_len]]
        titles = [self.item_titles[item_id] for item_id in item_ids if item_id in self.item_titles]
        return " ".join(titles), item_ids

    def recommend_for_user(self, history_df, user_id, topk=10, max_history=5):
        query, history_items = self.build_user_query(history_df, user_id, max_len=max_history)
        return self.search(query, topk=topk, exclude_item_ids=history_items)
