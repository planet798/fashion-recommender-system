import json
import os

import numpy as np


TEXT_DIM = 384
IMAGE_DIM = 512


def normalize(vec):
    vec = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def weighted_fusion(text_vec, image_vec, alpha=1.0, beta=1.0, normalize_final=True):
    weighted_text = normalize(text_vec) * float(alpha)
    weighted_image = normalize(image_vec) * float(beta)
    fused = np.concatenate([weighted_text, weighted_image]).astype(np.float32)
    if normalize_final:
        fused = normalize(fused)
    return fused


def build_text_query_fusion(text_vec, alpha=1.0, beta=1.0, normalize_final=True):
    return weighted_fusion(
        text_vec=text_vec,
        image_vec=np.zeros(IMAGE_DIM, dtype=np.float32),
        alpha=alpha,
        beta=beta,
        normalize_final=normalize_final,
    )


def load_fusion_config(feature_path=None, config_path=None):
    candidates = []
    if config_path:
        candidates.append(config_path)
    if feature_path:
        candidates.append(os.path.join(os.path.dirname(feature_path), "fusion_config.json"))

    for path in candidates:
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return {
                "alpha": float(payload.get("alpha", payload.get("text_weight", 1.0))),
                "beta": float(payload.get("beta", payload.get("image_weight", 1.0))),
                "normalize_final": bool(payload.get("normalize_final", True)),
            }

    return {
        "alpha": 1.0,
        "beta": 1.0,
        "normalize_final": True,
    }
