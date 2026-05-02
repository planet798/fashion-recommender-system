import os
import tempfile

os.environ["TEMP"] = "D:\\TEMP"
os.environ["TMP"] = "D:\\TEMP"
os.environ["TMPDIR"] = "D:\\TEMP"
tempfile.tempdir = "D:\\TEMP"

# Work around PyTorch mega-cache duplicate artifact registration under hot-reload.
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, CLIPModel

from fusion_utils import build_text_query_fusion, load_fusion_config
from hybrid_recall_v3 import HybridRecallV3
from ranking_model import PairwiseFeatureRanker
from recall_faiss_v2 import FaissRecallV2
from user_model import UserInterestModel
from user_auth import UserAuth
from user_feedback import UserFeedback, get_all_interests, chinese_to_english_tag, REAL_INTEREST_TAGS, TAG_KEYWORDS, tag_to_display
from interest_updater import InterestUpdater
from cold_start import ColdStartRecommender
from data_config import config
from llm_service import LLMService
from llm_query_understanding import QueryUnderstanding


def _patch_torch_megacache_registration_once():
    """Make CacheArtifact registration idempotent for Streamlit reruns/hot reload."""
    try:
        from torch.compiler import _cache as torch_cache
    except Exception:
        return

    marker = "_recommendation_safe_register_patched"
    if getattr(torch_cache.CacheArtifactFactory, marker, False):
        return

    original_register = torch_cache.CacheArtifactFactory.register.__func__

    def _safe_register(cls, artifact_cls):
        artifact_type_key = artifact_cls.type()
        existing = cls._artifact_types.get(artifact_type_key)
        if existing is not None:
            return artifact_cls
        return original_register(cls, artifact_cls)

    torch_cache.CacheArtifactFactory.register = classmethod(_safe_register)
    setattr(torch_cache.CacheArtifactFactory, marker, True)


_patch_torch_megacache_registration_once()


def inject_global_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Noto+Sans+SC:wght@300;400;500;700&family=Share+Tech+Mono&display=swap');

        :root {
            --bg-deep: #060b14;
            --bg-panel: rgba(10, 18, 38, 0.85);
            --bg-card: rgba(14, 24, 48, 0.78);
            --text-primary: #e0f0ff;
            --text-secondary: #7eb8da;
            --accent-cyan: #00e5ff;
            --accent-blue: #2979ff;
            --accent-purple: #b388ff;
            --glow-cyan: 0 0 18px rgba(0, 229, 255, 0.25);
            --glow-blue: 0 0 22px rgba(41, 121, 255, 0.22);
            --border-glow: rgba(0, 229, 255, 0.18);
            --border-subtle: rgba(100, 180, 220, 0.10);
            --font-display: "Orbitron", "Noto Sans SC", sans-serif;
            --font-body: "Noto Sans SC", sans-serif;
            --font-mono: "Share Tech Mono", monospace;
        }

        /* === GLOBAL BACKGROUND === */
        .stApp {
            background:
                radial-gradient(ellipse at 20% 10%, rgba(0, 200, 255, 0.06), transparent 35%),
                radial-gradient(ellipse at 80% 5%, rgba(120, 80, 255, 0.05), transparent 30%),
                radial-gradient(ellipse at 50% 60%, rgba(0, 160, 220, 0.03), transparent 40%),
                linear-gradient(175deg, #060b14 0%, #0a1226 40%, #0c162e 100%);
            color: var(--text-primary);
            font-family: var(--font-body);
        }

        .stApp::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            z-index: 0;
            background-image:
                linear-gradient(rgba(0, 229, 255, 0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0, 229, 255, 0.03) 1px, transparent 1px);
            background-size: 48px 48px;
            mask-image: radial-gradient(ellipse at 50% 0%, rgba(0,0,0,0.5), transparent 70%);
        }

        /* Hide Streamlit default header / toolbar */
        header[data-testid="stHeader"] { display: none !important; }
        [data-testid="stToolbar"] { display: none !important; }
        #MainMenu { display: none !important; }
        [data-testid="stDecoration"] { display: none !important; }
        [data-testid="stDeployButton"] { display: none !important; }
        .stApp > header { display: none !important; }

        .block-container {
            padding-top: 0.3rem;
            padding-bottom: 2.4rem;
            max-width: 1340px;
            position: relative;
            z-index: 1;
        }

        h1, h2, h3, h4 {
            font-family: var(--font-display) !important;
            letter-spacing: 0.03em;
        }

        .stMarkdown, .stCaption, .stTextInput label, .stSelectbox label {
            font-family: var(--font-body) !important;
        }

        /* === SIDEBAR === */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(6, 14, 30, 0.98), rgba(10, 22, 48, 0.97));
            border-right: 1px solid rgba(0, 229, 255, 0.10);
            box-shadow: 2px 0 30px rgba(0, 0, 0, 0.4);
        }

        [data-testid="stSidebar"] * {
            color: #d0e8ff;
        }

        [data-testid="stSidebar"] h2 {
            font-family: var(--font-display) !important;
            color: var(--accent-cyan) !important;
            text-shadow: 0 0 12px rgba(0, 229, 255, 0.3);
            letter-spacing: 0.06em;
        }

        [data-testid="stSidebar"] .stSlider, [data-testid="stSidebar"] .stCheckbox, [data-testid="stSidebar"] .stRadio {
            background: rgba(0, 229, 255, 0.03);
            border: 1px solid rgba(0, 229, 255, 0.08);
            border-radius: 12px;
            padding: 0.5rem 0.7rem;
        }

        [data-testid="stSidebar"] .stSlider:hover, [data-testid="stSidebar"] .stCheckbox:hover {
            border-color: rgba(0, 229, 255, 0.2);
            box-shadow: 0 0 14px rgba(0, 229, 255, 0.06);
        }

        /* === HERO SHELL === */
        .hero-shell {
            position: relative;
            background:
                radial-gradient(ellipse at 30% 20%, rgba(0, 200, 255, 0.10), transparent 40%),
                radial-gradient(ellipse at 80% 10%, rgba(120, 80, 255, 0.08), transparent 35%),
                linear-gradient(160deg, rgba(8, 20, 44, 0.92), rgba(12, 30, 60, 0.85));
            border: 1px solid var(--border-glow);
            border-radius: 20px;
            padding: 1.2rem 1.4rem;
            margin-bottom: 0.9rem;
            overflow: hidden;
            box-shadow: var(--glow-cyan), inset 0 1px 0 rgba(0, 229, 255, 0.06);
        }

        .hero-shell::after {
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(0, 229, 255, 0.5), transparent);
            animation: scan-line 3s ease-in-out infinite;
        }

        @keyframes scan-line {
            0%, 100% { opacity: 0.3; }
            50% { opacity: 1; }
        }

        .hero-eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.22rem 0.65rem;
            border-radius: 4px;
            background: rgba(0, 229, 255, 0.08);
            border: 1px solid rgba(0, 229, 255, 0.2);
            color: var(--accent-cyan);
            font-family: var(--font-mono);
            font-size: 0.7rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }

        .hero-eyebrow::before {
            content: ">";
            color: var(--accent-cyan);
            animation: blink-cursor 1s step-end infinite;
        }

        @keyframes blink-cursor {
            0%, 100% { opacity: 1; }
            50% { opacity: 0; }
        }

        .hero-title {
            font-family: var(--font-display);
            font-size: 2.1rem;
            line-height: 1.12;
            font-weight: 700;
            margin: 0.6rem 0 0.3rem;
            max-width: 11em;
            background: linear-gradient(135deg, #e0f0ff, #80d8ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .hero-text {
            max-width: 44rem;
            color: #7eb8da;
            font-size: 0.9rem;
            line-height: 1.65;
            margin-bottom: 0.7rem;
        }

        .hero-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.6rem;
            margin-top: 0.7rem;
        }

        .hero-chip {
            border-radius: 12px;
            background: rgba(0, 229, 255, 0.04);
            border: 1px solid rgba(0, 229, 255, 0.10);
            padding: 0.65rem 0.8rem;
            transition: all 0.3s ease;
        }

        .hero-chip:hover {
            border-color: rgba(0, 229, 255, 0.25);
            box-shadow: 0 0 16px rgba(0, 229, 255, 0.08);
            background: rgba(0, 229, 255, 0.06);
        }

        .hero-chip strong {
            display: block;
            font-family: var(--font-display);
            font-size: 0.82rem;
            font-weight: 600;
            color: var(--accent-cyan);
            margin-bottom: 0.1rem;
            letter-spacing: 0.03em;
        }

        .hero-chip p, .hero-chip span {
            color: #7eb8da;
            font-size: 0.8rem;
        }

        /* === SECTION CARDS === */
        .section-card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            border-radius: 16px;
            padding: 0.85rem 0.95rem;
            backdrop-filter: blur(20px);
        }

        .section-title {
            font-family: var(--font-display);
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 0.15rem;
            color: var(--accent-cyan);
            letter-spacing: 0.04em;
        }

        .section-text {
            color: #7eb8da;
            font-size: 0.88rem;
            line-height: 1.6;
        }

        /* === EXAMPLE PILLS === */
        .example-shell {
            display: grid;
            gap: 0.5rem;
            margin-top: 0.7rem;
        }

        .example-pill {
            background: rgba(0, 229, 255, 0.04);
            border: 1px solid rgba(0, 229, 255, 0.10);
            border-radius: 10px;
            padding: 0.55rem 0.7rem;
            color: #90caf9;
            font-size: 0.85rem;
            transition: all 0.25s ease;
            cursor: pointer;
        }

        .example-pill:hover {
            border-color: rgba(0, 229, 255, 0.3);
            background: rgba(0, 229, 255, 0.07);
            box-shadow: 0 0 12px rgba(0, 229, 255, 0.06);
        }

        /* === RESULT CARDS === */
        .result-card {
            background: var(--bg-card);
            border: 1px solid rgba(0, 229, 255, 0.10);
            border-radius: 14px;
            padding: 0.8rem 0.9rem;
            box-shadow: 0 6px 24px rgba(0, 0, 0, 0.3);
            margin-bottom: 0.65rem;
            transition: all 0.3s ease;
        }

        .result-card:hover {
            border-color: rgba(0, 229, 255, 0.22);
            box-shadow: 0 8px 28px rgba(0, 229, 255, 0.06), 0 6px 24px rgba(0, 0, 0, 0.4);
        }

        .result-meta {
            display: flex;
            gap: 0.4rem;
            flex-wrap: wrap;
            margin: 0.4rem 0 0.15rem;
        }

        .metric-tag {
            display: inline-flex;
            align-items: center;
            padding: 0.18rem 0.55rem;
            border-radius: 4px;
            background: rgba(0, 229, 255, 0.07);
            border: 1px solid rgba(0, 229, 255, 0.12);
            color: var(--accent-cyan);
            font-family: var(--font-mono);
            font-size: 0.7rem;
            letter-spacing: 0.03em;
        }

        /* === SIDE PANEL === */
        .side-panel {
            background: linear-gradient(180deg, rgba(10, 20, 44, 0.9), rgba(14, 26, 54, 0.85));
            border: 1px solid rgba(0, 229, 255, 0.10);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.35);
            border-radius: 18px;
            padding: 0.9rem 0.9rem 0.5rem;
        }

        /* === CHAT MESSAGE === */
        .stChatMessage {
            background: rgba(14, 24, 50, 0.7);
            border: 1px solid rgba(0, 229, 255, 0.10);
            border-radius: 14px;
            padding: 0.55rem 0.75rem;
        }

        /* === INPUT FIELDS === */
        .stTextInput input, .stChatInput textarea, .stSelectbox div[data-baseweb="select"] > div,
        .stMultiSelect div[data-baseweb="select"] > div {
            border-radius: 10px !important;
            border: 1px solid rgba(0, 229, 255, 0.14) !important;
            background: rgba(10, 20, 40, 0.8) !important;
            color: #e0f0ff !important;
            font-family: var(--font-body) !important;
        }

        .stTextInput input:focus, .stChatInput textarea:focus {
            border-color: var(--accent-cyan) !important;
            box-shadow: 0 0 14px rgba(0, 229, 255, 0.12) !important;
        }

        .stTextInput input::placeholder, .stChatInput textarea::placeholder {
            color: rgba(120, 180, 220, 0.4) !important;
        }

        /* === BUTTONS === */
        .stButton > button {
            border-radius: 8px;
            border: 1px solid rgba(0, 229, 255, 0.25) !important;
            background: linear-gradient(135deg, rgba(0, 180, 220, 0.15), rgba(0, 100, 200, 0.12)) !important;
            color: var(--accent-cyan) !important;
            font-family: var(--font-display) !important;
            font-weight: 600 !important;
            font-size: 0.82rem !important;
            letter-spacing: 0.05em !important;
            box-shadow: 0 0 14px rgba(0, 229, 255, 0.08);
            transition: all 0.3s ease;
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, rgba(0, 200, 240, 0.25), rgba(0, 140, 230, 0.2)) !important;
            border-color: var(--accent-cyan) !important;
            box-shadow: 0 0 24px rgba(0, 229, 255, 0.2), 0 0 8px rgba(0, 229, 255, 0.1) !important;
            color: #ffffff !important;
        }

        /* === AUTH / LOGIN === */
        .auth-shell {
            /* Hidden: this decorative placeholder shell pushes the real login block downward */
            display: none;
        }

        .auth-hero {
            position: relative;
            overflow: hidden;
            background:
                radial-gradient(ellipse at 20% 20%, rgba(0, 200, 255, 0.10), transparent 30%),
                radial-gradient(ellipse at 85% 15%, rgba(120, 80, 255, 0.08), transparent 28%),
                linear-gradient(160deg, rgba(6, 16, 36, 0.95), rgba(10, 28, 56, 0.9));
            color: #e0f0ff;
            border-radius: 22px;
            border: 1px solid var(--border-glow);
            box-shadow: var(--glow-cyan), 0 18px 50px rgba(0, 0, 0, 0.4);
            padding: 1.4rem;
        }

        .auth-hero::before {
            content: "";
            position: absolute;
            inset: 0;
            background-image:
                linear-gradient(rgba(0, 229, 255, 0.04) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0, 229, 255, 0.04) 1px, transparent 1px);
            background-size: 40px 40px;
            mask-image: linear-gradient(180deg, rgba(255,255,255,0.5), transparent 90%);
        }

        .auth-panel {
            background: rgba(10, 20, 42, 0.88);
            border: 1px solid rgba(0, 229, 255, 0.12);
            border-radius: 22px;
            box-shadow: 0 18px 44px rgba(0, 0, 0, 0.4);
            padding: 1.1rem;
            backdrop-filter: blur(22px);
        }

        .auth-kicker, .status-kicker {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.22rem 0.6rem;
            border-radius: 4px;
            background: rgba(0, 229, 255, 0.07);
            border: 1px solid rgba(0, 229, 255, 0.15);
            color: var(--accent-cyan);
            font-family: var(--font-mono);
            font-size: 0.7rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }

        .auth-kicker::before, .status-kicker::before {
            content: "●";
            font-size: 0.5rem;
            animation: pulse-dot 2s ease-in-out infinite;
        }

        @keyframes pulse-dot {
            0%, 100% { opacity: 1; text-shadow: 0 0 6px var(--accent-cyan); }
            50% { opacity: 0.3; text-shadow: none; }
        }

        .auth-title {
            font-family: var(--font-display);
            font-size: 2.4rem;
            line-height: 1.08;
            font-weight: 700;
            margin: 0.8rem 0 0.5rem;
            max-width: 11ch;
            background: linear-gradient(135deg, #e0f0ff, #80d8ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .auth-text {
            color: rgba(180, 215, 240, 0.78);
            max-width: 40rem;
            line-height: 1.7;
            font-size: 0.9rem;
        }

        .auth-stat-grid, .compare-grid, .profile-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.65rem;
            margin-top: 0.9rem;
        }

        .auth-stat, .compare-card, .profile-card, .signal-card {
            border-radius: 14px;
            padding: 0.8rem;
            border: 1px solid rgba(0, 229, 255, 0.10);
        }

        .auth-stat {
            background: rgba(0, 229, 255, 0.04);
        }

        .auth-stat:hover {
            border-color: rgba(0, 229, 255, 0.2);
            box-shadow: 0 0 14px rgba(0, 229, 255, 0.05);
        }

        .auth-stat strong, .compare-card strong, .profile-card strong {
            display: block;
            font-family: var(--font-display);
            font-size: 0.9rem;
            font-weight: 600;
            margin-bottom: 0.15rem;
            color: var(--accent-cyan);
            letter-spacing: 0.03em;
        }

        .auth-stat p, .auth-stat span, .compare-card p, .profile-card p {
            color: #90caf9;
            font-size: 0.8rem;
        }

        .login-divider {
            height: 1px;
            margin: 0.9rem 0;
            background: linear-gradient(90deg, transparent, rgba(0, 229, 255, 0.3), transparent);
        }

        .lab-shell {
            background: linear-gradient(180deg, rgba(10, 20, 42, 0.88), rgba(14, 26, 54, 0.82));
            border-radius: 18px;
            border: 1px solid rgba(0, 229, 255, 0.10);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.35);
            padding: 0.9rem;
        }

        .reason-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem;
            margin: 0.5rem 0 0.1rem;
        }

        .reason-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.2rem;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            background: rgba(0, 229, 255, 0.07);
            border: 1px solid rgba(0, 229, 255, 0.12);
            color: var(--accent-cyan);
            font-family: var(--font-mono);
            font-size: 0.7rem;
            letter-spacing: 0.02em;
        }

        .compare-card {
            background: rgba(10, 20, 42, 0.7);
            border: 1px solid rgba(0, 229, 255, 0.10);
        }

        .compare-card:hover {
            border-color: rgba(0, 229, 255, 0.2);
            box-shadow: 0 0 14px rgba(0, 229, 255, 0.05);
        }

        .compare-rank {
            font-family: var(--font-display);
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--accent-cyan);
            text-shadow: 0 0 10px rgba(0, 229, 255, 0.15);
        }

        .compare-label {
            color: #7eb8da;
            font-size: 0.78rem;
        }

        /* === BANNER === */
        .stage-banner {
            background: linear-gradient(135deg, rgba(6, 16, 36, 0.95), rgba(8, 50, 100, 0.85));
            color: #e0f0ff;
            border-radius: 16px;
            padding: 0.85rem 1rem;
            border: 1px solid rgba(0, 229, 255, 0.12);
            box-shadow: 0 12px 36px rgba(0, 0, 0, 0.35), var(--glow-cyan);
        }

        .stage-banner p {
            color: rgba(200, 225, 245, 0.78);
            margin: 0.35rem 0 0;
            line-height: 1.6;
        }

        /* === METRICS / STREAMLIT OVERRIDES === */
        [data-testid="stMetricValue"] {
            font-family: var(--font-display) !important;
            color: var(--accent-cyan) !important;
        }

        [data-testid="stMetricLabel"] {
            font-family: var(--font-mono) !important;
            color: #7eb8da !important;
            font-size: 0.75rem !important;
            letter-spacing: 0.04em !important;
        }

        [data-testid="stTabs"] button {
            font-family: var(--font-display) !important;
            letter-spacing: 0.04em !important;
            color: #7eb8da !important;
        }

        [data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--accent-cyan) !important;
            border-bottom-color: var(--accent-cyan) !important;
        }

        /* Tab content area background */
        [data-testid="stTabs"] [role="tabpanel"] {
            background: transparent !important;
        }

        /* === SCROLLBAR === */
        ::-webkit-scrollbar {
            width: 6px;
        }
        ::-webkit-scrollbar-track {
            background: rgba(0, 0, 0, 0.2);
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(0, 229, 255, 0.15);
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(0, 229, 255, 0.25);
        }

        /* === RESPONSIVE === */
        @media (max-width: 900px) {
            .auth-shell,
            .auth-stat-grid,
            .compare-grid,
            .profile-grid {
                grid-template-columns: 1fr;
            }
            .hero-title {
                font-size: 1.7rem;
            }
            .hero-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    st.markdown(
        """
        <div class="hero-shell">
            <span class="hero-eyebrow">Multimodal Recommendation Engine v3.0</span>
            <div class="hero-title">多模态时尚商品推荐系统</div>
            <div class="hero-grid">
                <div class="hero-chip">
                    <strong>◆ 语义理解</strong>
                    <span>LLM 意图解析 · 品类 / 风格 / 场景提取</span>
                </div>
                <div class="hero-chip">
                    <strong>◆ 多模态融合</strong>
                    <span>CLIP 视觉编码 · 文本-图像联合检索</span>
                </div>
                <div class="hero-chip">
                    <strong>◆ 个性化重排</strong>
                    <span>兴趣标签注入 · 实时反馈权重调整</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_signal_console():
    components.html(
        """
        <div style="position:relative;height:220px;border-radius:16px;overflow:hidden;
                    border:1px solid rgba(0,229,255,.20);
                    background:linear-gradient(160deg,#060e1a,#0a1a36);
                    box-shadow: 0 0 30px rgba(0,229,255,.06), inset 0 1px 0 rgba(0,229,255,.04);">
          <canvas id="signal-canvas" style="position:absolute;inset:0;width:100%;height:100%;"></canvas>
          <div style="position:absolute;inset:0;padding:18px 20px;color:#e0f0ff;
                      font-family:'Share Tech Mono','Noto Sans SC',monospace;">
            <div style="font-size:10px;letter-spacing:.2em;text-transform:uppercase;color:#00e5ff;
                        text-shadow:0 0 10px rgba(0,229,255,.25);">
              ● Live Signal Console
            </div>
            <div id="signal-text" style="font-size:24px;font-weight:700;margin-top:18px;
                        font-family:'Share Tech Mono',monospace;color:#e0f0ff;
                        text-shadow:0 0 14px rgba(0,229,255,.15);">INITIALIZING...</div>
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:26px;">
              <div style="padding:8px 10px;border-radius:8px;background:rgba(0,229,255,.03);
                          border:1px solid rgba(0,229,255,.08);">
                <div style="font-size:10px;color:#00e5ff;letter-spacing:.08em;">INTENT</div>
                <div id="metric-1" style="font-size:18px;font-weight:700;font-family:'Share Tech Mono',monospace;
                            color:#7ec8ff;text-shadow:0 0 8px rgba(0,200,255,.2);">0</div>
              </div>
              <div style="padding:8px 10px;border-radius:8px;background:rgba(0,229,255,.03);
                          border:1px solid rgba(0,229,255,.08);">
                <div style="font-size:10px;color:#00e5ff;letter-spacing:.08em;">RECALL</div>
                <div id="metric-2" style="font-size:18px;font-weight:700;font-family:'Share Tech Mono',monospace;
                            color:#7ec8ff;text-shadow:0 0 8px rgba(0,200,255,.2);">0</div>
              </div>
              <div style="padding:8px 10px;border-radius:8px;background:rgba(0,229,255,.03);
                          border:1px solid rgba(0,229,255,.08);">
                <div style="font-size:10px;color:#00e5ff;letter-spacing:.08em;">PERSONA</div>
                <div id="metric-3" style="font-size:18px;font-weight:700;font-family:'Share Tech Mono',monospace;
                            color:#7ec8ff;text-shadow:0 0 8px rgba(0,200,255,.2);">0</div>
              </div>
            </div>
          </div>
        </div>
        <script>
          const canvas = document.getElementById("signal-canvas");
          const ctx = canvas.getContext("2d");
          const dpr = window.devicePixelRatio || 1;
          const w = canvas.clientWidth;
          const h = canvas.clientHeight;
          canvas.width = w * dpr;
          canvas.height = h * dpr;
          ctx.scale(dpr, dpr);
          const dots = Array.from({length: 50}, () => ({
            x: Math.random() * w,
            y: Math.random() * h,
            vx: (Math.random() - .5) * .5,
            vy: (Math.random() - .5) * .5,
          }));
          function draw() {
            ctx.clearRect(0, 0, w, h);
            // grid lines
            ctx.strokeStyle = "rgba(0,229,255,0.04)";
            ctx.lineWidth = 0.5;
            for (let x = 0; x < w; x += 32) {
              ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
            }
            for (let y = 0; y < h; y += 32) {
              ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
            }
            // scanning line effect
            const t = Date.now() / 3000;
            const scanY = (t % 1) * h * 1.5 - h * 0.25;
            ctx.strokeStyle = "rgba(0,229,255,0.06)";
            ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(0, scanY); ctx.lineTo(w, scanY); ctx.stroke();
            // particles
            dots.forEach((d, i) => {
              d.x += d.vx; d.y += d.vy;
              if (d.x < 0 || d.x > w) d.vx *= -1;
              if (d.y < 0 || d.y > h) d.vy *= -1;
              // glow particle
              const gradient = ctx.createRadialGradient(d.x, d.y, 0, d.x, d.y, 2.5);
              gradient.addColorStop(0, "rgba(0,229,255,0.9)");
              gradient.addColorStop(0.4, "rgba(0,180,240,0.4)");
              gradient.addColorStop(1, "rgba(0,100,200,0)");
              ctx.fillStyle = gradient;
              ctx.beginPath(); ctx.arc(d.x, d.y, 2.5, 0, Math.PI * 2); ctx.fill();
              // connections
              dots.slice(i + 1).forEach((n) => {
                const dx = d.x - n.x, dy = d.y - n.y;
                const dist = Math.hypot(dx, dy);
                if (dist < 80) {
                  ctx.strokeStyle = `rgba(0,229,255,${0.10 - dist / 1000})`;
                  ctx.lineWidth = 0.5;
                  ctx.beginPath(); ctx.moveTo(d.x, d.y); ctx.lineTo(n.x, n.y); ctx.stroke();
                }
              });
            });
            requestAnimationFrame(draw);
          }
          draw();
          const phrases = [
            "NEURAL PERSONA ENGINE ONLINE",
            "MULTIMODAL SIGNAL FUSION ACTIVE",
            "REAL-TIME PERSONALIZATION LIVE"
          ];
          let idx = 0;
          const el = document.getElementById("signal-text");
          setInterval(() => {
            idx = (idx + 1) % phrases.length;
            el.textContent = phrases[idx];
          }, 2000);
          let m1 = 12, m2 = 64, m3 = 4;
          setInterval(() => {
            m1 = 10 + Math.floor(Math.random() * 8);
            m2 = 48 + Math.floor(Math.random() * 32);
            m3 = 3 + Math.floor(Math.random() * 4);
            document.getElementById("metric-1").textContent = m1 + " TAGS";
            document.getElementById("metric-2").textContent = m2 + " ITEMS";
            document.getElementById("metric-3").textContent = m3 + " HEADS";
          }, 1500);
        </script>
        """,
        height=280,
        scrolling=False,
    )


user_auth = UserAuth()
user_feedback = UserFeedback()
interest_updater = InterestUpdater()


def apply_demo_query_params():
    try:
        demo_user = st.query_params.get("demo_user")
        if demo_user and not st.session_state.get("current_user"):
            success, _, user_id = user_auth.login(str(demo_user))
            if success:
                st.session_state.current_user = user_id

        demo_query = st.query_params.get("demo_lab_query")
        demo_run_lab = str(st.query_params.get("demo_run_lab", "0")).lower() in {"1", "true", "yes"}
        if demo_query:
            st.session_state["lab_query"] = str(demo_query)

        return {
            "demo_user": str(demo_user) if demo_user else None,
            "demo_view": str(st.query_params.get("demo_view", "")).strip().lower(),
            "demo_lab_query": str(demo_query) if demo_query else None,
            "demo_run_lab": demo_run_lab,
        }
    except Exception:
        return {
            "demo_user": None,
            "demo_view": "",
            "demo_lab_query": None,
            "demo_run_lab": False,
        }


def ensure_session_defaults():
    defaults = {
        "current_user": None,
        "chat_history": [],
        "recommendations": [],
        "diverse_recommendations": [],
        "personalization_lab_results": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_login_screen():
    st.markdown(
        """
        <div class="auth-shell">
            <div class="auth-hero">
                <span class="auth-kicker">Personalization Gateway</span>
                <div class="auth-title">多模态个性化推荐工作台</div>
                <div class="auth-stat-grid">
                    <div class="auth-stat">
                        <strong>◆ 身份认证</strong>
                        <span>登录后绑定兴趣档案与反馈信号</span>
                    </div>
                    <div class="auth-stat">
                        <strong>◆ 画像可见</strong>
                        <span>兴趣标签与权重实时透明展示</span>
                    </div>
                    <div class="auth-stat">
                        <strong>◆ 对照演示</strong>
                        <span>通用排序 vs 个性化排序同屏对比</span>
                    </div>
                </div>
            </div>
            <div class="auth-panel"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1.18, 0.82], gap="large")
    with left_col:
        render_signal_console()

    with right_col:
        st.markdown('<div class="auth-panel">', unsafe_allow_html=True)
        st.markdown('<span class="status-kicker">Access Control</span>', unsafe_allow_html=True)
        auth_tab1, auth_tab2 = st.tabs(["登录", "注册"])
        with auth_tab1:
            login_nickname = st.text_input("昵称", key="login_nickname", placeholder="例如 demo_sports")
            login_password = st.text_input("密码", key="login_password", type="password")
            if st.button("进入推荐工作台", key="login_btn", use_container_width=True):
                if login_nickname and login_password:
                    success, msg, user_id = user_auth.login(login_nickname, login_password)
                    if success:
                        st.session_state.current_user = user_id
                        st.rerun()
                    else:
                        st.error(msg)
                else:
                    st.warning("请输入昵称和密码")

        with auth_tab2:
            reg_nickname = st.text_input("昵称", key="reg_nickname", placeholder="例如 demo_beauty")
            reg_password = st.text_input("密码", key="reg_password", type="password")
            reg_password2 = st.text_input("确认密码", key="reg_password2", type="password")
            all_interests = get_all_interests()
            interest_options = [tag for tags in all_interests.values() for tag in tags]
            selected_interests = st.multiselect("初始兴趣标签", options=interest_options, default=[])
            if st.button("创建账号并进入", key="register_btn", use_container_width=True):
                if not reg_nickname:
                    st.warning("请输入昵称")
                elif len(reg_nickname) < 3:
                    st.warning("昵称至少 3 个字符")
                elif not reg_password:
                    st.warning("请输入密码")
                elif len(reg_password) < 6:
                    st.warning("密码至少 6 个字符")
                elif reg_password != reg_password2:
                    st.warning("两次输入的密码不一致")
                else:
                    success, msg, user_id = user_auth.register(
                        nickname=reg_nickname,
                        password=reg_password,
                        interests=selected_interests
                    )
                    if success:
                        st.session_state.current_user = user_id
                        st.rerun()
                    else:
                        st.error(msg)
        st.markdown("</div>", unsafe_allow_html=True)


# =================================
# 初始化应用
# =================================
st.set_page_config(layout="wide")
inject_global_styles()
ensure_session_defaults()
demo_context = apply_demo_query_params()

if not st.session_state.current_user:
    render_login_screen()
    st.stop()


# =================================
# 懒加载模型 (按需加载，减少启动时间)
# =================================
@st.cache_resource
def load_text_model():
    return SentenceTransformer("models/paraphrase-MiniLM-L3-v2")


@st.cache_resource
def load_clip_model():
    clip_path = "models/clip-vit-base-patch32"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(clip_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(clip_path, use_fast=True)
    return model, tokenizer, device


@st.cache_resource
def load_user_model():
    return UserInterestModel(
        history_path=config.user_history_csv,
        text_feature_path=config.text_features,
        image_feature_path=config.image_features
    )


@st.cache_resource
def load_faiss_recall(_user_model, _text_model=None):
    return FaissRecallV2(
        user_model=_user_model,
        text_index_path=config.text_index,
        text_ids_path=config.text_ids,
        image_index_path=config.image_index,
        image_ids_path=config.image_ids,
        text_encoder=_text_model
    )


@st.cache_resource
def load_hybrid_recall(_faiss_recall, _text_encoder=None):
    text_enc = _text_encoder or getattr(_faiss_recall, 'text_encoder', None)
    return HybridRecallV3(
        faiss_recall=_faiss_recall,
        bm25=None,
        mode="faiss_only",
        text_encoder=text_enc
    )


@st.cache_resource
def load_ranking_model():
    model_path = config.ranking_model
    if not os.path.exists(model_path):
        return None
    try:
        return PairwiseFeatureRanker(
            model_path=model_path,
            text_feature_path=config.text_features,
            image_feature_path=config.image_features
        ).load()
    except (ValueError, FileNotFoundError) as exc:
        return None


@st.cache_resource
def load_cold_recommender(_items_count, _features_count):
    items = pd.read_csv(config.items_csv)
    items["item_id"] = items["item_id"].astype(str)
    text_features = {
        str(item_id): np.array(vec, dtype=np.float32)
        for item_id, vec in np.load(config.text_features, allow_pickle=True).item().items()
    }
    return ColdStartRecommender(
        items_df=items,
        text_features=text_features,
        interest_tags=get_all_interests()
    )


@st.cache_data
def load_llm_service():
    from data_config import DataConfig
    backend = DataConfig.LLM_BACKEND
    api_key = DataConfig.ZHIPU_API_KEY or os.environ.get("ZHIPU_API_KEY", "")
    local_path = DataConfig.LOCAL_LLM_MODEL_PATH
    service = LLMService(backend=backend, api_key=api_key, local_model_path=local_path)
    return service


@st.cache_data
def load_data():
    items = pd.read_csv(config.items_csv)
    items["item_id"] = items["item_id"].astype(str)
    text_features = {
        str(item_id): np.array(vec, dtype=np.float32)
        for item_id, vec in np.load(config.text_features, allow_pickle=True).item().items()
    }
    image_features = {
        str(item_id): np.array(vec, dtype=np.float32)
        for item_id, vec in np.load(config.image_features, allow_pickle=True).item().items()
    }
    multimodal_features = {
        str(item_id): np.array(vec, dtype=np.float32)
        for item_id, vec in np.load(config.multimodal_features, allow_pickle=True).item().items()
    }
    fusion_config = load_fusion_config(feature_path=config.multimodal_features)

    valid_image_ids = set()
    if os.path.isdir(config.images_dir):
        valid_image_ids = set(
            f.replace(".jpg", "") for f in os.listdir(config.images_dir) if f.endswith(".jpg")
        )

    return items, text_features, image_features, multimodal_features, fusion_config, valid_image_ids


def get_models():
    """按需获取模型，使用session_state缓存避免重复加载"""
    if "models_loaded" not in st.session_state:
        st.session_state.models_loaded = {}

    cached = st.session_state.models_loaded

    if "data" not in cached:
        with st.spinner("Loading data..."):
            cached["data"] = load_data()

    items, text_features, image_features, multimodal_features, fusion_config, valid_image_ids = cached["data"]

    if "text_model" not in cached:
        with st.spinner("Loading text encoder..."):
            cached["text_model"] = load_text_model()
    if "clip" not in cached:
        cached["clip"] = None  # defer: 600MB CLIP loads on first query
    if "user_model" not in cached:
        with st.spinner("Loading user model..."):
            cached["user_model"] = load_user_model()
    if "faiss_recall" not in cached:
        text_m = cached.get("text_model")
        with st.spinner("Loading FAISS index..."):
            cached["faiss_recall"] = load_faiss_recall(cached["user_model"], text_m)
    if "hybrid" not in cached:
        text_m = cached.get("text_model")
        with st.spinner("Loading hybrid recall..."):
            cached["hybrid"] = load_hybrid_recall(cached["faiss_recall"], text_m)
    if "ranking_model" not in cached:
        cached["ranking_model"] = None  # defer: load on first rerank call
    if "llm_service" not in cached:
        with st.spinner("Loading LLM service..."):
            cached["llm_service"] = load_llm_service()
    if "query_understanding" not in cached:
        cached["query_understanding"] = QueryUnderstanding(cached["llm_service"])

    return (
        cached["text_model"], cached["clip"],
        cached["user_model"], None, cached["hybrid"],
        cached["ranking_model"], cached["faiss_recall"],
        items, text_features, image_features, multimodal_features, fusion_config,
        valid_image_ids,
        cached["llm_service"], cached["query_understanding"], None
    )


# =================================
# 左侧控制栏
# =================================
st.sidebar.markdown("## ◆ CONTROL PANEL")
mode = "Smart Personalization"
st.session_state.current_mode = mode

recall_k = st.sidebar.slider(
    "Recall TopK",
    10, 200, 50
)

rank_k = st.sidebar.slider(
    "Final TopK",
    1, 20, 8
)

use_prompt_enhance = st.sidebar.checkbox(
    "Use Prompt Enhancement",
    value=True
)

show_debug_controls = st.sidebar.checkbox(
    "Show Debug Controls",
    value=False
)

if show_debug_controls:
    recall_strategy = st.sidebar.radio(
        "Our Recall Strategy",
        ["FAISS", "Hybrid"]
    )
    user_rerank_strategy = st.sidebar.radio(
        "User Rerank Strategy",
        ["Similarity", "Learned Ranker"]
    )
else:
    recall_strategy = "Hybrid"
    user_rerank_strategy = "Learned Ranker"

with st.sidebar.expander("LLM Config", expanded=False):
    from data_config import DataConfig
    llm_backend = st.selectbox(
        "LLM Backend",
        ["zhipu", "local_qwen"],
        index=0 if DataConfig.LLM_BACKEND == "zhipu" else 1,
        help="zhipu: GLM-4-Flash | local_qwen: Qwen2.5-1.5B"
    )
    if llm_backend == "zhipu":
        zhipu_api_key = st.text_input(
            "Zhipu API Key",
            value=DataConfig.ZHIPU_API_KEY,
            type="password",
            help="从 https://open.bigmodel.cn 获取免费API Key"
        )
        if zhipu_api_key != DataConfig.ZHIPU_API_KEY:
            DataConfig.ZHIPU_API_KEY = zhipu_api_key
            if "llm_service" in st.session_state.get("models_loaded", {}):
                del st.session_state.models_loaded["llm_service"]
                del st.session_state.models_loaded["query_understanding"]
    else:
        st.caption("本地Qwen2.5-1.5B需要~3.2GB显存")
        st.caption("与CLIP共用GPU，可能较慢")


# =================================
# 按需加载模型 (在sidebar之后，首次渲染即触发)
# =================================
(text_model, clip_model, user_model,
 _, hybrid_recall, ranking_model,
 faiss_recall,
 items, text_features, image_features, multimodal_features, fusion_config,
 valid_image_ids,
 llm_service, query_understanding, bert_retriever) = get_models()

cold_recommender = load_cold_recommender(len(items), len(text_features))

if clip_model is not None:
    clip_model_obj, clip_tokenizer, device = clip_model
else:
    clip_model_obj, clip_tokenizer, device = None, None, "cpu"

if llm_service is not None and llm_service.is_available():
    st.sidebar.success(f"LLM: {llm_service.backend} ready")
else:
    st.sidebar.warning("LLM unavailable - configure API Key in LLM Config")


# =================================
# 显示推荐结果工具函数
# =================================
def normalize(vec):
    vec = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def display_score(score):
    return max(0.0, min(1.0, (score + 1.0) / 2.0))


def get_item_title(item_id):
    item_id = str(item_id)
    row = items.loc[items["item_id"] == item_id, "title"]
    if len(row) > 0:
        return row.values[0]
    return f"Item {item_id}"


def get_feature(store, item_id):
    item_id = str(item_id)
    return store.get(item_id)


def show_item(item_id, score, extra_text=None, use_display_transform=True, score_label="Score"):
    item_id = str(item_id)
    img_path = os.path.join(config.images_dir, f"{item_id}.jpg")

    if os.path.exists(img_path):
        img = Image.open(img_path)
        st.image(img, width=160)
    else:
        st.write("❌ Image Missing")

    st.write(get_item_title(item_id))
    st.write(f"Item ID: {item_id}")
    shown_score = display_score(score) if use_display_transform else score
    st.write(f"{score_label}: {shown_score:.3f}")

    if extra_text is not None:
        st.caption(extra_text)

    st.markdown("---")


def render_section_intro(title, text):
    st.markdown(
        f"""
        <div class="section-card">
            <div class="section-title">{title}</div>
            <div class="section-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =================================
# Query 检索与重排
# =================================
def prompt_enhance_query(query):
    return (
        f"This is a fashion product search query about {query}. "
        f"Please focus on product category, style, use case, and fine-grained visual semantics."
    )


def _extract_model_embedding(output):
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "text_embeds") and output.text_embeds is not None:
        return output.text_embeds
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        return output.last_hidden_state[:, 0, :]
    raise TypeError(f"Unsupported embedding output type: {type(output)!r}")


def encode_clip_text(text):
    # Lazy-load CLIP on first call (600MB model)
    cached = st.session_state.get("models_loaded", {})
    if cached.get("clip") is None:
        with st.spinner("Loading CLIP model..."):
            cached["clip"] = load_clip_model()

    clip_data = cached.get("clip")
    if clip_data is None:
        return np.zeros(512, dtype=np.float32)

    _clip_model_obj, _clip_tokenizer, _device = clip_data
    inputs = _clip_tokenizer(
        [text],
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    inputs = {key: value.to(_device) for key, value in inputs.items()}
    with torch.no_grad():
        text_emb = _clip_model_obj.get_text_features(**inputs)
        text_emb = _extract_model_embedding(text_emb)
    text_emb = text_emb.detach().cpu().numpy()[0]
    return normalize(text_emb)


def build_text_candidates(query_text, recall_topk):
    if recall_strategy == "Hybrid":
        ours_candidates = hybrid_recall.recall_by_text(
            query_text,
            topk=recall_topk
        )
    else:
        ours_candidates = faiss_recall.recall_by_text(query_text, topk=recall_topk)

    results = [(item_id, float(score)) for item_id, score in ours_candidates]
    return results


def rerank_text_query_ours(query_text, candidates, topk=8, alpha=0.6, beta=0.4, use_prompt=True):
    if use_prompt:
        enhanced_query = prompt_enhance_query(query_text)
    else:
        enhanced_query = query_text
    query_text_vec = text_model.encode(enhanced_query)
    query_text_vec = normalize(query_text_vec)
    query_clip_vec = encode_clip_text(enhanced_query)
    rows = []
    for candidate in candidates:
        item_id = str(candidate[0]) if isinstance(candidate, (tuple, list)) else str(candidate)
        recall_score = float(candidate[1]) if isinstance(candidate, (tuple, list)) and len(candidate) > 1 else 0.0
        item_text = get_feature(text_features, item_id)
        item_img = get_feature(image_features, item_id)
        if item_text is None or item_img is None:
            continue
        item_text_vec = normalize(item_text)
        item_img_vec = normalize(item_img)
        text_score = cosine_similarity([query_text_vec], [item_text_vec])[0][0]
        clip_score = cosine_similarity([query_clip_vec], [item_img_vec])[0][0]
        rank_score = alpha * text_score + beta * clip_score
        rows.append({
            "item_id": item_id,
            "rank_score": float(rank_score),
            "text_score": float(text_score),
            "clip_score": float(clip_score),
            "recall_score": float(recall_score)
        })
    rows.sort(key=lambda x: x["rank_score"], reverse=True)
    rows = rows[:topk]
    for row in rows:
        display_score_raw = (
            0.5 * row["text_score"] +
            0.3 * row["clip_score"] +
            0.2 * row["recall_score"]
        )
        row["display_score"] = float(display_score_raw)
    return rows


def baseline_rank_user(user_id, topk=8):
    return faiss_recall.recall_by_user(user_id, topk=topk)


def rerank_user_ours(user_id, candidates, topk=8):
    user_vec = normalize(user_model.build_user_vector(user_id))
    results = []
    for item_id, recall_score in candidates:
        item_id = str(item_id)
        item_vec = get_feature(text_features, item_id)
        if item_vec is None:
            continue
        item_vec = normalize(item_vec)
        final_score = cosine_similarity([user_vec], [item_vec])[0][0]
        results.append({
            "item_id": item_id,
            "final_score": float(final_score),
            "recall_score": float(recall_score)
        })
    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results[:topk]


def rerank_user_learned(user_id, candidates, topk=8, ground_truth=None):
    # Lazy-load ranking model on first call
    cached = st.session_state.get("models_loaded", {})
    if cached.get("ranking_model") is None:
        with st.spinner("Loading ranking model..."):
            cached["ranking_model"] = load_ranking_model()
    _ranking_model = cached.get("ranking_model")
    if _ranking_model is None:
        return None
    return _ranking_model.rerank_candidates(
        user_id=user_id,
        candidates=candidates,
        user_model=user_model,
        bm25=None,
        topk=topk,
        return_features=True,
        ground_truth=ground_truth
    )


def rerank_text_query_v2(query_text, candidates, topk=8, use_prompt=True):
    if use_prompt:
        enhanced_query = prompt_enhance_query(query_text)
    else:
        enhanced_query = query_text
    query_text_vec = normalize(text_model.encode(enhanced_query))
    query_clip_vec = encode_clip_text(enhanced_query)
    query_fusion_vec = build_text_query_fusion(
        text_vec=query_text_vec,
        alpha=fusion_config["alpha"],
        beta=fusion_config["beta"],
        normalize_final=fusion_config["normalize_final"],
    )
    query_fusion_vec = normalize(query_fusion_vec)

    bert_score_dict = {}
    if bert_retriever is not None:
        bert_results = bert_retriever.search(query_text, topk=max(topk * 5, 50))
        bert_score_dict = {item_id: score for item_id, score in bert_results}

    rows = []
    for candidate in candidates:
        item_id = str(candidate[0]) if isinstance(candidate, (tuple, list)) else str(candidate)
        recall_score = float(candidate[1]) if isinstance(candidate, (tuple, list)) and len(candidate) > 1 else 0.0
        item_text = get_feature(text_features, item_id)
        item_img = get_feature(image_features, item_id)
        item_multi = get_feature(multimodal_features, item_id)
        if item_text is None or item_img is None or item_multi is None:
            continue
        item_text_vec = normalize(item_text)
        item_img_vec = normalize(item_img)
        item_multi_vec = normalize(item_multi)
        text_score = cosine_similarity([query_text_vec], [item_text_vec])[0][0]
        clip_score = cosine_similarity([query_clip_vec], [item_img_vec])[0][0]
        fusion_score = cosine_similarity([query_fusion_vec], [item_multi_vec])[0][0]
        bert_score = bert_score_dict.get(item_id, 0.0)

        signal_strengths = np.array([text_score, clip_score, fusion_score, bert_score])
        signal_weights_base = np.array([0.30, 0.20, 0.25, 0.25])
        attention_weights = signal_weights_base * (0.5 + 0.5 * signal_strengths)
        attention_weights = attention_weights / attention_weights.sum()

        rank_score = (
            attention_weights[0] * text_score +
            attention_weights[1] * clip_score +
            attention_weights[2] * fusion_score +
            attention_weights[3] * bert_score
        )
        rows.append({
            "item_id": item_id,
            "rank_score": float(rank_score),
            "text_score": float(text_score),
            "clip_score": float(clip_score),
            "fusion_score": float(fusion_score),
            "bert_score": float(bert_score),
            "recall_score": float(recall_score),
            "attention_w": attention_weights.tolist()
        })
    rows.sort(key=lambda x: x["rank_score"], reverse=True)
    rows = rows[:topk]
    for row in rows:
        row["display_score"] = float(
            0.30 * row["text_score"] +
            0.20 * row["clip_score"] +
            0.25 * row["fusion_score"] +
            0.25 * row["bert_score"]
        )
    return rows


# =================================
# Smart Chat 主界面
# =================================
def get_user_interest_weights(user_auth_obj, user_id):
    if not user_id:
        return {}
    interests = user_auth_obj.get_user_interests(user_id)
    return {row["tag"]: float(row["weight"]) for row in interests}


def build_negative_interest_profile(user_id, cold_recommender_obj, limit=60):
    if not user_id:
        return {}

    negative_scores = {}
    feedback_rows = user_feedback.get_user_feedback(user_id, limit=limit)
    for feedback in feedback_rows:
        rating = feedback.get("rating")
        liked = feedback.get("liked")
        is_negative = liked == 0 or liked is False or (rating is not None and rating <= 2)
        if not is_negative:
            continue

        for tag, score in cold_recommender_obj.get_item_interest_matches(
            str(feedback["item_id"]),
            topn=3,
            min_score=0.0,
        ):
            negative_scores[tag] = negative_scores.get(tag, 0.0) + float(score)

    if not negative_scores:
        return {}

    max_score = max(negative_scores.values()) or 1.0
    return {tag: score / max_score for tag, score in negative_scores.items()}


def get_persona_snapshot(user_id, topn=5):
    interests = user_auth.get_user_interests(user_id) if user_id else []
    return interests[:topn]


def build_item_reason_pills(user_id, item_id, row=None):
    pills = []
    interest_weights = get_user_interest_weights(user_auth, user_id) if user_id else {}
    if interest_weights:
        matches = cold_recommender.get_item_interest_matches(
            item_id,
            interest_weights=interest_weights,
            topn=2,
            min_score=0.08,
        )
        if matches:
            pills.append("匹配 " + " / ".join(tag_to_display(tag) for tag, _ in matches))

    if isinstance(row, dict):
        personalization_score = float(row.get("personalization_score", 0.0))
        negative_score = float(row.get("negative_score", 0.0))
        if personalization_score > 0.12:
            pills.append(f"个性化提升 {personalization_score:.2f}")
        if negative_score > 0.18:
            pills.append(f"已抑制负向偏好 {negative_score:.2f}")

    return pills[:3]


def personalize_chat_results(user_id, base_results, cold_recommender_obj):
    if not user_id or not base_results:
        return base_results

    # === STEP 1: Build explicit blacklist from disliked / low-rated items ===
    blacklist = set()
    liked_set = set()
    feedback_rows = user_feedback.get_user_feedback(user_id, limit=200)
    for fb in feedback_rows:
        item_id = str(fb["item_id"])
        rating = fb.get("rating")
        liked = fb.get("liked")
        is_negative = liked == 0 or liked is False or (rating is not None and rating <= 2)
        if is_negative:
            blacklist.add(item_id)
        if liked == 1 or liked is True or (rating is not None and rating >= 4):
            liked_set.add(item_id)

    # Remove blacklisted items from candidates (they were explicitly rejected)
    filtered = [r for r in base_results if str(r["item_id"]) not in blacklist]
    if not filtered:
        filtered = base_results

    # === STEP 2: Build interest score map from interest-based recommendations ===
    interest_weights = get_user_interest_weights(user_auth, user_id)
    if not interest_weights:
        return filtered[:rank_k]

    interest_results = cold_recommender_obj.generate_from_interest_weights(
        interest_weights=interest_weights,
        topk=max(recall_k, rank_k * 6)
    )

    interest_score_map = {}
    if interest_results:
        max_interest_score = max(score for _, score in interest_results) or 1.0
        interest_score_map = {
            str(item_id): float(score) / max_interest_score
            for item_id, score in interest_results
        }

    # Boost items the user previously liked
    for liked_id in liked_set:
        if liked_id in interest_score_map:
            interest_score_map[liked_id] = max(interest_score_map[liked_id], 0.8)
        else:
            interest_score_map[liked_id] = 0.7

    # === STEP 3: Build negative score map ===
    negative_interest_weights = build_negative_interest_profile(user_id, cold_recommender_obj)
    negative_score_map = {}
    if negative_interest_weights:
        raw_negative_scores = {}
        for row in filtered:
            iid = str(row["item_id"])
            raw_negative_scores[iid] = cold_recommender_obj.score_item_against_interests(
                iid, negative_interest_weights
            )
        max_negative_score = max(raw_negative_scores.values()) or 1.0
        negative_score_map = {
            item_id: float(score) / max_negative_score
            for item_id, score in raw_negative_scores.items()
            if max_negative_score > 0
        }

    # === STEP 4: Rerank with stronger personalization ===
    reranked = []
    for row in filtered:
        item_id = str(row["item_id"])
        row_copy = dict(row)
        base_score = float(row_copy.get("rank_score", row_copy.get("display_score", 0.0)))
        interest_score = interest_score_map.get(item_id, 0.0)
        negative_score = negative_score_map.get(item_id, 0.0)

        row_copy["personalization_score"] = interest_score
        row_copy["negative_score"] = negative_score

        # Strong personalization: 50% interest, 35% base, heavy negative penalty
        if negative_score > 0.3:
            # Items match user's disliked interests → heavy demotion
            row_copy["final_score"] = 0.30 * base_score + 0.50 * interest_score - 0.35 * negative_score
        elif interest_score > 0.1:
            # Items match user's interests → boost
            row_copy["final_score"] = 0.35 * base_score + 0.50 * interest_score - 0.10 * negative_score + 0.05
        else:
            row_copy["final_score"] = 0.45 * base_score + 0.35 * interest_score - 0.10 * negative_score

        row_copy["display_score"] = (
            0.40 * float(row_copy.get("display_score", base_score)) +
            0.45 * interest_score -
            0.10 * negative_score
        )
        reranked.append(row_copy)

    reranked.sort(key=lambda x: x["final_score"], reverse=True)

    # === STEP 5: Inject top interest-based items not in base results ===
    base_ids = {str(r["item_id"]) for r in reranked}
    inject_count = max(2, rank_k // 3)  # inject at least 2 interest-driven items
    injected = []
    for item_id, interest_score in interest_results:
        sid = str(item_id)
        if sid in blacklist:
            continue
        if sid not in base_ids:
            row_copy = {
                "item_id": sid,
                "rank_score": interest_score * 0.5,
                "recall_score": 0.0,
                "text_score": 0.0,
                "clip_score": 0.0,
                "fusion_score": 0.0,
                "display_score": interest_score * 0.6,
                "personalization_score": interest_score,
                "negative_score": 0.0,
                "final_score": interest_score * 0.65 + 0.05,
            }
            injected.append(row_copy)
            base_ids.add(sid)
            if len(injected) >= inject_count:
                break

    if injected:
        reranked.extend(injected)

    reranked.sort(key=lambda x: x["final_score"], reverse=True)
    return reranked[:rank_k]


def render_feedback_actions(user_id, item_id, prefix):
    if not user_id:
        return

    if "liked_items" not in st.session_state:
        st.session_state.liked_items = set()
    if "disliked_items" not in st.session_state:
        st.session_state.disliked_items = set()

    action_col1, action_col2 = st.columns(2)
    with action_col1:
        is_liked = item_id in st.session_state.liked_items
        like_label = "已喜欢" if is_liked else "喜欢"
        if st.button(like_label, key=f"{prefix}_like_{item_id}", use_container_width=True):
            user_feedback.add_feedback(user_id, item_id, liked=True)
            interest_updater.update_interest_weights(user_id, [{"item_id": item_id, "liked": True}])
            st.session_state.liked_items.add(item_id)
            st.session_state.disliked_items.discard(item_id)
            st.rerun()
    with action_col2:
        is_disliked = item_id in st.session_state.disliked_items
        dislike_label = "已点踩" if is_disliked else "点踩"
        if st.button(dislike_label, key=f"{prefix}_dislike_{item_id}", use_container_width=True):
            user_feedback.add_feedback(user_id, item_id, liked=False, rating=1)
            interest_updater.update_interest_weights(user_id, [{"item_id": item_id, "liked": False, "rating": 1}])
            st.session_state.disliked_items.add(item_id)
            st.session_state.liked_items.discard(item_id)
            st.rerun()

    rating_key = f"{prefix}_rating_{item_id}"
    current_rating = st.session_state.get(rating_key, 0)
    st.caption(f"当前评分: {current_rating}/5")
    star_cols = st.columns(5)
    for i, col in enumerate(star_cols, 1):
        with col:
            star_label = "★" if i <= current_rating else "☆"
            if st.button(star_label, key=f"{prefix}_star_{i}_{item_id}", use_container_width=True):
                st.session_state[rating_key] = i
                user_feedback.add_feedback(user_id, item_id, liked=i >= 4, rating=i)
                if i >= 4:
                    interest_updater.update_interest_weights(user_id, [{"item_id": item_id, "liked": True, "rating": i}])
                elif i <= 2:
                    interest_updater.update_interest_weights(user_id, [{"item_id": item_id, "liked": False, "rating": i}])
                st.rerun()


def render_result_cards(results, user_id=None, prefix="result", show_feedback=True):
    if not results:
        return

    for idx, row in enumerate(results):
        item_id = str(row["item_id"]) if isinstance(row, dict) else str(row[0])
        img_path = os.path.join(config.images_dir, f"{item_id}.jpg")

        col_left, col_right = st.columns([1, 3])
        with col_left:
            if os.path.exists(img_path):
                st.image(Image.open(img_path), width=150)
            else:
                st.write("Image missing")

        with col_right:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.write(f"**{get_item_title(item_id)}**")
            st.caption(f"Item ID: {item_id}")

            if isinstance(row, dict):
                recall_score = float(row.get("recall_score", 0.0))
                fusion_score = float(row.get("fusion_score", 0.0))
                personalization_score = float(row.get("personalization_score", 0.0))
                negative_score = float(row.get("negative_score", 0.0))
                st.markdown(
                    (
                        '<div class="result-meta">'
                        f'<span class="metric-tag">Recall {recall_score:.3f}</span>'
                        f'<span class="metric-tag">Fusion {fusion_score:.3f}</span>'
                        f'<span class="metric-tag">Personalization {personalization_score:.3f}</span>'
                        f'<span class="metric-tag">Negative {negative_score:.3f}</span>'
                        '</div>'
                    ),
                    unsafe_allow_html=True,
                )

                reason_pills = build_item_reason_pills(user_id, item_id, row)
                if reason_pills:
                    st.markdown(
                        '<div class="reason-row">' +
                        "".join(f'<span class="reason-pill">{pill}</span>' for pill in reason_pills) +
                        '</div>',
                        unsafe_allow_html=True,
                    )

            if show_feedback:
                render_feedback_actions(user_id, item_id, f"{prefix}_{idx}")
            st.markdown("</div>", unsafe_allow_html=True)
        st.divider()


def run_personalization_lab(query_text, user_id):
    enhanced_query = query_text
    intent = None
    if query_understanding is not None and llm_service is not None and llm_service.is_available():
        intent = query_understanding.parse_intent(query_text)
        if intent:
            enhanced_query = query_understanding.build_search_query(intent)

    recall_results = build_text_candidates(enhanced_query, recall_k)
    generic_results = rerank_text_query_v2(
        query_text=enhanced_query,
        candidates=recall_results,
        topk=rank_k,
        use_prompt=use_prompt_enhance,
    )
    personalized_results = personalize_chat_results(user_id, generic_results, cold_recommender)

    generic_rank = {str(row["item_id"]): idx + 1 for idx, row in enumerate(generic_results)}
    personalized_rank = {str(row["item_id"]): idx + 1 for idx, row in enumerate(personalized_results)}
    changed_items = sum(
        1 for item_id, rank in personalized_rank.items()
        if generic_rank.get(item_id) != rank
    )
    overlap_count = len(set(generic_rank.keys()) & set(personalized_rank.keys()))

    return {
        "query": query_text,
        "enhanced_query": enhanced_query,
        "generic_results": generic_results,
        "personalized_results": personalized_results,
        "changed_items": changed_items,
        "overlap_count": overlap_count,
        "top1_changed": (
            bool(generic_results and personalized_results) and
            generic_results[0]["item_id"] != personalized_results[0]["item_id"]
        ),
    }


def render_personalization_lab():
    st.markdown(
        """
        <div class="lab-shell">
            <span class="status-kicker">Visible Personalization Lab</span>
            <div class="login-divider"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    demo_query = st.text_input(
        "演示查询",
        value="送女生的夏季礼物",
        key="lab_query",
        placeholder="例如：适合通勤的轻薄外套",
    )
    if demo_context.get("demo_run_lab") and demo_query:
        current_lab = st.session_state.get("personalization_lab_results") or {}
        if current_lab.get("query") != demo_query:
            st.session_state.personalization_lab_results = run_personalization_lab(
                demo_query,
                st.session_state.current_user,
            )
    if st.button("生成显性个性化对比", key="run_lab_compare", use_container_width=True):
        st.session_state.personalization_lab_results = run_personalization_lab(
            demo_query,
            st.session_state.current_user,
        )

    lab_results = st.session_state.get("personalization_lab_results")
    if not lab_results:
        return

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("排序变化商品数", lab_results["changed_items"])
    metric_col2.metric("Top1 是否变化", "是" if lab_results["top1_changed"] else "否")
    metric_col3.metric("结果重合数", lab_results["overlap_count"])
    st.caption(f"检索词: {lab_results['enhanced_query']}")

    left_col, right_col = st.columns(2, gap="large")
    with left_col:
        render_section_intro("◆ 通用排序", "基于查询语义与多模态相似度的基础排序")
        render_result_cards(lab_results["generic_results"][:4], prefix="lab_generic", show_feedback=False)
    with right_col:
        render_section_intro("◆ 个性化排序", "叠加兴趣标签、实时反馈与负向抑制信号")
        render_result_cards(
            lab_results["personalized_results"][:4],
            user_id=st.session_state.current_user,
            prefix="lab_personalized",
        )


def render_persona_snapshot():
    persona_rows = get_persona_snapshot(st.session_state.current_user, topn=6)
    if not persona_rows:
        st.caption("暂无画像数据，请设置兴趣标签或进行交互")
        return

    st.markdown('<div class="profile-grid">', unsafe_allow_html=True)
    cols = st.columns(3)
    for idx, row in enumerate(persona_rows):
        with cols[idx % 3]:
            st.markdown(
                f"""
                <div class="profile-card">
                    <strong>{tag_to_display(row['tag'])}</strong>
                    <div class="compare-label">类别: {tag_to_display(row['category'])}</div>
                    <div class="compare-rank">{row['weight']:.2f}</div>
                    <div class="compare-label">兴趣权重</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown('</div>', unsafe_allow_html=True)


def render_auth_panel():
    user_info = user_auth.get_user_info(st.session_state.current_user)
    if user_info:
        st.write(f"**当前用户**: {user_info['nickname']}")
        st.caption(f"用户ID: {st.session_state.current_user}")

    all_interests = get_all_interests()
    interest_options = [tag for tags in all_interests.values() for tag in tags]
    current_interests = user_auth.get_user_interests(st.session_state.current_user)
    current_tags = [row["tag"] for row in current_interests if row["tag"] in interest_options]
    selected_tags = st.multiselect(
        "兴趣标签",
        options=interest_options,
        default=current_tags,
        key="edit_interests"
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("保存兴趣", key="save_interests", use_container_width=True):
            if user_auth.update_interests(st.session_state.current_user, selected_tags):
                st.success("兴趣标签已更新")
                st.rerun()
            else:
                st.error("兴趣标签更新失败")
    with col2:
        if st.button("退出登录", key="logout_btn", use_container_width=True):
            st.session_state.current_user = None
            st.session_state.chat_history = []
            st.session_state.personalization_lab_results = None
            st.rerun()

st.sidebar.markdown(
    """
    <div class="section-card">
        <div class="section-title">◆ QUERY EXAMPLES</div>
        <div class="example-shell">
            <div class="example-pill">冬天户外跑步穿的外套</div>
            <div class="example-pill">约会穿的裙子</div>
            <div class="example-pill">碎花长裙</div>
            <div class="example-pill">夏天穿的凉鞋</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

render_hero()
st.markdown(
    """
    <div class="stage-banner">
        <span class="auth-kicker" style="background:rgba(0,229,255,0.1);color:#00e5ff;">System Status</span>
        <p>多模态召回 · 个性化重排 · 实时反馈闭环 ｜ 所有子系统运行中</p>
    </div>
    """,
    unsafe_allow_html=True,
)

main_col, side_col = st.columns([2.1, 1], gap="large")

with side_col:
    st.markdown('<div class="side-panel">', unsafe_allow_html=True)
    render_section_intro("账户与兴趣", "管理兴趣标签与交互偏好")
    render_auth_panel()

    if st.session_state.current_user:
        user_id = st.session_state.current_user
        stats = user_feedback.get_interaction_stats(user_id)
        stat_col1, stat_col2 = st.columns(2)
        stat_col1.metric("曝光", stats["total_exposures"])
        stat_col2.metric("喜欢", stats["likes"])

        if st.button("刷新兴趣推荐", key="refresh_interest_recs", use_container_width=True):
            interest_weights = get_user_interest_weights(user_auth, user_id)
            if interest_weights:
                st.session_state.recommendations = cold_recommender.generate_from_interest_weights(
                    interest_weights=interest_weights,
                    topk=6
                )
                if st.session_state.recommendations:
                    user_feedback.record_exposure(
                        user_id=user_id,
                        item_ids=[item_id for item_id, _ in st.session_state.recommendations]
                    )
            else:
                st.warning("当前用户还没有兴趣标签")

        if st.button("探索新品", key="explore_btn", use_container_width=True):
            interest_weights = get_user_interest_weights(user_auth, user_id)
            diverse_seed = list(interest_weights.keys()) if interest_weights else ["时尚", "运动户外", "家居生活"]
            st.session_state.diverse_recommendations = cold_recommender.get_diverse_recommendations(
                user_interests=diverse_seed,
                topk=6,
                diversity_weight=0.4
            )

        if st.session_state.recommendations:
            st.markdown("**兴趣推荐结果**")
            for item_id, score in st.session_state.recommendations[:4]:
                st.caption(f"{get_item_title(item_id)} | {score:.3f}")

        if st.session_state.diverse_recommendations:
            st.markdown("**新品探索结果**")
            for row in st.session_state.diverse_recommendations[:4]:
                st.caption(f"{get_item_title(row['item_id'])}")

        st.markdown("**最近反馈**")
        recent_feedback = user_feedback.get_user_feedback(user_id, limit=5)
        if recent_feedback:
            for fb in recent_feedback:
                rating_text = f" | 评分 {fb['rating']}" if fb.get("rating") else ""
                st.caption(f"{get_item_title(fb['item_id'])}{rating_text}")
        else:
            st.caption("暂无反馈记录")
    st.markdown("</div>", unsafe_allow_html=True)

with main_col:
    demo_view = demo_context.get("demo_view", "")
    if demo_view == "lab":
        render_section_intro("显性个性化实验台", "通用排序与个性化排序同屏对比")
        render_personalization_lab()
        st.stop()
    if demo_view == "profile":
        render_section_intro("当前用户画像", "当前账号的兴趣标签与权重分布")
        render_persona_snapshot()
        st.stop()

    search_tab, lab_tab, profile_tab = st.tabs(["智能搜索", "显性个性化实验台", "用户画像"])

    with search_tab:
        render_section_intro("对话式个性化推荐", "LLM 意图理解 → 多模态召回 → 个性化重排")
        if llm_service is None or query_understanding is None:
            st.error("LLM 服务未加载，请先在侧边栏配置。")
        elif not llm_service.is_available():
            st.warning("当前 LLM 不可用，请在侧边栏补充 API Key。")
        else:
            chat_container = st.container()
            with chat_container:
                for msg in st.session_state.chat_history:
                    role = msg["role"]
                    content = msg["content"]
                    if role == "user":
                        with st.chat_message("user"):
                            st.write(content)
                    elif role == "assistant_results":
                        with st.chat_message("assistant"):
                            st.write(content.get("summary", ""))
                            if content.get("intent_summary"):
                                st.caption(content["intent_summary"])
                            results = content.get("results", [])
                            if results:
                                render_result_cards(
                                    results,
                                    user_id=st.session_state.current_user,
                                    prefix=content.get("result_key", "chat")
                                )

            user_input = st.chat_input("描述你想要的商品、风格、场景或预算...")

            if user_input:
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                with chat_container:
                    with st.chat_message("user"):
                        st.write(user_input)

                    with st.chat_message("assistant"):
                        with st.spinner("正在理解你的需求..."):
                            intent = query_understanding.parse_intent(user_input)

                        enhanced_query = user_input
                        intent_summary = ""
                        if intent:
                            enhanced_query = query_understanding.build_search_query(intent)
                            attrs_display = [f"{k}: {v}" for k, v in intent.get("attributes", {}).items() if v]
                            intent_summary = f"类别: {intent.get('category', 'fashion')}"
                            if attrs_display:
                                intent_summary += f" | 属性: {', '.join(attrs_display)}"
                            if intent.get("occasion") and intent["occasion"] != "general":
                                intent_summary += f" | 场景: {intent['occasion']}"
                            intent_summary += f" | 检索词: {enhanced_query}"
                            st.caption(intent_summary)

                        with st.spinner("正在进行多模态召回与重排..."):
                            recall_results = build_text_candidates(enhanced_query, recall_k)
                            chat_results = rerank_text_query_v2(
                                query_text=enhanced_query,
                                candidates=recall_results,
                                topk=rank_k,
                                use_prompt=use_prompt_enhance
                            )
                            chat_results = personalize_chat_results(
                                st.session_state.current_user,
                                chat_results,
                                cold_recommender
                            )

                        if st.session_state.current_user and chat_results:
                            user_feedback.record_exposure(
                                user_id=st.session_state.current_user,
                                item_ids=[row["item_id"] for row in chat_results]
                            )

                        chat_response = query_understanding.generate_chat_response(
                            user_input, intent, len(chat_results)
                        )
                        st.write(chat_response)

                        if chat_results:
                            render_result_cards(
                                chat_results,
                                user_id=st.session_state.current_user,
                                prefix=f"chat_{len(st.session_state.chat_history)}"
                            )

                        st.session_state.chat_history.append({
                            "role": "assistant_results",
                            "content": {
                                "summary": chat_response,
                                "intent_summary": intent_summary,
                                "results": chat_results,
                                "result_key": f"chat_history_{len(st.session_state.chat_history)}"
                            }
                        })

    with lab_tab:
        render_personalization_lab()

    with profile_tab:
        render_section_intro("当前用户画像", "高权重兴趣标签与类别分布")
        render_persona_snapshot()
