# Fashion Recommender System V2

A production-oriented multimodal fashion recommendation system.

## Goal

Build a complete two-stage recommendation pipeline using:

- user behaviour modelling
- multimodal product representations
- candidate retrieval
- personalized ranking
- cold-start recommendation
- model serving and containerized deployment

## Planned Architecture

User History
    |
    v
User Tower
    |
    v
Two-Tower Retrieval
    |
    v
FAISS Candidate Retrieval
    |
    v
DIN Ranking
    |
    v
Top-K Recommendations

Product representations will combine:

- Item ID
- Category
- CLIP text embedding
- CLIP image embedding

## Roadmap

### Phase 1 — Data
- Amazon Reviews 2023
- Data cleaning
- Temporal split
- EDA
- Evaluation protocol

### Phase 2 — Baselines
- Popularity
- ItemCF
- LightGCN

### Phase 3 — Retrieval
- Two-Tower
- Negative sampling
- FAISS ANN

### Phase 4 — Multimodal
- CLIP text encoder
- CLIP vision encoder
- Multimodal fusion

### Phase 5 — Ranking
- DIN

### Phase 6 — Cold Start
- New-item recommendation
- Multimodal cold-start evaluation

### Phase 7 — Serving
- FastAPI
- Streamlit
- Docker

## Status

V2 reconstruction in progress.
