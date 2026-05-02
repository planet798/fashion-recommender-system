from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

# Keep Amazon data isolated from the current toy pipeline.
AMAZON_ROOT = PROJECT_ROOT / "datasets" / "amazon_reviews23"
AMAZON_RAW = AMAZON_ROOT / "raw"
AMAZON_PROCESSED = AMAZON_ROOT / "processed"
AMAZON_IMAGES = AMAZON_ROOT / "images"


def ensure_amazon_dirs():
    for path in [AMAZON_ROOT, AMAZON_RAW, AMAZON_PROCESSED, AMAZON_IMAGES]:
        path.mkdir(parents=True, exist_ok=True)
