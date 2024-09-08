import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_HOME = Path(
    os.getenv("PROJECT_HOME", str(Path.home() / "scale-free-visual-cortex")),
)
cache_path = PROJECT_HOME / "cache"

MANUSCRIPT_HOME = PROJECT_HOME / "manuscript"
MANUSCRIPT_HOME.mkdir(parents=True, exist_ok=True)

BONNER_CACHING_HOME = Path(
    os.getenv("BONNER_CACHING_HOME", str(cache_path / "bonner-caching")),
)
BONNER_MODELS_HOME = Path(
    os.getenv("BONNER_MODELS_HOME", str(cache_path / "bonner-models")),
)
BONNER_DATASETS_HOME = Path(
    os.getenv("BONNER_DATASETS_HOME", str(cache_path / "bonner-datasets")),
)
BONNER_BRAINIO_HOME = Path(
    os.getenv("BONNER_BRAINIO_HOME", str(cache_path / "bonner-brainio")),
)
