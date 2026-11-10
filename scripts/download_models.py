"""Download the NER models once (so the first run of the app is fast / works offline later).

    python scripts/download_models.py            # all models
    python scripts/download_models.py en_ner ar_zero
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from masker.config import MODELS  # noqa: E402
from masker.pipeline import ModelProvider  # noqa: E402

if __name__ == "__main__":
    roles = sys.argv[1:] or ["en_ner", "en_zero", "ar_zero"]
    provider = ModelProvider("cpu")
    for role in roles:
        cfg = MODELS[role]
        print(f"-> {cfg['id']} ({cfg['size']}) ...", flush=True)
        provider.get(role).load()
        print("   ok")
    print("done. Models are cached in ~/.cache/huggingface")
