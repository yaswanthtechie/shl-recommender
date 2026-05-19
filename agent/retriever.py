import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

_EMBEDDINGS_DIR = Path(__file__).parent.parent / "embeddings"
_MODEL_NAME = "all-MiniLM-L6-v2"

# Load once at import time — never per-request
_chunks: list[dict] = json.loads(
    (_EMBEDDINGS_DIR / "chunks.json").read_text(encoding="utf-8")
)
_index: faiss.Index = faiss.read_index(str(_EMBEDDINGS_DIR / "index.faiss"))
_model: SentenceTransformer = SentenceTransformer(_MODEL_NAME)

# Pre-built set of every valid URL in the catalog
VALID_URLS: set[str] = {item["link"] for item in _chunks}


def retrieve(query: str, top_k: int = 15) -> list[dict]:
    query_vec = _model.encode([query], normalize_embeddings=True).astype("float32")
    _, indices = _index.search(query_vec, top_k)
    return [_chunks[i] for i in indices[0] if i < len(_chunks)]


def retrieve_by_names(names: list[str]) -> list[dict]:
    lower_names = {n.lower() for n in names}
    return [item for item in _chunks if item["name"].lower() in lower_names]
