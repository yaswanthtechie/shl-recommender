import json
import sys
import faiss
from pathlib import Path

# Import the shared ONNX embedder from the build script.
# Works whether called from project root or as a submodule.
sys.path.insert(0, str(Path(__file__).parent.parent))
from embeddings.build_index import _embed, _get_session_and_tokenizer

_EMBEDDINGS_DIR = Path(__file__).parent.parent / "embeddings"

# Load once at import time — never per-request
_chunks: list[dict] = json.loads(
    (_EMBEDDINGS_DIR / "chunks.json").read_text(encoding="utf-8")
)
_index: faiss.Index = faiss.read_index(str(_EMBEDDINGS_DIR / "index.faiss"))

# Warm up ONNX session at import time (downloads model if not cached)
_get_session_and_tokenizer()

# Pre-built set of every valid URL in the catalog
VALID_URLS: set[str] = {item["link"] for item in _chunks}


def retrieve(query: str, top_k: int = 15) -> list[dict]:
    query_vec = _embed([query])
    _, indices = _index.search(query_vec, top_k)
    return [_chunks[i] for i in indices[0] if i < len(_chunks)]


def retrieve_by_names(names: list[str]) -> list[dict]:
    lower_names = {n.lower() for n in names}
    return [item for item in _chunks if item["name"].lower() in lower_names]
