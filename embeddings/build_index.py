import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).parent.parent
CATALOG_PATH = ROOT / "data" / "catalog.json"
INDEX_PATH = Path(__file__).parent / "index.faiss"
CHUNKS_PATH = Path(__file__).parent / "chunks.json"

MODEL_NAME = "all-MiniLM-L6-v2"

_model: SentenceTransformer | None = None
_index: faiss.Index | None = None
_chunks: list[dict] | None = None


def _load_catalog() -> list[dict]:
    raw = CATALOG_PATH.read_text(encoding="utf-8", errors="replace")
    return json.loads(raw, strict=False)


def _make_chunk_text(item: dict) -> str:
    name = item.get("name", "")
    keys = ", ".join(item.get("keys", []))
    job_levels = ", ".join(item.get("job_levels", []))
    description = item.get("description", "")
    return f"{name} | {keys} | {job_levels} | {description}"


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def load_index() -> tuple[faiss.Index, list[dict]]:
    """Load the pre-built FAISS index and chunk metadata from disk."""
    global _index, _chunks
    if _index is None or _chunks is None:
        _index = faiss.read_index(str(INDEX_PATH))
        _chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    return _index, _chunks


def search(query: str, top_k: int = 15) -> list[dict]:
    """Embed query, search FAISS index, return top_k metadata objects."""
    model = _get_model()
    index, chunks = load_index()

    query_vec = model.encode([query], normalize_embeddings=True).astype("float32")
    _, indices = index.search(query_vec, top_k)

    return [chunks[i] for i in indices[0] if i < len(chunks)]


def build():
    catalog = _load_catalog()
    print(f"Loaded {len(catalog)} items from catalog.")

    model = _get_model()
    print(f"Model loaded: {MODEL_NAME}")

    chunks = []
    texts = []

    for i, item in enumerate(catalog):
        chunk_text = _make_chunk_text(item)
        texts.append(chunk_text)
        chunks.append({
            "chunk_text": chunk_text,
            "name": item.get("name", ""),
            "link": item.get("link", ""),
            "keys": item.get("keys", []),
            "job_levels": item.get("job_levels", []),
            "duration": item.get("duration", ""),
            "languages": item.get("languages", []),
            "remote": item.get("remote", ""),
            "adaptive": item.get("adaptive", ""),
        })

        if (i + 1) % 50 == 0 or (i + 1) == len(catalog):
            print(f"  Prepared {i + 1}/{len(catalog)} chunks...")

    print("Embedding all chunks...")
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")
    print(f"Embeddings shape: {embeddings.shape}")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vecs = cosine similarity
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors (dim={dim}).")

    faiss.write_index(index, str(INDEX_PATH))
    print(f"Index saved to {INDEX_PATH}")

    CHUNKS_PATH.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Chunks saved to {CHUNKS_PATH}")


if __name__ == "__main__":
    build()

    print("\n--- Quick test: 'cognitive test for senior Java developer' ---")
    results = search("cognitive test for senior Java developer", top_k=5)
    for r in results:
        print(f"  {r['name']}  |  {r['keys']}")
