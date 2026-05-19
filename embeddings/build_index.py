import json
import numpy as np
import faiss
from pathlib import Path
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
import onnxruntime as ort

ROOT = Path(__file__).parent.parent
CATALOG_PATH = ROOT / "data" / "catalog.json"
INDEX_PATH = Path(__file__).parent / "index.faiss"
CHUNKS_PATH = Path(__file__).parent / "chunks.json"

# Xenova maintains ready-to-use ONNX exports of popular embedding models.
# bge-small-en-v1.5: 384-dim, ~24 MB, no PyTorch, no Rust — fits in 512 MB free tier.
HF_REPO = "Xenova/bge-small-en-v1.5"
MODEL_FILE = "onnx/model.onnx"
TOKENIZER_FILE = "tokenizer.json"

_session: ort.InferenceSession | None = None
_tokenizer: Tokenizer | None = None
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


def _get_session_and_tokenizer() -> tuple[ort.InferenceSession, Tokenizer]:
    global _session, _tokenizer
    if _session is None or _tokenizer is None:
        print(f"Downloading ONNX model from {HF_REPO} ...")
        model_path = hf_hub_download(repo_id=HF_REPO, filename=MODEL_FILE)
        tok_path = hf_hub_download(repo_id=HF_REPO, filename=TOKENIZER_FILE)
        _session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        _tokenizer = Tokenizer.from_file(tok_path)
        _tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=512)
        _tokenizer.enable_truncation(max_length=512)
        print("Model ready.")
    return _session, _tokenizer


def _embed(texts: list[str]) -> np.ndarray:
    session, tokenizer = _get_session_and_tokenizer()
    encodings = tokenizer.encode_batch(texts)
    input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
    token_type_ids = np.zeros_like(input_ids)

    outputs = session.run(None, {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    })

    # Mean pooling weighted by attention mask, then L2-normalise → cosine-ready
    token_embs = outputs[0].astype(np.float32)           # (batch, seq, 384)
    mask = attention_mask[:, :, np.newaxis].astype(np.float32)
    embeddings = np.sum(token_embs * mask, axis=1) / (np.sum(mask, axis=1) + 1e-9)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + 1e-9)


def load_index() -> tuple[faiss.Index, list[dict]]:
    global _index, _chunks
    if _index is None or _chunks is None:
        _index = faiss.read_index(str(INDEX_PATH))
        _chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    return _index, _chunks


def search(query: str, top_k: int = 15) -> list[dict]:
    index, chunks = load_index()
    query_vec = _embed([query])
    _, indices = index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]


def build():
    catalog = _load_catalog()
    print(f"Loaded {len(catalog)} items from catalog.")

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

    # Embed in batches of 32 so progress is visible
    print("Embedding all chunks...")
    batch_size = 32
    all_embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start: start + batch_size]
        all_embeddings.append(_embed(batch))
        print(f"  Embedded {min(start + batch_size, len(texts))}/{len(texts)}")

    embeddings = np.vstack(all_embeddings).astype("float32")
    print(f"Embeddings shape: {embeddings.shape}")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalised vecs = cosine similarity
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
