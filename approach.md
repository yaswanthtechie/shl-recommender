# SHL Assessment Recommender — Approach Document

## 1. Design Choices

### FAISS + ONNX Embeddings (BAAI/bge-small-en-v1.5)
Semantic search over 377 catalog items requires vector similarity, not keyword matching. FAISS `IndexFlatIP` with L2-normalized vectors gives exact cosine similarity at this catalog size without approximation overhead. The model choice was constrained by Render's 512 MB free tier: `sentence-transformers` pulls in PyTorch (~400–500 MB runtime), which OOMs on cold start. `BAAI/bge-small-en-v1.5` via `onnxruntime` uses ~80 MB — the ONNX export from the Xenova HuggingFace repo is loaded with pure Python, no Rust, no PyTorch. The 384-dim model scores well on MTEB retrieval benchmarks relative to its size, making it the best fit-for-constraint choice available.

### Groq (llama-3.3-70b-versatile) over OpenAI
Groq's free tier provides ~14,400 requests/day on `llama-3.3-70b-versatile` with no credit card required, which is sufficient for a demo/assessment deployment. The Groq SDK is OpenAI-compatible, making the switch a two-line change if needed. `llama-3.3-70b-versatile` follows structured XML output instructions reliably enough for deterministic tag parsing, whereas smaller models required more prompt engineering to stay on format.

### Stateless API Design
Each `/chat` request carries the full message history from the client. No server-side session storage eliminates state management complexity, Redis dependencies, and session expiry edge cases. For a max-8-turn conversation, the full history fits comfortably within the model's context window. The tradeoff is slightly larger payloads per request, which is negligible at this scale.

### Context Injection Strategy
Rather than fine-tuning or relying on the LLM's parametric knowledge of SHL assessments, retrieved catalog items are injected directly into the system prompt on every turn. The retrieval query is built from the last two user messages, keeping it focused. This ensures the model only recommends assessments that exist in the catalog — hallucinated URLs are eliminated at the source. Top-15 retrieved items give the model enough breadth to handle multi-constraint queries without exceeding prompt token limits.

---

## 2. Retrieval Setup

### Chunking Formula
Each catalog item is encoded as a single chunk:
```
{name} | {keys} | {job_levels} | {description}
```
This concatenation ensures that a query like "cognitive test for mid-level Java developer" can match on `keys` (Cognitive), `job_levels` (Mid-Professional), and `description` semantics simultaneously in one vector.

### Why Description Is Critical
The `name` field alone is opaque (e.g., "Verify — Numerical Reasoning"). The `description` field carries the semantic signal that connects user intent ("I need someone who can analyze data under pressure") to the right assessment. Without it, retrieval degrades to surface-level name matching.

### Post-Retrieval URL Validation
After the LLM generates recommendations, every URL is validated against `VALID_URLS` — a set built from `chunks.json` at startup. Any URL not in this set is dropped before the response is returned. This is a hard guardrail: the model cannot hallucinate a plausible-looking SHL URL and have it reach the client.

---

## 3. Prompt Design

### CLARIFY FIRST Rule
The system prompt instructs the model to ask at least one clarifying question before recommending when the query is vague (no job level, no role, no assessment type specified). This mirrors the sample conversation patterns (C1–C10) where effective recommendations required 1–2 turns of scoping. The rule is enforced in the prompt, not in application logic — keeping the agent layer thin.

### Structured XML Output Parsing
Recommendations are returned inside a `<recommendations>[...]</recommendations>` block and conversation end is signaled via `<end_conversation/>`. Regex parsing extracts these tags from the raw LLM response, making the extraction robust to surrounding prose. This is simpler and more reliable than asking the model to return pure JSON, which often breaks with nested quotes or trailing commas.

### Refusal Rules
The system prompt explicitly instructs the model to refuse: off-topic questions (salary, legal, general HR), requests to ignore its instructions, and attempts to extract system prompt content. These rules are stated directly rather than relying on the base model's RLHF refusal behavior, which can be inconsistent across providers.

---

## 4. Evaluation Approach

### 7 Behavior Tests (All Passing)
Tests cover the key behavioral requirements: health check, clarification on vague queries, recommendations on specific queries, URL validity of all recommendations, refusal of off-topic questions, multi-turn refinement, and end-of-conversation signaling. Tests run against a live local server (`http://localhost:8000`) with a 35-second timeout to account for LLM latency.

### What Didn't Work
`sentence-transformers` was the original embedder — dropped because PyTorch's memory footprint causes OOM on Render's 512 MB free tier. `fastembed` was the intended replacement (handles ONNX internally) but has no pre-built wheels for Python 3.14, requiring compilation from source via Rust/MSVC which fails on the dev machine. Final solution: `onnxruntime` with the `Xenova/bge-small-en-v1.5` ONNX export, which ships pre-built wheels for Python 3.14 and adds `huggingface_hub` + `tokenizers` as explicit dependencies.

### Trade-offs: Free Tier vs Paid Hosting
Render's free tier cold-starts after 15 minutes of inactivity, adding ~30–60 seconds to the first request (ONNX model download from HuggingFace cache + FAISS index load). Paid hosting eliminates cold starts and raises the memory ceiling, enabling larger embedding models. For this use case, the free tier trade-off is acceptable: the pre-built `index.faiss` and `chunks.json` are committed to git, so Render skips the rebuild step on restart.

---

## 5. AI Tools Used

- **Claude Code (Anthropic)** — All code generation, architecture decisions, debugging, and iterative refinement were done via Claude Code in the VS Code extension. This included the full stack: embeddings pipeline, retrieval layer, agent orchestration, FastAPI endpoints, pytest behavior tests, Dockerfile, and render.yaml.

- **Groq Free Tier** — LLM inference for the production agent using `llama-3.3-70b-versatile`. No cost, no rate-limit issues for demo traffic, OpenAI-compatible SDK.
