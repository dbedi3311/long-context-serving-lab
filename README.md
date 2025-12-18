# long-context-serving-lab

A minimal LLM serving lab showcasing PagedAttention-style KV paging, tiered KV offload (GPU→CPU→SSD), sliding-window attention, speculative decoding, and chunked RAG—complete with a multi-session FastAPI server and reproducible benchmarks.

## Why KV cache grows O(T)
Transformer decoders append one key/value pair per generated token, so the KV cache expands linearly with sequence length `T`. Without pruning or paging, long conversations accumulate tens of thousands of tokens and the cache consumes O(T) memory, quickly exhausting GPU capacity.

## Paging and offload prevent OOM
To stay within a fixed GPU budget, the lab models pages of KV slots. Cold pages spill from GPU → CPU → SSD using an LRU heuristic, while hot pages are pinned. This keeps memory bounded yet allows paging back in when context resumes.

## Sliding window and speculative decoding for latency
A sliding attention window truncates to the most recent tokens, bounding compute and reducing attention cost. Speculative decoding drafts multiple tokens then verifies them, trading a bit of extra work for lower end-to-end latency when drafts are accepted.

## Running locally on Apple Silicon
1. Install dependencies: `make setup-mac` (uses `requirements-mac.txt` and prefers CPU-friendly wheels).
2. Download placeholder models: `make models`.
3. Start the FastAPI server: `make run`.
4. Hit `/start` and `/step` via `curl` or the provided notebook to exercise paging and speculative decoding.

## Benchmarks
Run `make bench` to generate CSVs under `bench/` and visualize with `python scripts/plot_results.py --root bench` (uses matplotlib if installed, otherwise writes portable SVGs).

## Project layout
- `runtime/`: KV cache structures, pager, sliding window, and speculative decoding toy implementation.
- `rag/`: Lightweight vector index with FAISS-or-sklearn fallback and retrieval helper.
- `server/`: FastAPI app and session management.
- `bench/`: Synthetic benchmark generators.
- `scripts/`: Utility scripts for models and plotting.
- `notebooks/`: End-to-end exploration notebook.
