# Local Log

Notes about local, gitignored artifacts that are **not** tracked in this repo.

## Model weights

### Q8 quantized weights (present locally)

- **File:** `meta-llama-3-8B-instruct-Q8.gguf`
- **Size:** 8,540,770,560 bytes (7.95 GiB)

### Source instruct weights used to generate the Q8 quant (deleted locally)

The Q8 file above was quantized from the full-precision instruct GGUF below. That
source file has been **deleted locally to reclaim disk space** — recorded here so the
provenance is preserved.

- **File:** `meta-llama-3-8B-instruct.gguf`
- **Size:** 32,128,880,896 bytes (29.92 GiB)
- **Deleted:** 2026-07-16

> Both `.gguf` files are gitignored (`*.gguf`) and were never committed to git history.
