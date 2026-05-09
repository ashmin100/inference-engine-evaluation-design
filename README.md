# Inference Benchmark: Friendli Engine vs vLLM

Benchmarks the **Throughput–Latency efficiency frontier** of Friendli Engine against vLLM across concurrency levels 1 → 32.

## Quick Start

```bash
pip install -r requirements.txt

# Preview expected results (no engines needed)
python benchmark.py --mock

# Real benchmark (both engines must be serving on the OpenAI-compatible API)
python benchmark.py \
  --vllm-url     http://localhost:8000 \
  --friendli-url http://localhost:8001 \
  --model        meta-llama/Llama-3.1-8B-Instruct
```

Output: `results/benchmark_result.png` + `results/raw_metrics.json`

## All Options

| Flag | Default | Description |
|------|---------|-------------|
| `--vllm-url` | `http://localhost:8000` | vLLM server base URL |
| `--friendli-url` | `http://localhost:8001` | Friendli Engine server base URL |
| `--model` | `meta-llama/Llama-3.1-8B-Instruct` | Model name in API requests |
| `--requests-per-level` | `50` | Requests measured at each concurrency level |
| `--output` | `results/benchmark_result.png` | Graph output path |
| `--mock` | off | Generate graph from simulated data |

## Requirements

- Both engines must expose an **OpenAI-compatible** `/v1/chat/completions` endpoint.
- `stream: true` must be supported — TTFT measurement depends on it.
- Same model must be loaded in both engines for a fair comparison.

## What the Script Measures

For each concurrency level in `[1, 2, 4, 8, 16, 32]`:

1. Sends `WARMUP_REQUESTS` (5) to eliminate cold-start latency.
2. Sends 50 concurrent requests (same prompts to both engines).
3. Records **TTFT** (via streaming) and **output token count** per request.
4. Computes **P95 TTFT** and **Throughput** (total output tokens / wall-clock seconds).

---

## Why These Metrics?

### Time-to-First-Token (P95 TTFT)

TTFT is the delay between a user submitting a request and receiving the first token. It is the latency signal most directly perceived by end users in interactive applications (chat, copilots, search). P95 captures the tail experience that determines whether a product "feels" responsive at scale, not just on average.

TTFT degrades sharply under concurrent load because engines must queue and batch requests. The degree of degradation is the clearest signal of scheduling and batching efficiency — where Friendli Engine's continuous batching and memory management optimizations are most visible.

### Throughput (output tokens / second)

Throughput measures serving capacity: how many tokens all concurrent users receive per second in aggregate. It directly maps to cost efficiency — higher throughput on the same hardware means more users served per dollar.

Measuring throughput alone misses latency; measuring TTFT alone misses capacity. Together they define the efficiency envelope.

---

## Why the Throughput–Latency Frontier?

The efficiency frontier plot (Throughput on X, P95 TTFT on Y, one point per concurrency level) is the industry-standard visualization for comparing inference systems. It is used in vLLM's own published benchmarks and Anyscale's serving comparisons because it simultaneously communicates two dimensions of performance in a single curve.

**How to read the graph:**

- Each point represents one concurrency level (labeled `c=N`).
- As concurrency increases, throughput rises but latency also rises — the curve reveals each engine's efficiency envelope under growing load.
- A curve that sits **lower-right** is strictly better: higher throughput *and* lower latency at the same concurrency.
- The **gap between the two curves**, which widens at higher concurrency levels, shows that Friendli Engine's advantage is not an idle-state artifact — it grows as production load increases.

This asymmetric widening is the key insight: both engines look similar at c=1, but Friendli's throughput scales ~5× higher while its TTFT grows far more slowly than vLLM's. A single sub-graph captures this story completely.

---

## Sample Output

![Efficiency Frontier](results/benchmark_result.png)

*Generated with `--mock` using data approximated from published FriendliAI benchmarks
([friendli.ai/blog/comparing-friendli-engine-vllm](https://friendli.ai/blog/comparing-friendli-engine-vllm),
[friendli.ai/blog/quantized-mixtral-single-gpu](https://friendli.ai/blog/quantized-mixtral-single-gpu)).*
