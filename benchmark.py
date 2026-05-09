#!/usr/bin/env python3
"""
Inference Benchmark: Friendli Engine vs vLLM
Measures Throughput and P95 TTFT across concurrency levels.
Generates a single Throughput–Latency efficiency frontier plot.

Usage:
  # Real benchmark (both engines must be running)
  python benchmark.py \\
    --vllm-url http://localhost:8000 \\
    --friendli-url http://localhost:8001 \\
    --model meta-llama/Llama-3.1-8B-Instruct

  # Preview expected results (no live engines needed)
  python benchmark.py --mock
"""

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import aiohttp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Configuration ──────────────────────────────────────────────────────────

CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32]
REQUESTS_PER_LEVEL = 50
WARMUP_REQUESTS = 5
REQUEST_TIMEOUT_S = 120

# Varied prompts to approximate real workload input/output distributions
SAMPLE_PROMPTS = [
    "Explain the difference between TCP and UDP in simple terms.",
    "Write a Python function that checks if a string is a palindrome.",
    "What are the main causes of the French Revolution?",
    "How does a transformer model process text? Give a brief explanation.",
    "Describe the water cycle in three sentences.",
    "What is the time complexity of quicksort in the worst case, and why?",
    "Translate 'The early bird catches the worm' into Spanish and explain it.",
    "What is gradient descent and how is it used in machine learning?",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "Explain what a REST API is to a non-technical person.",
]


# ── Data Types ─────────────────────────────────────────────────────────────

@dataclass
class RequestResult:
    ttft_ms: float
    output_tokens: int
    total_duration_s: float
    success: bool
    error: str = ""


@dataclass
class LevelMetrics:
    concurrency: int
    throughput_tps: float    # output tokens / wall-clock seconds
    p50_ttft_ms: float
    p95_ttft_ms: float
    p99_ttft_ms: float
    success_rate: float


# ── Async Benchmark Core ───────────────────────────────────────────────────

async def send_streaming_request(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: str,
) -> RequestResult:
    """Send a single streaming chat request and measure TTFT + output tokens."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "stream": True,
    }
    t_start = time.perf_counter()
    ttft_ms: Optional[float] = None
    output_tokens = 0

    try:
        async with session.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S),
        ) as resp:
            resp.raise_for_status()
            async for raw_line in resp.content:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                content = (
                    chunk.get("choices", [{}])[0]
                    .get("delta", {})
                    .get("content", "")
                ) or ""
                if not content:
                    continue
                # Record TTFT on the first chunk that carries actual content
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t_start) * 1000
                output_tokens += 1

        total_s = time.perf_counter() - t_start
        return RequestResult(
            ttft_ms=ttft_ms if ttft_ms is not None else total_s * 1000,
            output_tokens=output_tokens,
            total_duration_s=total_s,
            success=True,
        )
    except Exception as exc:
        return RequestResult(
            ttft_ms=0,
            output_tokens=0,
            total_duration_s=time.perf_counter() - t_start,
            success=False,
            error=str(exc),
        )


async def run_level(
    base_url: str,
    model: str,
    concurrency: int,
    n_requests: int,
) -> tuple[List[RequestResult], float]:
    """Run warmup then measured requests. Returns (results, wall_time_s)."""
    semaphore = asyncio.Semaphore(concurrency)
    prompts = [SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)] for i in range(n_requests)]

    async def bounded(prompt: str) -> RequestResult:
        async with semaphore:
            return await send_streaming_request(session, base_url, model, prompt)

    async with aiohttp.ClientSession() as session:
        # Warmup — results discarded, wall time NOT started yet
        await asyncio.gather(*[bounded(SAMPLE_PROMPTS[0]) for _ in range(WARMUP_REQUESTS)])
        # Measured run — wall time covers only this section
        t0 = time.perf_counter()
        results = await asyncio.gather(*[bounded(p) for p in prompts])
        wall_time_s = time.perf_counter() - t0

    return list(results), wall_time_s


def compute_metrics(
    results: List[RequestResult], wall_time_s: float, concurrency: int
) -> LevelMetrics:
    good = [r for r in results if r.success and r.ttft_ms > 0]
    if not good:
        return LevelMetrics(concurrency, 0.0, 0.0, 0.0, 0.0, 0.0)
    ttfts = [r.ttft_ms for r in good]
    total_tokens = sum(r.output_tokens for r in good)
    return LevelMetrics(
        concurrency=concurrency,
        throughput_tps=total_tokens / wall_time_s,
        p50_ttft_ms=float(np.percentile(ttfts, 50)),
        p95_ttft_ms=float(np.percentile(ttfts, 95)),
        p99_ttft_ms=float(np.percentile(ttfts, 99)),
        success_rate=len(good) / len(results),
    )


async def benchmark_engine(
    name: str, base_url: str, model: str, requests_per_level: int
) -> List[LevelMetrics]:
    metrics_list: List[LevelMetrics] = []
    for c in CONCURRENCY_LEVELS:
        print(f"  [{name}] concurrency={c:>2} ...", end=" ", flush=True)
        results, wall = await run_level(base_url, model, c, requests_per_level)
        m = compute_metrics(results, wall, c)
        metrics_list.append(m)
        print(
            f"throughput={m.throughput_tps:>7.1f} tok/s  "
            f"P95-TTFT={m.p95_ttft_ms:>6.0f} ms  "
            f"success={m.success_rate:.0%}"
        )
    return metrics_list


# ── Mock Data ──────────────────────────────────────────────────────────────

def generate_mock_data() -> tuple[List[LevelMetrics], List[LevelMetrics]]:
    """
    Simulated data based on FriendliAI published benchmarks.
    Approximates Llama-3-8B performance on A100 80 GB.

    Key sources:
      friendli.ai/blog/comparing-friendli-engine-vllm
        → 2–4× latency reduction, up to 6× higher throughput at same latency
      friendli.ai/blog/quantized-mixtral-single-gpu
        → TTFT 4.1× faster, TPOT 3.8–23.8× faster
    """
    rng = np.random.default_rng(42)

    # (throughput tok/s, P95 TTFT ms) at each concurrency level in CONCURRENCY_LEVELS
    vllm_curve = [
        (55,   140),
        (110,  310),
        (185,  720),
        (230, 1850),
        (245, 4600),
        (252, 9200),
    ]
    friendli_curve = [
        (80,    90),
        (230,  155),
        (490,  240),
        (870,  390),
        (1150, 580),
        (1380, 820),
    ]

    def make_metrics(pairs: list) -> List[LevelMetrics]:
        return [
            LevelMetrics(
                concurrency=CONCURRENCY_LEVELS[i],
                throughput_tps=tp * float(rng.uniform(0.97, 1.03)),
                p50_ttft_ms=lat * 0.82,
                p95_ttft_ms=lat * float(rng.uniform(0.97, 1.03)),
                p99_ttft_ms=lat * 1.22,
                success_rate=1.0,
            )
            for i, (tp, lat) in enumerate(pairs)
        ]

    return make_metrics(vllm_curve), make_metrics(friendli_curve)


# ── Visualization ──────────────────────────────────────────────────────────

def plot_frontier(
    vllm: List[LevelMetrics],
    friendli: List[LevelMetrics],
    output_path: str,
    is_mock: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    vx = [m.throughput_tps for m in vllm]
    vy = [m.p95_ttft_ms for m in vllm]
    fx = [m.throughput_tps for m in friendli]
    fy = [m.p95_ttft_ms for m in friendli]

    ax.plot(vx, vy, "o-", color="#E55353", linewidth=2.2, markersize=9, label="vLLM", zorder=3)
    ax.plot(fx, fy, "s-", color="#2563EB", linewidth=2.2, markersize=9, label="Friendli Engine", zorder=3)

    # Per-point offsets to avoid label overlap
    # vLLM c=1,2 go upper-left; rest go right
    vllm_offsets = [(-36, 8), (-36, 5), (8, 5), (8, 5), (8, 5), (8, -13)]
    # Friendli c=1,2 go below to clear the vLLM labels; c=32 goes above-left (right edge)
    friendli_offsets = [(8, -14), (4, -14), (8, 5), (8, 5), (8, 5), (-36, 8)]

    for m, off in zip(vllm, vllm_offsets):
        ax.annotate(
            f"c={m.concurrency}",
            (m.throughput_tps, m.p95_ttft_ms),
            textcoords="offset points", xytext=off,
            fontsize=8, color="#C0392B",
        )
    for m, off in zip(friendli, friendli_offsets):
        ax.annotate(
            f"c={m.concurrency}",
            (m.throughput_tps, m.p95_ttft_ms),
            textcoords="offset points", xytext=off,
            fontsize=8, color="#1D4ED8",
        )

    title_suffix = "  [mock — expected results]" if is_mock else ""
    ax.set_title(
        f"Inference Efficiency Frontier: Friendli Engine vs vLLM{title_suffix}\n"
        "Each point = one concurrency level (c).  Lower-right corner = better.",
        fontsize=13, pad=12,
    )
    ax.set_xlabel("Throughput  (output tokens / second)", fontsize=12)
    ax.set_ylabel("P95 Time-to-First-Token  (ms)", fontsize=12)
    ax.legend(fontsize=12, loc="upper left", framealpha=0.9, edgecolor="#dddddd")
    all_y = vy + fy
    ax.set_ylim(0, max(all_y) * 1.1)
    ax.grid(True, alpha=0.2, linestyle="--", which="major", color="#aaaaaa")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:,.0f}"))

    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(colors="#555555")

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Graph saved → {output_path}")
    try:
        plt.show()
    except Exception:
        pass  # non-interactive environments (e.g. headless servers)


# ── Entry Point ────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark Friendli Engine vs vLLM — Throughput/Latency frontier"
    )
    p.add_argument("--vllm-url", default="http://localhost:8000",
                   help="Base URL of the vLLM server (default: http://localhost:8000)")
    p.add_argument("--friendli-url", default="http://localhost:8001",
                   help="Base URL of the Friendli Engine server (default: http://localhost:8001)")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct",
                   help="Model name to pass in the API request")
    p.add_argument("--requests-per-level", type=int, default=REQUESTS_PER_LEVEL,
                   help=f"Requests per concurrency level (default: {REQUESTS_PER_LEVEL})")
    p.add_argument("--output", default="results/benchmark_result.png",
                   help="Output path for the graph (default: results/benchmark_result.png)")
    p.add_argument("--mock", action="store_true",
                   help="Generate graph from simulated data (no live engines needed)")
    return p.parse_args()


async def main() -> None:
    args = parse_args()

    if args.mock:
        print("Mock mode — generating expected results from published FriendliAI benchmarks.\n")
        vllm_metrics, friendli_metrics = generate_mock_data()
    else:
        print(f"Benchmarking vLLM        → {args.vllm_url}")
        print(f"Benchmarking Friendli    → {args.friendli_url}")
        print(f"Model: {args.model}\n")

        vllm_metrics = await benchmark_engine(
            "vLLM", args.vllm_url, args.model, args.requests_per_level
        )
        print()
        friendli_metrics = await benchmark_engine(
            "Friendli", args.friendli_url, args.model, args.requests_per_level
        )

    # Persist raw numbers
    Path("results").mkdir(exist_ok=True)
    raw_path = "results/raw_metrics.json"
    with open(raw_path, "w") as f:
        json.dump(
            {
                "vllm": [vars(m) for m in vllm_metrics],
                "friendli": [vars(m) for m in friendli_metrics],
            },
            f,
            indent=2,
        )
    print(f"Raw metrics saved → {raw_path}\n")

    plot_frontier(vllm_metrics, friendli_metrics, args.output, is_mock=args.mock)


if __name__ == "__main__":
    asyncio.run(main())
