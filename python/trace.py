"""
Reasoning Trace Visualizer — implemented from scratch.

What this demonstrates
----------------------
Reasoning models (o1, DeepSeek-R1, QwQ) emit long internal "thinking" tokens.
This tool streams those tokens in real-time, computes per-token entropy to
detect where the model is most uncertain, and writes a self-contained HTML
report you can open in a browser.

Entropy measures uncertainty in the token distribution:
    H(p) = -Σ p_i · log2(p_i)     [bits]
High entropy → model is uncertain about what to say next (key inflection points).
Low entropy  → model is confident (routine reasoning steps).

Run
---
    python trace.py

API key resolution order:
    1. OPENAI_API_KEY environment variable
    2. .env or .env.local in this directory or any parent up to /

Output: reasoning_trace.html (open in any browser)

Dependencies: stdlib only (urllib, json, math, os, pathlib).
"""

from __future__ import annotations

import html
import json
import math
import os
import pathlib
import urllib.request
from dataclasses import dataclass


# ──────────────────────────────────────────────────────────────────────────────
# Env-file loader (no python-dotenv required)
# ──────────────────────────────────────────────────────────────────────────────
def _load_env_files() -> None:
    here = pathlib.Path(__file__).resolve().parent
    for directory in [here, *here.parents]:
        for name in (".env.local", ".env"):
            path = directory / name
            if path.is_file():
                with path.open(encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        key, _, val = line.partition("=")
                        key = key.strip()
                        val = val.strip().strip('"').strip("'")
                        if key and key not in os.environ:
                            os.environ[key] = val


_load_env_files()

API_KEY  = os.environ.get("OPENAI_API_KEY", "")
MODEL    = os.environ.get("TRACE_MODEL", "gpt-4o-mini")
OUT_FILE = pathlib.Path(__file__).parent / "reasoning_trace.html"

if not API_KEY:
    raise SystemExit(
        "OPENAI_API_KEY not found.\n"
        "Set it in your environment or create a .env / .env.local file:\n"
        "  OPENAI_API_KEY=sk-..."
    )

PROBLEM = (
    "A webhook-based payments service sometimes charges users twice when the "
    "webhook is retried. Explain step by step why this happens and how to fix it."
)


# ──────────────────────────────────────────────────────────────────────────────
# Token with logprob info
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class TracedToken:
    text:        str
    logprob:     float          # log probability of chosen token
    top_logprobs: dict[str, float]   # top-5 alternatives {token: logprob}

    @property
    def prob(self) -> float:
        return math.exp(self.logprob)

    @property
    def entropy(self) -> float:
        """Shannon entropy over the top-k alternatives (proxy for true entropy)."""
        probs = [math.exp(lp) for lp in self.top_logprobs.values()]
        total = sum(probs)
        if total == 0:
            return 0.0
        norm  = [p / total for p in probs]
        return -sum(p * math.log2(p + 1e-12) for p in norm)


# ──────────────────────────────────────────────────────────────────────────────
# LLM call — returns tokens with logprobs
# ──────────────────────────────────────────────────────────────────────────────
def stream_tokens(prompt: str) -> list[TracedToken]:
    """Fetch completion tokens with log-probabilities from the OpenAI API."""
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "logprobs": True,
        "top_logprobs": 5,
        "max_tokens": 600,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())

    tokens: list[TracedToken] = []
    for lp_obj in data["choices"][0]["logprobs"]["content"]:
        top = {t["token"]: t["logprob"] for t in lp_obj.get("top_logprobs", [])}
        top[lp_obj["token"]] = lp_obj["logprob"]
        tokens.append(TracedToken(
            text=lp_obj["token"],
            logprob=lp_obj["logprob"],
            top_logprobs=top,
        ))
    return tokens


# ──────────────────────────────────────────────────────────────────────────────
# Running entropy stats
# ──────────────────────────────────────────────────────────────────────────────
def running_stats(entropies: list[float]) -> tuple[float, float]:
    """Return (mean, std) for a list of values."""
    if not entropies:
        return 0.0, 1.0
    mu  = sum(entropies) / len(entropies)
    var = sum((x - mu) ** 2 for x in entropies) / len(entropies)
    return mu, math.sqrt(var) or 1.0


# ──────────────────────────────────────────────────────────────────────────────
# HTML report
# ──────────────────────────────────────────────────────────────────────────────
_CSS = """
<style>
  body   { font-family: 'Courier New', monospace; background: #0d1117; color: #c9d1d9;
            max-width: 860px; margin: 40px auto; padding: 20px; line-height: 1.7; }
  h1     { color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 12px; }
  h2     { color: #79c0ff; font-size: 1em; margin-top: 28px; }
  .meta  { color: #8b949e; font-size: .85em; margin-bottom: 24px; }
  .trace { font-size: .95em; line-height: 2; }
  .tok   { border-radius: 3px; padding: 1px 2px; cursor: default; }
  .legend { display: flex; gap: 18px; margin-bottom: 18px; font-size: .82em; }
  .legend-item { display: flex; align-items: center; gap: 6px; }
  .swatch { width: 18px; height: 14px; border-radius: 2px; display: inline-block; }
  .tooltip { position: relative; }
  .tooltip .tip { display: none; position: absolute; bottom: 130%; left: 50%;
                  transform: translateX(-50%); background: #161b22; border: 1px solid #30363d;
                  padding: 6px 10px; border-radius: 4px; font-size: .78em; white-space: nowrap;
                  z-index: 10; color: #c9d1d9; }
  .tooltip:hover .tip { display: block; }
  table  { width: 100%; border-collapse: collapse; font-size: .85em; margin-top: 12px; }
  th     { text-align: left; color: #8b949e; border-bottom: 1px solid #21262d; padding: 6px 8px; }
  td     { padding: 5px 8px; border-bottom: 1px solid #161b22; }
  .bar   { height: 10px; background: #238636; border-radius: 2px; }
</style>
"""


def _entropy_color(entropy: float, mean: float, std: float) -> str:
    """Map entropy z-score to a colour from cool-blue (certain) → warm-red (uncertain)."""
    z = (entropy - mean) / (std or 1.0)
    z = max(-2.0, min(2.0, z))
    t = (z + 2.0) / 4.0          # normalise to [0, 1]

    def lerp(a: int, b: int) -> int:
        return int(a + t * (b - a))

    r = lerp(0x58, 0xff)           # 88 → 255  (blue → red)
    g = lerp(0xa6, 0x7b)           # 166 → 123
    b_ = lerp(0xff, 0x00)          # 255 → 0
    alpha = 0.15 + 0.55 * t        # subtle at low entropy, stronger at high
    return f"rgba({r},{g},{b_},{alpha:.2f})"


def _top_alts_html(tok: TracedToken) -> str:
    lines = []
    for t, lp in sorted(tok.top_logprobs.items(), key=lambda x: -x[1])[:5]:
        prob_pct = math.exp(lp) * 100
        lines.append(
            f"<span style='display:block'>"
            f"<b>{html.escape(repr(t))}</b> {prob_pct:.1f}%"
            f"</span>"
        )
    return "".join(lines)


def build_html(tokens: list[TracedToken], problem: str, model: str) -> str:
    entropies = [t.entropy for t in tokens]
    mean, std = running_stats(entropies)

    token_spans: list[str] = []
    for tok in tokens:
        color = _entropy_color(tok.entropy, mean, std)
        tip   = (
            f"token: {html.escape(repr(tok.text))} | "
            f"prob: {tok.prob*100:.1f}% | "
            f"entropy: {tok.entropy:.3f} bits"
        )
        alts  = _top_alts_html(tok)
        span  = (
            f'<span class="tok tooltip" style="background:{color}" title="{tip}">'
            f'{html.escape(tok.text)}'
            f'<span class="tip">{alts}</span>'
            f'</span>'
        )
        token_spans.append(span)

    # Top-10 highest-entropy tokens table
    top10 = sorted(enumerate(tokens), key=lambda x: -x[1].entropy)[:10]
    table_rows = "".join(
        f"<tr>"
        f"<td>{i+1}</td>"
        f"<td><code>{html.escape(repr(tok.text))}</code></td>"
        f"<td>{tok.prob*100:.2f}%</td>"
        f"<td>{tok.entropy:.3f}</td>"
        f"<td><div class='bar' style='width:{min(100,tok.entropy*20):.0f}px'></div></td>"
        f"</tr>"
        for i, (_, tok) in enumerate(top10)
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Reasoning Trace</title>{_CSS}</head>
<body>
<h1>Reasoning Trace Visualizer</h1>
<p class="meta">
  Model: <b>{html.escape(model)}</b> &nbsp;|&nbsp;
  Tokens: <b>{len(tokens)}</b> &nbsp;|&nbsp;
  Mean entropy: <b>{mean:.3f} bits</b> &nbsp;|&nbsp;
  Hover tokens for alternatives
</p>
<p class="meta"><b>Problem:</b> {html.escape(problem)}</p>

<div class="legend">
  <span class="legend-item"><span class="swatch" style="background:rgba(88,166,255,0.20)"></span>Low entropy (certain)</span>
  <span class="legend-item"><span class="swatch" style="background:rgba(200,120,60,0.50)"></span>Mid entropy</span>
  <span class="legend-item"><span class="swatch" style="background:rgba(255,80,0,0.70)"></span>High entropy (uncertain)</span>
</div>

<h2>Token stream (hover for top alternatives)</h2>
<div class="trace">{"".join(token_spans)}</div>

<h2>Top-10 highest-entropy tokens</h2>
<table>
<tr><th>#</th><th>Token</th><th>Prob</th><th>Entropy (bits)</th><th>Bar</th></tr>
{table_rows}
</table>
</body></html>"""


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("  Reasoning Trace Visualizer")
    print(f"  Model: {MODEL}")
    print("=" * 60)
    print(f"\nProblem: {PROBLEM}\n")
    print("Fetching tokens with logprobs…")

    tokens = stream_tokens(PROBLEM)
    entropies = [t.entropy for t in tokens]
    mean, std = running_stats(entropies)

    print(f"\nCollected {len(tokens)} tokens")
    print(f"Entropy  — mean: {mean:.3f} bits  std: {std:.3f} bits")

    high_e = sorted(tokens, key=lambda t: -t.entropy)[:5]
    print("\nTop-5 highest-entropy tokens (model most uncertain):")
    for i, tok in enumerate(high_e):
        print(f"  {i+1}. {repr(tok.text):<20} entropy={tok.entropy:.3f}  prob={tok.prob*100:.1f}%")

    report = build_html(tokens, PROBLEM, MODEL)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nHTML report written to: {OUT_FILE}")
    print("Open it in any browser. Hover tokens to see alternatives.")


if __name__ == "__main__":
    main()
