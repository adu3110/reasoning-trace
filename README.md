# reasoning-trace

Token-level reasoning trace visualizer — color-coded entropy map of how an LLM thinks.

Streams tokens from an LLM with log-probabilities, computes per-token Shannon entropy, and generates a self-contained HTML report where high-entropy (uncertain) tokens are red and low-entropy (confident) tokens are blue.

## How it works

```
token logprobs  →  Shannon entropy H(p) = -Σ p_i · log₂(p_i)
                →  z-score per token
                →  color gradient: blue (certain) → red (uncertain)
                →  HTML report with hover-over alternatives
```

High entropy = decision point where the model could have gone a different way.

## Run

```bash
# Python
python python/trace.py
# → writes python/reasoning_trace.html

# TypeScript
npx ts-node typescript/src/trace.ts
# → writes typescript/reasoning_trace.html
```

API key loaded automatically from `.env` / `.env.local` in this directory or any parent:

```bash
OPENAI_API_KEY=sk-... python python/trace.py
```

## Output

`reasoning_trace.html` — open in any browser:
- Every token color-coded by entropy
- Hover any token to see the top-5 alternatives with probabilities
- Table of the 10 most uncertain tokens

## Config

| Variable | Default | Meaning |
|----------|---------|---------|
| `OPENAI_API_KEY` | (required) | API key |
| `TRACE_MODEL` | `gpt-4o-mini` | Model (must support `logprobs`) |

## Dependencies

- Python: stdlib only (`urllib`, `json`, `math`, `pathlib`)
- TypeScript: stdlib only (`https`, `fs`, `path`)
