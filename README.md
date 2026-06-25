# reasoning-trace

Stream tokens from an LLM with log-probabilities, compute per-token Shannon entropy, and render a color-coded HTML uncertainty map. No frontend framework. No external dependencies. Pure Python stdlib + TypeScript.

Every token the model outputs had alternatives. This tool shows you what they were and how confident the model was at each step.

---

## What entropy tells you

At each position in the output, the model assigns a probability distribution over its entire vocabulary. Shannon entropy measures how spread out that distribution is:

```
H(p) = -Σ  p_i · log₂(p_i)     [bits]
         i
```

| Entropy | Meaning | Visual |
|---------|---------|--------|
| ~0 bits | Model is certain — one token dominates | Blue |
| ~2 bits | Four plausible alternatives | Orange |
| ~3+ bits | Model is genuinely unsure | Red |

High-entropy tokens are the interesting ones: inflection points where the model switches logical tracks, introduces new concepts, or hedges.

---

## Run

```bash
# Python
python python/trace.py
# → writes python/reasoning_trace.html

# TypeScript
npx ts-node typescript/src/trace.ts
# → writes typescript/reasoning_trace.html
```

API key is read automatically from `.env` or `.env.local` anywhere in the directory tree:

```
OPENAI_API_KEY=sk-...
TRACE_MODEL=gpt-4o-mini     # optional, default: gpt-4o-mini
```

Then open the HTML file in any browser.

---

## What the HTML report shows

```
┌─────────────────────────────────────────────────────┐
│  Token stream (hover any token for alternatives)    │
│                                                     │
│  The [fix] is [to] store [a] unique [webhook] ID   │
│   ████  ██   ████  ██  ███████████████████         │
│   blue  blue  red  blue     red (high entropy)     │
│                                                     │
│  Top-10 most uncertain tokens                       │
│  1. "or"      entropy=2.51  prob=8.2%   ████████  │
│  2. "help"    entropy=2.50  prob=4.8%   ███████   │
│  3. "an"      entropy=2.48  prob=5.6%   ███████   │
└─────────────────────────────────────────────────────┘
```

Hover any token to see the top-5 alternatives the model considered, with their probabilities.

---

### Real output from a gpt-4o-mini run

```
Collected 600 tokens
Entropy — mean: 0.826 bits   std: 0.741 bits

Top-5 highest-entropy tokens:
  1. ' or'       entropy=2.513   prob=8.2%
  2. ' help'     entropy=2.503   prob=4.8%
  3. ' an'       entropy=2.477   prob=5.6%
  4. ' Perform'  entropy=2.355   prob=6.5%
  5. '-level'    entropy=2.315   prob=3.3%
```

---

## Configuration

| Variable | Default | Meaning |
|----------|---------|---------|
| `OPENAI_API_KEY` | required | Must support `logprobs` parameter |
| `TRACE_MODEL` | `gpt-4o-mini` | Any OpenAI chat model |

The `logprobs` + `top_logprobs` API parameters are standard across OpenAI-compatible endpoints (Together AI, Fireworks, etc.).

---

## Files

```
reasoning-trace/
├── python/trace.py              fetch tokens, compute entropy, build HTML
└── typescript/src/trace.ts      same logic, stdlib https only
```

---

## Related work

- **Logit Lens** (nostalgebraist, 2020) — inspecting intermediate layer predictions
- **Attention visualization** — a different (and often misleading) interpretability lens
- **Semantic Uncertainty** (Kuhn et al., 2023) — using entropy to detect hallucination
