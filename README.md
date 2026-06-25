# reasoning-trace

> **Mean token entropy: 0.83 bits. Top uncertainty spike: 2.51 bits at " or".**  
> Every token the model generates had alternatives. This shows you what they were.

Stream tokens from any OpenAI-compatible model with log-probabilities, compute per-token [Shannon entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)), and render a color-coded HTML uncertainty map. Zero dependencies — pure Python stdlib + TypeScript.

```bash
OPENAI_API_KEY=sk-... python python/trace.py
# → reasoning_trace.html (open in browser)
```

---

## What you get

```
Collected 600 tokens
Entropy — mean: 0.826 bits   std: 0.741 bits

Top-5 highest-entropy tokens:
  1. ' or'       entropy=2.513   prob=8.2%    ← model considered 4 directions here
  2. ' help'     entropy=2.503   prob=4.8%
  3. ' an'       entropy=2.477   prob=5.6%
  4. ' Perform'  entropy=2.355   prob=6.5%
  5. '-level'    entropy=2.315   prob=3.3%
```

The HTML report:

```
┌─────────────────────────────────────────────────────────────────┐
│  Token stream (hover any token for the top-5 alternatives)      │
│                                                                 │
│  The [fix] is [to] store [a] unique [webhook] ID               │
│   ████  ██   ████  ██  ███████████████████████                 │
│   blue  blue  red  blue       red  (high entropy)              │
│                                                                 │
│  Entropy:  █ 0–0.5 bits (certain)  █ 0.5–1.5  █ 1.5+ (unsure) │
└─────────────────────────────────────────────────────────────────┘
```

Hover any token to see the top-5 alternatives the model considered, with probabilities.

---

## What entropy tells you

At each token position, the model assigns a probability distribution over its vocabulary. Shannon entropy measures how spread out that distribution is:

```
H(p) = -Σ  p_i · log₂(p_i)     [bits]
         i
```

| Entropy | What it means | Color |
|---------|--------------|-------|
| ~0 bits | One token dominates — model is certain | Blue |
| ~1 bit | 2 plausible continuations | Yellow |
| ~2 bits | 4 plausible continuations | Orange |
| ~3+ bits | Model is genuinely unsure | Red |

**High-entropy tokens are the interesting ones** — inflection points where the model switches logical tracks, introduces new concepts, or hedges. Low-entropy runs are routine reasoning.

---

## Run

```bash
# Python — writes reasoning_trace.html
OPENAI_API_KEY=sk-... python python/trace.py

# TypeScript — writes reasoning_trace.html
OPENAI_API_KEY=sk-... npx ts-node typescript/src/trace.ts

# Custom prompt
OPENAI_API_KEY=sk-... python python/trace.py --prompt "Explain the CAP theorem"

# Different model
TRACE_MODEL=gpt-4o python python/trace.py

# Any OpenAI-compatible endpoint (Together, Fireworks, etc.)
OPENAI_API_KEY=... OPENAI_BASE_URL=https://api.together.ai/v1 TRACE_MODEL=mistral-7b python python/trace.py
```

---

## Environment

| Variable | Default | Notes |
|----------|---------|-------|
| `OPENAI_API_KEY` | required | Read from env, `.env`, or `.env.local` |
| `TRACE_MODEL` | `gpt-4o-mini` | Any model that supports `logprobs` |
| `TRACE_PROMPT` | built-in debug prompt | Override with `--prompt` flag |

---

## Why logprobs?

The `logprobs` + `top_logprobs` parameters are standard across OpenAI-compatible APIs:
- OpenAI (GPT-4o, GPT-4o-mini)
- Together AI, Fireworks AI, Anyscale
- Local models via Ollama, vLLM, LMStudio

Any model that supports these parameters works with this tool.

---

## Related work

- **Logit Lens** (nostalgebraist, 2020) — inspecting intermediate layer predictions, not final token distributions
- **Semantic Uncertainty** (Kuhn et al., 2023) — using entropy as a hallucination signal, consistent with what this shows at token level
- **Attention visualization** — a different (and often misleading) interpretability lens; entropy on the output distribution is more directly meaningful

---

## Files

```
reasoning-trace/
├── python/trace.py          fetch tokens, compute entropy, build HTML (stdlib only)
└── typescript/src/trace.ts  same logic, stdlib https only
```

---

## License

MIT
