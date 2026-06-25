/**
 * Reasoning Trace Visualizer — from scratch (TypeScript).
 *
 * Fetches tokens with logprobs from an LLM, computes per-token Shannon
 * entropy, and generates a self-contained HTML report with color-coded
 * tokens and hover-over alternatives.
 *
 * Run:
 *   npx ts-node src/trace.ts
 *   OPENAI_API_KEY=sk-... npx ts-node src/trace.ts   (explicit override)
 *
 * API key resolution order:
 *   1. OPENAI_API_KEY environment variable
 *   2. .env or .env.local in this directory or any parent
 *
 * Output: reasoning_trace.html
 *
 * Dependencies: stdlib (https, fs, path) only.
 */

import * as https from "https";
import * as fs    from "fs";
import * as path  from "path";

// ──────────────────────────────────────────────────────────────────────────────
// Env-file loader
// ──────────────────────────────────────────────────────────────────────────────
function loadEnvFiles(): void {
  let dir = __dirname;
  const visited = new Set<string>();
  while (dir && !visited.has(dir)) {
    visited.add(dir);
    for (const name of [".env.local", ".env"]) {
      const filePath = path.join(dir, name);
      if (fs.existsSync(filePath)) {
        const lines = fs.readFileSync(filePath, "utf-8").split("\n");
        for (const raw of lines) {
          const line = raw.trim();
          if (!line || line.startsWith("#") || !line.includes("=")) continue;
          const eqIdx = line.indexOf("=");
          const key   = line.slice(0, eqIdx).trim();
          const val   = line.slice(eqIdx + 1).trim().replace(/^["']|["']$/g, "");
          if (key && !(key in process.env)) process.env[key] = val;
        }
      }
    }
    const parent = path.dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
}

loadEnvFiles();

const API_KEY  = process.env.OPENAI_API_KEY ?? "";
const MODEL    = process.env.TRACE_MODEL    ?? "gpt-4o-mini";
const OUT_FILE = path.join(__dirname, "..", "reasoning_trace.html");

if (!API_KEY) {
  console.error(
    "OPENAI_API_KEY not found.\n" +
    "Set it in your environment or create a .env / .env.local file:\n" +
    "  OPENAI_API_KEY=sk-..."
  );
  process.exit(1);
}

const PROBLEM =
  "A webhook-based payments service sometimes charges users twice when the " +
  "webhook is retried. Explain step by step why this happens and how to fix it.";

// ──────────────────────────────────────────────────────────────────────────────
// TracedToken
// ──────────────────────────────────────────────────────────────────────────────
interface TracedToken {
  text:         string;
  logprob:      number;
  topLogprobs:  Record<string, number>;   // token → logprob
  prob:         number;
  entropy:      number;
}

function makeToken(text: string, logprob: number, topLogprobs: Record<string, number>): TracedToken {
  const prob    = Math.exp(logprob);
  const probs   = Object.values(topLogprobs).map(lp => Math.exp(lp));
  const total   = probs.reduce((a, b) => a + b, 0) || 1;
  const entropy = -probs.reduce((s, p) => {
    const pn = p / total;
    return s + (pn > 0 ? pn * Math.log2(pn) : 0);
  }, 0);
  return { text, logprob, topLogprobs, prob, entropy };
}

// ──────────────────────────────────────────────────────────────────────────────
// LLM fetch
// ──────────────────────────────────────────────────────────────────────────────
async function streamTokens(prompt: string): Promise<TracedToken[]> {
  const body = JSON.stringify({
    model:        MODEL,
    messages:     [{ role: "user", content: prompt }],
    logprobs:     true,
    top_logprobs: 5,
    max_tokens:   600,
    stream:       false,
  });

  return new Promise((resolve, reject) => {
    const buf  = Buffer.from(body, "utf-8");
    const opts: https.RequestOptions = {
      hostname: "api.openai.com",
      path:     "/v1/chat/completions",
      method:   "POST",
      headers:  {
        "Content-Type":   "application/json",
        "Authorization":  `Bearer ${API_KEY}`,
        "Content-Length": buf.length,
      },
    };
    const req = https.request(opts, res => {
      let raw = "";
      res.on("data", c => raw += c);
      res.on("end", () => {
        try {
          const data = JSON.parse(raw);
          if (data.error) { reject(new Error(data.error.message)); return; }
          const tokens = (data.choices[0].logprobs.content as any[]).map((lp: any) => {
            const top: Record<string, number> = {};
            for (const t of (lp.top_logprobs ?? [])) top[t.token] = t.logprob;
            top[lp.token] = lp.logprob;
            return makeToken(lp.token, lp.logprob, top);
          });
          resolve(tokens);
        } catch (e) { reject(e); }
      });
    });
    req.on("error", reject);
    req.write(buf);
    req.end();
  });
}

// ──────────────────────────────────────────────────────────────────────────────
// Stats
// ──────────────────────────────────────────────────────────────────────────────
function runningStats(arr: number[]): [number, number] {
  const mu  = arr.reduce((a, b) => a + b, 0) / (arr.length || 1);
  const std = Math.sqrt(arr.reduce((s, x) => s + (x - mu) ** 2, 0) / (arr.length || 1)) || 1;
  return [mu, std];
}

// ──────────────────────────────────────────────────────────────────────────────
// HTML generation
// ──────────────────────────────────────────────────────────────────────────────
const CSS = `
<style>
  body{font-family:'Courier New',monospace;background:#0d1117;color:#c9d1d9;max-width:860px;margin:40px auto;padding:20px;line-height:1.7}
  h1{color:#58a6ff;border-bottom:1px solid #30363d;padding-bottom:12px}
  h2{color:#79c0ff;font-size:1em;margin-top:28px}
  .meta{color:#8b949e;font-size:.85em;margin-bottom:24px}
  .trace{font-size:.95em;line-height:2}
  .tok{border-radius:3px;padding:1px 2px;cursor:default;position:relative}
  .tok .tip{display:none;position:absolute;bottom:130%;left:50%;transform:translateX(-50%);
            background:#161b22;border:1px solid #30363d;padding:6px 10px;border-radius:4px;
            font-size:.78em;white-space:nowrap;z-index:10;color:#c9d1d9}
  .tok:hover .tip{display:block}
  .legend{display:flex;gap:18px;margin-bottom:18px;font-size:.82em}
  .legend-item{display:flex;align-items:center;gap:6px}
  .swatch{width:18px;height:14px;border-radius:2px;display:inline-block}
  table{width:100%;border-collapse:collapse;font-size:.85em;margin-top:12px}
  th{text-align:left;color:#8b949e;border-bottom:1px solid #21262d;padding:6px 8px}
  td{padding:5px 8px;border-bottom:1px solid #161b22}
  .bar{height:10px;background:#238636;border-radius:2px}
</style>`;

function escapeHtml(s: string): string {
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

function entropyColor(entropy: number, mean: number, std: number): string {
  const z = Math.max(-2, Math.min(2, (entropy - mean) / std));
  const t = (z + 2) / 4;
  const lerp = (a: number, b: number) => Math.round(a + t * (b - a));
  const r = lerp(0x58, 0xff), g = lerp(0xa6, 0x7b), b = lerp(0xff, 0x00);
  const a = (0.15 + 0.55 * t).toFixed(2);
  return `rgba(${r},${g},${b},${a})`;
}

function buildHtml(tokens: TracedToken[], problem: string, model: string): string {
  const entropies = tokens.map(t => t.entropy);
  const [mean, std] = runningStats(entropies);

  const spans = tokens.map(tok => {
    const color = entropyColor(tok.entropy, mean, std);
    const alts  = Object.entries(tok.topLogprobs)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([t, lp]) => `<span style="display:block"><b>${escapeHtml(JSON.stringify(t))}</b> ${(Math.exp(lp)*100).toFixed(1)}%</span>`)
      .join("");
    return `<span class="tok" style="background:${color}">${escapeHtml(tok.text)}<span class="tip">${alts}</span></span>`;
  }).join("");

  const top10 = [...tokens].sort((a, b) => b.entropy - a.entropy).slice(0, 10);
  const tableRows = top10.map((tok, i) =>
    `<tr><td>${i+1}</td><td><code>${escapeHtml(JSON.stringify(tok.text))}</code></td>` +
    `<td>${(tok.prob*100).toFixed(2)}%</td><td>${tok.entropy.toFixed(3)}</td>` +
    `<td><div class="bar" style="width:${Math.min(100,tok.entropy*20).toFixed(0)}px"></div></td></tr>`
  ).join("\n");

  return `<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Reasoning Trace</title>${CSS}</head>
<body>
<h1>Reasoning Trace Visualizer</h1>
<p class="meta">Model: <b>${escapeHtml(model)}</b> &nbsp;|&nbsp; Tokens: <b>${tokens.length}</b> &nbsp;|&nbsp; Mean entropy: <b>${mean.toFixed(3)} bits</b> &nbsp;|&nbsp; Hover tokens for alternatives</p>
<p class="meta"><b>Problem:</b> ${escapeHtml(problem)}</p>
<div class="legend">
  <span class="legend-item"><span class="swatch" style="background:rgba(88,166,255,0.20)"></span>Low entropy (certain)</span>
  <span class="legend-item"><span class="swatch" style="background:rgba(200,120,60,0.50)"></span>Mid entropy</span>
  <span class="legend-item"><span class="swatch" style="background:rgba(255,80,0,0.70)"></span>High entropy (uncertain)</span>
</div>
<h2>Token stream (hover for top alternatives)</h2>
<div class="trace">${spans}</div>
<h2>Top-10 highest-entropy tokens</h2>
<table><tr><th>#</th><th>Token</th><th>Prob</th><th>Entropy (bits)</th><th>Bar</th></tr>
${tableRows}
</table>
</body></html>`;
}

// ──────────────────────────────────────────────────────────────────────────────
// Main
// ──────────────────────────────────────────────────────────────────────────────
async function main(): Promise<void> {
  const HR = "=".repeat(60);
  console.log(HR);
  console.log("  Reasoning Trace Visualizer  (TypeScript)");
  console.log(`  Model: ${MODEL}`);
  console.log(HR);
  console.log(`\nProblem: ${PROBLEM}\n`);
  console.log("Streaming tokens…");

  const tokens    = await streamTokens(PROBLEM);
  const entropies = tokens.map(t => t.entropy);
  const [mean, std] = runningStats(entropies);

  console.log(`\nCollected ${tokens.length} tokens`);
  console.log(`Entropy  — mean: ${mean.toFixed(3)} bits  std: ${std.toFixed(3)} bits`);

  const top5 = [...tokens].sort((a, b) => b.entropy - a.entropy).slice(0, 5);
  console.log("\nTop-5 highest-entropy tokens:");
  top5.forEach((tok, i) => {
    console.log(`  ${i+1}. ${JSON.stringify(tok.text).padEnd(22)} entropy=${tok.entropy.toFixed(3)}  prob=${(tok.prob*100).toFixed(1)}%`);
  });

  const report = buildHtml(tokens, PROBLEM, MOCK_MODE ? "mock" : MODEL);
  fs.writeFileSync(OUT_FILE, report, "utf-8");
  console.log(`\nHTML report written to: ${OUT_FILE}`);
  console.log("Open it in any browser. Hover tokens to see alternatives.");
}

main().catch(console.error);
