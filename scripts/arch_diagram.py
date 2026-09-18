"""Render docs/architecture.svg for polyvoice (stdlib only). Run: python scripts/arch_diagram.py"""
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

W, H = 1180, 560
F = "font-family='Helvetica, Arial, sans-serif'"
INK, MUTE, LINE, LANE = "#1f2933", "#52606d", "#9aa5b1", "#8a94a6"
FILL = {"coval": "#fde8e8", "proxy": "#fff4d6", "server": "#e9f7ef", "loop": "#f3eefc", "store": "#f5f7fa", "gpu": "#eef2f7"}
out: list[str] = []


def text(x, y, s, size=12, weight="400", fill=MUTE, anchor="start"):
    out.append(f"<text x='{x}' y='{y}' {F} font-size='{size}' font-weight='{weight}' fill='{fill}' text-anchor='{anchor}'>{escape(s)}</text>")


def box(x, y, w, h, title, lines=(), kind="store", dashed=False):
    d = " stroke-dasharray='6 4'" if dashed else ""
    out.append(f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='10' fill='{FILL[kind]}' stroke='{LINE}' stroke-width='1.2'{d}/>")
    text(x + 12, y + 22, title, 14, "700", INK)
    for i, t in enumerate(lines):
        text(x + 12, y + 42 + 16 * i, t)


def route(points, label="", dash=False, head=True, lx=None, ly=None, anchor="middle"):
    d = " stroke-dasharray='5 4'" if dash else ""
    m = " marker-end='url(#a)'" if head else ""
    out.append(f"<polyline points='{' '.join(f'{x},{y}' for x, y in points)}' fill='none' stroke='{INK}' stroke-width='1.4'{d}{m}/>")
    if label:
        text(lx, ly, label, 11, anchor=anchor)


out.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}'>")
out.append("<defs><marker id='a' markerWidth='10' markerHeight='10' refX='9' refY='5' orient='auto'>"
           f"<path d='M0,0 L10,5 L0,10 z' fill='{INK}'/></marker></defs>")
out.append(f"<rect width='{W}' height='{H}' fill='white'/>")
text(24, 34, "polyvoice: a voice agent's LLM improves from its own calls", 20, "700", INK)
text(24, 54, "Coval simulates the caller and scores the call; polyloop trains on the captured turns and gates the result; the proxy serves the winner", 12)

# GPU box outline
out.append(f"<rect x='420' y='80' width='736' height='200' rx='12' fill='{FILL['gpu']}' stroke='{LINE}' stroke-width='1' stroke-dasharray='6 4'/>")
text(432, 98, "your GPU node (Modal H100:2, Lambda, …)", 11, "700", LANE)

box(24, 110, 330, 120, "Coval", ["persona (\"Standard Customer\") phones the agent", "test sets: pool 16 scenarios · held-out 8", "metrics: expected behaviours met (composite),", "never states a price / diagnosis (binary judge)"], "coval")
box(440, 110, 320, 120, "polyloop proxy  (public https)", ["receptionist system prompt prepended", "session id = Coval's simulation id", "records every turn token-exact", "serves the adapter under test (/admin/serve)"], "proxy")
box(830, 110, 310, 120, "SkyRL Tinker server", ["trainer GPU: Megatron LoRA", "sampler GPU: vLLM, multi-adapter", "Qwen3.5-4B (PhoneLLM: trainer loads,", "sampler pending)"], "server")
route([(354, 150), (440, 150)], "chat completions", lx=397, ly=144)
route([(440, 190), (354, 190)], "replies", lx=397, ly=206)
route([(760, 170), (830, 170)], "sample", lx=795, ly=164)

# lane 2: loop stages
y = 330
text(24, y - 14, "one cycle  (polyloop run --loop recipes/dental/loop.yaml)", 13, "700", INK)
stages = [("filter", ["8 scenarios × 2 calls", "under the incumbent"]), ("train (opsd)", ["rows from the captured", "turns + judge hints"]),
          ("evaluate", ["8 held-out × 4, base", "then candidate, paired"]), ("gate", ["bootstrap CI, regressions;", "receipt.json"]), ("promote", ["approve → live.json →", "proxy serves it"])]
xs = [24 + 232 * i for i in range(len(stages))]
for x, (name, lines) in zip(xs, stages):
    box(x, y, 210, 74, name, lines, "loop")
for i in range(len(xs) - 1):
    route([(xs[i] + 210, y + 37), (xs[i + 1], y + 37)])

# flows
route([(129, 230), (129, y)], "runs: launch, poll, rewards by simulation id", lx=140, ly=290, anchor="start")
route([(361, 230), (361, y)], "held-out runs", lx=372, ly=290, anchor="start")
route([(600, 230), (600, y)], "traces + ledger → rows", lx=612, ly=290, anchor="start")
route([(1060, y), (1060, 250), (985, 250), (985, 230)], "candidate adapter", lx=1070, ly=290, anchor="start")

# lane 3: artifacts
ay = 450
box(24, ay, 440, 84, "coval_ledger.jsonl", ["every simulated call: session, scenario, policy, label,", "reward, judge explanation; held-out sessions", "excluded from training"], "store")
box(490, ay, 320, 84, "cycles/<id>/receipt.json", ["cycle 1: base 0.888 → candidate 0.938,", "delta +0.049, CI [−0.021, 0.117], 4/1/3", "→ rejected"], "store")
box(836, ay, 320, 84, "Coval dashboard", ["the same runs, transcripts and", "metric verdicts, per simulation"], "coval")
route([(129, y + 74), (129, ay)])
route([(750, y + 74), (750, ay)], "receipt", lx=762, ly=420, anchor="start")
route([(361, y + 74), (361, 430), (996, 430), (996, ay)], dash=True)
out.append("</svg>")
Path(__file__).resolve().parents[1].joinpath("docs", "architecture.svg").write_text("\n".join(out))
print("wrote docs/architecture.svg")
