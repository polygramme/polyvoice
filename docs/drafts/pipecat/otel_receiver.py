"""OTLP/HTTP receiver: maps Pipecat `llm` spans to polyvoice trace records.

pip install fastapi uvicorn opentelemetry-proto. Bot side: setup_tracing(service_name, exporter=OTLPSpanExporter())
and PipelineWorker(enable_tracing=True, conversation_id=session_id). `llm` spans carry no conversation.id;
resolve via parent ids, and keep the pending list across batches because llm spans arrive before their turn
span closes. Missing vs the observer: tool results, finish reason, end reason, node name; speculative requests
are indistinguishable.
"""
import json

from fastapi import FastAPI, Request, Response
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

app = FastAPI()
turns: dict[bytes, dict] = {}
convs: dict[bytes, str] = {}
pending: list[tuple] = []
records: list[dict] = []


def attrs(span) -> dict:
    out = {}
    for kv in span.attributes:
        v = kv.value
        k = v.WhichOneof("value")
        out[kv.key] = getattr(v, k) if k else None
    return out


@app.post("/v1/traces")
async def traces(req: Request):
    msg = ExportTraceServiceRequest()
    msg.ParseFromString(await req.body())
    for rs in msg.resource_spans:
        for ss in rs.scope_spans:
            for s in ss.spans:
                a = attrs(s)
                if s.name == "conversation":
                    convs[s.span_id] = a.get("conversation.id")
                elif s.name == "turn":
                    turns[s.span_id] = {"session": a.get("conversation.id"), "turn": a.get("turn.number"),
                                        "interrupted": a.get("turn.was_interrupted")}
                elif s.name == "llm":
                    pending.append((s, a))
    still = []
    for s, a in pending:
        t = turns.get(s.parent_span_id)
        if t is None and s.parent_span_id not in convs:
            still.append((s, a))  # parent not seen yet
            continue
        t = t or {"session": convs.get(s.parent_span_id), "turn": None, "interrupted": None}
        records.append({
            "session_id": t["session"], "turn": t["turn"], "was_interrupted": t["interrupted"],
            "messages": json.loads(a["input"]) if "input" in a else [],
            "tools": json.loads(a["tools"]) if "tools" in a else None,
            "reply": a.get("output", ""), "llm_ttfb_s": a.get("metrics.ttfb"),
            "model": a.get("gen_ai.request.model"), "system": a.get("gen_ai.system_instructions"),
            "usage": {"prompt_tokens": a.get("gen_ai.usage.input_tokens"),
                      "completion_tokens": a.get("gen_ai.usage.output_tokens")},
            "started_at": s.start_time_unix_nano / 1e9, "ended_at": s.end_time_unix_nano / 1e9,
            "token_exact": False, "source": "otel",
        })
    pending[:] = still
    return Response(status_code=200, media_type="application/x-protobuf")
