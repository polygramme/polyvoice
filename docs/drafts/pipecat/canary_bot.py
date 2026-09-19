"""Canary in one bot: incumbent vs candidate behind an LLMSwitcher, chosen per session by a stable hash.

LLMSwitcher is a ParallelPipeline with per-branch filters; index 0 is active at start; register_function and
LLMContext(tools=) handlers propagate to all members; both branches share the one LLMContext. Queue the switch
frame BEFORE LLMRunFrame. Give the services distinct names so MetricsData.processor splits TTFB per branch.
"""
import hashlib
import os

from pipecat.frames.frames import LLMRunFrame, ManuallySwitchServiceFrame
from pipecat.pipeline.llm_switcher import LLMSwitcher
from pipecat.pipeline.service_switcher import ServiceSwitcherStrategyManual
from pipecat.services.openai.llm import OpenAILLMService

PROXY = os.environ["POLYVOICE_PROXY_URL"]
CANARY_PCT = int(os.getenv("CANARY_PCT", "10"))


def build_switcher(session_id: str, system: str) -> tuple[LLMSwitcher, OpenAILLMService, OpenAILLMService]:
    common = dict(api_key=os.getenv("POLYVOICE_API_KEY", "polyvoice"), base_url=PROXY)
    incumbent = OpenAILLMService(name="llm-incumbent", **common,
                                 default_headers={"X-Session-Id": session_id, "X-Polyvoice-Branch": "incumbent"},
                                 settings=OpenAILLMService.Settings(model="incumbent", system_instruction=system))
    candidate = OpenAILLMService(name="llm-candidate", **common,
                                 default_headers={"X-Session-Id": session_id, "X-Polyvoice-Branch": "candidate"},
                                 settings=OpenAILLMService.Settings(model="candidate", system_instruction=system))
    return LLMSwitcher(llms=[incumbent, candidate], strategy_type=ServiceSwitcherStrategyManual), incumbent, candidate


def pick_branch(session_id: str, incumbent, candidate):
    bucket = int(hashlib.sha1(session_id.encode()).hexdigest(), 16) % 100  # not hash(): PYTHONHASHSEED
    return candidate if bucket < CANARY_PCT else incumbent


# in bot():
#   llm_switcher, incumbent, candidate = build_switcher(session_id, SYSTEM)
#   pipeline = Pipeline([transport.input(), stt, user_agg, llm_switcher, tts, transport.output(), assistant_agg])
#   @transport.event_handler("on_client_connected")
#   async def on_client_connected(transport, client):
#       branch = pick_branch(session_id, incumbent, candidate)
#       if branch is not llm_switcher.active_llm:
#           await worker.queue_frames([ManuallySwitchServiceFrame(service=branch)])
#       context.add_message({"role": "developer", "content": "Greet the caller briefly."})
#       await worker.queue_frames([LLMRunFrame()])
