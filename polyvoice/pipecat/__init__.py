"""Pipecat integration: the LLM slot (proxy-backed OpenAI service with session/turn headers) and the trace
observer. Depends only on pipecat-ai; install with `pip install polyvoice[pipecat]`."""
from polyvoice.pipecat.llm import PolyvoiceLLMService, polyvoice_llm
from polyvoice.pipecat.observer import PolyvoiceObserver

__all__ = ["PolyvoiceLLMService", "PolyvoiceObserver", "polyvoice_llm"]
