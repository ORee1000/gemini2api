"""Synthetic regressions for retries, state commits and stream assembly."""
import asyncio
import json
from types import SimpleNamespace as NS

import pytest

from app.core.gemini_client import HTTPStatusError
from app.core.conversation_store import Conversation
from app.routers import openai as api
from app.utils.tools import is_malformed_tool_result, parse_tool_response

TOOLS = [{"type": "function", "function": {"name": "probe", "parameters": {"type": "object"}}}]
GOOD = json.dumps({"tool_calls": [{"name": "probe", "arguments": {"value": 1}}]})
BAD = '{"tool_calls": ['
AUTH = {"Authorization": "Bearer sk-test-key"}


def frames(chunks):
    return [json.loads(line[6:]) for chunk in chunks for line in chunk.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"]


def test_malformed_calls_cannot_hide_behind_text_status():
    text = json.dumps({"status": "text", "content": "ready", "tool_calls": [
        {"name": "probe", "arguments": {}}, {"name": "probe", "arguments": []}]})
    assert is_malformed_tool_result(parse_tool_response(text))


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("status", [429, 502])
def test_tool_regeneration_preserves_upstream_failure(gem_client, monkeypatch, stream, status):
    calls = []
    async def generate(*args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            return {"text": BAD}
        raise HTTPStatusError(status, "synthetic upstream resource failure")
    monkeypatch.setattr(api.gemini_client, "generate", generate)
    res = gem_client.post("/openai/v1/chat/completions", headers=AUTH, json={
        "model": "gemini-flash", "stream": stream, "tools": TOOLS,
        "tool_choice": "required", "messages": [{"role": "user", "content": "Probe."}]})
    errors = [f["error"] for f in frames([res.text]) if "error" in f] if stream else [res.json()["error"]]
    assert len(calls) == 2
    assert len(errors) == 1
    assert "synthetic upstream resource failure" in errors[0]["message"]
    assert (errors[0]["code"] if stream else res.status_code) == status


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("valid", [False, True])
def test_only_accepted_generation_is_committed(gem_client, monkeypatch, stream, valid):
    calls = []
    updates = []
    conv = Conversation("synthetic", "before")
    async def get(*args):return conv
    async def update(value):updates.append(value.to_dict())
    async def generate(*args, **kwargs):
        calls.append(args)
        success = valid and len(calls) == 2
        return {"text": GOOD if success else BAD, "conversation_id": "accepted" if success else "rejected",
                "thoughts": "accepted-reasoning" if success else "rejected-reasoning"}
    monkeypatch.setattr(api.conversation_store, "get", get)
    monkeypatch.setattr(api.conversation_store, "update", update)
    monkeypatch.setattr(api.gemini_client, "generate", generate)
    res = gem_client.post("/openai/v1/chat/completions", headers=AUTH, json={
        "model": "gemini-flash-thinking", "stream": stream, "conversation_id": "synthetic",
        "tools": TOOLS, "tool_choice": "required", "messages": [{"role": "user", "content": "Probe."}]})
    assert len(calls) == 2
    if valid:
        assert len(updates) == 1
        assert conv.gemini_conv_id == "accepted"
        assert conv.messages[-1]["content"] == GOOD
        assert "accepted-reasoning" in res.text and "rejected-reasoning" not in res.text
    else:
        assert not updates and not conv.messages and conv.gemini_conv_id == "before"


@pytest.mark.parametrize("event", [{"type": "thoughts", "text": "working"},
                                    {"type": "final", "text": "ready", "thoughts": ""}])
def test_partial_stream_never_replays_or_switches_provider(monkeypatch, event):
    extra_calls = []
    async def generate_stream(*args, **kwargs):
        yield event
        raise ValueError("synthetic interrupted stream")
    async def generate(*args, **kwargs):
        extra_calls.append("regenerate")
        return {"text": "a different answer"}
    async def fallback(*args, **kwargs):
        extra_calls.append("fallback")
        if False:yield ""
    monkeypatch.setattr(api.gemini_client, "generate_stream", generate_stream)
    monkeypatch.setattr(api.gemini_client, "generate", generate)
    monkeypatch.setattr(api, "_maybe_fallback_stream", fallback)
    async def run():
        return [s async for s in api._stream_response("Probe.", "gemini-flash-thinking", False,
                 "old", messages_raw=[{"role": "user", "content": "Probe."}])]
    data = frames(asyncio.run(run()))
    assert not extra_calls
    assert len([f for f in data if "error" in f]) == 1


def test_thoughts_only_is_error_without_splicing_fallback(monkeypatch):
    calls = []
    async def generate_stream(*args, **kwargs):
        yield {"type": "thoughts", "text": "working"}
        yield {"type": "final", "text": "", "thoughts": "working"}
    async def fallback(*args, **kwargs):
        calls.append("fallback")
        if False:yield ""
    monkeypatch.setattr(api.gemini_client, "generate_stream", generate_stream)
    monkeypatch.setattr(api, "_maybe_fallback_stream", fallback)
    async def run():return [s async for s in api._stream_response("Probe.", "gemini-flash-thinking", False)]
    data = frames(asyncio.run(run()))
    assert not calls
    assert len([f for f in data if "error" in f]) == 1
    assert not any(c.get("finish_reason") == "stop" for f in data for c in f.get("choices", []))


def test_successful_stream_recovery_actually_delivers_answer(monkeypatch):
    calls = []
    async def generate_stream(*args, **kwargs):
        raise ValueError("synthetic expired conversation")
        yield
    async def generate(*args, **kwargs):
        calls.append(kwargs)
        return {"text": "recovered answer", "thoughts": "recovered reasoning"}
    monkeypatch.setattr(api.gemini_client, "generate_stream", generate_stream)
    monkeypatch.setattr(api.gemini_client, "generate", generate)
    async def run():
        return [s async for s in api._stream_response("Probe.", "gemini-flash-thinking", False,
                    "old", messages_raw=[{"role": "user", "content": "Probe."}], extended_thinking=True)]
    data = frames(asyncio.run(run()))
    deltas = [c["delta"] for f in data for c in f.get("choices", [])]
    assert "".join(d.get("content") or "" for d in deltas) == "recovered answer"
    assert "".join(d.get("reasoning_content") or "" for d in deltas) == "recovered reasoning"
    assert calls[0]["extended_thinking"] is True


def test_stream_tool_calls_have_separate_indices(monkeypatch):
    async def generate(*args, **kwargs):
        return {"text": json.dumps({"tool_calls": [{"name": "probe", "arguments": {"value": i}} for i in range(2)]})}
    monkeypatch.setattr(api.gemini_client, "generate", generate)
    req = NS(tools=[NS(model_dump=lambda: TOOLS[0])], tool_choice="required")
    async def run():return [s async for s in api._stream_response_buffered("Probe.", "gemini-flash", True, req=req)]
    calls = [t for f in frames(asyncio.run(run())) for c in f.get("choices", []) for t in c["delta"].get("tool_calls") or []]
    assert [t.get("index") for t in calls] == [0, 1]
    assert [json.loads(t["function"]["arguments"])["value"] for t in calls] == [0, 1]
    assert len({t["id"] for t in calls}) == 2


def test_http_recovery_preserves_terminal_resource_error(gem_client, monkeypatch):
    calls = []
    async def get(*args):return Conversation("synthetic", "old")
    async def generate(*args, **kwargs):
        calls.append(args)
        if len(calls) == 1:raise ValueError("synthetic expired conversation")
        raise HTTPStatusError(429, "synthetic terminal resource error")
    async def fallback(*args, **kwargs):return None
    monkeypatch.setattr(api.conversation_store, "get", get)
    monkeypatch.setattr(api.gemini_client, "generate", generate)
    monkeypatch.setattr(api, "_fallback_result", fallback)
    res = gem_client.post("/openai/v1/chat/completions", headers=AUTH, json={
        "model": "gemini-flash", "conversation_id": "synthetic",
        "messages": [{"role": "user", "content": "Probe."}]})
    assert res.status_code == 429
    assert "synthetic terminal resource error" in res.json()["error"]["message"]
    assert len(calls) == 2


def test_buffered_tool_retry_uses_recovered_prompt_and_conversation(monkeypatch):
    calls = []
    async def generate(prompt, model, conversation_id="", *args, **kwargs):
        calls.append((prompt, conversation_id))
        if len(calls) == 1:raise ValueError("synthetic expired conversation")
        return {"text": BAD if len(calls) == 2 else GOOD}
    monkeypatch.setattr(api.gemini_client, "generate", generate)
    req = NS(tools=[NS(model_dump=lambda: TOOLS[0])], tool_choice="required")
    async def run():
        return [s async for s in api._stream_response_buffered("short", "gemini-flash", True,
               "old", messages_raw=[{"role": "user", "content": "retained synthetic context"}], req=req)]
    data = frames(asyncio.run(run()))
    assert len(calls) == 3
    assert calls[1] == calls[2]
    assert calls[2][1] == "" and "retained synthetic context" in calls[2][0]
    assert any(c.get("finish_reason") == "tool_calls" for f in data for c in f.get("choices", []))
