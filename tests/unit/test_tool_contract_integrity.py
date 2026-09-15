import json
import pytest
from app.utils.tools import build_tool_prompt, parse_tool_response, validate_tool_result, is_malformed_tool_result

TOOLS = [{"type":"function","function":{"name":"record_test","parameters":{"type":"object","properties":{"value":{"type":"string"}}}}}]

def response(arguments=None, name="record_test"):
    return json.dumps({"tool_calls":[{"name":name,"arguments":arguments if arguments is not None else {"value":"ready"}}]})

@pytest.mark.parametrize("arguments", ["{broken", [1, 2], 7, True])
def test_bad_arguments_are_never_replaced_with_executable_defaults(arguments):
    parsed = parse_tool_response(response(arguments))
    assert is_malformed_tool_result(parsed)
    assert validate_tool_result(parsed, TOOLS) is not None

def test_mixed_batch_is_atomic():
    parsed = parse_tool_response(json.dumps({"tool_calls":[{"name":"record_test","arguments":{}}, {"name":"record_test","arguments":[]}]}))
    assert is_malformed_tool_result(parsed)

@pytest.mark.parametrize("choice", ["required", {"type":"function","function":{"name":"record_test"}}])
def test_required_prompt_has_no_conflicting_plain_text_example(choice):
    prompt = build_tool_prompt("Readiness check.", TOOLS, choice)
    assert '"status": "text"' not in prompt
    assert '"tool_calls"' in prompt

def test_choice_and_declared_names_are_enforced():
    assert validate_tool_result(parse_tool_response(response()), TOOLS, "required") is None
    assert validate_tool_result(parse_tool_response("A normal answer"), TOOLS, "required")
    assert validate_tool_result(parse_tool_response(response(name="not_declared")), TOOLS)
    assert validate_tool_result(parse_tool_response(response()), TOOLS, "none")
    assert validate_tool_result(parse_tool_response("A normal answer"), TOOLS, "auto") is None

@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("scenario", ["required_plain", "unknown_tool", "bad_arguments", "forced_image", "none", "valid"])
def test_openai_wire_contract(client, monkeypatch, stream, scenario):
    from app.routers import openai as api
    calls=[]
    reply = {"required_plain":"A normal answer", "unknown_tool":response(name="not_declared"), "bad_arguments":response([1]), "forced_image":response(), "none":"A normal answer", "valid":response()}[scenario]
    async def generate(prompt, model, *args, **kwargs):
        calls.append(prompt)
        return {"text":reply,"images":[],"thoughts":""}
    async def generate_stream(*args, **kwargs):
        yield {"type":"delta","text":reply}
        yield {"type":"final","text":reply,"images":[],"thoughts":""}
    monkeypatch.setattr(api.gemini_client,"generate",generate)
    monkeypatch.setattr(api.gemini_client,"generate_stream",generate_stream)
    choice = "none" if scenario=="none" else "required"
    if scenario=="forced_image": choice={"type":"function","function":{"name":"record_test"}}
    req={"model":"gemini-flash","stream":stream,"messages":[{"role":"user","content":"generate an image of a cat" if scenario=="forced_image" else "Readiness check."}],"tools":TOOLS,"tool_choice":choice}
    r=client.post("/openai/v1/chat/completions",json=req,headers={"Authorization":"Bearer sk-test-key"})
    frames=[json.loads(l[6:]) for l in r.text.splitlines() if l.startswith("data: ") and l!="data: [DONE]"] if stream else [r.json()]
    errors=[f["error"] for f in frames if f.get("error")]
    failed=scenario in {"required_plain","unknown_tool","bad_arguments"}
    assert bool(errors)==failed
    choices=[c for f in frames for c in f.get("choices",[])]
    if failed:
        assert not any(c.get("finish_reason") in {"stop","tool_calls"} for c in choices)
        assert not any((c.get("delta") or c.get("message") or {}).get("tool_calls") for c in choices)
    elif scenario!="none":
        assert any(c.get("finish_reason")=="tool_calls" for c in choices)
        assert calls and '"tool_calls"' in calls[0]
    else:
        assert not any((c.get("delta") or c.get("message") or {}).get("tool_calls") for c in choices)


@pytest.mark.parametrize("status", [429, 502])
def test_resource_error_does_not_downgrade_extra_thinking(client, monkeypatch, status):
    from app.routers import openai as api
    from app.core.gemini_client import HTTPStatusError
    calls=[]
    async def generate(*args, **kwargs):
        calls.append(kwargs.get("extended_thinking"))
        raise HTTPStatusError(status, "synthetic resource failure")
    async def fallback(*args, **kwargs):return None
    monkeypatch.setattr(api.gemini_client,"generate",generate)
    monkeypatch.setattr(api,"_fallback_result",fallback)
    r=client.post("/openai/v1/chat/completions",json={"model":"gemini-flash-thinking","messages":[{"role":"user","content":"Readiness check."}],"reasoning_effort":"high"},headers={"Authorization":"Bearer sk-test-key"})
    assert r.status_code==status
    assert calls==[True]

@pytest.mark.parametrize("status", [429,502])
def test_resource_error_does_not_replay_full_history(status, monkeypatch):
    import asyncio
    from app.routers import openai as api
    from app.core.gemini_client import HTTPStatusError
    calls=[]
    async def generate(*args, **kwargs):
        calls.append(args)
        raise HTTPStatusError(status,"synthetic resource failure")
    async def fallback(*args, **kwargs):
        if False:yield None
    monkeypatch.setattr(api.gemini_client,"generate",generate)
    monkeypatch.setattr(api,"_maybe_fallback_stream",fallback)
    async def run():
        return [event async for event in api._stream_response_buffered("short continuation","gemini-flash",True,"existing-conversation",messages_raw=[{"role":"user","content":"retained history"}])]
    chunks=asyncio.run(run())
    assert len(calls)==1
    assert any('"error"' in chunk for chunk in chunks)


def test_conversation_recovery_retains_tools(monkeypatch):
    import asyncio
    from types import SimpleNamespace as NS
    from app.routers import openai as api
    calls=[]
    async def generate(prompt,*args,**kwargs):
        calls.append(prompt)
        if len(calls)==1:raise ValueError("synthetic expired conversation")
        return {"text":response(),"images":[],"thoughts":""}
    monkeypatch.setattr(api.gemini_client,"generate",generate)
    req=NS(tools=[NS(model_dump=lambda:TOOLS[0])],tool_choice="required")
    async def run():
        return [event async for event in api._stream_response_buffered("short continuation","gemini-flash",True,"expired-conversation",messages_raw=[{"role":"user","content":"retained history"}],req=req)]
    chunks=asyncio.run(run())
    assert len(calls)==2
    assert "record_test" in calls[1] and "retained history" in calls[1]
    assert any('"tool_calls"' in chunk for chunk in chunks)
