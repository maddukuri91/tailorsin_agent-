"""Anti-hallucination regression tests (all CRM/LLM calls mocked)."""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agents._utils import (
    context_mobile,
    execute_tool_call,
    extract_success_message,
    generate_agent_replies,
    is_success_result,
    is_tool_failure_content,
    looks_like_success_claim,
    resolve_primary_contact,
    validate_tool_args,
)
from agents.signup import _latest_signup_name
from services.conversation_service import (
    _build_replies,
    _had_verified_tool_success,
    _has_tool_error,
)


def _tool_call_response(content="calling"):
    return AIMessage(
        content=content,
        tool_calls=[{"name": "t", "args": {}, "id": "call-1"}],
    )


def test_logical_crm_failure_is_not_completed():
    out = execute_tool_call(
        _tool_call_response(),
        [HumanMessage(content="hi")],
        {"t": lambda: {"status": "error", "message": "duplicate"}},
    )
    assert out.get("completed") is False
    tool_text = out["messages"][-1].content
    assert tool_text.startswith("Tool error:")
    assert is_tool_failure_content(tool_text)


def test_transport_exception_is_not_completed():
    def boom():
        raise RuntimeError("CRM down")

    out = execute_tool_call(
        _tool_call_response(), [HumanMessage(content="hi")], {"t": boom}
    )
    assert out.get("completed") is False
    assert out["messages"][-1].content.startswith("Tool error:")


def test_unknown_tool_is_not_completed():
    out = execute_tool_call(
        _tool_call_response(), [HumanMessage(content="hi")], {}
    )
    assert out.get("completed") is False
    assert is_tool_failure_content(out["messages"][-1].content)


def test_verified_success_is_completed():
    out = execute_tool_call(
        _tool_call_response(),
        [HumanMessage(content="hi")],
        {"t": lambda: {"status": "success", "message": "thanks!"}},
    )
    assert out.get("completed") is True
    assert extract_success_message(out["api_result"]) == "thanks!"


def test_is_success_result_gate():
    assert is_success_result({"status": "success", "message": "ok"})
    assert is_success_result({"status": "SUCCESS"})
    assert not is_success_result({"status": "error", "message": "bad"})
    assert not is_success_result({"error": "bad"})
    assert not is_success_result(None)
    assert not is_success_result("ok")
    assert not is_success_result({})


def test_hallucinated_claim_filtered_on_tool_error():
    before = {
        "messages": [HumanMessage(content="hi")],
        "completed": False,
        "api_result": None,
    }
    result = {
        "messages": before["messages"]
        + [
            AIMessage(content="Done! You are registered!"),
            ToolMessage(content="Tool error: boom", tool_call_id="call-1"),
        ],
        "completed": False,
        "api_result": None,
    }
    assert _has_tool_error(before, result)
    assert not _had_verified_tool_success(before, result)
    replies = _build_replies(before, result)
    assert all("done" not in r.text.lower() for r in replies)
    assert all("register" not in r.text.lower() for r in replies)


def test_logical_error_json_never_leaks_raw_payload():
    before = {
        "messages": [HumanMessage(content="hi")],
        "completed": False,
        "api_result": {"status": "error"},
    }
    result = {
        "messages": before["messages"]
        + [
            AIMessage(content="All done!"),
            ToolMessage(
                content=json.dumps({"status": "error", "message": "bad"}),
                tool_call_id="call-1",
            ),
        ],
        "completed": False,
        "api_result": {"status": "error"},
    }
    replies = _build_replies(before, result)
    assert all("status" not in r.text for r in replies)
    assert all("done" not in r.text.lower() for r in replies)


def test_verified_success_surfaces_friendly_message():
    before = {
        "messages": [HumanMessage(content="hi")],
        "completed": False,
        "api_result": None,
    }
    payload = {"status": "success", "message": "thank you! our agent will call"}
    result = {
        "messages": before["messages"]
        + [ToolMessage(content=json.dumps(payload), tool_call_id="call-1")],
        "completed": True,
        "api_result": payload,
    }
    assert _had_verified_tool_success(before, result)
    replies = _build_replies(before, result)
    assert any("thank you" in r.text.lower() for r in replies)
    assert all("status" not in r.text for r in replies)


def test_verified_success_without_crm_message_is_not_generic_retry():
    before = {
        "messages": [HumanMessage(content="hi")],
        "completed": False,
        "api_result": None,
    }
    result = {
        "messages": before["messages"]
        + [
            ToolMessage(
                content=json.dumps({"status": "success", "type": "client"}),
                tool_call_id="call-1",
            )
        ],
        "completed": True,
        "api_result": {"status": "success", "type": "client"},
    }
    replies = _build_replies(before, result)
    assert replies
    assert "not sure I caught" not in replies[0].text.lower()


def test_generic_success_fallback_only_on_verified_success():
    before = {
        "messages": [HumanMessage(content="hi")],
        "completed": False,
        "api_result": None,
    }
    result = {
        "messages": before["messages"]
        + [ToolMessage(content="Tool error: CRM down", tool_call_id="call-1")],
        "completed": False,
        "api_result": None,
    }
    replies = _build_replies(before, result)
    assert all("yay" not in r.text.lower() for r in replies)
    assert all("locked in" not in r.text.lower() for r in replies)


def test_generate_agent_replies_never_hallucinates_success():
    msgs = [
        AIMessage(content="Done! registered!"),
        ToolMessage(content="Tool error: x", tool_call_id="call-1"),
    ]
    out = generate_agent_replies(msgs, {})
    assert all("done" not in t.lower() for t in out)
    assert all("register" not in t.lower() for t in out)

    msgs2 = [
        ToolMessage(
            content=json.dumps({"status": "success", "message": "welcome aboard"}),
            tool_call_id="call-1",
        )
    ]
    out2 = generate_agent_replies(msgs2, {})
    assert any("welcome" in t.lower() for t in out2)


def test_looks_like_success_claim():
    assert looks_like_success_claim("Done! You are registered!")
    assert looks_like_success_claim("Your request has been received")
    assert not looks_like_success_claim("What is your name?")
    assert not looks_like_success_claim("")


def test_validate_tool_args_still_rejects_missing_details():
    ok, hint = validate_tool_args("register_client", {"client_name": "  "}, ["client_name"])
    assert ok is False and "client_name" in hint


def test_signup_recovers_name_from_follow_up_message():
    messages = [
        HumanMessage(content="NEW CUSTOMER MESSAGE:\nI want to register as a new Tailorsin client."),
        HumanMessage(content="NEW CUSTOMER MESSAGE:\nRamakrishna G"),
    ]
    assert _latest_signup_name(messages) == "Ramakrishna G"


def test_signup_contact_resolution_returns_phone_and_history_separately():
    messages = [
        SystemMessage(
            content=(
                "Customer context for this conversation:\n"
                "mobile: 919640864111"
            )
        ),
        HumanMessage(content="NEW CUSTOMER MESSAGE:\nRamakrishna G"),
    ]
    args = {"client_name": "Ramakrishna G"}
    history = resolve_primary_contact(args, messages)
    assert context_mobile(messages) == "919640864111"
    assert args["primary_no"] == "919640864111"
    assert "Ramakrishna G" in history
    ok, hint = validate_tool_args(
        "register_client",
        args,
        ["client_name", "primary_no"],
        history=history,
    )
    assert ok is True and hint is None
