# agents/_utils.py

import json
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import ToolMessage


def execute_tool_call(response, messages, tool_map):
    """Execute the tool requested in `response` (if any) and return state updates.

    Returns a dict that always contains an updated 'messages' list. When a
    tool call was made it also includes 'api_result', 'completed' and the
    parsed 'args' so the calling node can persist the extracted fields.
    """
    calls = getattr(response, "tool_calls", None)
    if not calls:
        return {"messages": messages + [response]}

    call = calls[0]
    name = call.get("name")
    args = call.get("args") or {}
    call_id = call.get("id")
    func = tool_map.get(name)

    if func is None:
        return {
            "messages": messages + [
                response,
                ToolMessage(
                    content=f"Tool error: unknown tool requested: {name}",
                    tool_call_id=call_id,
                ),
            ],
            "api_result": None,
            "completed": False,
            "args": args,
        }

    try:
        result = func(**args)
    except Exception as exc:  # noqa: BLE001 - surface as controlled error
        return {
            "messages": messages + [
                response,
                ToolMessage(
                    content=f"Tool error: {exc}",
                    tool_call_id=call_id,
                ),
            ],
            "api_result": None,
            "completed": False,
            "args": args,
        }

    # Only an explicit CRM success counts as completed. A logical CRM
    # failure (e.g. {"status": "error", ...} with HTTP 200) is normalised
    # to "Tool error: ..." so downstream error handling treats it as a
    # failure instead of letting the model claim success.
    if is_success_result(result):
        content = json.dumps(result, default=str)
        completed = True
    else:
        try:
            detail = json.dumps(result, default=str)
        except Exception:
            detail = str(result)
        content = f"Tool error: CRM reported failure: {detail}"
        completed = False

    return {
        "messages": messages + [
            response,
            ToolMessage(content=content, tool_call_id=call_id),
        ],
        "api_result": result,
        "completed": completed,
        "args": args,
    }


def is_success_result(result: Any) -> bool:
    """True only when the CRM payload explicitly reports success.

    The CRM always returns a dict with a ``status`` field. A ``status``
    of ``\"success\"`` (case-insensitive) is the *only* thing that counts
    as success — anything else (``\"error\"``, missing status with an
    ``error`` key, ``None``, a plain string, ...) is a failure. This is
    the core anti-hallucination gate: the agent must never claim a
    booking/registration/estimate succeeded unless this returns True.
    """
    if not isinstance(result, dict):
        return False
    status = str(result.get("status", "")).strip().lower()
    if status:
        return status == "success"
    # No explicit status — only accept legacy ``{"success": true}`` shape.
    if "success" in result:
        return bool(result.get("success"))
    return False


def is_tool_failure_content(text: str) -> bool:
    """True when a *tool* message payload indicates failure.

    Covers both transport errors (``\"Tool error: ...\"``) and logical CRM
    failures returned with HTTP 200, e.g. ``{\"status\": \"error\", ...}``.
    Must only be applied to tool messages, never to user/AI text.
    """
    stripped = (text or "").strip()
    if not stripped:
        return False
    if stripped.lower().startswith("tool error"):
        return True
    try:
        payload = json.loads(stripped)
    except Exception:
        return False
    if isinstance(payload, dict):
        status = str(payload.get("status", "")).strip().lower()
        if status:
            return status != "success"
        if "success" in payload:
            return not bool(payload.get("success"))
        if "error" in payload and payload.get("error"):
            return True
    return False


def _history_text(messages) -> str:
    """Flatten human + AI conversation history into one searchable string."""
    parts = []
    for message in messages or []:
        if getattr(message, "type", "") not in ("human", "user", "ai"):
            continue
        content = getattr(message, "content", "")
        if isinstance(content, str) and content.strip():
            parts.append(content)
    return "\n".join(parts)


def context_mobile(messages) -> Optional[str]:
    """Extract the customer's shared mobile from the injected context message.

    The conversation service prepends a SystemMessage such as
    "Customer context: ...\\nmobile: <number>\\n..." so agents can read the
    number the customer shared via the 'Share contact' button.
    """
    for message in messages or []:
        if getattr(message, "type", "") != "system":
            continue
        content = getattr(message, "content", "")
        if not isinstance(content, str):
            continue
        for line in content.splitlines():
            if line.strip().lower().startswith("mobile:"):
                value = line.split(":", 1)[1].strip()
                if value:
                    return value
    return None


def resolve_primary_contact(args: Dict[str, Any], messages) -> str:
    """Route the customer's shared number into the tool args as primary_no.

    - The shared Telegram number (from the customer context) is always
      primary_no.
    - Any *different* number the customer types in chat is moved to
      secondary_no instead of overwriting primary_no.

    Returns a history string that includes the shared number, so groundedness
    validation accepts primary_no without the customer re-typing it.
    """
    mobile = context_mobile(messages)
    if mobile and args.get("primary_no") and args["primary_no"] != mobile \
            and not args.get("secondary_no"):
        args["secondary_no"] = args["primary_no"]
    if mobile:
        args["primary_no"] = mobile
    history = _history_text(messages)
    if mobile:
        history = f"{history}\n{mobile}"
    return history


def _mentions(value: str, history: str) -> bool:
    """True when a tool-arg value is grounded in the conversation history.

    Numeric values are compared digit-wise so "+91 98765" still matches
    "9198765". Non-numeric values need a case-insensitive substring match.
    """
    needle = (value or "").strip().lower()
    haystack = (history or "").lower()
    if not needle or not haystack:
        return False
    if needle in haystack:
        return True
    digits_needle = "".join(ch for ch in needle if ch.isdigit())
    if digits_needle:
        digits_hay = "".join(ch for ch in haystack if ch.isdigit())
        return bool(digits_needle) and digits_needle in digits_hay
    return False


def validate_tool_args(
    tool_name: str,
    args: Dict[str, Any],
    required: List[str],
    validator: Optional[Dict[str, Any]] = None,
    history: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """Validate tool arguments *before* the tool is called.

    Parameters
    ----------
    tool_name
        The name of the tool (used only for error messages).
    args
        The parsed arguments the model produced.
    required
        Keys that must be present and non-empty.
    validator
        Optional dict mapping a key to a callable ``fn(value) -> bool``
        that returns True for a *valid* value. If the value is invalid
        the helper returns ``(False, "please re-supply ...")``.
    history
        Optional flattened conversation text. When given, each required
        value must also appear (case-insensitive substring match) in the
        history — otherwise the model hallucinated it and validation
        fails so the agent asks the customer instead of calling the CRM.

    Returns
    -------
    (ok, message)
        ``message`` is None when ok, otherwise a short user-facing hint.
    """
    validator = validator or {}
    for key in required:
        val = args.get(key)
        if val is None or (isinstance(val, str) and not val.strip()):
            return False, (
                f"Please provide {key} so I can complete your "
                f"{tool_name.replace('_', ' ')} request."
            )
        if history and isinstance(val, str) and val.strip():
            # Groundedness check: the value must have been supplied by the
            # customer (or confirmed by them) in the visible conversation.
            # Numeric values are normalised so "+91 98765" still matches.
            needle = val.strip().lower()
            haystack = history.lower()
            if needle not in haystack:
                digits_needle = "".join(ch for ch in needle if ch.isdigit())
                digits_hay = "".join(ch for ch in haystack if ch.isdigit())
                if not (digits_needle and digits_needle in digits_hay):
                    return False, (
                        f"Please provide {key} so I can complete your "
                        f"{tool_name.replace('_', ' ')} request."
                    )
    for key, fn in validator.items():
        val = args.get(key)
        if val is not None and not fn(val):
            return False, (
                f"The {key} you provided — I couldn't use it. "
                "Mind sharing a valid one?"
            )
    return True, None


def extract_success_message(api_result: Any) -> Optional[str]:
    """Pull a friendly human-readable message out of a CRM success response.

    Tailorsin's CRM returns dicts like:
      {"status": "success", "message": "thank you! ...", "data": {...}}
    We surface just the ``message`` so the customer doesn't see raw JSON.
    Returns None for non-dict / non-success / missing-message results.
    """
    if not is_success_result(api_result):
        return None
    raw = api_result.get("message")
    if not raw or not isinstance(raw, str):
        return None
    return raw.strip()


def generate_agent_replies(
    messages: list,
    tool_map: Dict[str, Any],
    default_success: str = (
        "✅ Yay, done! 😄 Your request is locked in 📌 — our team "
        "will reach out to you shortly. Sit tight! 🙌"
    ),
) -> List[str]:
    """Convert a tool call's messages into user-facing strings.

    - Skips internal "skipping tool call" and "Tool error" bookkeeping.
    - Drops AI success claims unless a tool success exists (anti-hallucination).
    - Prefers the CRM success ``message`` when available.
    - Falls back to ``default_success`` ONLY on verified tool success.
    - Returns [] when nothing succeeded (caller decides the error reply).
    """
    replies: List[str] = []
    api_result = None
    completed = False
    ai_texts: List[str] = []

    for message in messages:
        msg_type = getattr(message, "type", "")
        text = getattr(message, "content", None)
        if hasattr(text, "strip"):
            text = text.strip()
        else:
            text = str(text).strip() if text else ""

        if msg_type == "ai" and text:
            ai_texts.append(text)
        elif msg_type == "tool":
            if is_tool_failure_content(text):
                continue
            if text.lower().startswith("skipping tool call"):
                continue
            # Only verified CRM success counts — anything else is ignored.
            try:
                parsed = json.loads(text)
            except Exception:
                continue  # non-JSON tool output is never a verified success
            if is_success_result(parsed):
                api_result = parsed
                completed = True

    if completed:
        # Verified success: keep genuine AI text, else CRM/default message.
        genuine = [
            t for t in ai_texts
            if t and not t.lower().startswith("skipping tool call")
        ]
        if genuine:
            replies.extend(genuine)
        else:
            msg = extract_success_message(api_result)
            replies.append(msg or default_success)
    else:
        # No verified success: only keep AI text that does NOT claim success.
        for t in ai_texts:
            if t.lower().startswith("skipping tool call"):
                continue
            if looks_like_success_claim(t):
                continue
            replies.append(t)

    return replies


def looks_like_success_claim(text: str) -> bool:
    """True when AI text claims the request already succeeded."""
    import re

    if not text or not isinstance(text, str):
        return False
    return bool(
        re.search(
            r"\b(done|registered|registration\s+(complete|completed|successful|success)|"
            r"successfully|success|locked\s*in|confirmed|booked|booked\s*in|"
            r"request\s+(has\s+been\s+|have\s+been\s+|was\s+|were\s+|is\s+|are\s+)?"
            r"(received|submitted|confirmed|done|successful)|"
            r"estimate\s+(ready|done|sent)|order\s+(placed|confirmed|received)|"
            r"you\s*(are|'re)\s*registered)\b",
            text,
            re.IGNORECASE,
        )
    )