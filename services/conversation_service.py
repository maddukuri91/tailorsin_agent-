# services/conversation_service.py
import asyncio
import json
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from graph import graph
from agents._utils import (
    extract_success_message as _extract_crm_success_message,
    is_success_result as _is_success_result,
    is_tool_failure_content as _is_tool_failure_payload,
    looks_like_success_claim as _is_success_claim,
)
from services.customer_context import CustomerContextStore, classify_mobile
from services.menu_router import menu_router
from tools.tailorsin_tools import get_client_address
@dataclass
class IncomingMessage:
    user_id: Any
    text: str = ""
    contact_phone: Optional[str] = None
    contact_user_id: Optional[str] = None
    source_user_id: Optional[Any] = None
    is_start_command: bool = False
    location_lat: Optional[float] = None
    location_lng: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    # Telegram photo attachment. ``photo_file_id`` is the raw Telegram file
    # id; ``photo_url`` is the resolved, directly-downloadable https URL
    # (resolved by the Telegram channel layer). Photos are not consumed by
    # any agent tool — fabric estimation no longer accepts a reference
    # picture — so they are carried for completeness only; the caption text
    # is what reaches the agent.
    photo_file_id: Optional[str] = None
    photo_url: Optional[str] = None
@dataclass
class OutgoingMessage:
    text: str
    reply_markup: Any = None
WELCOME_TEXT = (
    "👋 Hey hey! Welcome to Tailorsin — your spot for custom fits 🧵✨\n\n"
    "We pick up your fabric, stitch your threads, and drop them at your door 🚪📦 "
    "(usually within 24 hrs ⚡ after you approve the design).\n\n"
    "To tailor everything to YOU 🫵, share your number 👇\n"
    "Tap \"📱 Share my phone number\" or just type it — takes 5 seconds ⏱️💨"
)
# Telegram Markdown special characters that can break parsing.
_MARKDOWN_CHARS = re.compile(r"([_*\[\]()~`>#+\-=|{}.!\\])")
# A keyboard that asks the user to share their Telegram contact (phone).
_CONTACT_REQUEST_KEYBOARD = {
    "keyboard": [[{"text": "📱 Share my phone number", "request_contact": True}]],
    "one_time_keyboard": True,
    "resize_keyboard": True,
}
# Per-user conversation state, keyed by session id. Held in memory so it is
# suitable for a single-process dev / demo deployment.
_conversations: Dict[str, dict] = {}
# Customer context store: per-session customer_type / mobile / customer_id.
_context_store = CustomerContextStore()
# Maps telegram user key -> active session id.
_sessions: Dict[str, str] = {}
_lock = threading.Lock()
def _fresh_state() -> dict:
    # `next_agent` doubles as "the agent currently serving this customer":
    # it is set to the routed agent after every turn and must never be reset
    # to None by a later turn — only a menu-launched (fresh=True) turn or a
    # new routed value may change it.
    return {
        "messages": [],
        "next_agent": None,
        "client_name": None,
        "primary_no": None,
        "secondary_no": None,
        "mobile": None,
        "store_id": None,
        "bookdate": None,
        "booktime": None,
        "order_action": None,
        "address_id": None,
        "api_result": None,
        "completed": False,
    }
def _escape_markdown(text: str) -> str:
    """Escape characters that interfere with Telegram's Markdown parser."""
    return _MARKDOWN_CHARS.sub(r"\\\1", text)
def _extract_phone(message: IncomingMessage) -> Optional[str]:
    """Pull a phone number from a shared contact or typed text."""
    if message.contact_phone:
        return _normalize_phone(message.contact_phone)
    digits = re.sub(r"\D", "", message.text or "")
    # A plausible mobile number is at least 8 digits.
    if len(digits) >= 8:
        return digits
    return None
def _normalize_phone(phone: str) -> str:
    """Strip separators and a leading '+' from a phone number."""
    return re.sub(r"\D", "", phone or "")
def _looks_like_menu_choice(text: str) -> Optional[int]:
    """If the text is a bare menu index (e.g. '3'), return it as an int."""
    text = (text or "").strip()
    if not text.isdigit():
        return None
    value = int(text)
    return value if value >= 1 else None
def _should_route_to_agent(message: IncomingMessage, session_id: str) -> bool:
    """True when free input belongs to an agent turn, not the menu router.
    While an agent turn is still collecting details, a bare number is a
    parameter (e.g. a phone number) rather than a menu tap.
    """
    with _lock:
        prior = _conversations.get(session_id)
    if not prior or not prior.get("messages"):
        return False
    return not prior.get("completed")
def _build_menu_reply(menu: dict) -> OutgoingMessage:
    """
    Render a menu (from the Menu Router) as a Telegram inline keyboard.
    Each button carries callback_data 'menu_<index>' which Telegram's
    callback parser converts into the plain index for us to map back.
    """
    title = menu["title"]
    options = menu["options"]
    buttons = [
        [{"text": option["label"], "callback_data": f"menu_{i + 1}"}]
        for i, option in enumerate(options)
    ]
    return OutgoingMessage(
        text=_escape_markdown(title),
        reply_markup={"inline_keyboard": buttons},
    )
def _main_menu_reply(ctx) -> List[OutgoingMessage]:
    """Reset the customer to their root/main menu and render it."""
    menu_id = menu_router.root_menu_id(ctx.customer_type)
    ctx.current_menu_id = menu_id
    _context_store.save(ctx)
    return [_build_menu_reply(menu_router.get_menu(menu_id))]


def _address_items(payload: Any) -> list[dict[str, Any]]:
    """Extract address records from the CRM's varying response envelopes."""
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("addresses", "data", "results", "address"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            nested = _address_items(value)
            if nested:
                return nested
    return []


def _address_label(address: dict[str, Any], index: int) -> str:
    text = " ".join(
        str(address.get(key, "")).strip()
        for key in ("address", "address2", "locality", "city", "pincode")
        if address.get(key)
    ).strip()
    return text or f"Saved address {index}"


def _address_id(payload: Any) -> Optional[int]:
    """Find a newly-created address id in a CRM response."""
    if isinstance(payload, dict):
        for key in ("id", "address_id"):
            value = payload.get(key)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass
        for key in ("data", "address"):
            found = _address_id(payload.get(key))
            if found is not None:
                return found
    return None


async def _start_pickup_flow(session_id: str, ctx) -> List[OutgoingMessage]:
    """Look up saved addresses before asking for pickup date and time."""
    try:
        payload = await asyncio.to_thread(get_client_address.func, mobile=ctx.mobile)
    except Exception:
        payload = None
    addresses = _address_items(payload)
    if not addresses:
        with _lock:
            state = dict(_conversations.get(session_id) or _fresh_state())
            state["order_action"] = "add_address"
            _conversations[session_id] = state
        return await _run_agent_turn(
            session_id=session_id,
            ctx=ctx,
            user_text=(
                "I want to schedule a fabric pickup, but I have no saved "
                "address. Please collect my new address first."
            ),
            forced_agent="order",
        )

    menu_id = f"saved_addresses_{session_id}"
    options = [
        {
            "label": _address_label(address, index),
            "intent": "pickup_address",
            "action": "pickup_address",
            "address_id": address.get("id") or address.get("address_id"),
        }
        for index, address in enumerate(addresses, start=1)
    ]
    options.append({
        "label": "Add a new address",
        "intent": "add_address",
        "action": "agent",
        "prompt": "I want to add a new pickup address.",
    })
    menu_router.register_menu(menu_id, {
        "title": "📍 Select a saved pickup address:",
        "options": options,
    })
    ctx.current_menu_id = menu_id
    _context_store.save(ctx)
    return [_build_menu_reply(menu_router.get_menu(menu_id))]
def _is_tool_error(text: str) -> bool:
    """True if a *tool* message text indicates a failure (CRM/API error).
    Covers both transport errors (``"Tool error: ..."``) and logical CRM
    failures returned with HTTP 200 (``{"status": "error", ...}``), which
    ``execute_tool_call`` normalises to ``"Tool error: ..."``. Must only be
    called with tool-message content, never user/AI text.
    """
    return _is_tool_failure_payload(text)
def _has_tool_error(state_before: dict, result: dict) -> bool:
    """Scan the latest agent turn for any tool-call failure message."""
    from_count = len(state_before["messages"])
    for message in result["messages"][from_count:]:
        if getattr(message, "type", "") != "tool":
            continue
        text = _content_to_text(getattr(message, "content", None))
        if _is_tool_error(text):
            return True
    return False
def _had_verified_tool_success(state_before: dict, result: dict) -> bool:
    """True only when the latest turn has a verified CRM success payload."""
    if result.get("completed") and _is_success_result(result.get("api_result")):
        return True
    from_count = len(state_before["messages"])
    for message in result["messages"][from_count:]:
        if getattr(message, "type", "") != "tool":
            continue
        text = _content_to_text(getattr(message, "content", None)).strip()
        if not text or _is_tool_error(text):
            continue
        if _crm_success_from_text(text) is not None:
            return True
    return False
async def _handle_menu_choice(
    session_id: str,
    ctx,
    choice: int,
) -> List[OutgoingMessage]:
    """Send a menu selection through the Menu Router and act on it."""
    menu_id = ctx.current_menu_id or menu_router.root_menu_id(ctx.customer_type)
    action = menu_router.resolve(menu_id, choice)
    if action["action"] == "retry":
        return [OutgoingMessage(text="Oops — that's not on the menu 🤔 Try one of the buttons above 👆")]
    if action["action"] == "agent":
        # Hand the intent to the supervisor agent -> sub-agents.
        forced_agent = {
            "register": "signup",
            "fabric_estimation": "fabric_estimation",
            "bulk_order": "bulk_order",
            "appointment": "book_visit",
            "talk_to_human": "human_support",
            "ship_fabric": "order",
            "schedule_pickup": "order",
            "add_address": "order",
            # Order management (track, cancel, modify existing orders)
            "order_status": "order_management",
            "order_changes": "order_management",
            "order_cancel": "order_management",
        }.get(action.get("intent"))
        return await _run_agent_turn(
            session_id=session_id,
            ctx=ctx,
            user_text=action["prompt"],
            fresh=True,
            forced_agent=forced_agent,
        )
    if action["action"] == "lookup_addresses":
        return await _start_pickup_flow(session_id, ctx)
    if action["action"] == "pickup_address":
        with _lock:
            state = dict(_conversations.get(session_id) or _fresh_state())
            state["order_action"] = "schedule_pickup"
            state["address_id"] = action.get("address_id")
            _conversations[session_id] = state
        return await _run_agent_turn(
            session_id=session_id,
            ctx=ctx,
            user_text="I selected a saved address and want to schedule a fabric pickup.",
            forced_agent="order",
        )
    if action["action"] == "content":
        # Curated onboarding copy — sent RAW (already Telegram Markdown).
        messages: List[OutgoingMessage] = [
            OutgoingMessage(text=block) for block in action["blocks"]
        ]
        # Advance to the "next" (sub)menu, or stay on the current one.
        next_menu_id = action.get("next") or action["menu_id"]
        ctx.current_menu_id = next_menu_id
        _context_store.save(ctx)
        messages.append(_build_menu_reply(menu_router.get_menu(next_menu_id)))
        return messages
    if action["action"] == "menu":
        # Return the customer to the root/main menu.
        return _main_menu_reply(ctx)
    if action["action"] == "reply":
        return [OutgoingMessage(text=_escape_markdown(action["text"]))]
    # action == "human": needs a follow-up from the team.
    return [
        OutgoingMessage(
            text=(
                f"✍️ Got it — \"{action['label']}\"! A human 🧑‍💼 from our team "
                "will reach out to sort this with you. Anything else meanwhile? 😊"
            )
        )
    ]
def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    # Some models return a list of content parts.
    parts = []
    for part in content if isinstance(content, list) else [content]:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and "text" in part:
            parts.append(str(part["text"]))
    return "".join(parts)
def _crm_success_from_text(text: str) -> Optional[str]:
    """Pull the CRM's friendly success `message` out of a tool-result text.
    CRM tools return JSON like {"status": "success", "message": "..."}. The
    customer should see just the `message`, not the raw JSON payload.
    Returns None when `text` is not a CRM success payload.
    """
    try:
        payload = json.loads(text)
    except Exception:  # noqa: BLE001 - not JSON, surface text as-is upstream
        return None
    return _extract_crm_success_message(payload)
def _build_replies(state_before: dict, result: dict) -> List[OutgoingMessage]:
    """
    Extract messages the agent added during the last turn (the assistant
    replies and any tool result confirmation) and turn them into outgoing
    Telegram messages.
    Anti-hallucination rules:
    - Raw tool payloads are never surfaced: tool success shows only the
      CRM's friendly ``message``; tool failures are swallowed here (the
      caller sends a controlled error reply instead).
    - AI text that claims success ("done!", "registered", ...) is dropped
      unless the turn has a verified CRM success payload.
    """
    prior_count = len(state_before["messages"])
    new_messages = result["messages"][prior_count:]
    ai_texts: List[str] = []
    crm_success: Optional[str] = None
    for message in new_messages:
        msg_type = getattr(message, "type", "")
        text = _content_to_text(getattr(message, "content", None)).strip()
        if msg_type == "ai" and text:
            ai_texts.append(text)
        elif msg_type == "tool":
            # Don't surface raw API error text; the caller routes back to menu.
            if _is_tool_error(text):
                continue
            # Don't surface our internal "skipped tool call" bookkeeping —
            # the agent's natural follow-up question (added as an 'ai' msg)
            # will be picked up instead.
            if text.lower().startswith("skipping tool call"):
                continue
            # Surface the CRM's friendly success `message` instead of raw
            # JSON. Anything that is not a verified CRM success payload is
            # ignored (never leak raw JSON / unverified content).
            friendly = _crm_success_from_text(text)
            if friendly:
                crm_success = crm_success or friendly
    verified = _had_verified_tool_success(state_before, result)
    tool_failed = _has_tool_error(state_before, result)
    # If the tool failed, return nothing — the caller is responsible for
    # sending a controlled error reply + main menu. Never let a failed tool
    # call produce a generic "not sure I caught that" fallback.
    if tool_failed:
        return []
    replies: List[OutgoingMessage] = []
    if verified:
        # Verified CRM success: prefer the agent's own confirmation text
        # (skipping internal bookkeeping), else the CRM message.
        genuine = [
            t for t in ai_texts
            if t and not t.lower().startswith("skipping tool call")
        ]
        for t in genuine:
            replies.append(OutgoingMessage(text=_escape_markdown(t)))
        if not replies and crm_success:
            replies.append(OutgoingMessage(text=_escape_markdown(crm_success)))
    else:
        # No verified success: keep clarifying questions, but drop any AI
        # text that claims the request already succeeded (hallucination).
        for t in ai_texts:
            if not t or t.lower().startswith("skipping tool call"):
                continue
            if _is_success_claim(t):
                continue
            replies.append(OutgoingMessage(text=_escape_markdown(t)))
        # If the agent executed a tool but produced no readable reply, see
        # whether the CRM gave us a verified success message we can surface.
        # The generic "Yay, done!" fallback is ONLY used on verified success —
        # never on failure, so a hallucinated confirmation can't slip through.
    if not replies and verified:
        success_msg = _extract_crm_success_message(result.get("api_result"))
        replies.append(
            OutgoingMessage(
                text=_escape_markdown(success_msg or crm_success) if (success_msg or crm_success) else (
                    "Yay, done! Your request is locked in — our team "
                    "will reach out to you shortly. Sit tight!"
                )
            )
        )
    elif not replies:
        # No verified success and nothing safe to say: ask the customer to
        # retry. Tool-failure turns are replaced by the controlled error
        # reply in _run_agent_turn; this is only for empty/unclear turns.
        replies.append(
            OutgoingMessage(
                text=(
                    "Hmm, not sure I caught that 🤔 Mind typing it again? "
                    "I'm all ears 👂✨"
                )
            )
        )
    return replies
def _context_system_message(ctx) -> Optional[SystemMessage]:
    """
    Build a system message that orients the supervisor agent / sub-agents
    with the current customer context.
    """
    if ctx is None or not ctx.is_classified:
        return None
    serving = ""
    with _lock:
        prior_ctx = _conversations.get(ctx.session_id) if ctx else None
    if prior_ctx and prior_ctx.get("next_agent"):
        serving = prior_ctx["next_agent"]
    return SystemMessage(
        content=(
            "Customer context for this conversation:\n"
            f"{ctx.describe()}\n\n"
            + (f"currently_serving: {serving}\n\n" if serving else "")
            + "Use this context to personalise your response and route "
            "appropriately."
        )
    )


def _is_registration_success(state: dict, result: dict) -> bool:
    """True only when the signup agent verified a CRM registration success."""
    return (
        result.get("next_agent") == "signup"
        and result.get("completed") is True
        and _is_success_result(result.get("api_result"))
    )


async def _refresh_after_registration(
    ctx,
    session_id: str,
    result: dict,
    replies: List[OutgoingMessage],
) -> List[OutgoingMessage]:
    """After a successful registration, reclassify the customer (they are no
    longer a new user) and bring them straight to the client menu with a fresh
    conversation so the next turn no longer continues inside the signup agent.
    """
    mobile = result.get("primary_no") or ctx.mobile
    customer_type = "client"
    customer_id = ctx.customer_id
    if mobile:
        try:
            new_type, new_id = await asyncio.to_thread(classify_mobile, mobile)
        except Exception:  # noqa: BLE001 - fall back to client on lookup error
            new_type, new_id = None, None
        if new_type in ("client", "active_client"):
            customer_type = new_type
            customer_id = new_id or customer_id
    ctx.customer_type = customer_type
    ctx.customer_id = customer_id
    ctx.current_menu_id = menu_router.root_menu_id(customer_type)
    _context_store.save(ctx)
    with _lock:
        # Fresh state keeps the next turn from continuing inside signup.
        _conversations[session_id] = _fresh_state()
    client_menu = _build_menu_reply(menu_router.get_menu(ctx.current_menu_id))
    success_message = _extract_crm_success_message(result.get("api_result"))
    confirmation = success_message or (
        "🎉 Registration successful! Welcome to Tailorsin. "
        "You can now choose what you would like to do next:"
    )
    # Do not forward the signup model's final text here. The registration
    # result is already verified, so send one deterministic confirmation and
    # then the refreshed client menu.
    return [OutgoingMessage(text=_escape_markdown(confirmation)), client_menu]


async def _run_agent_turn(
    session_id: str,
    ctx,
    user_text: str,
    fresh: bool = False,
    forced_agent: Optional[str] = None,
) -> List[OutgoingMessage]:
    """
    Feed a human message through the supervisor agent -> sub-agents (the
    langgraph graph) and return the replies.
    The customer context is injected as a system message when available.
    When `fresh` is True, start a brand-new conversation state (used when a
    menu selection is re-routed into an agent request).
    """
    with _lock:
        prior_state = None if fresh else _conversations.get(session_id)
    context_message = _context_system_message(ctx)
    if prior_state is None or fresh:
        state = _fresh_state()
    else:
        state = dict(prior_state)
    # Keep the verified Telegram contact in graph state as well as the
    # injected prompt. Tool validation must not depend on the LLM preserving
    # a system-message detail in its trace.
    if ctx and ctx.mobile:
        state["mobile"] = ctx.mobile
        state["primary_no"] = ctx.mobile
    if forced_agent:
        state["next_agent"] = forced_agent
    base_messages = list(state["messages"])
    if context_message is not None:
        base_messages = [context_message] + base_messages
    rendered = (user_text or "").strip()
    if rendered:
        # Marker the LLM supervisors understand: this is the newest customer
        # line in the conversation.
        rendered = f"NEW CUSTOMER MESSAGE:\n{rendered}"
    base_messages = base_messages + [HumanMessage(content=rendered)]
    state["messages"] = base_messages
    # graph.invoke is synchronous (it makes blocking Groq + CRM calls),
    # so run it off the event loop.
    # NOTE: `next_agent` must NOT be reset — it is the only stable record of
    # the in-progress agent. It is kept inside _fresh_state() when starting a
    # turn and inside the stored state across turns.
    result = await asyncio.to_thread(graph.invoke, state)
    # Store the result without the injected context system message so the
    # context is rebuilt freshly each turn (no duplicate context headers).
    stored = dict(result)
    stored["messages"] = [
        m for m in stored["messages"]
        if getattr(m, "type", "") != "system"
    ]
    with _lock:
        _conversations[session_id] = stored
    replies = _build_replies(state, result)
    # If a sub-agent's API call failed (e.g. CRM error), spare the customer
    # the raw error and bring them back to the main menu.
    if _has_tool_error(state, result):
        replies = [
            OutgoingMessage(
                text=(
                    "😅 Oops — hit a small snag on our end. "
                    "Let's get you back to the menu!"
                )
            ),
            *_main_menu_reply(ctx),
        ]
    if _is_registration_success(state, result):
        replies = await _refresh_after_registration(ctx, session_id, result, replies)
    elif (
        result.get("next_agent") == "order"
        and state.get("order_action") == "add_address"
        and result.get("completed") is True
    ):
        address_id = _address_id(result.get("api_result"))
        with _lock:
            refreshed = dict(_conversations.get(session_id) or result)
            refreshed["order_action"] = "schedule_pickup"
            refreshed["address_id"] = address_id
            _conversations[session_id] = refreshed
        replies.append(
            OutgoingMessage(
                text=(
                    "✅ Address saved! Please send your preferred pickup "
                    "date and time 📅🕐."
                )
            )
        )
    return replies
async def handle_incoming_message(message: IncomingMessage) -> List[OutgoingMessage]:
    user_key = str(message.source_user_id or message.user_id)
    with _lock:
        session_id = _sessions.get(user_key)
    ctx = _context_store.get(session_id)
    # /start (or /reset) — start a fresh session + context.
    if message.is_start_command or message.text.strip().lower() == "/reset":
        if session_id:
            _context_store.delete(session_id)
        with _lock:
            _sessions.pop(user_key, None)
            _conversations.pop(session_id, None)
        return [OutgoingMessage(text=WELCOME_TEXT, reply_markup=_CONTACT_REQUEST_KEYBOARD)]
    # --- Step 0: ensure a session + context exists ------------------------
    if ctx is None:
        ctx = _context_store.create_session()
        session_id = ctx.session_id
        with _lock:
            _sessions[user_key] = session_id
    # --- Step 1: Customer Context — classify the number if unknown --------
    if not ctx.is_classified:
        phone = _extract_phone(message)
        if phone is None:
            return [
                OutgoingMessage(
                    text=(
                        "Almost there! 🤗 Just need your number to set you up 🫵 — "
                        "tap the 📱 button below or type it in. ⚡"
                    ),
                    reply_markup=_CONTACT_REQUEST_KEYBOARD,
                )
            ]
        customer_type, customer_id = await asyncio.to_thread(classify_mobile, phone)
        if customer_type == "unknown":
            # CRM lookup failed: do NOT guess new_user (that misroutes the
            # customer and can trigger hallucinated agent replies). Keep them
            # unclassified and ask for a retry — no state is stored.
            return [
                OutgoingMessage(
                    text=(
                        "😅 Oops — I couldn't verify that number just now. "
                        "Please check it and try again, or tap the 📱 button below."
                    ),
                    reply_markup=_CONTACT_REQUEST_KEYBOARD,
                )
            ]
        ctx.mobile = phone
        ctx.customer_type = customer_type
        ctx.customer_id = customer_id
        ctx.current_menu_id = menu_router.root_menu_id(customer_type)
        _context_store.save(ctx)
        # --- Step 2: Menu Router — show the root menu for this type -------
        return [_build_menu_reply(menu_router.get_menu(ctx.current_menu_id))]
    # --- Step 3: explicit "main menu" request -----------------------------
    if (message.text or "").strip().lower() in {
        "menu", "/menu", "main menu", "mainmenu", "home", "/home", "start", "🏠",
    }:
        return _main_menu_reply(ctx)
    # --- Step 4: handle a menu selection ---------------------------------
    # A bare number is only a menu tap when the customer is NOT mid-flow with
    # an agent. Otherwise it is a parameter (e.g. a phone number) and must
    # reach the agent in Step 5. Swallowing it here is exactly the "keeps
    # asking" loop: out-of-range numbers got a "not on the menu" retry and
    # never reached the agent, so the agent kept asking for the same details.
    choice = _looks_like_menu_choice(message.text)
    if choice is not None and not _should_route_to_agent(message, session_id):
        return await _handle_menu_choice(session_id, ctx, choice)
    # --- Step 5: free text -> supervisor agent -> sub-agents ---------------
    # Only the typed text (caption) is sent to the agent; photos are no
    # longer a fabric-estimation reference and are ignored by the agents.
    user_text = (message.text or "").strip()
    return await _run_agent_turn(session_id, ctx, user_text)