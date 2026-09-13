# agents/signup.py

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from state import AgentState
from tools.tailorsin_tools import register_client
from agents._utils import (
    context_mobile,
    execute_tool_call,
    resolve_primary_contact,
    validate_tool_args,
)

load_dotenv()


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


SIGNUP_PROMPT = """
You are the Tailorsin Signup Agent (for new customers).

Your job is to register customers.

Required information (matching the register_client tool):
- client_name (the customer's name)
- primary_no (the number the customer shared via their Telegram contact —
  if it is present in the customer context, use it and DO NOT ask for it
  again)
- secondary_no (optional — only if the customer provides a *different*
  number for calls)

Rules:
1. Extract whatever you can from the conversation.
2. Ask for the missing details (usually just the name) in ONE short message.
3. primary_no comes from the customer context (the number they shared). Do
   NOT ask for a number if one is already available.
4. If the customer enters a different number for calls, send it as
   secondary_no — never overwrite the shared primary_no with it.
5. NEVER invent, guess, or fill in customer details (name, numbers) the
   customer has not actually provided. Missing info means you ASK, never assume.
6. NEVER claim the registration succeeded ("done", "registered",
   "confirmed", ...) unless the register_client tool result you just
   received explicitly reports status success. If the tool errored, say
   there was a snag and ask to retry — do not pretend it worked.
7. Never call register_client while client_name is missing, or while
   primary_no is unavailable (no shared-context number and none provided).
8. Call register_client once all required information is collected.

Customer-message style:
- Short, natural and friendly — one or two sentences maximum.
- Light emojis only (🤗 😊 🙌).
- Example of a good message:
  "Nice to meet you! What name should I put on your order? 🤗
   And if your preferred contact number is different from the one you're
   texting from now, just share it here."

Tone:
- Friendly, upbeat and casual (our users are Gen Z and Gen Alpha).
"""


def signup_agent(state: AgentState):

    messages = state["messages"]

    # --------------------------------------------------------
    # Don't call the API again once a previous turn already
    # succeeded. Without this guard, if this node ever gets
    # invoked again before the graph routes onward (retry,
    # re-entrant edge, etc.), the customer could get registered
    # twice.
    # --------------------------------------------------------

    previous_result = state.get("api_result")

    if _api_success(previous_result):
        return {
            "api_result": previous_result,
            "completed": True,
        }

    response = llm.bind_tools(
        [register_client]
    ).invoke([
        SystemMessage(content=SIGNUP_PROMPT),
        *messages
    ])

    calls = getattr(response, "tool_calls", None)

    if not calls:
        recovered_name = _latest_signup_name(messages)
        shared_mobile = context_mobile(messages) or state.get("mobile")
        if recovered_name and shared_mobile:
            response = AIMessage(
                content="",
                tool_calls=[{
                    "name": "register_client",
                    "args": {
                        "client_name": recovered_name,
                        "primary_no": shared_mobile,
                    },
                    "id": "signup-grounded-details",
                }],
            )
            calls = response.tool_calls
        else:
            # Model is asking for missing information (or just chatting).
            return {
                "messages": messages + [response],
                "completed": False,
            }

    call = calls[0]
    args = dict(call.get("args") or {})

    # --------------------------------------------------------
    # The primary number comes from the customer's shared
    # Telegram contact, not from something the model typed out
    # itself. The old code computed this and then threw it away
    # (passed it into validate_tool_args' `history` param, which
    # expects conversation history, not a phone number). Fill it
    # into args here so:
    #   (a) validation doesn't reject a customer we already have
    #       a number for, and
    #   (b) the actual API call below carries primary_no too.
    # --------------------------------------------------------

    history = resolve_primary_contact(args, messages)
    primary_no = context_mobile(messages) or state.get("mobile")
    if primary_no and primary_no not in history:
        history = f"{history}\n{primary_no}"

    if primary_no and not args.get("primary_no"):
        args["primary_no"] = primary_no
    elif primary_no:
        # The verified Telegram contact is always the primary number. A
        # different number typed in chat belongs in secondary_no.
        typed_primary = str(args["primary_no"])
        if typed_primary != str(primary_no) and not args.get("secondary_no"):
            args["secondary_no"] = typed_primary
        args["primary_no"] = primary_no

    # Feed the (possibly enriched) args back into the tool call
    # object so execute_tool_call sends the real values, not
    # whatever the model originally produced.
    call["args"] = args

    ok, hint = validate_tool_args(
        tool_name="register_client",
        args=args,
        required=["client_name", "primary_no"],
        history=history,
    )

    if not ok:
        return {
            "messages": messages + [
                response,
                ToolMessage(
                    content=f"Skipping tool call — user needs to supply: {hint}",
                    tool_call_id=call.get("id"),
                ),
            ],
            # Reset any stale result so a previous turn's success can
            # never be mistaken for this turn's outcome.
            "api_result": None,
            "completed": False,
        }

    updates = execute_tool_call(
        response,
        messages,
        {"register_client": register_client.func}
    )

    tool_args = updates.pop("args", None) or {}

    # Keep our grounded values if the executed tool call didn't
    # echo them back for some reason.
    client_name = tool_args.get("client_name") or args.get("client_name")
    primary_no = tool_args.get("primary_no") or args.get("primary_no")
    secondary_no = tool_args.get("secondary_no") or args.get("secondary_no")

    api_result = updates.get("api_result")
    if api_result is None:
        api_result = updates.get("result")

    # --------------------------------------------------------
    # Explicitly decide completion from the tool result — never
    # from what the model says. This is also what tells the
    # graph whether to loop back here again or move on.
    # --------------------------------------------------------

    updates["client_name"] = client_name
    updates["primary_no"] = primary_no
    updates["secondary_no"] = secondary_no
    updates["api_result"] = api_result
    updates["completed"] = _api_success(api_result)

    return updates


def _api_success(result):
    """
    Only an explicit {"status": "success", ...} counts.
    """

    if not isinstance(result, dict):
        return False

    status = result.get("status")

    if isinstance(status, str):
        return status.strip().lower() == "success"

    return False


def _latest_signup_name(messages) -> str | None:
    """Return the latest grounded customer name from the conversation."""
    for message in reversed(messages or []):
        if getattr(message, "type", "") not in ("human", "user"):
            continue
        text = getattr(message, "content", "")
        if not isinstance(text, str):
            continue
        if "NEW CUSTOMER MESSAGE:" in text:
            text = text.split("NEW CUSTOMER MESSAGE:", 1)[1].strip()
        text = text.strip()
        if not text or any(char.isdigit() for char in text):
            continue
        if text.lower() in {"hi", "hello", "hey", "yes", "no"}:
            continue
        if text.lower().startswith(("i want to register", "place an order")):
            continue
        for prefix in ("my name is ", "i am ", "i'm "):
            if text.lower().startswith(prefix):
                text = text[len(prefix):].strip()
                break
        return text or None
    return None