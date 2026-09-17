# agents/order_management.py

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, ToolMessage

from state import AgentState
from tools.tailorsin_tools import (
    get_order_status,
    cancel_order,
    modify_order,
)
from agents._utils import (
    context_mobile,
    execute_tool_call,
    validate_tool_args,
    _history_text,
)

load_dotenv()


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


ORDER_MANAGEMENT_PROMPT = """
You are the Tailorsin Order Management Agent.

You help customers with three order-related actions:
1. Track an order — check the current status of an existing order.
2. Cancel an order — cancel an existing order with a reason.
3. Modify an order — request changes to an existing order.

Required information for each action:

- get_order_status:
  - mobile (the customer's contact number — use the number from the customer
    context when available, and only ask if it is not already provided)

- cancel_order:
  - mobile (the customer's contact number)
  - order_id (the ID of the order to cancel)
  - reason (why the customer wants to cancel, e.g. 'Changed my mind')

- modify_order:
  - mobile (the customer's contact number)
  - order_id (the ID of the order to modify)
  - comment (the modification request, e.g. 'Please make the sleeves half instead of full')

Rules:
1. Use the customer's mobile from the conversation context. Never invent or
   guess missing details — ask the customer if information is missing.
2. Ask for the missing details in ONE short, friendly message.
3. NEVER claim an action succeeded ("done", "cancelled", "tracked", "modified",
   "confirmed", ...) unless the tool result you just received explicitly reports
   status success. If the tool errored, say there was a snag and ask to retry —
   do not pretend it worked.
4. Call the appropriate tool once all required info is available.
5. Only one action per turn — handle track, cancel, or modify, not multiple at once.

Customer-message style:
- Short, natural and friendly — one or two sentences maximum.
- Light emojis only (📦 🔍 ❌ ✏️).
- Example messages:
  - Tracking: "Let me check your order status! What's your mobile number? 📱"
  - Cancelling: "I can help with that! What's the order ID and the reason for
    cancelling? 🤔"
  - Modifying: "Sure! What's the order ID and what changes do you need? ✏️"

Tone:
- Friendly, upbeat and casual (our users are Gen Z and Gen Alpha).
"""


def order_management_agent(state: AgentState):

    messages = state["messages"]

    # Pre-extract mobile from context so we can tell the LLM it's available
    mobile = context_mobile(messages) or state.get("mobile") or state.get("primary_no")

    context = SystemMessage(
        content=(
            f"Customer mobile (from context): {mobile or 'NOT PROVIDED'}\n"
            f"Do NOT ask for mobile if it is shown above — it is already known.\n"
        )
    )

    response = llm.bind_tools(
        [get_order_status, cancel_order, modify_order]
    ).invoke([
        context,
        SystemMessage(content=ORDER_MANAGEMENT_PROMPT),
        *messages
    ])

    calls = getattr(response, "tool_calls", None)

    if not calls:
        return {
            "messages": messages + [response],
            "completed": False,
        }

    call = calls[0]
    args = dict(call.get("args") or {})

    # Inject mobile from context if not provided
    if mobile and not args.get("mobile"):
        args["mobile"] = mobile

    call["args"] = args

    name = call.get("name")

    required_map = {
        "get_order_status": ["mobile"],
        "cancel_order": ["mobile", "order_id", "reason"],
        "modify_order": ["mobile", "order_id", "comment"],
    }

    required = required_map.get(name)
    if required is None:
        return {
            "messages": messages + [
                response,
                ToolMessage(
                    content=f"Tool error: unsupported order management tool {name}",
                    tool_call_id=call.get("id"),
                ),
            ],
            "completed": False,
            "api_result": None,
        }

    history = _history_text(messages)
    # Include mobile from state in history so validation recognizes it
    mobile = state.get("mobile") or state.get("primary_no")
    if mobile and mobile not in history:
        history = f"{history}\nCustomer mobile: {mobile}"
    ok, hint = validate_tool_args(
        tool_name=name,
        args=args,
        required=required,
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
            "completed": False,
            "api_result": None,
        }

    updates = execute_tool_call(
        response,
        messages,
        {
            "get_order_status": get_order_status.func,
            "cancel_order": cancel_order.func,
            "modify_order": modify_order.func,
        },
    )

    tool_args = updates.pop("args", None) or {}
    api_result = updates.get("api_result")
    if api_result is None:
        api_result = updates.get("result")

    updates["mobile"] = tool_args.get("mobile") or mobile
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
