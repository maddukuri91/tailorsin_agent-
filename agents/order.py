from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_groq import ChatGroq

from agents._utils import execute_tool_call, validate_tool_args
from state import AgentState
from tools.tailorsin_tools import (
    add_client_address,
    schedule_pickup,
    ship_fabric_to_store,
    get_order_status,
    cancel_order,
    modify_order,
)

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

ORDER_PROMPT = """
You are the Tailorsin order agent.

For a scheduled pickup:
1. If the customer has no saved address, collect address, city and pincode,
   then call add_client_address.
2. If the customer has a saved address, ask if they want to use it or provide a new one. If they provide a new address, call add_client_address.
2. After an address is available, collect pickup_date and pickup_time and
   call schedule_pickup. Use the selected address_id from the context.

For sending fabric to a store, collect store_id if needed and call
ship_fabric_to_store.

Use the customer's mobile from context. Never invent missing details.
Ask one short question when information is missing. Use friendly emojis such
as 📍, 📅, 🕐, and 🏬 naturally. Only claim success when
the tool result explicitly has status success.
"""


def order_agent(state: AgentState):
    messages = state["messages"]
    context = SystemMessage(
        content=(
            f"Order action: {state.get('order_action') or 'ship_fabric'}\n"
            f"Customer mobile: {state.get('mobile') or state.get('primary_no')}\n"
            f"Selected address_id: {state.get('address_id')}\n"
        )
    )
    response = llm.bind_tools([
        add_client_address,
        schedule_pickup,
        ship_fabric_to_store,
        get_order_status,
        cancel_order,
        modify_order,
    ]).invoke([context, SystemMessage(content=ORDER_PROMPT), *messages])

    calls = getattr(response, "tool_calls", None)
    if not calls:
        return {"messages": messages + [response], "completed": False}

    call = calls[0]
    args = dict(call.get("args") or {})
    mobile = state.get("mobile") or state.get("primary_no")
    if mobile and not args.get("mobile"):
        args["mobile"] = mobile
    if state.get("address_id") and not args.get("address_id"):
        args["address_id"] = state["address_id"]
    call["args"] = args

    name = call.get("name")
    required = {
        "add_client_address": ["mobile", "address"],
        "schedule_pickup": ["mobile", "pickup_date", "pickup_time", "address_id"],
        "ship_fabric_to_store": ["mobile", "store_id"],
        "get_order_status": ["mobile"],
        "cancel_order": ["mobile", "order_id", "reason"],
        "modify_order": ["mobile", "order_id", "comment"],
    }.get(name)
    if required is None:
        return {
            "messages": messages + [
                response,
                ToolMessage(content=f"Tool error: unsupported order tool {name}",
                            tool_call_id=call.get("id")),
            ],
            "completed": False,
            "api_result": None,
        }

    history = "\n".join(
        str(getattr(message, "content", ""))
        for message in messages
        if getattr(message, "type", "") in ("human", "user", "ai")
    ) + f"\n{mobile or ''}\n{state.get('address_id') or ''}"
    ok, hint = validate_tool_args(name, args, required, history=history)
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
            "add_client_address": add_client_address.func,
            "schedule_pickup": schedule_pickup.func,
            "ship_fabric_to_store": ship_fabric_to_store.func,
            "get_order_status": get_order_status.func,
            "cancel_order": cancel_order.func,
            "modify_order": modify_order.func,
        },
    )
    updates.pop("args", None)
    return updates
