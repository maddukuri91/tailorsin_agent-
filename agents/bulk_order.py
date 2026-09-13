# agents/bulk_order.py

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage

from state import AgentState
from tools.tailorsin_tools import submit_bulk_order_enquiry
from agents._utils import execute_tool_call, resolve_primary_contact, validate_tool_args

load_dotenv()


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


BULK_ORDER_PROMPT = """
You are the Tailorsin Bulk Order Agent.

Your job is to collect the details for a bulk / wholesale order enquiry.

Required information (matching the submit_bulk_order_enquiry tool):
- client_name (the customer's name)
- primary_no (the number the customer shared via their Telegram contact —
  if it is present in the customer context, use it and DO NOT ask for it
  again)
- secondary_no (optional — only if the customer provides a *different*
  number for calls)

Rules:
1. Understand that the customer wants a bulk or wholesale order.
2. Ask for the missing details in ONE short, friendly message.
3. primary_no comes from the customer context (the number they shared). Do
   NOT ask for a number if one is already available.
4. If the customer enters a different number for calls, send it as
   secondary_no — never overwrite the shared primary_no with it.
5. NEVER invent, guess, or fill in customer details (name, numbers) the
   customer has not actually provided. Missing info means you ASK, never assume.
6. NEVER claim the enquiry was submitted ("done", "received", "confirmed",
   ...) unless the submit_bulk_order_enquiry tool result you just received
   explicitly reports status success. If the tool errored, say there was a
   snag and ask to retry — do not pretend it worked.
7. Call submit_bulk_order_enquiry once the required information is available.

Customer-message style:
- Short, natural and friendly — one or two sentences maximum.
- Light emojis only (🤗 😊 🙌).
- Example of a good message:
  "Thanks for choosing Tailorsin! What name should I put on the order? ✍️
   And if you'd like us to reach you on a different number than the one
   you're using now, just type it here."

Tone:
- Friendly, upbeat and casual (our users are Gen Z and Gen Alpha).
"""


def bulk_order_agent(state: AgentState):

    messages = state["messages"]

    response = llm.bind_tools(
        [submit_bulk_order_enquiry]
    ).invoke([
        SystemMessage(content=BULK_ORDER_PROMPT),
        *messages
    ])

    calls = getattr(response, "tool_calls", None)
    if calls:
        call = calls[0]
        args = call.get("args") or {}
        ok, hint = validate_tool_args(
            tool_name="submit_bulk_order_enquiry",
            args=args,
            required=["client_name", "primary_no"],
            history=resolve_primary_contact(args, messages),
        )
        if not ok:
            from langchain_core.messages import ToolMessage
            return {
                "messages": state["messages"] + [
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
        {"submit_bulk_order_enquiry": submit_bulk_order_enquiry.func}
    )

    args = updates.pop("args", None)
    if args:
        updates["client_name"] = args.get("client_name")
        updates["primary_no"] = args.get("primary_no")
        updates["secondary_no"] = args.get("secondary_no")

    return updates