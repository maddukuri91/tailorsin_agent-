# agents/book_visit.py

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage

from state import AgentState
from tools.tailorsin_tools import book_store_visit
from agents._utils import _history_text, execute_tool_call, validate_tool_args

load_dotenv()


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


BOOK_VISIT_PROMPT = """
You are the Tailorsin Book-a-Visit Agent.

You help customers book a store visit / appointment.

Required information (matching the book_store_visit tool):
- mobile (the customer's contact number — use the number from the customer
  context when available, and only ask if it is not already provided)
- store_id (which Tailorsin store the customer wants to visit)
- bookdate (the preferred visit date, e.g. '2026-07-10')
- booktime (the preferred visit slot, e.g. '11:00 AM - 12:00 PM')

Rules:
1. Ask for the missing details in ONE short, friendly message.
2. NEVER invent, guess, or fill in details the customer has not provided.
   Missing info means you ASK, never assume.
3. NEVER claim the visit was booked ("done", "booked", "confirmed", ...)
   unless the book_store_visit tool result you just received explicitly
   reports status success. If the tool errored, say there was a snag and
   ask to retry — do not pretend it worked.
4. Call book_store_visit once all required info is available.

Customer-message style:
- Short, natural and friendly.
- Light emojis only (📅 🕐 🙌).

Tone:
- Friendly, upbeat and casual (our users are Gen Z and Gen Alpha).
"""


def book_visit_agent(state: AgentState):

    messages = state["messages"]

    response = llm.bind_tools(
        [book_store_visit]
    ).invoke([
        SystemMessage(content=BOOK_VISIT_PROMPT),
        *messages
    ])

    calls = getattr(response, "tool_calls", None)
    if calls:
        call = calls[0]
        args = call.get("args") or {}
        ok, hint = validate_tool_args(
            tool_name="book_store_visit",
            args=args,
            required=["mobile", "store_id", "bookdate", "booktime"],
            history=_history_text(messages),
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
                "mobile": args.get("mobile"),
                "store_id": args.get("store_id"),
                "bookdate": args.get("bookdate"),
                "booktime": args.get("booktime"),
                "api_result": None,
                "completed": False,
            }

    updates = execute_tool_call(
        response,
        messages,
        {"book_store_visit": book_store_visit.func}
    )

    args = updates.pop("args", None)
    if args:
        updates["mobile"] = args.get("mobile")
        updates["store_id"] = args.get("store_id")
        updates["bookdate"] = args.get("bookdate")
        updates["booktime"] = args.get("booktime")

    return updates