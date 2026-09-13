# agents/fabric_estimation.py

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage

from state import AgentState
from tools.tailorsin_tools import get_custom_fabric_estimation
from agents._utils import execute_tool_call, resolve_primary_contact, validate_tool_args

load_dotenv()


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


FABRIC_PROMPT = """
You are the Tailorsin Custom Fabric Estimation Agent.

Your job is to collect the details needed to request a custom fabric estimate.

Required information (matching the get_custom_fabric_estimation tool):
- client_name (the customer's name)
- primary_no (the number the customer shared via their Telegram contact —
  if it is present in the customer context, use it and DO NOT ask for it
  again)
- secondary_no (optional — only if the customer provides a *different*
  number for calls)

Rules:
1. Ask for the missing details in ONE short, friendly message.
2. Keep the request simple and easy to follow — avoid over-numbering.
3. primary_no comes from the customer context (the number they shared). Do
   NOT ask for a number if one is already available.
4. If the customer enters a different number for calls, send it as
   secondary_no — never overwrite the shared primary_no with it.
5. NEVER invent, guess, or fill in customer details (name, numbers) the
   customer has not actually provided. Missing info means you ASK, never assume.
6. NEVER claim the estimate was requested/sent ("done", "estimate ready",
   "confirmed", ...) unless the get_custom_fabric_estimation tool result
   you just received explicitly reports status success. If the tool
   errored, say there was a snag and ask to retry — do not pretend it worked.
7. Call get_custom_fabric_estimation once all required info is available.

Customer-message style:
- Short, natural and friendly.
- Light emojis only (✂️ 📏 🙌).
- Example of a good message:
  "Sure! 🙌 Just tell me your name and the number we can reach you on,
   and I'll get your custom estimate started."

Tone:
- Friendly, upbeat and casual (our users are Gen Z and Gen Alpha).
"""


def fabric_estimation_agent(state: AgentState):

    messages = state["messages"]

    response = llm.bind_tools(
        [get_custom_fabric_estimation]
    ).invoke([
        SystemMessage(content=FABRIC_PROMPT),
        *messages
    ])

    calls = getattr(response, "tool_calls", None)
    if calls:
        call = calls[0]
        args = call.get("args") or {}
        ok, hint = validate_tool_args(
            tool_name="get_custom_fabric_estimation",
            args=args,
            required=["client_name", "primary_no"],
            history=resolve_primary_contact(args, messages),
        )
        if not ok:
            # Don't call the CRM — ask the user to fix the missing/invalid field.
            from langchain_core.messages import ToolMessage
            return {
                "messages": state["messages"] + [
                    response,
                    ToolMessage(
                        content=f"Skipping tool call — user needs to supply: {hint}",
                        tool_call_id=call.get("id"),
                    ),
                ],
                "client_name": args.get("client_name"),
                "primary_no": args.get("primary_no"),
                "secondary_no": args.get("secondary_no"),
                # Reset any stale result so a previous turn's success can
                # never be mistaken for this turn's outcome.
                "api_result": None,
                "completed": False,
            }

    updates = execute_tool_call(
        response,
        messages,
        {"get_custom_fabric_estimation": get_custom_fabric_estimation.func}
    )

    args = updates.pop("args", None)
    if args:
        updates["client_name"] = args.get("client_name")
        updates["primary_no"] = args.get("primary_no")
        updates["secondary_no"] = args.get("secondary_no")

    return updates