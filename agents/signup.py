# agents/signup.py

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage

from state import AgentState
from tools.tailorsin_tools import register_client
from agents._utils import execute_tool_call, resolve_primary_contact, validate_tool_args

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

    response = llm.bind_tools(
        [register_client]
    ).invoke([
        SystemMessage(content=SIGNUP_PROMPT),
        *messages
    ])

    calls = getattr(response, "tool_calls", None)
    if calls:
        call = calls[0]
        args = call.get("args") or {}
        ok, hint = validate_tool_args(
            tool_name="register_client",
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
        {"register_client": register_client.func}
    )

    args = updates.pop("args", None)
    if args:
        updates["client_name"] = args.get("client_name")
        updates["primary_no"] = args.get("primary_no")
        updates["secondary_no"] = args.get("secondary_no")

    return updates