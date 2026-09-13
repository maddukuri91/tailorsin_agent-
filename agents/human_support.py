# agents/human_support.py

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage

from state import AgentState
from tools.tailorsin_tools import human_handover
from agents._utils import _history_text, execute_tool_call, validate_tool_args

load_dotenv()


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


HUMAN_SUPPORT_PROMPT = """
You are the Tailorsin Human Support Agent.

When a customer asks to speak to a real person, or needs help that only a
human can handle, collect the needed contact number and trigger the handover.

Required information (matching the human_handover tool):
- mobile (the customer's contact number — use the number from the customer
  context when available, and only ask if it is not already provided)

Rules:
1. Reassure the customer that a real human will take over, and confirm you
   have a valid contact number.
2. NEVER invent, guess, or fill in a mobile number the customer has not
   provided. Missing info means you ASK, never assume.
3. NEVER claim the handover was requested ("done", "handed over", ...)
   unless the human_handover tool result you just received explicitly
   reports status success. If the tool errored, say there was a snag and
   ask to retry — do not pretend it worked.
4. Call human_handover once a valid mobile number is available.

Customer-message style:
- Short, natural and friendly.
- Light emojis only (🧑‍💼 🤝 🙌).

Tone:
- Friendly, upbeat and casual (our users are Gen Z and Gen Alpha).
"""


def human_support_agent(state: AgentState):

    messages = state["messages"]

    response = llm.bind_tools(
        [human_handover]
    ).invoke([
        SystemMessage(content=HUMAN_SUPPORT_PROMPT),
        *messages
    ])

    calls = getattr(response, "tool_calls", None)
    if calls:
        call = calls[0]
        args = call.get("args") or {}
        ok, hint = validate_tool_args(
            tool_name="human_handover",
            args=args,
            required=["mobile"],
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
                "api_result": None,
                "completed": False,
            }

    updates = execute_tool_call(
        response,
        messages,
        {"human_handover": human_handover.func}
    )

    args = updates.pop("args", None)
    if args:
        updates["mobile"] = args.get("mobile")

    return updates