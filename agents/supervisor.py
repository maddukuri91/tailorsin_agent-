# agents/supervisor.py

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq

from state import AgentState

load_dotenv()


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


SUPERVISOR_PROMPT = """
You are the supervisor for Tailorsin customer services.

Route the user request to exactly one of these agents:

1. signup
   - customer registration
   - creating a new customer account

2. bulk_order
   - bulk order
   - wholesale order
   - large quantity order
   - business order enquiry

3. fabric_estimation
   - custom fabric estimation
   - a customer asking for a custom fabric estimate

4. book_visit
   - booking a store visit / appointment at a Tailorsin store

5. human_support
   - talking to a human / human support
   - human handover

If the customer is already being served by one agent (see
"currently_serving") and the newest customer line is just a bare detail for
that request — a phone number or a name — KEEP routing to that same
in-progress agent on its own line. Do NOT switch agents on a follow-up
number or name.

Also always respect an explicit "currently_serving: <agent>" line in the
conversation: it names the in-progress agent.

Return ONLY one of:

signup
bulk_order
fabric_estimation
book_visit
human_support
"""


def supervisor_node(state: AgentState):

    messages = state["messages"]
    active_agents = {
        "signup",
        "bulk_order",
        "fabric_estimation",
        "book_visit",
        "human_support",
        "order",
    }

    # Once a menu selection has started an agent flow, every subsequent
    # customer detail belongs to that same flow. Do not ask the supervisor
    # LLM to classify a name, phone number, date, or other follow-up again.
    prior_agent = state.get("next_agent")
    if prior_agent in active_agents:
        return {"next_agent": prior_agent}

    # The newest customer line in this service always starts with
    # "REFERENCE_PICTURE_URL:" (photo turn) or "NEW CUSTOMER MESSAGE:"
    # (typed turn). When it is a bare follow-up parameter — a number or a
    # name — keep the in-progress agent instead of asking the LLM to guess
    # from a one-word line.
    last_user_text = ""
    for message in reversed(messages or []):
        msg_type = getattr(message, "type", "")
        msg_text = message.content if isinstance(
            getattr(message, "content", ""), str
        ) else ""
        if msg_type in ("human", "user") and msg_text and msg_text.strip():
            last_user_text = msg_text.strip()
            break
    detail = ""
    marker = "NEW CUSTOMER MESSAGE:"
    if marker in last_user_text:
        detail = last_user_text.split(marker, 1)[1].strip()
    elif last_user_text.startswith("REFERENCE_PICTURE_URL:"):
        detail = last_user_text.split("REFERENCE_PICTURE_URL:", 1)[1]
        detail = (detail.splitlines() or [""])[0].strip()
    bare = detail and detail.split()[0].rstrip(",").isdigit()
    if bare:
        prior = state.get("next_agent")
        if prior in active_agents:
            return {"next_agent": prior}
        served = None
        for message in reversed(messages or []):
            msg_text = message.content if isinstance(
                getattr(message, "content", ""), str
            ) else ""
            if not msg_text or "currently_serving" not in msg_text:
                continue
            for agent_name in active_agents:
                if agent_name in msg_text:
                    served = agent_name
                    break
            if served:
                break
        if served:
            return {"next_agent": served}
        return {"next_agent": "fabric_estimation"}

    response = llm.invoke([
        SystemMessage(content=SUPERVISOR_PROMPT),
        *messages
    ])

    route = response.content.strip().lower()

    if route not in {
        "signup",
        "bulk_order",
        "fabric_estimation",
        "book_visit",
        "human_support",
        "order"
    }:
        route = "signup"

    return {
        "next_agent": route
    }