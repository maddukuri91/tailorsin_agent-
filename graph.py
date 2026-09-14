# graph.py

from langgraph.graph import StateGraph, START, END

from state import AgentState

from agents.supervisor import supervisor_node
from agents.signup import signup_agent
from agents.bulk_order import bulk_order_agent
from agents.fabric_estimation import fabric_estimation_agent
from agents.book_visit import book_visit_agent
from agents.human_support import human_support_agent
from agents.order import order_agent


def route_from_supervisor(state: AgentState):

    return state["next_agent"]


builder = StateGraph(AgentState)


# Nodes
builder.add_node(
    "supervisor",
    supervisor_node
)

builder.add_node(
    "signup",
    signup_agent
)

builder.add_node(
    "bulk_order",
    bulk_order_agent
)

builder.add_node(
    "fabric_estimation",
    fabric_estimation_agent
)

builder.add_node(
    "book_visit",
    book_visit_agent
)

builder.add_node(
    "human_support",
    human_support_agent
)

builder.add_node(
    "order",
    order_agent
)


# Entry
builder.add_edge(
    START,
    "supervisor"
)


# Supervisor routing
builder.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {
        "signup": "signup",
        "bulk_order": "bulk_order",
        "fabric_estimation": "fabric_estimation",
        "book_visit": "book_visit",
        "human_support": "human_support"
        ,"order": "order"
    }
)


# End after specialized agent
builder.add_edge(
    "signup",
    END
)

builder.add_edge(
    "bulk_order",
    END
)

builder.add_edge(
    "fabric_estimation",
    END
)

builder.add_edge(
    "book_visit",
    END
)

builder.add_edge(
    "human_support",
    END
)

builder.add_edge(
    "order",
    END
)


graph = builder.compile()