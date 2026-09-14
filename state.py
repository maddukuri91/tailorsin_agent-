# state.py

from typing import TypedDict, List, Optional


class AgentState(TypedDict):

    messages: List

    # routing
    next_agent: Optional[str]

    # customer information
    client_name: Optional[str]
    primary_no: Optional[str]
    secondary_no: Optional[str]

    # book a visit
    mobile: Optional[str]
    store_id: Optional[int]
    bookdate: Optional[str]
    booktime: Optional[str]

    # order flow
    order_action: Optional[str]
    address_id: Optional[int]

    # API result
    api_result: Optional[dict]

    # status
    completed: bool