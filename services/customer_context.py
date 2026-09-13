# services/customer_context.py
#
# The "Customer Context" layer. Each customer session carries a persistent
# context object so every downstream stage (menu router, supervisor agent,
# sub-agents) knows who the customer is:
#   - session_id     : unique id for the conversation session
#   - customer_type  : active_client | client | new_user
#   - mobile         : the customer's phone number
#   - customer_id    : the CRM client id (if known)

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from tools.tailorsin_tools import classify_client

from services.segments import KNOWN_CLIENT_TYPES

DEFAULT_CUSTOMER_TYPE = "new_user"


@dataclass
class CustomerContext:
    session_id: str
    customer_type: Optional[str] = None
    mobile: Optional[str] = None
    customer_id: Optional[str] = None
    current_menu_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_classified(self) -> bool:
        return self.customer_type is not None

    def describe(self) -> str:
        """A short human-readable summary used to orient the LLM agents."""
        parts = [f"session: {self.session_id}"]
        if self.customer_type:
            parts.append(f"customer_type: {self.customer_type}")
        if self.mobile:
            parts.append(f"mobile: {self.mobile}")
        if self.customer_id:
            parts.append(f"customer_id: {self.customer_id}")
        if self.current_menu_id:
            parts.append(f"current_menu: {self.current_menu_id}")
        return ", ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "customer_type": self.customer_type,
            "mobile": self.mobile,
            "customer_id": self.customer_id,
            "current_menu_id": self.current_menu_id,
        }


class CustomerContextStore:
    """In-memory, thread-safe store of customer contexts by session id."""

    def __init__(self):
        self._contexts: Dict[str, CustomerContext] = {}
        self._lock = threading.Lock()

    def create_session(self) -> CustomerContext:
        ctx = CustomerContext(session_id=str(uuid.uuid4()))
        with self._lock:
            self._contexts[ctx.session_id] = ctx
        return ctx

    def get(self, session_id: Optional[str]) -> Optional[CustomerContext]:
        if not session_id:
            return None
        with self._lock:
            return self._contexts.get(session_id)

    def save(self, ctx: CustomerContext) -> None:
        with self._lock:
            self._contexts[ctx.session_id] = ctx

    def delete(self, session_id: Optional[str]) -> None:
        if not session_id:
            return
        with self._lock:
            self._contexts.pop(session_id, None)


def classify_mobile(mobile: str) -> Tuple[str, Optional[str]]:
    """
    Ask the CRM to classify a mobile number.

    Returns (customer_type, customer_id). Distinguishes hard failures
    (exception / unparseable CRM response) from a genuine "unknown number"
    answer:

    - hard failure -> ('unknown', None) so the caller can ask the customer
      to retry instead of silently treating them as a new user.
    - CRM answers with an unknown type -> 'new_user' (genuine new user).
    """
    try:
        data = classify_client.func(mobile=mobile)
    except Exception:  # noqa: BLE001 - transport / CRM error
        return "unknown", None

    if not isinstance(data, dict):
        return "unknown", None

    customer_type = data.get("type")
    if customer_type not in KNOWN_CLIENT_TYPES:
        if customer_type is None and data.get("status") is not None:
            # CRM answered but without a usable type -> lookup failed.
            return "unknown", None
        customer_type = DEFAULT_CUSTOMER_TYPE

    client = data.get("client") or {}
    customer_id = client.get("id")
    return customer_type, (str(customer_id) if customer_id is not None else None)