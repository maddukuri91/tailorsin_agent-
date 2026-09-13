# services/classification_service.py
#
# Async classification helpers.  conversation_service.py imports both
# ``CustomerContextStore`` and ``classify_mobile`` from this module, so it
# re-exports them from services.customer_context_store (the canonical async
# implementation) to keep a single source of truth.
from __future__ import annotations

from services.customer_context_store import (
    CustomerContextStore,
    classify_mobile,
)

__all__ = ["CustomerContextStore", "classify_mobile"]
