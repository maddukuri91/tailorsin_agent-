# services/customer_context_store.py
#
# Async adapter around the in-memory CustomerContextStore (defined in
# services/customer_context.py) and the agent graph state kept per session.
#
# This module exists because conversation_service.py expects an async API
# with the following call signatures:
#
#   await CustomerContextStore.get_state(session_id)     -> dict | None
#   await CustomerContextStore.save_state(session_id, d) -> None
#   await CustomerContextStore.clear(session_id)         -> None
#   await CustomerContextStore.get(session_id)           -> CustomerContext | None
#   await CustomerContextStore.set(session_id, ctx)      -> None   (ctx may be
#                                                                   a CustomerContext
#                                                                   or a (type, id)
#                                                                   tuple from
#                                                                   classify_mobile)
#
# The sync CustomerContextStore (get / save / delete) and classify_mobile
# live in services/customer_context.py and are reused here.

from __future__ import annotations

import asyncio
from typing import Any, Optional

from services.customer_context import (
    CustomerContext,
    CustomerContextStore as _SyncStore,
    classify_mobile as _sync_classify_mobile,
)

# ── singletons ────────────────────────────────────────────────────────────────

_store = _SyncStore()

# Agent graph state keyed by session_id.  _run_agent_turn writes the
# langgraph result dict here; handle_incoming_message reads it back on the
# next turn so the human-turn "previous_state" can be reconstructed.
_agent_states: dict[str, Any] = {}


# ── async store helpers ───────────────────────────────────────────────────────

async def _to_thread(func, *args: Any, **kwargs: Any) -> Any:
    """Run a (potentially blocking) callable in a thread pool.

    Use asyncio.to_thread when it exists (Python 3.9+), otherwise fall back
    to run_in_executor.  This keeps the wrapper importable on 3.8+ if ever
    needed, although this project targets 3.10+.
    """
    try:
        return await asyncio.to_thread(func, *args, **kwargs)
    except AttributeError:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


def _ensure_customer_context(ctx: Any) -> CustomerContext:
    """Normalise a ctx value to a CustomerContext instance.

    The conversation_service layer sometimes stores the raw return value of
    classify_mobile -- a ``(customer_type, customer_id)`` tuple -- as the
    session context.  Downstream helpers (_context_system_message, _main_menu_reply)
    expect a CustomerContext (or something with .customer_type / .describe()).
    Normalise inside the store so every reader gets a real CustomerContext.
    """
    if isinstance(ctx, CustomerContext):
        return ctx
    if isinstance(ctx, tuple) and len(ctx) == 2:
        customer_type, customer_id = ctx
        return CustomerContext(
            session_id="",  # session id is not available at classify time;
                             # it is filled in when handle_incoming_message stores
                             # the context under the real session_id key.
            customer_type=customer_type,
            customer_id=customer_id,
        )
    # Unknown shape — wrap the bare value so getattr(key, default) still works.
    return CustomerContext(session_id="", customer_type=str(ctx))


class CustomerContextStore:
    """Async facade over the in-memory customer-context store.

    All methods that touch the store or the CRM are run in a thread so the
    event loop is never blocked by I/O.
    """

    @classmethod
    async def get_state(cls, session_id: str) -> Optional[dict[str, Any]]:
        """Return the previously saved agent graph state for *session_id*."""
        return _agent_states.get(session_id)

    @classmethod
    async def save_state(
        cls, session_id: str, state: dict[str, Any]
    ) -> None:
        """Persist the agent graph state for *session_id*."""
        _agent_states[session_id] = state

    @classmethod
    async def clear(cls, session_id: str) -> None:
        """Drop both the customer context and the agent graph state."""
        await _to_thread(_store.delete, session_id)
        _agent_states.pop(session_id, None)

    @classmethod
    async def get(cls, session_id: str) -> Optional[CustomerContext]:
        """Return the CustomerContext for *session_id* (or None)."""
        raw = await _to_thread(_store.get, session_id)
        if raw is None:
            return None
        return _ensure_customer_context(raw)

    @classmethod
    async def set(
        cls, session_id: str, ctx: Any
    ) -> None:
        """Store *ctx* for *session_id*.

        *ctx* may be a CustomerContext or the (type, id) tuple returned by
        classify_mobile.  Either way a CustomerContext is persisted.
        """
        normalized = _ensure_customer_context(ctx)
        # Restore the real session id on the normalised context so that
        # .describe() and other helpers that reference session_id remain
        # meaningful.
        if isinstance(normalized, CustomerContext):
            object.__setattr__(normalized, "session_id", session_id)
        await _to_thread(_store.save, normalized)


# Re-export classify_mobile as an async function so callers can keep using
# ``await classify_mobile(mobile)`` without another import path change.
async def classify_mobile(mobile: str) -> tuple[str, Optional[str]]:
    """Async wrapper around services.customer_context.classify_mobile."""
    return await _to_thread(_sync_classify_mobile, mobile)
