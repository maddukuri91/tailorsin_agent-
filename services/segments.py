# services/segments.py
#
# Customer-segment menu definitions. A customer is classified by the CRM
# (getclient.php) into one of the known client types, and each type sees a
# different set of options. Selecting an option sends back the matching
# intent, which the conversation service routes accordingly.

COMMON_OPTIONS: list[dict[str, str]] = [
    {"label": "Get a fabric estimate",     "intent": "fabric_estimation"},
    {"label": "Bulk / wholesale enquiry",  "intent": "bulk_order"},
    {"label": "Talk to our team",          "intent": "talk_to_human"},
]

SEGMENT_MENU_OPTIONS: dict[str, list[dict[str, str]]] = {

    "active_client": [
        {"label": "Track my current order",      "intent": "order_status"},
        {"label": "Modify my order",             "intent": "order_changes"},
        {"label": "Cancel my order",             "intent": "order_cancel"},
        {"label": "Place a new order / Pickup",  "intent": "new_order"},
        *COMMON_OPTIONS,
    ],

    "client": [
        {"label": "Place an order",     "intent": "new_order"},
        {"label": "Book a store visit",     "intent": "appointment"},
        {"label": "Drop off fabric at store",    "intent": "fabric_delivery"},
        *COMMON_OPTIONS,
    ],

    "new_user": [
        {"label": "How does tailorsin.com work?",   "intent": "about"},
        {"label": "Place an order",                 "intent": "register"},
        *COMMON_OPTIONS,
    ],
}

KNOWN_CLIENT_TYPES: set[str] = {"active_client", "client", "new_user"}

# Headers shown above each segment's menu.
SEGMENT_TITLES: dict[str, str] = {
    "active_client": "Welcome back! 👋 Here's how I can help with your orders:",
    "client": "Welcome back! 👋 Here's how I can help you today:",
    "new_user": "Welcome to Tailorsin! 👋 Here's how I can help you get started:",
}

# Human-friendly labels for internal reporting / logs.
SEGMENT_LABELS: dict[str, str] = {
    "active_client": "Active Client",
    "client": "Client",
    "new_user": "New User",
}

# Intents that the agent can already handle end-to-end. They are rewritten
# into a natural-language request and routed through the existing graph so
# the relevant agent completes the action (registration, fabric estimation,
# bulk order).
ROUTABLE_INTENTS: dict[str, str] = {
    "register": "I want to register as a new Tailorsin client.",
    "fabric_estimation": "I would like a custom fabric estimation.",
    "bulk_order": "I want to make a bulk / wholesale order enquiry.",
}

# A couple of intents have nice built-in replies.
STATIC_REPLIES: dict[str, str] = {
    "about": (
        "Tailorsin is a bespoke tailoring service handling measurements, "
        "fabric selection, and delivery — so you get perfectly fitted garments. "
        "Visit https://tailorsin.com to learn more!"
    ),
    "talk_to_human": (
        "Sure — our team will get in touch with you shortly. In the meantime, "
        "is there anything I can help you with?"
    ),
}


def options_for(client_type: str) -> list[dict[str, str]]:
    """Return the menu options for a client type (defaults to new_user)."""
    if client_type in SEGMENT_MENU_OPTIONS:
        return SEGMENT_MENU_OPTIONS[client_type]
    return SEGMENT_MENU_OPTIONS["new_user"]


def title_for(client_type: str) -> str:
    return SEGMENT_TITLES.get(client_type, SEGMENT_TITLES["new_user"])