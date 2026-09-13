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
        {"label": "How it works",       "intent": "how_it_works"},
        {"label": "Price catalogue",    "intent": "price_catalogue"},
        {"label": "Place an order",     "intent": "register"},
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
from services.company_content import TAILORSIN_ABOUT, TAILORSIN_PRICE_CATALOGUE


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


def _menu_option(option: dict[str, str]) -> dict[str, object]:
    """Convert a segment option into the menu-router action format."""
    intent = option["intent"]
    if intent in ROUTABLE_INTENTS:
        return {
            "label": option["label"],
            "intent": intent,
            "action": "agent",
            "prompt": ROUTABLE_INTENTS[intent],
        }
    if intent == "appointment":
        return {
            "label": option["label"],
            "intent": intent,
            "action": "agent",
            "prompt": "I want to book a store visit.",
        }
    if intent in STATIC_REPLIES:
        return {
            "label": option["label"],
            "intent": intent,
            "action": "reply",
            "text": STATIC_REPLIES[intent],
        }
    return {
        "label": option["label"],
        "intent": intent,
        "action": "human",
    }


ROOT_MENU_BY_TYPE: dict[str, str] = {
    customer_type: f"{customer_type}_root"
    for customer_type in SEGMENT_MENU_OPTIONS
}

MENUS: dict[str, dict[str, object]] = {
    menu_id: {
        "title": SEGMENT_TITLES[customer_type],
        "options": [
            _menu_option(option)
            for option in SEGMENT_MENU_OPTIONS[customer_type]
        ],
    }
    for customer_type, menu_id in ROOT_MENU_BY_TYPE.items()
}

# New-user onboarding is intentionally a nested menu. The root choices match
# the numbered flow shown to customers, while the leaf choices route to the
# existing agents.
MENUS["new_user_root"] = {
    "title": SEGMENT_TITLES["new_user"],
    "options": [
        {
            "label": "How it works",
            "intent": "how_it_works",
            "action": "content",
            "blocks": [TAILORSIN_ABOUT],
            "next": "new_user_how_it_works",
        },
        {
            "label": "Price catalogue",
            "intent": "price_catalogue",
            "action": "content",
            "blocks": [TAILORSIN_PRICE_CATALOGUE],
            "next": "new_user_price_catalogue",
        },
        _menu_option({"label": "Place an order", "intent": "register"}),
    ],
}

MENUS["new_user_how_it_works"] = {
    "title": "What would you like to explore next?",
    "options": [
        {
            "label": "Price catalogue",
            "intent": "price_catalogue",
            "action": "content",
            "blocks": [TAILORSIN_PRICE_CATALOGUE],
            "next": "new_user_price_catalogue",
        },
        _menu_option({"label": "Place an order", "intent": "register"}),
    ],
}

MENUS["new_user_price_catalogue"] = {
    "title": "Choose an option to continue:",
    "options": [
        _menu_option({
            "label": "Custom fabric estimation",
            "intent": "fabric_estimation",
        }),
        _menu_option({
            "label": "Bulk order enquiry",
            "intent": "bulk_order",
        }),
        _menu_option({"label": "Place an order", "intent": "register"}),
    ],
}


def root_menu_id(customer_type: str) -> str:
    """Return the root menu id for a customer segment."""
    return ROOT_MENU_BY_TYPE.get(customer_type, ROOT_MENU_BY_TYPE["new_user"])


def menu_for(menu_id: str) -> dict[str, object]:
    """Return a menu by id, defaulting safely to the new-user menu."""
    return MENUS.get(menu_id, MENUS[ROOT_MENU_BY_TYPE["new_user"]])


def options_for(client_type: str) -> list[dict[str, str]]:
    """Return the menu options for a client type (defaults to new_user)."""
    if client_type in SEGMENT_MENU_OPTIONS:
        return SEGMENT_MENU_OPTIONS[client_type]
    return SEGMENT_MENU_OPTIONS["new_user"]


def title_for(client_type: str) -> str:
    return SEGMENT_TITLES.get(client_type, SEGMENT_TITLES["new_user"])