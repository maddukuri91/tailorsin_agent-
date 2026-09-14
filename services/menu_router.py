# services/menu_router.py
#
# The "Menu Router" layer. It serves the right menu for a customer type and
# resolves a user's selection against the *current* (possibly nested) menu
# into a downstream action:
#   - {"action": "content", "blocks": ..., "next": ...} -> show curated
#     content, then surface the "next" (sub)menu
#   - {"action": "agent",    ...}                        -> hand the intent to
#     the supervisor agent so the right sub-agent runs
#   - {"action": "reply",    "text": ...}                -> immediate answer
#   - {"action": "human",    ...}                        -> needs a human
#   - {"action": "retry"}                                -> invalid selection

from typing import Any, Dict

from services.segments import MENUS, ROOT_MENU_BY_TYPE, menu_for, root_menu_id


class MenuRouter:
    """Serve segment menus and resolve nested-menu selections into actions."""

    def root_menu_id(self, customer_type: str) -> str:
        return root_menu_id(customer_type)

    def get_menu(self, menu_id: str) -> Dict[str, Any]:
        return menu_for(menu_id)

    def register_menu(self, menu_id: str, menu: Dict[str, Any]) -> None:
        """Register a short-lived menu, such as a customer's saved addresses."""
        from services import segments

        segments.MENUS[menu_id] = menu

    def resolve(self, menu_id: str, choice: int) -> Dict[str, Any]:
        """
        Map a 1-based menu index (within the given menu) to an action.

        The result always carries the originating `menu_id` so the caller can
        keep the user in the right place.
        """
        menu = menu_for(menu_id)
        options = menu["options"]

        if choice < 1 or choice > len(options):
            return {"action": "retry", "menu_id": menu_id}

        option = options[choice - 1]
        return {
            "action": option.get("action", "human"),
            "menu_id": menu_id,
            "label": option.get("label", ""),
            "intent": option.get("intent"),
            "prompt": option.get("prompt"),
            "blocks": option.get("blocks"),
            "next": option.get("next"),
            "text": option.get("text"),
        }


menu_router = MenuRouter()