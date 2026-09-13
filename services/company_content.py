# services/company_content.py
#
# Curated onboarding / marketing copy shown to customers — especially new
# users. These blocks are written in Telegram Markdown (e.g. *bold*), so they
# are sent to Telegram RAW (not run through the markdown-escape helper) and
# rely on Telegram's Markdown parser + the plain-text fallback in
# channels/telegram.py if a parse error ever occurs.

TAILORSIN_ABOUT = """\
tailorsin.com collects your fabric, stitches your garments, and delivers them—\
typically within 24 hours after design approval and cloth pickup. ⚡

*How tailorsin.com Works* 🧵

1. *📅 Schedule a Pickup*
• We collect your fabric from your location, or you can drop it off at our store.

2. *🧺 Share Your Fabric & Reference Garment*
• Provide your fabric along with a sample/reference garment.
• Within 6 business hours, our team contacts you to confirm the design and share a stitching estimate.

3. *✅ Approve the Estimate & Make Payment*
• Once you approve the estimate and complete the payment, we begin stitching.

4. *🚗 Stitching & Delivery*
• Your stitched garments are delivered to your doorstep.

5. *👕 Free Alterations*
• Free fitting alterations are available within 7 days of delivery.
"""


TAILORSIN_PRICE_CATALOGUE = """\
*Price List & T&C* 💸

https://drive.google.com/file/d/1s67qOzn2n22lN670ir0Le462FcgGyCGL/view?usp=sharing
"""


# Blocks shown when a new user picks "Learn about tailorsin.com".
NEW_USER_ABOUT_BLOCKS: list[str] = [
    TAILORSIN_ABOUT,
    TAILORSIN_PRICE_CATALOGUE,
]