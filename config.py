# config.py

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    telegram_bot_token: str = field(
        default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    )
    telegram_webhook_secret: str = field(
        default_factory=lambda: os.getenv("TELEGRAM_WEBHOOK_SECRET", "").strip()
    )
    telegram_webhook_url: str = field(
        default_factory=lambda: os.getenv("TELEGRAM_WEBHOOK_URL", "").strip()
    )
    # ── LangSmith tracking & observability ──
    langsmith_tracing: str = field(
        default_factory=lambda: os.getenv("LANGSMITH_TRACING", "false").strip()
    )
    langsmith_endpoint: str = field(
        default_factory=lambda: os.getenv(
            "LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"
        ).strip()
    )
    langsmith_api_key: str = field(
        default_factory=lambda: os.getenv("LANGSMITH_API_KEY", "").strip()
    )
    langsmith_project: str = field(
        default_factory=lambda: os.getenv("LANGSMITH_PROJECT", "tailorsin.com")
        .strip()
        .strip('"')
        .strip("'")
    )
    # ── Anthropic (optional) ──
    anthropic_api_key: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "").strip()
    )


settings = Settings()