# app.py

from fastapi import FastAPI

import config  # noqa: F401  (loads .env via config.py)
from config import settings
from channels.telegram import router, webhook_router


app = FastAPI(title="Tailorsin Agent")

app.include_router(router)
app.include_router(webhook_router)


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "tailorsin-agent",
        "endpoints": [
            "/telegram/webhook",
            "/telegram/webhook/{secret}",
            "/webhook/telegram",
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)