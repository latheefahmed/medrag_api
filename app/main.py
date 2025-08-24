from dotenv import load_dotenv
load_dotenv()  # load .env before anything else

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import auth, sessions, ask, debug
from .db import ensure_cosmos

app = FastAPI(title="MedRAG API", version="0.1.0")

# CORS for FE (with cookies)
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(auth.router)
app.include_router(sessions.router)
app.include_router(ask.router)
app.include_router(debug.router)

@app.on_event("startup")
def _startup_probe():
    try:
        ensure_cosmos()
    except Exception as e:
        # In dev, log and continue so the server can still start
        print(f"[WARN] Cosmos not available at startup: {e}")
