
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from dotenv import load_dotenv 
    load_dotenv()
except Exception:
    pass


from app.routers import auth, sessions, ask, debug
from app.db import ensure_cosmos


def _parse_cors_origins() -> list[str]:
    """
    Accepts either:
      - CORS_ORIGINS="https://foo,https://bar"
      - and/or FRONTEND_ORIGIN="https://foo"
    Returns a distinct, non-empty list.
    """
    origins = set()
    cors = (os.getenv("CORS_ORIGINS") or "").strip()
    if cors:
        for o in cors.split(","):
            o = o.strip()
            if o:
                origins.add(o)
    fe = (os.getenv("FRONTEND_ORIGIN") or "").strip()
    if fe:
        origins.add(fe)
    if not origins:
        origins.add("http://localhost:3000")
    return sorted(origins)


def create_app() -> FastAPI:
    app = FastAPI(title="MedRAG API", version="0.1.0")

    allow = _parse_cors_origins()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(auth.router)
    app.include_router(sessions.router)
    app.include_router(ask.router)
    app.include_router(debug.router)

    @app.on_event("startup")
    def _startup_probe():
        try:
            ensure_cosmos()
        except Exception as e:
            print(f"[WARN] Cosmos not available at startup: {e}")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=os.getenv("RELOAD", "0") == "1")
