"""FastAPI application.  Run with:  uvicorn helmet_monitor.main:app"""

from __future__ import annotations

import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from helmet_monitor.api import routes, websocket
from helmet_monitor.config import load_settings
from helmet_monitor.services import Services, build_services

STATIC_DIR = Path(__file__).parent / "static"
SNAPSHOT_NAME = re.compile(r"^[\w-]+\.jpg$")


def create_app(services: Services | None = None) -> FastAPI:
    """Pass ``services`` to inject fakes in tests; otherwise real models load at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.services = services or build_services(load_settings())
        yield
        app.state.services.database.close()

    app = FastAPI(title="Helmet Monitoring & Face Recognition", version="1.0.0", lifespan=lifespan)
    app.include_router(routes.router)
    app.include_router(websocket.router)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/snapshots/{name}", include_in_schema=False)
    def snapshot(name: str, request: Request):
        path = request.app.state.services.settings.snapshot_dir / name
        if not SNAPSHOT_NAME.match(name) or not path.is_file():
            raise HTTPException(404, "Snapshot not found.")
        return FileResponse(path, media_type="image/jpeg")

    return app


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
app = create_app()
