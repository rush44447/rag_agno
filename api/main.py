from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
import os

from api.settings import api_settings
from api.routes.v1_router import v1_router
from api.middleware.structured_response import StructuredResponseMiddleware
from dotenv import load_dotenv

load_dotenv()

def create_app() -> FastAPI:
    """Create a FastAPI App"""
    app: FastAPI = FastAPI(
        title=api_settings.title,
        version=api_settings.version,
        docs_url="/docs" if api_settings.docs_enabled else None,
        redoc_url="/redoc" if api_settings.docs_enabled else None,
        openapi_url="/openapi.json" if api_settings.docs_enabled else None,
    )

    origins = [
        "http://localhost:8000",     # if testing direct on backend
        "http://127.0.0.1:8000",
        os.getenv("FRONTEND_ORIGIN"),              # production IP (Angular frontend)
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,              # ✅ use list, no wildcard
        allow_credentials=True,             # ✅ cookies or auth headers
        allow_methods=["*"],
        allow_headers=["*"],
        max_age=3600
    )

    app.add_middleware(StructuredResponseMiddleware)

    # Routers
    app.include_router(v1_router)

    return app

app = create_app()