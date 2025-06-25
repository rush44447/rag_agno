from fastapi import APIRouter

from api.routes.playground import playground_router
from api.routes.health import health_check_router
from api.routes.chatbot import chatbot_router
from api.routes.summarize import summarizer_router

v1_router = APIRouter(prefix="/v1")
v1_router.include_router(playground_router)
v1_router.include_router(health_check_router)
v1_router.include_router(chatbot_router)
v1_router.include_router(summarizer_router)
