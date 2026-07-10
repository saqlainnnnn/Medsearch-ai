from fastapi import APIRouter

from app.api.routes import router as base_router

api_router = APIRouter()

api_router.include_router(base_router)