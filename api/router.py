from fastapi import APIRouter
from api.endpoint import prompt, recommend

api_router = APIRouter()

api_router.include_router(prompt.router)
api_router.include_router(recommend.router)