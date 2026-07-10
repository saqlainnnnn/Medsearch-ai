import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from app.api import api_router
from app.core import setup_logging

setup_logging()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown events.
    """

    logger.info("Starting MedSearch AI...")

    yield

    logger.info("Shutting down MedSearch AI...")


app = FastAPI(
    title="MedSearch AI",
    description="Agentic Retrieval-Augmented Generation for Medical Question Answering",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(api_router)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log every incoming request.
    """

    start = time.perf_counter()

    response = await call_next(request)

    duration = time.perf_counter() - start

    logger.info(
        "%s %s | %s | %.3f sec",
        request.method,
        request.url.path,
        response.status_code,
        duration,
    )

    return response