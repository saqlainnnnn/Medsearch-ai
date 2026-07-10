from fastapi import APIRouter

router = APIRouter()


@router.get("/", tags=["Root"])
async def root():
    """
    Root endpoint.
    """

    return {
        "project": "MedSearch AI",
        "version": "1.0.0",
        "status": "running",
    }


@router.get("/health", tags=["Health"])
async def health():
    """
    Health check endpoint.
    """

    return {
        "status": "healthy"
    }