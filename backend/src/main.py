import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


# Import routers
from backend.src.routes.admin import admin_router
from backend.src.routes.user import user_router

# Import services
from backend.src.services.chroma_db import ChromaDBHandler
from backend.src.services.transcriber import PodcastTranscriber
from backend.src.services.scraper import WebScraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Custom middleware for logging requests and responses."""

    async def dispatch(self, request, call_next):
        # Log incoming request
        logger.info(f"Incoming request: {request.method} {request.url}")

        try:
            response = await call_next(request)

            # Log response status
            logger.info(f"Response status: {response.status_code}")
            return response

        except Exception as exc:
            logger.error(f"Unhandled exception: {exc}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"message": "Internal server error"}
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown events.

    This context manager handles:
    - Initializing services
    - Performing startup checks
    - Closing resources on shutdown
    """
    try:
        # Initialize services
        app.state.chroma_handler = ChromaDBHandler()
        app.state.transcriber = PodcastTranscriber()
        app.state.scraper = WebScraper()

        # Perform startup checks
        logger.info("Performing startup checks...")
        startup_checks = {
            "ChromaDB": app.state.chroma_handler.check_connection(),
            "Transcriber": app.state.transcriber.check_dependencies(),
            "Web Scraper": app.state.scraper.check_configuration()
        }

        # Log startup check results
        for service, status in startup_checks.items():
            if not status:
                logger.warning(f"Startup check failed for {service}")

        logger.info("Application startup complete")
        yield

    except Exception as e:
        logger.error(f"Error during application startup: {e}", exc_info=True)
    finally:
        # Cleanup resources if needed
        logger.info("Shutting down application resources")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured application instance
    """
    app = FastAPI(
        title="AI Content Repurposer",
        description="A powerful platform for processing and querying content",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )

    # Configure CORS with more restrictive settings
    origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # Add request logging middleware
    app.add_middleware(RequestLoggingMiddleware)

    # Include API routes
    app.include_router(admin_router, prefix="/admin", tags=["Admin"])
    app.include_router(user_router, prefix="/user", tags=["User"])

    @app.get("/", response_model=Dict[str, Any])
    async def home():
        """
        Home route providing application overview.

        Returns:
            Dict: Application information and service descriptions
        """
        return {
            "message": "Welcome to AI Content Repurposer",
            "version": "0.1.0",
            "services": {
                "admin": {
                    "description": "Manage content extraction and processing",
                    "endpoints": [
                        "/admin/content",
                        "/admin/sources"
                    ]
                },
                "user": {
                    "description": "Query and retrieve processed content",
                    "endpoints": [
                        "/user/search",
                        "/user/content"
                    ]
                }
            },
            "documentation": {
                "swagger": "/docs",
                "redoc": "/redoc"
            }
        }

    # Global exception handler
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc):
        """
        Global HTTP exception handler for consistent error responses.

        Args:
            request: The incoming request
            exc: The HTTP exception

        Returns:
            JSONResponse with error details
        """
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "status_code": exc.status_code
            }
        )

    return app


# Create the app
app = create_app()

# Optional: Run the server if this script is the main entry point
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", 8000)),
        reload=os.getenv("DEBUG", "False").lower() == "true"
    )