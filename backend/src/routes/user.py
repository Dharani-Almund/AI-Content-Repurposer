from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, HttpUrl
from typing import Dict, List, Any

from backend.src.services.chroma_db import ChromaDBHandler
from backend.src.services.rag import process_query
from backend.src.dependencies import RoleChecker, get_current_user

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# User role checker
user_role_checker = RoleChecker(["user"])

# Initialize router with prefix
user_router = APIRouter()


# Input validation models
class QueryRequest(BaseModel):
    question: str
    context_type: str = "all"  # Allow filtering by content type


class ContentFilterRequest(BaseModel):
    content_type: str = "all"
    limit: int = 10
    offset: int = 0


# Dependency function for service injection
def get_chroma_handler():
    return ChromaDBHandler()


# Add a debugging endpoint that shows the user's token information
@user_router.get("/auth-debug")
async def auth_debug(user: Dict[str, Any] = Depends(get_current_user)):
    """Debug endpoint to check token data"""
    return {
        "authenticated": True,
        "user_id": user.get("user_id"),
        "role": user.get("role"),
        "email": user.get("email")
    }


@user_router.get("/", dependencies=[Depends(user_role_checker)])
async def user_home():
    """User home page"""
    return {"message": "Welcome to the User Portal"}


@user_router.get("/search", dependencies=[Depends(user_role_checker)])
async def search_page():
    """User search page"""
    return {"message": "Search Portal"}


@user_router.post("/query", dependencies=[Depends(user_role_checker)])
async def query_rag(
        request: QueryRequest,
        chroma_handler: ChromaDBHandler = Depends(get_chroma_handler)
):
    """
    Advanced query endpoint using Retrieval-Augmented Generation (RAG)

    Args:
        request (QueryRequest): Query details
        chroma_handler (ChromaDBHandler): ChromaDB service for content retrieval

    Returns:
        Dict[str, str]: Generated response based on query
    """
    try:
        # Optional: Retrieve context based on content type
        if request.context_type != "all":
            # FIX: Pass the question as the query parameter
            context = chroma_handler.get_context_by_type(query=request.question, content_type=request.context_type)
        else:
            # FIX: Pass the question as the query parameter
            context = chroma_handler.get_all_context(query=request.question)

        result = process_query(request.question, context)

        if not result:
            logger.warning(f"No response generated for query: {request.question}")
            raise HTTPException(status_code=204, detail="No response generated")

        return {"response": result}

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@user_router.post("/content", dependencies=[Depends(user_role_checker)])
async def list_content(
        request: ContentFilterRequest,
        chroma_handler: ChromaDBHandler = Depends(get_chroma_handler)
):
    """
    Retrieve content with optional filtering

    Args:
        request (ContentFilterRequest): Content retrieval parameters
        chroma_handler (ChromaDBHandler): ChromaDB service for content retrieval

    Returns:
        Dict[str, List[Dict]]: List of content items
    """
    try:
        content = chroma_handler.list_content(
            content_type=request.content_type
        )

        # Apply pagination after retrieving
        start = request.offset
        end = request.offset + request.limit
        paginated_content = content[start:end] if content else []

        return {
            "content": paginated_content,
            "total_count": len(content),
            "limit": request.limit,
            "offset": request.offset
        }

    except Exception as e:
        logger.error(f"Error retrieving content: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving content: {str(e)}")


@user_router.get("/content/{content_id}", dependencies=[Depends(user_role_checker)])
async def get_content_details(
        content_id: str,
        chroma_handler: ChromaDBHandler = Depends(get_chroma_handler)
):
    """
    Retrieve details of a specific content item

    Args:
        content_id (str): Unique identifier for content
        chroma_handler (ChromaDBHandler): ChromaDB service for content retrieval

    Returns:
        Dict[str, Any]: Detailed content information
    """
    try:
        # Note: This assumes you have a get_content_by_id method in ChromaDBHandler
        # If you don't, you'll need to implement it
        content_details = chroma_handler.get_content_by_id(content_id)

        if not content_details:
            raise HTTPException(status_code=404, detail="Content not found")

        return content_details

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving content details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving content details: {str(e)}")