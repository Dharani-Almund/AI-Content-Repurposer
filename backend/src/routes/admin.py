import logging
import traceback
import uuid
import datetime
from fastapi import Request
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, HttpUrl

from backend.src.services.chroma_db import ChromaDBHandler
from backend.src.services.scraper import WebScraper
from backend.src.services.transcriber import PodcastTranscriber
from backend.src.dependencies import RoleChecker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Dependency functions for service injection
def get_chroma_handler():
    return ChromaDBHandler()


def get_web_scraper():
    return WebScraper()


def get_transcriber():
    return PodcastTranscriber()


# Custom exception classes
class ContentExtractionError(Exception):
    """Custom exception for content extraction failures"""
    pass


class PodcastProcessingError(Exception):
    """Custom exception for podcast processing failures"""
    pass


# Input validation models
class BlogExtractRequest(BaseModel):
    url: HttpUrl


class PodcastProcessRequest(BaseModel):
    rss_feed_url: HttpUrl


# Initialize Router with prefix
admin_router = APIRouter(prefix="/admin")

# Role Checker for Admin
admin_role_checker = RoleChecker("admin")


@admin_router.get("/", dependencies=[Depends(admin_role_checker)])
def get_admin_dashboard() -> Dict[str, str]:
    """Returns the admin dashboard message."""
    return {"message": "Admin Dashboard"}


@admin_router.get("/dashboard", dependencies=[Depends(admin_role_checker)])
def get_admin_dashboard_legacy() -> Dict[str, str]:
    """Legacy endpoint for admin dashboard."""
    return {"message": "Admin Dashboard"}


@admin_router.get("/content", dependencies=[Depends(admin_role_checker)])
async def admin_list_content(
        chroma_handler: ChromaDBHandler = Depends(get_chroma_handler)
) -> Dict[str, Any]:
    """
    Admin endpoint to list all content.

    Args:
        chroma_handler (ChromaDBHandler): ChromaDB service for content retrieval

    Returns:
        Dict[str, Any]: All content items
    """
    try:
        content = chroma_handler.list_content(content_type="all")
        return {"content": content, "total_count": len(content)}
    except Exception as e:
        logger.error(f"Error retrieving admin content: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving content")


@admin_router.post("/extract-blog", dependencies=[Depends(admin_role_checker)])
def extract_blog(
        request: BlogExtractRequest,
        chroma_handler: ChromaDBHandler = Depends(get_chroma_handler),
        web_scraper: WebScraper = Depends(get_web_scraper)
) -> Dict[str, str]:
    """
    Extracts blog content from a URL and stores it in ChromaDB.
    """
    try:
        content_dict = web_scraper.extract_blog_content(str(request.url))

        if not content_dict:
            raise ContentExtractionError("Failed to extract blog content")

        # Extract the actual text content from the first available strategy
        text_content = None
        metadata = {}
        for strategy in ['article', 'semantic', 'soup']:
            if strategy in content_dict and 'content' in content_dict[strategy]:
                text_content = content_dict[strategy]['content']
                # Preserve metadata
                metadata = {k: v for k, v in content_dict[strategy].items() if k != 'content'}
                break

        if not text_content:
            raise ContentExtractionError("No content found in any extraction strategy")

        # Pass the text content to store_text instead of the whole dictionary
        chroma_handler.store_text(text_content, "blog", str(request.url), metadata=metadata)

        logger.info(f"Successfully extracted blog from {request.url}")
        return {"message": "Blog extracted and stored in ChromaDB"}

    except ContentExtractionError as e:
        logger.warning(f"Content extraction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error extracting blog: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@admin_router.post("/process-podcast", dependencies=[Depends(admin_role_checker)])
def process_podcast_endpoint(
        request: PodcastProcessRequest,
        chroma_handler: ChromaDBHandler = Depends(get_chroma_handler),
        transcriber: PodcastTranscriber = Depends(get_transcriber),
        web_scraper: WebScraper = Depends(get_web_scraper)
) -> Dict[str, str]:
    """
    Processes a podcast RSS feed by downloading and transcribing.

    Args:
        request (PodcastProcessRequest): Request containing RSS feed URL
        chroma_handler (ChromaDBHandler): ChromaDB service for storing content
        transcriber (PodcastTranscriber): Podcast transcription service
        web_scraper (WebScraper): Web scraping service

    Returns:
        Dict[str, str]: Confirmation message
    """
    try:
        # Parse RSS feed
        feed_data = web_scraper.parse_rss_feed(str(request.rss_feed_url))

        # Check for errors in feed parsing
        if 'error' in feed_data:
            raise PodcastProcessingError(f"Error parsing RSS feed: {feed_data['error']}")

        # Check if we have entries
        if not feed_data.get('entries') or len(feed_data['entries']) == 0:
            raise PodcastProcessingError("No entries found in RSS feed")

        # Get the latest episode (first entry)
        episode = feed_data['entries'][0]

        # Get podcast name from feed title
        podcast_name = feed_data.get('title', 'Unknown Podcast')

        # Get episode title
        episode_title = episode.get('title', 'Untitled Episode')

        # Find audio URL in media list
        audio_url = None
        for media in episode.get('media', []):
            # Look for audio types
            if media.get('type', '').startswith('audio/'):
                audio_url = media.get('url')
                break

        # If no specific audio type is found, use the first media URL
        if not audio_url and episode.get('media'):
            audio_url = episode['media'][0].get('url')

        if not audio_url:
            raise PodcastProcessingError("No audio URL found in RSS feed")

        # Download and transcribe
        audio_path = transcriber.download_audio(audio_url)
        if not audio_path:
            raise PodcastProcessingError(f"Failed to download audio from URL: {audio_url}")

        transcription = transcriber.transcribe_audio(audio_path)
        if not transcription:
            raise PodcastProcessingError("Failed to transcribe audio")

        # Store in ChromaDB
        chroma_handler.store_podcast_transcription(
            transcription,
            episode_title,
            podcast_name,
            str(request.rss_feed_url),
            audio_url
        )

        logger.info(f"Successfully processed podcast from {request.rss_feed_url}")
        return {"message": "Podcast processed and stored successfully"}

    except PodcastProcessingError as e:
        logger.warning(f"Podcast processing failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error processing podcast: {str(e)}")
        traceback.print_exc()  # This will print the full stack trace
        raise HTTPException(status_code=500, detail="Internal Server Error")


@admin_router.get("/list-podcasts", dependencies=[Depends(admin_role_checker)])
async def list_podcasts(
        chroma_handler: ChromaDBHandler = Depends(get_chroma_handler)
) -> Dict[str, Any]:
    """
    Fetches stored podcast metadata from ChromaDB.

    Args:
        chroma_handler (ChromaDBHandler): ChromaDB service for retrieving podcasts

    Returns:
        Dict[str, Any]: List of podcasts or empty list
    """
    try:
        podcasts = chroma_handler.get_podcasts()
        logger.info(f"Retrieved {len(podcasts)} podcasts")
        return {"podcasts": podcasts or []}
    except Exception as e:
        logger.error(f"Error fetching podcasts: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving podcasts")


@admin_router.delete("/delete-content", dependencies=[Depends(admin_role_checker)])
async def delete_content(
        content_id: str,
        chroma_handler: ChromaDBHandler = Depends(get_chroma_handler)
) -> Dict[str, str]:
    """
    Deletes specific stored content (blog or podcast) from ChromaDB.

    Args:
        content_id (str): Unique identifier of content to delete
        chroma_handler (ChromaDBHandler): ChromaDB service for deleting content

    Returns:
        Dict[str, str]: Confirmation message
    """
    try:
        chroma_handler.delete_content(content_id)
        logger.info(f"Successfully deleted content with ID: {content_id}")
        return {"message": "Content deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting content: {str(e)}")
        raise HTTPException(status_code=500, detail="Error deleting content")


# Add this endpoint at the end of the file, after your delete_content endpoint
@admin_router.get("/db-diagnostic", dependencies=[Depends(admin_role_checker)])
async def database_diagnostic(
        request: Request,
        chroma_handler: ChromaDBHandler = Depends(get_chroma_handler)
) -> Dict[str, Any]:
    """
    Diagnostic endpoint for ChromaDB connection and status.
    """
    try:
        # Get full ChromaDB diagnostic info
        db_status = chroma_handler.debug_chroma_status()

        # Test adding a simple document to verify write functionality
        test_result = {"success": False, "message": ""}

        try:
            # Only run this test if explicitly requested (to avoid creating test data)
            run_test = request.query_params.get("run_test", "").lower() == "true"
            if run_test:
                test_id = f"test_{uuid.uuid4()}"
                test_collection = chroma_handler.client.get_or_create_collection("test_collection")
                test_collection.add(
                    documents=["Test document to verify ChromaDB is working properly"],
                    metadatas=[{"test": True, "timestamp": str(datetime.datetime.now())}],
                    ids=[test_id]
                )
                # Clean up test data
                test_collection.delete(ids=[test_id])
                test_result = {"success": True, "message": "Successfully wrote and deleted test document"}
        except Exception as e:
            test_result = {"success": False, "message": str(e)}

        return {
            "db_status": db_status,
            "write_test": test_result,
            "collections": [c.name for c in chroma_handler.client.list_collections()],
            "connection_status": "Connected" if hasattr(chroma_handler,
                                                        "is_connected") and chroma_handler.is_connected else "Disconnected"
        }
    except Exception as e:
        logger.error(f"Error in database diagnostic: {str(e)}")
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }