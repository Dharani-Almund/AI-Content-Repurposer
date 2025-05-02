import os
import logging
from typing import Dict, List, Any, Optional
import uuid
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from datetime import datetime

from backend.src.config import config

logger = logging.getLogger(__name__)


class ChromaDBHandler:
    """Handles interactions with ChromaDB for vector storage and retrieval."""

    def __init__(self):
        """Initialize the ChromaDB client with proper configuration and connection."""
        self.db_path = config.get_database_path()
        logger.info(f"Initializing ChromaDB with path: {self.db_path}")

        try:
            # Configure the embedding function based on config
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=config.EMBEDDING_MODEL
            )

            # Initialize ChromaDB with persistent storage
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # Create/get collections for different content types
            self._initialize_collections()

            # Validate connection by checking collections
            collections = self.client.list_collections()
            logger.info(f"ChromaDB connected successfully. Found {len(collections)} collections.")

            self.is_connected = True
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            self.is_connected = False
            # Don't raise here, as we want the application to start even if DB connection fails initially
            # We'll handle connection errors in specific methods

    def _initialize_collections(self):
        """Initialize or get the necessary collections."""
        try:
            # Blog collection
            self.blog_collection = self.client.get_or_create_collection(
                name="blog_collection",
                embedding_function=self.embedding_function,
                metadata={"description": "Collection for blog content"}
            )

            # Podcast collection
            self.podcast_collection = self.client.get_or_create_collection(
                name="podcast_collection",
                embedding_function=self.embedding_function,
                metadata={"description": "Collection for podcast transcriptions"}
            )

            logger.info("ChromaDB collections initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing collections: {str(e)}")
            raise

    def check_connection(self):
        """Ensures database connection is active, reconnects if necessary."""
        if not self.is_connected:
            logger.info("Database connection was lost. Attempting to reconnect...")
            try:
                self.__init__()  # Reinitialize the connection
                if not self.is_connected:
                    raise ConnectionError("Failed to reconnect to ChromaDB")
            except Exception as e:
                logger.error(f"Reconnection failed: {str(e)}")
                raise ConnectionError(f"Could not connect to ChromaDB: {str(e)}")
        return self.is_connected

    def store_text(self, text: str, content_type: str, source_url: str, metadata: Optional[Dict] = None):
        """
        Store text content in ChromaDB with metadata.

        Args:
            text (str): The text content to store
            content_type (str): Type of content ('blog', 'podcast', etc.)
            source_url (str): URL source of the content
            metadata (Dict, optional): Additional metadata
        """
        self.check_connection()

        if not metadata:
            metadata = {}

        # Add required metadata fields
        metadata.update({
            "content_type": content_type,
            "source_url": source_url,
            "timestamp": str(datetime.now())
        })

        # Generate a unique ID based on content source
        content_id = f"{content_type}_{uuid.uuid4()}"

        try:
            if content_type == "blog":
                collection = self.blog_collection
            else:
                # Default to blog collection for other types
                collection = self.blog_collection

            # Store in ChromaDB
            collection.add(
                documents=[text],
                metadatas=[metadata],
                ids=[content_id]
            )

            # Log success with collection stats
            logger.info(
                f"Content stored successfully. Collection '{collection.name}' now has {collection.count()} items.")
            return content_id
        except Exception as e:
            logger.error(f"Error storing content in ChromaDB: {str(e)}")
            raise

    def store_podcast_transcription(self, transcription, episode_title: str,
                                    podcast_name: str, rss_feed_url: str, audio_url: str):
        """
        Store podcast transcription in ChromaDB.

        Args:
            transcription (str or dict): Podcast transcription text or dict with 'text' key
            episode_title (str): Title of the podcast episode
            podcast_name (str): Name of the podcast
            rss_feed_url (str): URL of the RSS feed
            audio_url (str): URL of the audio file
        """
        self.check_connection()

        # Extract text if transcription is a dictionary
        if isinstance(transcription, dict) and 'text' in transcription:
            document_text = transcription['text']
            # Extract additional metadata if available
            additional_metadata = {
                "language": transcription.get('language', 'en'),
                "audio_duration": transcription.get('audio_duration', 0),
                "transcript_path": transcription.get('transcript_path', '')
            }
        else:
            # Use transcription directly if it's a string
            document_text = transcription
            additional_metadata = {}

        # Create metadata
        metadata = {
            "content_type": "podcast",
            "episode_title": episode_title,
            "podcast_name": podcast_name,
            "rss_feed_url": rss_feed_url,
            "audio_url": audio_url,
            "timestamp": str(datetime.now()),
            **additional_metadata  # Include any additional metadata from the transcription dict
        }

        # Generate a unique ID
        content_id = f"podcast_{uuid.uuid4()}"

        try:
            # Store in the podcast collection
            self.podcast_collection.add(
                documents=[document_text],  # Use the extracted text string
                metadatas=[metadata],
                ids=[content_id]
            )

            logger.info(f"Podcast stored successfully. Collection now has {self.podcast_collection.count()} items.")
            return content_id
        except Exception as e:
            logger.error(f"Error storing podcast in ChromaDB: {str(e)}")
            raise

    def list_content(self, content_type="all"):
        """
        List content from ChromaDB based on content type.

        Args:
            content_type (str): Type of content to list ("all", "blog", "podcast")

        Returns:
            List[Dict]: Content items with metadata
        """
        self.check_connection()

        try:
            if content_type == "all":
                # Get content from all collections
                all_content = []

                # Get blog content
                blog_results = self._query_collection(self.blog_collection)
                if blog_results:
                    all_content.extend(blog_results)

                # Get podcast content
                podcast_results = self._query_collection(self.podcast_collection)
                if podcast_results:
                    all_content.extend(podcast_results)

                return all_content

            elif content_type == "blog":
                return self._query_collection(self.blog_collection)

            elif content_type == "podcast":
                return self._query_collection(self.podcast_collection)

            else:
                logger.warning(f"Unknown content type: {content_type}")
                return []

        except Exception as e:
            logger.error(f"Error listing content: {str(e)}")
            return []

    def _query_collection(self, collection):
        """Helper method to query a collection and format results."""
        if collection.count() == 0:
            logger.info(f"Collection '{collection.name}' is empty")
            return []

        try:
            # Query all documents in the collection
            # Using empty query returns everything
            result = collection.query(
                query_texts=[""],  # Empty query to match all documents
                n_results=1000,  # Limit results (adjust as needed)
                include=["metadatas", "documents", "distances"]
            )

            # Format results
            content_items = []
            for i in range(len(result["ids"][0])):
                item_id = result["ids"][0][i]
                metadata = result["metadatas"][0][i].copy() if i < len(result["metadatas"][0]) else {}
                document = result["documents"][0][i] if i < len(result["documents"][0]) else ""

                # Add ID and document content to the metadata
                metadata["id"] = item_id
                metadata["content"] = document
                metadata["collection"] = collection.name

                content_items.append(metadata)

            return content_items
        except Exception as e:
            logger.error(f"Error querying collection {collection.name}: {str(e)}")
            return []

    def get_podcasts(self):
        """Get all podcast entries."""
        return self.list_content(content_type="podcast")

    def delete_content(self, content_id: str):
        """Delete content by ID."""
        self.check_connection()

        # Try to delete from each collection
        try:
            self.blog_collection.delete(ids=[content_id])
            logger.info(f"Content {content_id} deleted from blog collection")
            return
        except Exception as e:
            # It's normal if content is not in this collection
            pass

        try:
            self.podcast_collection.delete(ids=[content_id])
            logger.info(f"Content {content_id} deleted from podcast collection")
            return
        except Exception as e:
            # If we get here, the content wasn't found in either collection
            logger.warning(f"Content with ID {content_id} not found in any collection")
            raise ValueError(f"Content with ID {content_id} not found")

    def debug_chroma_status(self):
        """Returns diagnostic information about ChromaDB collections and counts."""
        self.check_connection()  # Changed from ensure_connection to check_connection

        try:
            # Get all collection names
            collections = self.client.list_collections()
            collection_info = {}

            for collection in collections:
                try:
                    # Get the actual collection
                    coll = self.client.get_collection(collection.name)
                    # Get item count
                    count = coll.count()
                    collection_info[collection.name] = {
                        "count": count,
                        "metadata": collection.metadata
                    }
                except Exception as e:
                    collection_info[collection.name] = {"error": str(e)}

            return {
                "total_collections": len(collections),
                "collections": collection_info,
                "db_path": self.db_path,
                "is_connected": self.is_connected
            }
        except Exception as e:
            return {"error": str(e), "is_connected": False}

    def get_context_by_type(self, query: str, content_type: str = "all", num_results: int = 5):
        """
        Search for contextually relevant documents based on a query and content type.

        Args:
            query (str): The search query text
            content_type (str): Type of content to search ("all", "blog", "podcast")
            num_results (int): Maximum number of results to return

        Returns:
            List[Dict]: Relevant content items with metadata and similarity scores
        """
        self.check_connection()

        results = []

        try:
            # Search in appropriate collections based on content_type
            if content_type == "all" or content_type == "blog":
                blog_results = self._search_collection(
                    collection=self.blog_collection,
                    query=query,
                    num_results=num_results
                )
                results.extend(blog_results)

            if content_type == "all" or content_type == "podcast":
                podcast_results = self._search_collection(
                    collection=self.podcast_collection,
                    query=query,
                    num_results=num_results
                )
                results.extend(podcast_results)

            # Sort results by relevance (distance) if we have results from multiple collections
            if len(results) > num_results:
                results = sorted(results, key=lambda x: x.get("distance", 1.0))[:num_results]

            return results

        except Exception as e:
            logger.error(f"Error searching context by type: {str(e)}")
            return []

    def _search_collection(self, collection, query: str, num_results: int = 5):
        """Helper method to search a specific collection and format results."""
        if collection.count() == 0:
            logger.info(f"Collection '{collection.name}' is empty")
            return []

        try:
            # Query the collection with the search query
            result = collection.query(
                query_texts=[query],
                n_results=num_results,
                include=["metadatas", "documents", "distances"]
            )

            # Format results
            content_items = []
            if result["ids"] and len(result["ids"][0]) > 0:
                for i in range(len(result["ids"][0])):
                    item_id = result["ids"][0][i]
                    metadata = result["metadatas"][0][i].copy() if i < len(result["metadatas"][0]) else {}
                    document = result["documents"][0][i] if i < len(result["documents"][0]) else ""
                    distance = result["distances"][0][i] if "distances" in result and i < len(
                        result["distances"][0]) else 1.0

                    # Add ID, document content, and distance to the metadata
                    metadata["id"] = item_id
                    metadata["content"] = document
                    metadata["collection"] = collection.name
                    metadata["distance"] = distance  # Lower distance means better match

                    content_items.append(metadata)

            return content_items
        except Exception as e:
            logger.error(f"Error searching collection {collection.name}: {str(e)}")
            return []

    def get_all_context(self, query: str, num_results: int = 5):
        """
        Search for contextually relevant documents across all collections.
        This is a wrapper around get_context_by_type with content_type="all".

        Args:
            query (str): The search query text
            num_results (int): Maximum number of results to return

        Returns:
            List[Dict]: Relevant content items with metadata and similarity scores
        """
        return self.get_context_by_type(query=query, content_type="all", num_results=num_results)

    def get_content_by_id(self, content_id: str):
        """
        Get content by its ID from any collection.

        Args:
            content_id (str): The unique ID of the content

        Returns:
            Dict or None: Content details if found, None otherwise
        """
        self.check_connection()

        # Try each collection
        collections = [self.blog_collection, self.podcast_collection]

        for collection in collections:
            try:
                result = collection.get(ids=[content_id], include=["metadatas", "documents"])

                if result and result["ids"] and len(result["ids"]) > 0:
                    metadata = result["metadatas"][0] if "metadatas" in result and result["metadatas"] else {}
                    document = result["documents"][0] if "documents" in result and result["documents"] else ""

                    # Add ID and content to metadata
                    metadata["id"] = content_id
                    metadata["content"] = document
                    metadata["collection"] = collection.name

                    return metadata
            except Exception as e:
                logger.debug(f"Content {content_id} not found in {collection.name}: {str(e)}")

        return None