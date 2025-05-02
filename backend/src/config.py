import os
import logging
from dotenv import load_dotenv
from typing import Optional
import pathlib

# Load environment variables
load_dotenv()


class Config:
    """
    Centralized configuration management for the application.
    Loads environment variables with fallback defaults and validation.
    """
    # Application Settings
    API_URL: str = os.getenv("API_URL", "http://localhost:8000")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

    # Database Configuration
    # Default to a 'chroma_db' directory within the project root
    PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.absolute()
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", str(PROJECT_ROOT / "chroma_db"))

    # Ensure the path is absolute
    if not os.path.isabs(CHROMA_DB_PATH):
        CHROMA_DB_PATH = str(PROJECT_ROOT / CHROMA_DB_PATH)

    # API Keys and Credentials
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    SECRET_KEY: str = os.getenv("SECRET_KEY", os.urandom(32).hex())
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")

    # Embedding Model Configuration
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Logging Configuration
    LOG_LEVEL: int = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper())

    def __init__(self):
        """
        Initialize configuration and perform validation.
        Logs warnings for missing critical configurations.
        """
        self._validate_config()
        self._setup_logging()

    def _validate_config(self):
        """
        Validate critical configuration parameters.
        Raise warnings or errors for missing essential configurations.
        """
        warnings = []

        if not self.GROQ_API_KEY:
            warnings.append("GROQ_API_KEY is not set. Some AI features may not work.")

        if not self.SECRET_KEY:
            warnings.append("SECRET_KEY is using a randomly generated value. This is not recommended for production.")

        # Validate ChromaDB path
        if not os.path.exists(os.path.dirname(self.CHROMA_DB_PATH)):
            warnings.append(
                f"Parent directory for CHROMA_DB_PATH does not exist: {os.path.dirname(self.CHROMA_DB_PATH)}")

        if warnings:
            for warning in warnings:
                logging.warning(f"⚠️ Configuration Warning: {warning}")

    def _setup_logging(self):
        """
        Configure application-wide logging based on configuration.
        """
        logging.basicConfig(
            level=self.LOG_LEVEL,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    @classmethod
    def get_database_path(cls) -> str:
        """
        Retrieve the database path, ensuring it exists.

        Returns:
            str: Path to the database directory
        """
        db_path = cls.CHROMA_DB_PATH
        os.makedirs(db_path, exist_ok=True)
        logging.info(f"Using ChromaDB path: {db_path}")
        return db_path

    def to_dict(self) -> dict:
        """
        Convert configuration to a dictionary for easy inspection.

        Returns:
            dict: Configuration key-value pairs
        """
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


# Create a singleton configuration instance
config = Config()