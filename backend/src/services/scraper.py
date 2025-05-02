import os
import re
import json
import requests
import feedparser
import logging
import traceback
from typing import Dict, List, Optional, Union, Tuple
from urllib.parse import quote, urlparse, urlunparse
from urllib.parse import urlparse, unquote, quote, urlunparse
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import json
from bs4 import BeautifulSoup
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('web_scraper.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class WebScraper:
    """
    Advanced web scraping utility with multiple extraction strategies,
    robust error handling, and comprehensive content extraction.
    """

    def __init__(self, config_path: str = 'scraper_config.yaml'):
        """
        Initialize the WebScraper with configuration.

        Args:
            config_path (str): Path to the configuration file
        """
        self.config_path = config_path
        self.config = self.load_config()

        # Default configuration if none exists
        if not self.config:
            self.config = {
                'user_agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                'timeout': 20,
                'max_retries': 3,
                'default_strategies': ['article', 'semantic', 'soup'],
                'max_content_length': 10000,
                'allowed_domains': [],  # Empty list means all domains are allowed
                'rate_limiting': {
                    'enabled': False,
                    'requests_per_minute': 10
                }
            }
            self.save_config()

    def check_configuration(self) -> Dict[str, bool]:
        """
        Check if the scraper configuration is valid and all dependencies are available.

        Returns:
            Dict[str, bool]: Status of configuration checks
        """
        results = {
            "config_file_exists": os.path.exists(self.config_path),
            "requests_available": self._check_dependency("requests"),
            "bs4_available": self._check_dependency("bs4"),
            "feedparser_available": self._check_dependency("feedparser"),
            "yaml_available": self._check_dependency("yaml")
        }

        # Verify configuration values
        if results["config_file_exists"]:
            results["valid_config"] = self._validate_config()
        else:
            # Create default configuration if it doesn't exist
            self.save_config()
            results["config_created"] = True
            results["valid_config"] = True

        logger.info(f"WebScraper configuration check: {results}")
        return results

    def _check_dependency(self, module_name: str) -> bool:
        """
        Check if a dependency is available.

        Args:
            module_name (str): Name of the module to check

        Returns:
            bool: Whether the dependency is available
        """
        try:
            if module_name == "requests":
                import requests
            elif module_name == "bs4":
                from bs4 import BeautifulSoup
            elif module_name == "feedparser":
                import feedparser
            elif module_name == "yaml":
                import yaml
            return True
        except ImportError:
            logger.error(f"Dependency {module_name} is not available")
            return False

    def _validate_config(self) -> bool:
        """
        Validate the configuration.

        Returns:
            bool: Whether the configuration is valid
        """
        required_keys = ['user_agent', 'timeout', 'max_retries', 'default_strategies', 'max_content_length']

        # Check if all required keys exist
        for key in required_keys:
            if key not in self.config:
                logger.error(f"Missing required configuration key: {key}")
                return False

        # Validate types
        if not isinstance(self.config.get('timeout', 0), (int, float)):
            logger.error("Timeout must be a number")
            return False

        if not isinstance(self.config.get('max_retries', 0), int):
            logger.error("Max retries must be an integer")
            return False

        if not isinstance(self.config.get('default_strategies', []), list):
            logger.error("Default strategies must be a list")
            return False

        return True

    def save_config(self) -> None:
        """Save configuration to YAML file."""
        with open(self.config_path, 'w') as f:
            yaml.safe_dump(self.config, f)

    def load_config(self) -> Dict:
        """
        Load configuration from YAML file.

        Returns:
            Dict: Configuration dictionary
        """
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Configuration file {self.config_path} not found, using defaults")
            return {}

    @staticmethod
    def validate_url(url: str) -> bool:
        """
        Comprehensive URL validation.

        Args:
            url (str): URL to validate

        Returns:
            bool: Whether URL is valid
        """
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    @staticmethod
    def clean_url(url: str, encode: bool = True) -> str:
        """
        Enhanced URL cleaning and normalization with proper handling of emojis.

        Args:
            url (str): URL to clean
            encode (bool): Whether to URL encode components

        Returns:
            str: Cleaned and normalized URL
        """
        try:
            # First decode the URL to handle any double-encoding
            parsed = urlparse(url)

            # Handle path with special characters and emojis
            path = unquote(parsed.path)  # First decode any already encoded parts

            scheme = parsed.scheme or 'https'
            netloc = parsed.netloc or parsed.path

            # If netloc is in path, adjust accordingly
            if not parsed.netloc and '/' in parsed.path:
                parts = parsed.path.split('/', 1)
                netloc = parts[0]
                path = '/' + parts[1] if len(parts) > 1 else ''

            # Properly encode the path, preserving slashes
            if encode:
                path = quote(path, safe='/')

            # Remove tracking parameters
            query_parts = []
            if parsed.query:
                for param in parsed.query.split('&'):
                    if param and not any(track in param.lower() for track in ['utm_', 'ref=', 'click_id']):
                        query_parts.append(param)

            query = '&'.join(query_parts)

            # Reconstruct the URL
            cleaned_url = urlunparse((scheme, netloc, path, parsed.params, query, parsed.fragment))
            return cleaned_url
        except Exception as e:
            logger.error(f"URL cleaning error: {e}")
            return url


    @staticmethod
    def extract_metadata_from_html(soup: BeautifulSoup) -> Dict[str, str]:
        """
        Extract metadata from HTML using various meta tags.

        Args:
            soup (BeautifulSoup): Parsed HTML soup

        Returns:
            Dict containing extracted metadata
        """
        metadata = {}

        # Extract title
        title = soup.find('title')
        metadata['title'] = title.get_text(strip=True) if title else ''

        # Extract authors from common meta tags and structures
        authors_meta = [
            soup.find('meta', property='article:author'),
            soup.find('meta', attrs={'name': 'author'}),
            soup.find('span', class_=re.compile(r'author|byline', re.I)),
            soup.find('div', class_=re.compile(r'author|byline', re.I))
        ]

        for author_tag in authors_meta:
            if author_tag:
                metadata['author'] = author_tag.get('content', author_tag.get_text(strip=True))
                break

        # Extract publication date from various possible meta tags
        date_tags = [
            soup.find('meta', property='article:published_time'),
            soup.find('meta', attrs={'name': 'date'}),
            soup.find('time')
        ]

        for date_tag in date_tags:
            if date_tag:
                metadata['publish_date'] = date_tag.get('datetime',
                                                        date_tag.get('content', date_tag.get_text(strip=True)))
                break

        return metadata

    def extract_blog_content(
            self,
            url: str,
            max_length: Optional[int] = None,
            strategies: Optional[List[str]] = None
    ) -> Dict[str, Union[str, List[str]]]:
        """
        Multi-strategy content extraction with metadata.

        Args:
            url (str): URL to extract content from
            max_length (int, optional): Maximum content length. If None, uses config value.
            strategies (List[str], optional): Extraction strategies. If None, uses config value.

        Returns:
            Dict containing extracted content and metadata
        """
        # Use config values if parameters not provided
        max_length = max_length or self.config.get('max_content_length', 10000)
        strategies = strategies or self.config.get('default_strategies', ['article', 'semantic', 'soup'])

        try:
            # Validate and clean the URL
            if not self.validate_url(url):
                return {'error': 'Invalid URL format', 'url': url}

            url = self.clean_url(url)

            headers = {
                "User-Agent": self.config.get('user_agent',
                                              "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"),
                "Accept-Language": "en-US,en;q=0.9"
            }

            # Add retry logic for transient errors
            max_retries = self.config.get('max_retries', 3)
            retry_strategy = Retry(
                total=max_retries,
                backoff_factor=0.5,
                status_forcelist=[429, 500, 502, 503, 504],
            )

            with requests.Session() as session:
                adapter = HTTPAdapter(max_retries=retry_strategy)
                session.mount("https://", adapter)
                session.mount("http://", adapter)

                response = session.get(url, headers=headers, timeout=self.config.get('timeout', 20))
                response.raise_for_status()

                soup = BeautifulSoup(response.text, 'lxml')
                content_results = {}

                # Extract metadata first
                metadata = self.extract_metadata_from_html(soup)

                if 'article' in strategies:
                    article_tags = soup.find_all('article')
                    if article_tags:
                        paragraphs = []
                        for article in article_tags:
                            paras = article.find_all(['p', 'h1', 'h2', 'h3'])
                            paragraphs.extend([p.get_text().strip() for p in paras if p.get_text().strip()])

                        content_results['article'] = {
                            'content': '\n'.join(paragraphs)[:max_length],
                            'tags': [tag.name for tag in article_tags[0].parents if tag.name],
                            **metadata
                        }

                if 'semantic' in strategies:
                    # Try semantic HTML5 tags and common blog content containers
                    semantic_tags = soup.find_all(['main', 'div'],
                                                  class_=re.compile(r'content|post|article|entry|blog', re.I))
                    if semantic_tags:
                        semantic_content = []
                        for tag in semantic_tags:
                            paras = tag.find_all(['p', 'h1', 'h2', 'h3'])
                            semantic_content.extend([p.get_text().strip() for p in paras if p.get_text().strip()])

                        content_results['semantic'] = {
                            'content': '\n'.join(semantic_content)[:max_length],
                            **metadata
                        }

                if 'soup' in strategies:
                    paragraphs = soup.find_all(['p', 'div'])
                    filtered_text = [
                        p.get_text().strip()
                        for p in paragraphs
                        if p.get_text().strip() and len(p.get_text()) > 50
                    ]

                    content_results['soup'] = {
                        'content': '\n'.join(filtered_text)[:max_length],
                        'paragraphs_count': len(filtered_text),
                        **metadata
                    }

                # Validate if we extracted any content
                if not content_results:
                    logger.warning(f"No content found in any extraction strategy for {url}")
                    return {'error': 'No content found in any extraction strategy', 'url': url}

                return content_results

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if hasattr(e, 'response') else 'unknown'
            logger.error(f"HTTP error ({status_code}) for {url}: {e}")
            return {'error': f"HTTP error: {status_code}", 'url': url, 'details': str(e)}
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error for {url}: {e}")
            return {'error': 'Connection error', 'url': url, 'details': str(e)}
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout error for {url}: {e}")
            return {'error': 'Timeout error', 'url': url, 'details': str(e)}
        except Exception as e:
            logger.error(f"Content extraction error for {url}: {e}")
            return {'error': str(e), 'url': url, 'traceback': traceback.format_exc()}

    def parse_rss_feed(self, feed_url: str) -> Dict[str, Union[str, List[Dict]]]:
        """
        Parse RSS/Atom feed with rich metadata extraction and URL validation.

        Args:
            feed_url (str): URL of the feed to parse

        Returns:
            Dict containing feed information and entries
        """
        try:
            feed_url = self.clean_url(feed_url)
            feed = feedparser.parse(feed_url)

            # Extract feed metadata
            feed_info = {
                'title': feed.feed.get('title', ''),
                'description': feed.feed.get('description', feed.feed.get('subtitle', '')),
                'link': self._validate_and_clean_url(feed.feed.get('link', '')),
                'updated': feed.feed.get('updated', ''),
                'entries_count': len(feed.entries),
                'entries': []
            }

            # Process entries
            for entry in feed.entries:
                # Clean and validate the link URL
                link = self._validate_and_clean_url(entry.get('link', ''))

                # Extract enclosure/media URLs (commonly used for podcast audio files)
                media_urls = []
                for enclosure in entry.get('enclosures', []):
                    media_url = self._validate_and_clean_url(enclosure.get('href', ''))
                    if media_url:
                        media_urls.append({
                            'url': media_url,
                            'type': enclosure.get('type', ''),
                            'length': enclosure.get('length', '')
                        })

                # Look for media content in other fields if no enclosures found
                if not media_urls:
                    for media in entry.get('media_content', []):
                        media_url = self._validate_and_clean_url(media.get('url', ''))
                        if media_url:
                            media_urls.append({
                                'url': media_url,
                                'type': media.get('type', ''),
                                'length': media.get('fileSize', '')
                            })

                entry_data = {
                    'title': entry.get('title', ''),
                    'link': link,
                    'published': entry.get('published', entry.get('updated', '')),
                    'authors': [author.get('name', '') for author in entry.get('authors', [])],
                    'summary': entry.get('summary', ''),
                    'content': entry.get('content', [{}])[0].get('value', '') if entry.get('content') else '',
                    'tags': [tag.get('term', '') for tag in entry.get('tags', [])],
                    'media': media_urls  # Add media URLs to entry data
                }
                feed_info['entries'].append(entry_data)

            return feed_info

        except Exception as e:
            logger.error(f"RSS feed parsing error for {feed_url}: {e}")
            return {'error': str(e), 'traceback': traceback.format_exc()}

    def _validate_and_clean_url(self, url: str) -> str:
        """
        Validate and clean URL, return empty string if invalid.

        Args:
            url (str): URL to validate and clean

        Returns:
            str: Cleaned URL or empty string if invalid
        """
        if not url:
            return ''

        # Check if URL has a scheme, add https if missing
        if not urlparse(url).scheme:
            url = f"https://{url}"

        # Validate the URL
        if not self.validate_url(url):
            logger.warning(f"Invalid URL found: {url}")
            return ''

        return self.clean_url(url)


# Standalone helper functions that were in the original code
def save_config(config_dict: Dict, filename: str = 'scraper_config.yaml'):
    """Save configuration to YAML file."""
    with open(filename, 'w') as f:
        yaml.safe_dump(config_dict, f)


def load_config(filename: str = 'scraper_config.yaml') -> Dict:
    """Load configuration from YAML file."""
    try:
        with open(filename, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}

# Example usage
# if __name__ == "__main__":
#     # Example of how to use the WebScraper
#     scraper = WebScraper()
#     test_url = "https://example.com/blog-post"
#     content = scraper.extract_blog_content(test_url)
#     print(json.dumps(content, indent=2))
#
#     rss_url = "https://example.com/podcast-feed"
#     feed_info = scraper.parse_rss_feed(rss_url)
#     print(json.dumps(feed_info, indent=2))