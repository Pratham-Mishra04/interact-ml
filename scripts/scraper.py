import subprocess
from parsel import Selector
import requests
from typing import Dict, Any, Optional
from dataclasses import dataclass
from functools import lru_cache
import json
from datetime import datetime
from urllib.parse import urljoin

def logger(level , title, description, path):
    subprocess.run(['python3', 'api_logger.py', level, title, description, path], cwd='utils')

@dataclass
class PlaceholderConfig:
    value: str
    selector_type: str

@dataclass
class ScraperConfig:
    """Data class to hold scraper configuration"""
    url: str
    data: Dict[str, PlaceholderConfig]

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ScraperConfig':
        """Create ScraperConfig from dictionary"""
        data = {
            key: PlaceholderConfig(
                value=item['value'],
                selector_type=item['selector_type']
            )
            for key, item in config_dict.get('data', {}).items()
        }

        print(data)
        return cls(
            url=config_dict['url'],
            data=data
        )

class WebPageFetcher:
    """Responsible for fetching web pages"""
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    @lru_cache(maxsize=100)
    def fetch(self, url: str) -> requests.Response:
        """Fetch webpage content with caching"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            logger("error",f"Error fetching {url}: {str(e)}", "", "scraper.py")
            raise

class ParselScraper:
    """Main scraper class using Parsel"""
    def __init__(self):
        self.fetcher = WebPageFetcher()
    
    def _extract_with_selector(self, selector: Selector, 
                             placeholder_config: PlaceholderConfig) -> Optional[str]:
        """Extract content using specified selector type"""
        try:
            if placeholder_config.selector_type == 'css':
                result = selector.css(f"{placeholder_config.value}::text").get()
            elif placeholder_config.selector_type == 'xpath':
                # Append /text() to get text content for XPath selectors
                xpath_query = placeholder_config.value
                if not xpath_query.endswith('/text()'):
                    xpath_query = f"{xpath_query}/text()"
                result = selector.xpath(xpath_query).get()
            else:
                raise ValueError(f"Unsupported selector type: {placeholder_config.selector_type}")
            
            return result.strip() if result else None
            
        except Exception as e:
            logger("error",f"Error extracting content using selector: {str(e)}", "", "scraper.py")
            return None

    def _process_value(self, key: str, value: str, base_url: str) -> str:
        """Process scraped value based on key type"""
        if 'image' in key.lower() and value:  # Check if key contains 'image'
            return urljoin(base_url, value)
        return value

    def _extract_structured_data(self, selector: Selector) -> Dict[str, Any]:
        """Extract JSON-LD structured data if available"""
        try:
            script = selector.css('script[type="application/ld+json"]::text').get()
            if script:
                return json.loads(script)
        except Exception as e:
            logger("error",f"Error extracting structured data: {str(e)}", "", "scraper.py")
        return {}

    def scrape(self, config: ScraperConfig) -> Dict[str, Any]:
        """Main scraping method"""
        try:
            # Fetch content
            response = self.fetcher.fetch(config.url)
            selector = Selector(text=response.text)
            
            # Try to get structured data first
            structured_data = self._extract_structured_data(selector)
            
            result = {}
            
            # Process each placeholder
            for placeholder, placeholder_config in config.data.items():
                try:
                    # Check if the data is available in structured data
                    structured_key = placeholder.replace('placeholder', '').lower()
                    if structured_key in structured_data:
                        result[placeholder] = structured_data[structured_key]
                        continue
                        
                    value = self._extract_with_selector(
                        selector, placeholder_config
                    )
                    
                    if value:
                        result[placeholder] = self._process_value(placeholder, value, config.url)
                    
                except Exception as e:
                    logger("error",f"Error processing placeholder {placeholder}: {str(e)}", "", "scraper.py")
                    result[placeholder] = None
            
            return result
            
        except Exception as e:
            logger("error",f"Error scraping {config.url}: {str(e)}", "", "scraper.py")
            return {
                'success': False,
                'error': str(e),
                'url': config.url,
                'timestamp': datetime.now().isoformat()
            }
