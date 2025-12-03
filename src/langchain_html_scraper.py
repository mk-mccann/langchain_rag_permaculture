import re
import time
import json
import yaml
import requests
from typing import List
from pathlib import Path

import scrapy
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
from langdetect import detect, LangDetectException
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.document_loaders import SitemapLoader, RecursiveUrlLoader, WebBaseLoader


class WebsiteSpider(scrapy.Spider):
    name = "website_spider"

    def __init__(self, 
                 start_url: str,
                 base_output_dir: Path | str, 
                 target_langs: List[str] | str = ['en'], 

                 *args, **kwargs):

        super(WebsiteSpider, self).__init__(*args, **kwargs)

        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.start_url = start_url
        self.allowed_domains = urlparse(start_url).netloc

        # Handle target languages - must be list
        if isinstance(target_langs, str):
            self.target_langs = [target_langs]
        else:
            self.target_langs = target_langs

        self.visited_urls = set()
        self.max_depth = 2  # Set your desired crawl depth

        self.disallowed_file_types = [
            '.png', '.jpg', '.jpeg', '.gif', '.pdf', '.docx', '.zip', '.fcstd', 
            '.stl', '.kdenlive', '.mp4', '.mp3', '.avi', '.mov', '.svg', '.skp',
            '.exe', '.dmg', '.iso', '.tar', '.gz', '.rar', '.7z', '.csv',
            '.xlsx', '.pptx', '.ini', '.sys', '.dll', '.dxf', '.odt', 
            '.ods', '.odp', '.epub', '.mobi', '.dae', '.fbx', '.3ds'
        ]


    @staticmethod
    def _is_allowed_by_robots(url):
        robots_url = urljoin(url, "robots.txt")
        rp = RobotFileParser()
        rp.set_url(robots_url)

        try:
            rp.read()
            return rp.can_fetch("*", url)
        except Exception as e:
            print(f"Error reading robots.txt for {url}: {e}")
            return False


    def is_target_lang(self, text):
        try:
            return detect(text) in self.target_langs
        except LangDetectException:
            return False


    def parse(self, response):
        if response.url in self.visited_urls:
            return
        self.visited_urls.add(response.url)

        # Extract text from the page
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove navigation, headers, footers, scripts
        for element in soup(["nav", "header", "footer", "script"]):
            element.decompose()

        # Remove hyperlinks
        for a in soup.find_all('a', href=True):
            a.decompose()

        text = soup.get_text(separator='\n', strip=True)

        yield {
            'url': response.url,
            'text': text,
        }

        # Follow links recursively
        if response.meta.get('depth', 0) < self.max_depth:
            for link in response.css('a::attr(href)').getall():
                absolute_url = urljoin(response.url, link)
                if absolute_url not in self.visited_urls and self.allowed_domains[0] in absolute_url:
                    yield response.follow(absolute_url, self.parse)









class LangChainHTMLScraper:
    def __init__(self, 
                 base_output_dir: Path | str, 
                 config_path: str = "config/urls.json",
                 target_langs: List[str] | str = ['en'], 
                 delay_seconds: int = 1
    ):
        
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

        self.delay_seconds = delay_seconds

        self.config = self._load_config(Path(config_path))
        self.visited_urls = set()

        # Handle target languages - must be list
        if isinstance(target_langs, str):
            self.target_langs = [target_langs]
        else:
            self.target_langs = target_langs

        self.disallowed_file_types = [
            '.png', '.jpg', '.jpeg', '.gif', '.pdf', '.docx', '.zip', '.fcstd', 
            '.stl', '.kdenlive', '.mp4', '.mp3', '.avi', '.mov', '.svg', '.skp',
            '.exe', '.dmg', '.iso', '.tar', '.gz', '.rar', '.7z', '.csv',
            '.xlsx', '.pptx', '.ini', '.sys', '.dll', '.dxf', '.odt', 
            '.ods', '.odp', '.epub', '.mobi', '.dae', '.fbx', '.3ds'
        ]

        if not self.base_output_dir.exists():
            self.base_output_dir.mkdir(parents=True, exist_ok=True)
    

    @staticmethod
    def _load_config(config_path) -> list[str]:
        """Load URLs from the configuration file."""
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config.get('urls', [])
    

    @staticmethod
    def _is_allowed_by_robots(url, robots_url):
        rp = RobotFileParser()
        rp.set_url(robots_url)

        try:
            rp.read()
            return rp.can_fetch("*", url)
        except Exception as e:
            print(f"Error reading robots.txt for {url}: {e}")
            return False
    

    @staticmethod
    def _extract_html_tags(soup):
        """Extract standard meta tags."""
        meta_data = {}

        for meta in soup.find_all('meta'):
            if meta.get('source'):
                meta_data[meta['source']] = meta.get('content', '')
            elif meta.get('http-equiv'):
                meta_data[f"http-equiv-{meta['http-equiv']}"] = meta.get('content', '')

        return meta_data


    def is_target_lang(self, text):
        try:
            return detect(text) in self.target_langs
        except LangDetectException:
            return False


    def assemble_metadata(self, soup, url, config):
        tags = self._extract_html_tags(soup)

        metadata = {
            "source": str(config.get("source", "")),
            "title": str(soup.title.string) if soup.title else "",
            "author": str(tags.get("author", "")),
            "url": str(url),
            "access_date": str(time.strftime("%Y-%m-%d")),
            "license": str(config.get("license", "")),
            "description": str(tags.get("description", ""))
        }

        return metadata


    @staticmethod
    def get_links(soup, base_url):
        links = set()
        
        for link in soup.find_all("a", href=True):
            href = link["href"]
            full_url = urljoin(base_url, href)
        
            if full_url.startswith(base_url) and "#" not in full_url and "?" not in full_url:
                links.add(full_url)
        
        return links


    def scrape_urls(self, output_dir: str = "data/scraped") -> None:
        """Scrape all URLs and save as markdown files."""
        output_path = Path(output_dir)

        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load HTML documents
        loader = AsyncChtmlLoader(self.urls)
        docs = loader.load()
        
        # Transform HTML to text/markdown
        html2text = Html2TextTransformer()
        docs_transformed = html2text.transform_documents(docs)
        
        # Save each document
        for i, doc in enumerate(docs_transformed):
            url = self.urls[i]
            filename = self._url_to_filename(url)
            filepath = output_path / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# Source: {url}\n\n")
                f.write(doc.page_content)
            
            print(f"Saved: {filepath}")
    
    def _url_to_filename(self, url: str) -> str:
        """Convert URL to a valid filename."""
        filename = re.sub(r'https?://', '', url)
        filename = re.sub(r'[^\w\-_.]', '_', filename)
        return f"{filename}.md"


if __name__ == "__main__":
    scraper = LangChainHTMLScraper("../data/raw/scraped", config_path="../config/urls.json")
    scraper.scrape()
