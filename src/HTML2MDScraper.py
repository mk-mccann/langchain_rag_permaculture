import os
import time
import json
import yaml
import requests
from typing import List, Dict
from pathlib import Path
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from bs4 import BeautifulSoup
from markdownify import MarkdownConverter
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
from langdetect import detect, LangDetectException


DISALLOWED_FILE_TYPES = [
    'png', 'jpg', 'jpeg', 'gif', 'pdf', 
    'docx', 'zip', 'fcstd', 'stl', 'kdenlive',
    'mp4', 'mp3', 'avi', 'mov', 'svg', 'skp',
    'exe', 'dmg', 'iso', 'tar', 'gz', '.rar', '7z', 'csv',
    'xlsx', 'pptx', 'ini', 'sys', 'dll', 'dxf', 'odt', 
    'ods', 'odp', 'epub', 'mobi', 'dae', 'fbx', '3ds',
    'ino', 'stp'
]

DISALLOWED_PAGE_TYPES = [
    'File:', 'Schematic:', 'Category:', 'Special:', 
    'Template:', 'one-community-welcomes'
]


class HTML2MDScraper:

    def __init__(
        self, 
        base_output_dir: Path | str, 
        target_langs: List[str] | str = ['en'], 
        delay_seconds: int = 1
        ):

        """
        Initialize the HTML to Markdown Scraper.
        
        Args:
            base_output_dir (Path | str): Base directory to save scraped Markdown files.
            target_langs (List[str] | str): List of target languages for filtering pages.
            delay_seconds (int): Delay in seconds between requests to the same site.
        """

        self.base_output_dir = Path(base_output_dir)
        self.delay_seconds = delay_seconds
        self.visited_urls = set()

        # Handle target languages - must be list
        if isinstance(target_langs, str):
            self.target_langs = [target_langs]
        else:
            self.target_langs = target_langs

        if not self.base_output_dir.exists():
            self.base_output_dir.mkdir(parents=True, exist_ok=True)


    @staticmethod
    def load_json_config(file_path: str | Path) -> Dict:
        """Load JSON configuration file."""

        with open(file_path, 'r') as file:
            return json.load(file)


    @staticmethod
    def is_allowed_by_robots(url: str, robots_url: str) -> bool:
        """Check if the URL is allowed to be scraped according to robots.txt."""

        rp = RobotFileParser()
        rp.set_url(robots_url)

        try:
            rp.read()
            return rp.can_fetch("*", url)
        except Exception as e:
            print(f"Error reading robots.txt for {url}: {e}")
            return False


    def is_target_lang(self, text: str) -> bool:
        """Check if the detected language of the text is in target languages."""

        try:
            return detect(text) in self.target_langs
        except LangDetectException:
            return False

    @staticmethod
    def extract_html_tags(soup: BeautifulSoup) -> Dict:
        """Extract standard metadata tags."""

        meta_data = {}

        for meta in soup.find_all('meta'):
            if meta.get('source'):
                meta_data[meta['source']] = meta.get('content', '')
            elif meta.get('http-equiv'):
                meta_data[f"http-equiv-{meta['http-equiv']}"] = meta.get('content', '')

        return meta_data


    def assemble_metadata(self, soup: BeautifulSoup, url: str, config: Dict) -> Dict:
        """
        Assemble metadata dictionary from HTML soup and config.
        
        Args:
            soup (BeautifulSoup): Parsed HTML content.
            url (str): URL of the webpage.
            config (Dict): Configuration dictionary for additional metadata.
            
        Returns:
            Dict: Metadata
        """
        tags = self.extract_html_tags(soup)

        metadata = {
            "source": str(config.get("source", "")),
            "title": str(soup.title.string) if soup.title else "",
            "author": str(tags.get("author", "")),
            "url": str(url),
            "access_date": str(time.strftime("%Y-%m-%d")),
            "license": str(config.get("license", "")),
            "description": str(tags.get("description", ""))
        }

        for key, value in metadata.items():
            if key != "url":
                metadata[key] = value.replace(':', ' -')

        return metadata


    @staticmethod
    def bs4_to_md(soup: BeautifulSoup, **options):
        """
        Convert BeautifulSoup object to Markdown string.
        
        Args:
            soup (BeautifulSoup): Parsed HTML content.
            **options: Additional options for MarkdownConverter.
        """

        return MarkdownConverter(**options).convert_soup(soup)


    def scrape_page_to_md(self, url: str, config: Dict, save_dir: str | Path) -> BeautifulSoup | None:
        """
        Scrape a single webpage, convert it to Markdown, and save it with metadata.
        
        Args:
            url (str): URL of the webpage to scrape.
            config (Dict): Configuration dictionary for metadata.
            save_dir (str | Path): Directory to save the Markdown file.
        
        Returns:
            BeautifulSoup | None: Parsed HTML content if successful, else None.
        """
        
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()

            page_name = f"{urlparse(url).path.replace('/', '') or 'index'}.md"
            if len(page_name.split('.')[0]) > 4 and page_name[:4] == "wiki":
                page_name = page_name[4:]

            if type(save_dir) is str:
                save_dir = Path(save_dir)

            page_save_path = save_dir / page_name

            soup = BeautifulSoup(response.text, "html.parser")
            metadata = self.assemble_metadata(soup, url, config)

            for tag in ["nav", "header", "footer", "footer-widgets", "script"]:
                for element in soup.find_all(tag):
                    element.decompose()

            # If already saved, scrape for urls only
            # This is a fast check to avoid redundant work, and a failsafe if the process is interrupted
            if page_save_path.exists():                
                return soup

            md_content = self.bs4_to_md(soup, heading_style="ATX")
            
            yaml_frontmatter = yaml.dump(metadata, 
                                         default_flow_style=False, 
                                         sort_keys=False, 
                                         allow_unicode=True)
            
            markdown_with_frontmatter = f"---\n{yaml_frontmatter}---\n\n{md_content}"

            with open(page_save_path, "w", encoding="utf-8") as file:
                file.write(markdown_with_frontmatter)
            
            print(f"Scraped: {url}")
            return soup
        
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None


    @staticmethod
    def get_links(soup, base_url: str) -> set:
        """
        Extract and return a set of valid links from the BeautifulSoup object.

        Args:
            soup (BeautifulSoup): Parsed HTML content.
            base_url (str): Base URL to resolve relative links.

        Returns:
            set: A set of valid URLs found in the page.
        """

        links = set()
        
        for link in soup.find_all("a", href=True):
            href = link["href"]
            full_url = urljoin(base_url, href)

            # Skip disallowed file types
            if any([file_type in str(full_url).lower().rsplit(".")[-1] for file_type in DISALLOWED_FILE_TYPES]):
                continue

            # Skip disallowed page types
            if any([page_type in full_url for page_type in DISALLOWED_PAGE_TYPES]):
                continue
        
            if full_url.startswith(base_url) and "#" not in full_url and "?" not in full_url:
                links.add(full_url)
        
        return links


    def crawl_site(self, config: Dict, save_dir: str | Path, workers: int = 1):
        """
        Crawl a website starting from the base URL in config, saving Markdown files to save_dir.
        Uses multithreading to speed up the process while respecting robots.txt and per-thread delays.

        Args:
            config (Dict): Configuration dictionary with at least 'url' key.
            save_dir (str | Path): Directory to save the scraped Markdown files.
            workers (int): Number of concurrent threads to use for crawling. Defaults to single thread.
        """

        base_url = config['url']
        url_root = f"{urlparse(base_url).scheme}://{urlparse(base_url).netloc}/"

        # Per-site state to avoid cross-site interference when running in parallel
        visited = set()
        queue = deque([base_url])
        lock = threading.Lock()

        def worker():
            while True:
                with lock:
                    if not queue:
                        return
                    url = queue.pop()
                    if url in visited:
                        continue
                    visited.add(url)

                robots_url = urljoin(url, "robots.txt")
                if not self.is_allowed_by_robots(url, robots_url):
                    print(f"Skipping {url}: Disallowed by robots.txt")
                    continue

                soup = self.scrape_page_to_md(url, config, save_dir)
                if soup:
                    new_links = self.get_links(soup, url_root)
                    with lock:
                        for link in new_links:
                            if link not in visited:
                                queue.append(link)

                # Maintain per-thread delay between requests
                time.sleep(self.delay_seconds)

        # Tune max_workers as needed; keep modest to respect target sites
        max_workers = min((os.cpu_count() or 4), workers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(worker) for _ in range(max_workers)]
            for f in futures:
                f.result()


    def process_config(self, config_file: str | Path):
        """
        Process a JSON configuration file to crawl multiple websites.

        Args:
            config_file (str | Path): Path to the JSON configuration file.
        """

        config = self.load_json_config(config_file)

        for cfg in config:
            site_source = cfg['source']
            base_url = cfg['url']

            output_dir = self.base_output_dir / site_source.replace(" ", "_")
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"\nCrawling {site_source} ({base_url})")
            self.crawl_site(cfg, output_dir)

            print(f"\nScraped {len(self.visited_urls)} pages to {output_dir}/")
            self.visited_urls.clear()

        print("\nScraping completed.")


if __name__ == "__main__":
    input_file = Path("../config/urls.json")
    base_output_dir = Path("../data/raw/md_scraped_pages")

    scraper = HTML2MDScraper(base_output_dir=base_output_dir, target_langs=['en'], delay_seconds=1)
    scraper.process_config(input_file)
