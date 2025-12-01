import time
import json
import yaml
import requests
from typing import List
from pathlib import Path

from bs4 import BeautifulSoup
from markdownify import MarkdownConverter
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
from langdetect import detect, LangDetectException
from langchain_community.document_loaders import WebBaseLoader


class HTML2MDScraper:

    def __init__(
        self, 
        base_output_dir: Path | str, 
        target_langs: List[str] | str = ['en'], 
        delay_seconds: int = 1
        ):

        self.base_output_dir = Path(base_output_dir)
        self.delay_seconds = delay_seconds
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
    def load_json_config(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)


    @staticmethod
    def is_allowed_by_robots(url, robots_url):
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

    @staticmethod
    def extract_html_tags(soup):
        """Extract standard meta tags."""
        meta_data = {}

        for meta in soup.find_all('meta'):
            if meta.get('source'):
                meta_data[meta['source']] = meta.get('content', '')
            elif meta.get('http-equiv'):
                meta_data[f"http-equiv-{meta['http-equiv']}"] = meta.get('content', '')

        return meta_data

    def assemble_metadata(self, soup, url, config):
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

        return metadata


    @staticmethod
    def bs4_to_md(soup, **options):
        """Convert BeautifulSoup object to Markdown string."""
        return MarkdownConverter(**options).convert_soup(soup)


    def scrape_page_to_md(self, url, config, save_dir):
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()

            page_source = f"{urlparse(url).path.replace('/', '') or 'index'}.md"
            page_save_path =  save_dir / page_source

            if page_save_path.exists():
                print(f"Already scraped. Skipping: {url}")
                soup = BeautifulSoup(response.text, "html.parser")
                for element in soup(["nav", "footer", "script", "style"]):
                    element.decompose()
                return soup

            if not self.is_target_lang(response.text):
                print(f"Page language not in target. Skipping: {url}")
                return None

            soup = BeautifulSoup(response.text, "html.parser")
            metadata = self.assemble_metadata(soup, url, config)

            for element in soup(["nav", "footer", "script", "style"]):
                element.decompose()

            md_content = self.bs4_to_md(soup, separator="\n", bullets='-')
            yaml_frontmatter = yaml.dump(metadata, default_flow_style=False, 
                                        sort_keys=False, allow_unicode=True)
            markdown_with_frontmatter = f"---\n{yaml_frontmatter}---\n\n{md_content}"

            with open(page_save_path, "w", encoding="utf-8") as file:
                file.write(markdown_with_frontmatter)
            
            print(f"Scraped: {url}")
            return soup
        
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None

    @staticmethod
    def get_links(soup, base_url):
        links = set()
        
        for link in soup.find_all("a", href=True):
            href = link["href"]
            full_url = urljoin(base_url, href)
        
            if full_url.startswith(base_url) and "#" not in full_url and "?" not in full_url:
                links.add(full_url)
        
        return links


    def crawl_site(self, config, save_dir):
        base_url = config['url']
        url_root = f"{urlparse(base_url).scheme}://{urlparse(base_url).netloc}/"

        queue = {base_url}

        while queue:
            url = queue.pop()

            robots_url = urljoin(url, "robots.txt")
            if not self.is_allowed_by_robots(url, robots_url):
                print(f"Skipping {url}: Disallowed by robots.txt")
                continue

            if url in self.visited_urls:
                continue

            if any([file_type in url.lower() for file_type in self.disallowed_file_types]):
                continue

            if "User" in url or "/users/" in url:
                continue

            if "File:" in url or "/files/" in url:
                continue

            self.visited_urls.add(url)

            soup = self.scrape_page_to_md(url, config, save_dir)
            if soup:
                new_links = self.get_links(soup, url_root)
                queue.update(new_links - self.visited_urls)
                time.sleep(self.delay_seconds)


    def scrape_from_config(self, config_file):
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


if __source__ == "__main__":
    input_file = Path("../config/urls.json")
    base_output_dir = Path("../data/raw/scraped_pages")

    scraper = HTML2MDScraper(base_output_dir=base_output_dir, target_langs=['en'], delay_seconds=1)
    scraper.scrape_from_config(input_file)
