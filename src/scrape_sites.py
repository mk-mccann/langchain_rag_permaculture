import os
import time
import json
import yaml
import requests

from bs4 import BeautifulSoup
from markdownify import MarkdownConverter
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
from langdetect import detect, LangDetectException



def load_json_config(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
    

def is_allowed_by_robots(url, robots_url):
    rp = RobotFileParser()
    rp.set_url(robots_url)

    try:
        rp.read()
        return rp.can_fetch("*", url)
    
    except Exception as e:
        print(f"Error reading robots.txt for {url}: {e}")
        return False


def is_target_lang(text, target_langs):
    try:
        return detect(text) in target_langs
    except LangDetectException:
        return False  # Skip if language detection fails


def extract_html_tags(soup):
    """Extract standard meta tags."""
    meta_data = {}

    # Standard meta tags
    for meta in soup.find_all('meta'):
        if meta.get('name'):
            meta_data[meta['name']] = meta.get('content', '')
        elif meta.get('http-equiv'):
            meta_data[f"http-equiv-{meta['http-equiv']}"] = meta.get('content', '')

    return meta_data


def assemble_metadata(soup, url, config):
    tags = extract_html_tags(soup)

    # Extract metadata
    metadata = {
        "url": str(url),
        "access_date": time.strftime("%Y-%m-%d"),
        "title": str(soup.title.string) if soup.title else "",
        "license": str(config.get("license", "")),
        "description": str(tags.get("description", ""))
    }

    return metadata


def bs4_to_md(soup, **options):
    """Convert BeautifulSoup object to Markdown string.
    This is from the markdownify library directly"""
    return MarkdownConverter(**options).convert_soup(soup)


def scrape_page_to_md(url, config, target_langs):

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()

        page_name = f"{urlparse(url).path.replace('/', '') or 'index'}.md"

        # If we already have this page, dont save it but return the bs4 object
        if os.path.exists(os.path.join(output_dir, page_name)):
            print(f"Already scraped. Skipping: {url}")

            # Get the beautiful soup object
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove unwanted elements (e.g., nav, footer, scripts) before converting to Markdown
            for element in soup(["nav", "footer", "script", "style"]):
                element.decompose()

            return soup

        # Check if the page is in English
        if not is_target_lang(response.text, target_langs):
            print(f"Page language not in target. Skipping: {url}")
            return None

        # Get the beautiful soup object
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract HTML metadata
        metadata = assemble_metadata(soup, url, config)

        # Remove unwanted elements (e.g., nav, footer, scripts) before converting to Markdown
        for element in soup(["nav", "footer", "script", "style"]):
            element.decompose()

        # Convert to Markdown
        md_content = bs4_to_md(soup, separator="\n", bullets='-')

        # Create YAML front matter for metadata
        yaml_frontmatter = yaml.dump(metadata, default_flow_style=False, sort_keys=False, allow_unicode=True)

        # Format the final Markdown with front matter
        markdown_with_frontmatter = f"---\n{yaml_frontmatter}---\n\n{md_content}"

        # Save the page content
        with open(os.path.join(output_dir, page_name), "w", encoding="utf-8") as file:
            file.write(markdown_with_frontmatter)
        
        print(f"Scraped: {url}")
        return soup
    
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None


def get_links(soup, base_url):
    links = set()
    
    for link in soup.find_all("a", href=True):
        href = link["href"]
        full_url = urljoin(base_url, href)
    
        if full_url.startswith(base_url) and "#" not in full_url and "?" not in full_url:
            links.add(full_url)
    
    return links


def crawl_site(config, target_langs=["en"]):
    base_url = config['url']      

    # Generally the supplied URL base poitns to a site map. Need to get the main URL from there.
    # This will also prevent issues with robots.txt checks and from scraping pages not on the main site.
    url_root = f"{urlparse(base_url).scheme}://{urlparse(base_url).netloc}/"

    # disallow crawling certain file types
    disallowed_file_types = ['.png', '.jpg', '.jpeg', '.gif', '.pdf', 
                             '.docx', '.zip', '.fcstd', '.stl', '.kdenlive',
                             '.mp4', '.mp3', '.avi', '.mov', '.svg', '.skp', '.stl',
                             '.exe', '.dmg', '.iso', '.tar', '.gz', '.rar', '.7z', '.csv',
                             '.xlsx', '.pptx', '.ini', '.sys', '.dll', '.dxf', '.odt', 
                             '.ods', '.odp', '.epub', '.mobi', '.dae', '.fbx', '.3ds', '.dxf']

    # Queue up the sites to scrape
    queue = {base_url}

    while queue:
        url = queue.pop()

        # Check to see if we are allowed to crawl the site and sub-pages
        robots_url = urljoin(url, "robots.txt")
    
        if not is_allowed_by_robots(url, robots_url):
            print(f"Skipping {url}: Disallowed by robots.txt")
            continue

        # If we've already visited this URL, skip it
        if url in visited_urls:
            continue

        # Skip disallowed file types
        if any([disallowed_file_types in url for disallowed_file_types in disallowed_file_types]):
            continue

        # Skip any URLs related to users
        if "User" in url or "/users/" in url:
            continue

        # If this is a file, skip it
        if "File:" in url or "/files/" in url:
            continue

        # Mark URL as visited
        visited_urls.add(url)

        # Scrape the page
        soup = scrape_page_to_md(url, config, target_langs)
        if soup:
            new_links = get_links(soup, url_root)
            queue.update(new_links - visited_urls)
            time.sleep(delay_seconds)


if __name__ == "__main__":

    # Configuration
    input_file = "../config/urls.json"
    base_output_dir = "../data/raw/scraped_pages"

    target_langs = ['en']  # Target language for scraping

    visited_urls = set()
    delay_seconds = 1  # Rate limiting

    # Make outptut directory if it doesn't exist
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)

    # Load URLs from config
    config = load_json_config(input_file)

    # Loop through each URL in the config and get sub-pages if allowed
    for cfg in config:
        site_name = cfg['name']
        base_url = cfg['url']  

        output_dir = os.path.join(base_output_dir, site_name.replace(" ", "_"))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"\nCrawling {site_name} ({base_url})")
        crawl_site(cfg, target_langs=target_langs)

        print(f"\nScraped {len(visited_urls)} pages to {output_dir}/")
        visited_urls.clear()  # Clear for next site

    print("\nScraping completed.")
