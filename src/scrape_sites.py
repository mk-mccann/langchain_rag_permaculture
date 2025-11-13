import os
import time
import json
import requests

from bs4 import BeautifulSoup
from urllib.robotparser import RobotFileParser
from urllib.parse import urljoin, urlparse
from langdetect import detect, LangDetectException



def load_json_config(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
    

def is_allowed(url, robots_url):
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


def scrape_page(url, site_name, target_langs):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()

        # Check if the page is in English
        if not is_target_lang(response.text, target_langs):
            print(f"Page language not in target. Skipping: {url}")
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove unwanted elements (e.g., nav, footer, scripts)
        for element in soup(["nav", "footer", "script", "style"]):
            element.decompose()

        # Save the page content
        page_name = f"{site_name}{urlparse(url).path.replace('/', '_') or 'index'}.txt"
        
        with open(os.path.join(output_dir, page_name), "w", encoding="utf-8") as file:
            file.write(soup.get_text(separator="\n", strip=True))
        
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


def crawl_site(base_url, site_name, target_langs=["en"]):

    # disallow crawling certain file types
    disallowed_file_types = ['.png', '.jpg', '.jpeg', '.gif', '.pdf', 
                             '.docx', '.zip', '.fcstd', '.stl', '.kdenlive']

    # Check to see if we are allowed to crawl the site and sub-pages
    robots_url = urljoin(base_url, "robots.txt")
    
    if not is_allowed(base_url, robots_url):
        print(f"Skipping {base_url}: Disallowed by robots.txt")
        return

    # Queue up the sites to scrape
    queue = {base_url}

    while queue:
        url = queue.pop()

        # If we've already visited this URL, skip it
        if url in visited_urls:
            continue

        # Skip disallowed file types
        if any([disallowed_file_types in url for disallowed_file_types in disallowed_file_types]):
            continue

        # Skip any URLs related to users
        if "User" in url or "/users/" in url:
            continue

        # Mark URL as visited
        visited_urls.add(url)

        # Scrape the page
        soup = scrape_page(url, site_name, target_langs)
        if soup:
            new_links = get_links(soup, base_url)
            queue.update(new_links - visited_urls)
            time.sleep(delay_seconds)


if __name__ == "__main__":

    # Configuration
    input_file = "../config/urls.json"
    output_dir = "../data/raw/scraped_pages"

    target_langs = ['en']  # Target language for scraping

    visited_urls = set()
    delay_seconds = 1  # Rate limiting

    # Make outptut directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load URLs from config
    urls_config = load_json_config(input_file)

    # Loop through each URL in the config and get sub-pages if allowed
    for url in urls_config:
        site_name = url['name']
        base_url = url['url']  
        print(f"\nCrawling {site_name} ({base_url})")
        crawl_site(base_url, site_name, target_langs=target_langs)

    print(f"\nDone! Scraped {len(visited_urls)} pages to {output_dir}/")
