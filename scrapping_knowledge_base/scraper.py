import os
import re
import json
import time
import hashlib
from urllib.parse import urljoin, urlparse
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime

# ----------------- CONFIG -----------------
BASE_DOMAIN = "bankofmaharashtra.in"
OUTPUT_DIR = "output"
HEADERS = {"User-Agent": "LoanKB-Scraper/1.0 (+your-email@example.com)"}
os.makedirs(f"{OUTPUT_DIR}/pages", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/tables", exist_ok=True)

# ----------------- HELPERS -----------------
def is_internal(url):
    return urlparse(url).netloc.endswith(BASE_DOMAIN)

def likely_loan_url(url):
    return bool(re.search(r"loan|interest|maha-super", url, re.I))

def get_hash(text):
    return hashlib.md5(text.encode()).hexdigest()[:10]

def save_html(url, html):
    fname = get_hash(url) + ".html"
    with open(f"{OUTPUT_DIR}/pages/{fname}", "w", encoding="utf-8") as f:
        f.write(html)

def clean_text(soup):
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.extract()
    texts = []
    for el in soup.find_all(["h1", "h2", "h3", "p", "li"]):
        txt = el.get_text(strip=True)
        if txt and len(txt) > 5:
            texts.append(txt)
    return "\n".join(texts)

def extract_tables(url, soup):
    tables = pd.read_html(str(soup))
    table_files = []
    for idx, table in enumerate(tables):
        filename = f"{get_hash(url)}_{idx}.csv"
        table.to_csv(f"{OUTPUT_DIR}/tables/{filename}", index=False)
        table_files.append(filename)
    return table_files

# ----------------- MAIN SCRAPER -----------------
def scrape(seed_urls, max_pages=50):
    visited, to_visit = set(), list(seed_urls)
    results = []
    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited: continue
        try:
            print(f"[+] Fetching: {url}")
            res = requests.get(url, headers=HEADERS, timeout=15)
            res.raise_for_status()
        except Exception as e:
            print(f"[!] Failed: {url} | {e}")
            continue

        visited.add(url)
        html = res.text
        save_html(url, html)
        soup = BeautifulSoup(html, "lxml")

        # Extract main content
        text = clean_text(soup)
        tables = extract_tables(url, soup) if "<table" in html else []

        # Store as JSON entry
        entry = {
            "id": get_hash(url),
            "url": url,
            "scrape_date": datetime.now().isoformat(),
            "title": soup.title.get_text() if soup.title else "",
            "text": text,
            "tables": tables,
        }
        results.append(entry)

        # Find links to follow
        for a in soup.find_all("a", href=True):
            link = urljoin(url, a["href"])
            if is_internal(link) and likely_loan_url(link) and link not in visited:
                to_visit.append(link)

        time.sleep(1.5)  # polite crawling

    # Save to JSONL
    with open(f"{OUTPUT_DIR}/loan_data.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[✔] Scraping finished. {len(results)} pages saved to loan_data.jsonl.")

# ----------------- ENTRY -----------------
if __name__ == "__main__":
    with open("seed_urls.txt") as f:
        seed_urls = [line.strip() for line in f if line.strip()]
    scrape(seed_urls, max_pages=50)
