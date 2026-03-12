#!/usr/bin/env python3
"""
FrankerFaceZ Emote Scraper
Scrapes emote names from the FrankerFaceZ emote library (first 20 pages)
Saves results to a JSON file
"""

import requests
from bs4 import BeautifulSoup
import json
from typing import Set, List
import time


def scrape_ffz_page(page: int) -> Set[str]:
    """
    Scrape emote names from a single FrankerFaceZ page
    
    Args:
        page: Page number (1-based)
        
    Returns:
        Set of emote names from that page
    """
    if page == 1:
        url = "https://www.frankerfacez.com/emoticons/"
    else:
        url = f"https://www.frankerfacez.com/emoticons/?page={page}"
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all emote rows in the table
        emotes = set()
        
        # Look for table rows with emote links
        rows = soup.find_all('tr')
        for row in rows:
            # Find the first link in the row (emote name link)
            link = row.find('a', href=True)
            if link and '/emoticon/' in link.get('href', ''):
                emote_name = link.text.strip()
                if emote_name:
                    emotes.add(emote_name)
        
        return emotes
    
    except Exception as e:
        print(f"Error scraping page {page}: {e}")
        return set()


def scrape_ffz_pages(num_pages: int = 20) -> List[str]:
    """
    Scrape emotes from multiple pages
    
    Args:
        num_pages: Number of pages to scrape (default 20)
        
    Returns:
        List of all unique emote names
    """
    all_emotes = set()
    
    print(f"Scraping FrankerFaceZ emote library (first {num_pages} pages)...\n")
    
    for page in range(1, num_pages + 1):
        print(f"Scraping page {page}/{num_pages}...", end=' ')
        emotes = scrape_ffz_page(page)
        all_emotes.update(emotes)
        print(f"({len(emotes)} emotes, {len(all_emotes)} total)")
        
        # Be respectful - small delay between requests
        time.sleep(0.5)
    
    return sorted(list(all_emotes))


def save_to_json(emotes: List[str], filename: str = 'ffz_emotes.json'):
    """
    Save emote names to JSON file
    
    Args:
        emotes: List of emote names
        filename: Output filename
    """
    data = {
        'count': len(emotes),
        'emotes': emotes
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✓ Saved {len(emotes)} emotes to {filename}")


if __name__ == '__main__':
    # Scrape first 20 pages
    emotes = scrape_ffz_pages(num_pages=20)
    
    # Save to JSON
    save_to_json(emotes)
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Total emotes: {len(emotes)}")
    print(f"  First 10: {', '.join(emotes[:10])}")
    print(f"  Last 10: {', '.join(emotes[-10:])}")
