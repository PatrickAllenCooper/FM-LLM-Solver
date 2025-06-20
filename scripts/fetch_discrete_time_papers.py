#!/usr/bin/env python3
"""
Fetch discrete-time barrier certificate papers for knowledge base enhancement.

This script identifies and downloads papers specifically about discrete-time 
barrier certificates to improve the knowledge base for discrete systems.
"""

import requests
import time
import json
from pathlib import Path

# Search terms for discrete-time barrier certificates
DISCRETE_TIME_SEARCH_TERMS = [
    "discrete time barrier certificate",
    "discrete barrier function", 
    "barrier certificate discrete system",
    "discrete time reachability",
    "discrete barrier method",
    "finite horizon barrier",
    "barrier certificate hybrid automata",
    "discrete time safety verification",
    "barrier function difference equation"
]

# arXiv API endpoint
ARXIV_API_URL = "http://export.arxiv.org/api/query"

def search_arxiv(search_term, max_results=10):
    """Search arXiv for papers matching the search term."""
    params = {
        'search_query': f'all:{search_term}',
        'start': 0,
        'max_results': max_results,
        'sortBy': 'relevance',
        'sortOrder': 'descending'
    }
    
    try:
        response = requests.get(ARXIV_API_URL, params=params)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error searching arXiv: {e}")
        return None

def parse_arxiv_response(xml_content):
    """Parse arXiv XML response to extract paper information."""
    import xml.etree.ElementTree as ET
    
    papers = []
    try:
        root = ET.fromstring(xml_content)
        namespace = {'atom': 'http://www.w3.org/2005/Atom'}
        
        for entry in root.findall('atom:entry', namespace):
            title_elem = entry.find('atom:title', namespace)
            summary_elem = entry.find('atom:summary', namespace)
            published_elem = entry.find('atom:published', namespace)
            pdf_link_elem = entry.find('.//atom:link[@type="application/pdf"]', namespace)
            
            if title_elem is not None and pdf_link_elem is not None:
                papers.append({
                    'title': title_elem.text.strip(),
                    'summary': summary_elem.text.strip() if summary_elem is not None else '',
                    'published': published_elem.text.strip() if published_elem is not None else '',
                    'pdf_url': pdf_link_elem.get('href'),
                    'relevant_terms': []
                })
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
    
    return papers

def download_paper(paper_info, output_dir):
    """Download a paper PDF to the output directory."""
    try:
        response = requests.get(paper_info['pdf_url'], stream=True)
        response.raise_for_status()
        
        # Create filename from title
        safe_title = "".join(c for c in paper_info['title'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"{safe_title[:50]}.pdf"  # Limit filename length
        filepath = output_dir / filename
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Downloaded: {filename}")
        return str(filepath)
    
    except requests.RequestException as e:
        print(f"Error downloading {paper_info['title']}: {e}")
        return None

def filter_relevant_papers(papers, min_relevance_score=2):
    """Filter papers based on relevance to discrete-time barrier certificates."""
    discrete_keywords = [
        'discrete', 'difference equation', 'discrete-time', 'finite horizon',
        'hybrid automata', 'discrete system', 'discrete dynamics',
        'barrier certificate', 'barrier function', 'safety verification'
    ]
    
    filtered_papers = []
    for paper in papers:
        score = 0
        matched_terms = []
        
        text_to_check = f"{paper['title']} {paper['summary']}".lower()
        
        for keyword in discrete_keywords:
            if keyword in text_to_check:
                score += 1
                matched_terms.append(keyword)
        
        if score >= min_relevance_score:
            paper['relevance_score'] = score
            paper['matched_terms'] = matched_terms
            filtered_papers.append(paper)
    
    return sorted(filtered_papers, key=lambda x: x['relevance_score'], reverse=True)

def main():
    """Main function to search and download discrete-time barrier certificate papers."""
    output_dir = Path("data/discrete_time_papers")
    output_dir.mkdir(exist_ok=True)
    
    all_papers = []
    
    print("Searching for discrete-time barrier certificate papers...")
    
    for search_term in DISCRETE_TIME_SEARCH_TERMS:
        print(f"\nSearching for: {search_term}")
        xml_content = search_arxiv(search_term, max_results=5)
        
        if xml_content:
            papers = parse_arxiv_response(xml_content)
            for paper in papers:
                paper['search_term'] = search_term
            all_papers.extend(papers)
            
            # Be respectful to arXiv API
            time.sleep(1)
    
    # Remove duplicates based on title
    unique_papers = []
    seen_titles = set()
    for paper in all_papers:
        if paper['title'] not in seen_titles:
            unique_papers.append(paper)
            seen_titles.add(paper['title'])
    
    print(f"\nFound {len(unique_papers)} unique papers")
    
    # Filter for relevance
    relevant_papers = filter_relevant_papers(unique_papers, min_relevance_score=2)
    print(f"Filtered to {len(relevant_papers)} relevant papers")
    
    # Save paper metadata
    metadata_file = output_dir / "discrete_papers_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(relevant_papers, f, indent=2)
    print(f"Saved metadata to: {metadata_file}")
    
    # Download top papers
    top_papers = relevant_papers[:10]  # Download top 10 most relevant
    print(f"\nDownloading top {len(top_papers)} papers...")
    
    downloaded_papers = []
    for i, paper in enumerate(top_papers, 1):
        print(f"[{i}/{len(top_papers)}] {paper['title'][:60]}...")
        filepath = download_paper(paper, output_dir)
        if filepath:
            downloaded_papers.append(filepath)
        
        # Be respectful - add delay between downloads
        time.sleep(2)
    
    print(f"\nDownload complete!")
    print(f"Downloaded {len(downloaded_papers)} papers to: {output_dir}")
    print("\nNext steps:")
    print("1. Run knowledge base builder on these papers")
    print("2. Rebuild the discrete knowledge base")
    print("3. Fine-tune the model with discrete-time training data")

if __name__ == "__main__":
    main() 