import requests
import os
import xml.etree.ElementTree as ET
from scholarly import scholarly
import time
import re
import json # For parsing Semantic Scholar response
import csv # Add csv import

# --- Configuration ---
# IMPORTANT: Replace with your actual email for Unpaywall/API politeness
UNPAYWALL_EMAIL = "your-email@example.com"
# Optional: Add your Semantic Scholar API Key if you have one for higher rate limits
SEMANTIC_SCHOLAR_API_KEY = None # "YOUR_API_KEY" or None

# Read user IDs from CSV
user_ids = []
script_dir = os.path.dirname(__file__) # Get the directory of the current script
csv_path = os.path.join(script_dir, "user_ids.csv")
try:
    with open(csv_path, mode='r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader, None) # Skip header row if it exists
        for row in reader:
            if row: # Ensure row is not empty
                user_ids.append(row[0].strip()) # Assume ID is in the first column
    if not user_ids:
        print(f"Warning: No user IDs found in {csv_path}")
except FileNotFoundError:
    print(f"Error: {csv_path} not found. Please create it with user IDs.")
    user_ids = [] # Ensure user_ids is an empty list if file not found
except Exception as e:
    print(f"Error reading {csv_path}: {e}")
    user_ids = []

output_dir = "recent_papers_all_sources_v2"
os.makedirs(output_dir, exist_ok=True)

# Be a good citizen: delays between external requests (in seconds)
SLEEP_TIME_SCHOLARLY = 2
SLEEP_TIME_API = 1

# Set a user agent to mimic a browser
REQUESTS_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36'
}
# --- End Configuration ---

def sanitize_filename(title):
    """Cleans a string to be safe for use as a filename."""
    # Remove invalid chars
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', title)
    # Replace multiple spaces/underscores with a single underscore
    sanitized = re.sub(r'[\s_]+', '_', sanitized)
    # Limit length
    return sanitized[:150] # Increased length slightly

def check_url_is_pdf(url):
    """Sends a HEAD request to check if the Content-Type is PDF."""
    if not url or not url.lower().startswith('http'):
        return False
    try:
        response = requests.head(url, headers=REQUESTS_HEADERS, allow_redirects=True, timeout=10)
        content_type = response.headers.get('Content-Type', '').lower()
        # Check common PDF content types
        if response.status_code == 200 and ('application/pdf' in content_type or 'application/x-pdf' in content_type):
             # Check if it redirects to a non-pdf landing page sometimes missed by HEAD
            if response.url.endswith('.pdf') or 'pdf' in content_type:
                 print(f"    [HEAD Check OK] URL Content-Type is PDF: {url}")
                 return True
        print(f"    [HEAD Check Failed] URL: {url}, Status: {response.status_code}, Content-Type: {content_type}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"    [HEAD Check Error] Failed to check URL {url}: {e}")
        return False

def download_pdf(url, title, author_id):
    """Downloads a PDF from a URL after verifying it's likely a PDF."""
    print(f"  Attempting download for: {title}")
    author_dir = os.path.join(output_dir, author_id)
    os.makedirs(author_dir, exist_ok=True)
    filename = f"{sanitize_filename(title)}.pdf"
    filepath = os.path.join(author_dir, filename)

    if os.path.exists(filepath):
        print(f"    Already downloaded: {filename}")
        return True

    # Final verification before full download
    if not check_url_is_pdf(url):
        print(f"    Verification failed or not a PDF: {url}")
        return False

    try:
        response = requests.get(url, stream=True, headers=REQUESTS_HEADERS, timeout=20) # Increased timeout
        response.raise_for_status() # Raise an exception for bad status codes

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(8192): # Larger chunk size
                f.write(chunk)
        print(f"    SUCCESS: Downloaded {filename}")
        return True

    except requests.exceptions.RequestException as e:
        print(f"    Error downloading {title} from {url}: {e}")
        # Clean up potentially incomplete file
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError:
                pass
        return False
    except Exception as e:
        print(f"    Unexpected error during download of {title}: {e}")
        return False

def extract_doi_from_url(url):
    """Extracts DOI from various URL patterns using regex."""
    if not url:
        return None
    # Regex for common DOI patterns in URLs (handles http/https, doi.org prefix, etc.)
    # Example: https://doi.org/10.1000/xyz123, http://dx.doi.org/10...
    doi_match = re.search(r'(10\.\d{4,}(\.\d+)*\/[-._;()/:A-Z0-9]+)', url, re.IGNORECASE)
    if doi_match:
        return doi_match.group(1).strip().rstrip('/') # Return the matched DOI part
    return None

def get_pdf_from_unpaywall(doi):
    """Queries Unpaywall API for the best Open Access PDF URL."""
    if not doi: return None
    print(f"    Querying Unpaywall for DOI: {doi}")
    api_url = f"https://api.unpaywall.org/v2/{doi}?email={UNPAYWALL_EMAIL}"
    try:
        resp = requests.get(api_url, headers=REQUESTS_HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        oa_location = data.get("best_oa_location")
        if oa_location and oa_location.get("url_for_pdf"):
            pdf_url = oa_location.get("url_for_pdf")
            print(f"      Unpaywall found PDF: {pdf_url}")
            return pdf_url
        else:
            print("      Unpaywall: No OA PDF found.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"      Unpaywall API error for DOI {doi}: {e}")
    except json.JSONDecodeError:
        print(f"      Unpaywall returned non-JSON response for DOI {doi}")
    except Exception as e:
        print(f"      Unexpected Unpaywall error for DOI {doi}: {e}")
    return None

def search_arxiv(title, author_names=None):
    """Searches arXiv API by title and optionally authors."""
    print(f"    Querying arXiv for title: '{title}'")
    # Try precise title search first
    query_url = f"http://export.arxiv.org/api/query?search_query=ti:\"{title}\"&start=0&max_results=1"
    pdf_url = _fetch_arxiv_results(query_url, title)
    if pdf_url:
        return pdf_url

    # Fallback: broader title search (remove quotes) and add authors if available
    print(f"    Querying arXiv (fallback) for title: {title}")
    search_terms = f"ti:{title.split(':')[0]}" # Use only main title part sometimes
    if author_names:
         # Extract last names for query
         last_names = [name.split()[-1] for name in author_names if name.strip()]
         if last_names:
             author_query = "+AND+".join([f'au:"{name}"' for name in last_names])
             search_terms += f"+AND+({author_query})"

    query_url = f"http://export.arxiv.org/api/query?search_query={search_terms}&start=0&max_results=3" # Get a few results
    return _fetch_arxiv_results(query_url, title)


def _fetch_arxiv_results(query_url, original_title):
    """Helper function to fetch and parse arXiv results."""
    try:
        resp = requests.get(query_url, headers=REQUESTS_HEADERS, timeout=15)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        atom_ns = "{http://www.w3.org/2005/Atom}"
        entries = root.findall(f"{atom_ns}entry")

        if not entries:
            print("      arXiv: No matching entries found.")
            return None

        # Look for the best match (often the first, but could check title similarity)
        # For simplicity, checking the first entry's PDF link
        for link in entries[0].findall(f"{atom_ns}link"):
            if link.get("title") == "pdf" and link.get("href"):
                 pdf_url = link.get("href")
                 print(f"      arXiv found potential PDF: {pdf_url}")
                 # ArXiv links often end in .pdf but sometimes redirect slightly, check carefully
                 return pdf_url + ".pdf" if not pdf_url.lower().endswith('.pdf') else pdf_url # Ensure .pdf extension

        print("      arXiv: Entry found but no PDF link.")
        return None

    except requests.exceptions.RequestException as e:
        print(f"      arXiv API search error: {e}")
    except ET.ParseError:
         print(f"      arXiv returned invalid XML.")
    except Exception as e:
        print(f"      Unexpected arXiv search error: {e}")
    return None

def search_semantic_scholar(title, doi=None):
    """Queries Semantic Scholar (S2) API for paper details and PDF link."""
    print(f"    Querying Semantic Scholar for: '{title}'")
    base_url = "https://api.semanticscholar.org/graph/v1"
    # Use DOI if available (more precise), otherwise use title search
    if doi:
        query = f"/paper/DOI:{doi}"
        params = {'fields': 'isOpenAccess,openAccessPdf,title'}
    else:
        # S2 title search can be noisy, use the title directly
        query = "/paper/search"
        params = {'query': title, 'limit': 1, 'fields': 'isOpenAccess,openAccessPdf,title'}

    api_url = base_url + query
    s2_headers = {**REQUESTS_HEADERS} # Copy base headers
    if SEMANTIC_SCHOLAR_API_KEY:
        s2_headers['x-api-key'] = SEMANTIC_SCHOLAR_API_KEY

    try:
        resp = requests.get(api_url, headers=s2_headers, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        paper_data = None
        if doi: # Direct paper lookup
            paper_data = data
        elif data.get('total', 0) > 0 and data.get('data'): # Search results
             # Basic check: Title similarity (can be improved)
            found_title = data['data'][0].get('title','').lower()
            if title.lower() in found_title or found_title in title.lower():
                 paper_data = data['data'][0]
            else:
                 print(f"      Semantic Scholar: Found paper '{found_title}', but title mismatch.")


        if paper_data:
            if paper_data.get("isOpenAccess") and paper_data.get("openAccessPdf", {}):
                pdf_url = paper_data["openAccessPdf"].get("url")
                if pdf_url:
                    print(f"      Semantic Scholar found OA PDF: {pdf_url}")
                    return pdf_url
            else:
                print(f"      Semantic Scholar: Paper found but no Open Access PDF link available.")
        else:
            print(f"      Semantic Scholar: No definitive match found.")

    except requests.exceptions.RequestException as e:
        print(f"      Semantic Scholar API error: {e}")
    except json.JSONDecodeError:
        print(f"      Semantic Scholar returned non-JSON response.")
    except Exception as e:
        print(f"      Unexpected Semantic Scholar error: {e}")

    return None


def safe_year(p):
    """Safely extracts publication year, returns 0 on failure."""
    year_str = p.get("bib", {}).get("pub_year", "0")
    try:
        return int(year_str)
    except (ValueError, TypeError):
        return 0

# --- Main Execution ---
for user_id in user_ids:
    print(f"\nProcessing researcher: {user_id}")
    try:
        author = scholarly.search_author_id(user_id)
        print(f"  Fetching publications for: {author.get('name', 'Unknown Name')}")
        # Fill author profile with publications section
        # Note: Scholarly might have rate limits or captchas. Add delays.
        author = scholarly.fill(author, sections=["publications"], sortby="year")
        time.sleep(SLEEP_TIME_SCHOLARLY) # Pause after fetching author data

        # Get the 5 most recent publications (already sorted by year by scholarly)
        publications = author.get("publications", [])[:5]
        print(f"  Found {len(publications)} recent publications to check.")

        for i, pub in enumerate(publications):
            pdf_found = False
            pdf_url = None
            print(f"\n  Checking publication {i+1}/{len(publications)}...")

            try:
                # Fill individual publication details (can be slow/trigger limits)
                print(f"    Fetching details...")
                pub_filled = scholarly.fill(pub)
                time.sleep(SLEEP_TIME_SCHOLARLY) # Pause after filling pub details

                title = pub_filled.get("bib", {}).get("title", f"untitled_{i}")
                print(f"    Title: {title}")
                authors = pub_filled.get("bib", {}).get("author", "").split(' and ') # Get authors for arXiv fallback


                # --- PDF Finding Strategy ---
                sources_tried = []

                # 1. Check eprint_url (often arXiv or direct link)
                sources_tried.append("eprint_url")
                pdf_url = pub_filled.get("eprint_url")
                if pdf_url:
                    print(f"    Found eprint_url: {pdf_url}")
                    if download_pdf(pdf_url, title, user_id):
                        pdf_found = True
                    else:
                        pdf_url = None # Reset if download failed/not PDF

                # 2. Try Unpaywall via DOI
                doi = None
                if not pdf_found:
                    sources_tried.append("DOI/Unpaywall")
                    pub_url = pub_filled.get("pub_url", "")
                    doi = extract_doi_from_url(pub_url)
                    if doi:
                        pdf_url = get_pdf_from_unpaywall(doi)
                        time.sleep(SLEEP_TIME_API) # Pause after Unpaywall API call
                        if pdf_url and download_pdf(pdf_url, title, user_id):
                             pdf_found = True
                        else:
                             pdf_url = None # Reset if failed
                    else:
                        print("      No DOI found in pub_url.")


                # 3. Try Semantic Scholar (using DOI if found, else title)
                if not pdf_found:
                    sources_tried.append("Semantic Scholar")
                    pdf_url = search_semantic_scholar(title, doi)
                    time.sleep(SLEEP_TIME_API) # Pause after S2 API call
                    if pdf_url and download_pdf(pdf_url, title, user_id):
                        pdf_found = True
                    else:
                        pdf_url = None # Reset

                # 4. Try arXiv Search (using title and maybe authors)
                if not pdf_found:
                     sources_tried.append("arXiv")
                     pdf_url = search_arxiv(title, authors)
                     time.sleep(SLEEP_TIME_API) # Pause after arXiv API call
                     if pdf_url and download_pdf(pdf_url, title, user_id):
                         pdf_found = True
                     else:
                         pdf_url = None # Reset

                # --- End PDF Finding ---

                if not pdf_found:
                    print(f"    ----> No verified PDF found for '{title}' after trying: {', '.join(sources_tried)}")

            except Exception as e:
                # Catch errors during processing of a single publication
                print(f"    ERROR processing publication {i+1}: {e}")
                # Optionally add a longer sleep here if it seems like a rate limit issue
                time.sleep(SLEEP_TIME_SCHOLARLY * 2)


    except Exception as e:
        # Catch errors during processing of a whole author (e.g., initial fetch failed)
        print(f"  FAILED to process user {user_id}: {e}")
        # Consider adding a sleep here too
        time.sleep(SLEEP_TIME_SCHOLARLY * 2)

print("\nScript finished.")