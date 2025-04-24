import os
import sys
import csv
import time
import requests
from bs4 import BeautifulSoup
import logging
import urllib.parse # For joining relative URLs
import argparse # For command line arguments

# Add project root to Python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

# Now we can import utils and scholarly
from utils.config_loader import load_config, DEFAULT_CONFIG_PATH # Import config loader
from scholarly import scholarly
try:
    from scholarly import MaxTriesExceededException, TimeoutException
except ImportError:
    # Create custom exceptions if not available in the installed version of scholarly
    class MaxTriesExceededException(Exception):
        pass
    class TimeoutException(Exception):
        pass

# Load configuration
cfg = load_config()

# --- Configuration from YAML ---
# IMPORTANT: Ensure UNPAYWALL_EMAIL environment variable is set.
UNPAYWALL_EMAIL = os.environ.get("UNPAYWALL_EMAIL")
if not UNPAYWALL_EMAIL:
    print("Error: UNPAYWALL_EMAIL environment variable not set. Please set it before running.", file=sys.stderr)
    sys.exit(1)

# Optional: Add your Semantic Scholar API Key if you have one for higher rate limits
SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY") # Also read from env var

# Get parameters from config
SLEEP_TIME_SCHOLARLY = cfg.data_fetching.sleep_time_scholarly
SLEEP_TIME_API = cfg.data_fetching.sleep_time_api
SLEEP_TIME_RETRY = cfg.data_fetching.sleep_time_retry
MAX_RETRIES = cfg.data_fetching.max_retries
REQUESTS_HEADERS = {
    'User-Agent': cfg.data_fetching.requests_user_agent
}
PUBLICATION_LIMIT = cfg.data_fetching.publication_limit_per_author

# Get paths from config
output_dir = cfg.paths.pdf_input_dir # Fetcher saves PDFs here
user_ids_csv_path = cfg.paths.user_ids_csv # Path to user IDs CSV

# --- End Configuration ---


# Read user IDs from CSV
user_ids = []
try:
    with open(user_ids_csv_path, mode='r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader, None) # Skip header row if it exists
        for row in reader:
            if row: # Ensure row is not empty
                user_ids.append(row[0].strip()) # Assume ID is in the first column
    if not user_ids:
        print(f"Warning: No user IDs found in {user_ids_csv_path}")
except FileNotFoundError:
    print(f"Error: {user_ids_csv_path} not found. Please create it with user IDs.")
    user_ids = [] # Ensure user_ids is an empty list if file not found
    # Optional: exit if no users?
    # sys.exit("Exiting: No user IDs provided.")
except Exception as e:
    print(f"Error reading {user_ids_csv_path}: {e}")
    user_ids = []

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)


def sanitize_filename(title):
    """Cleans a string to be safe for use as a filename."""
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', title)
    sanitized = re.sub(r'[\s_]+', '_', sanitized)
    return sanitized[:150]

def check_url_is_pdf(url):
    """Sends a HEAD request to check if the Content-Type is PDF."""
    if not url or not url.lower().startswith('http'):
        return False, None # Return URL as well
    try:
        response = requests.head(url, headers=REQUESTS_HEADERS, allow_redirects=True, timeout=15)
        final_url = response.url # Get the URL after redirects
        content_type = response.headers.get('Content-Type', '').lower()
        content_length = response.headers.get('Content-Length')

        if response.status_code == 200 and ('application/pdf' in content_type or 'application/x-pdf' in content_type):
             # Basic check for zero length PDFs which might indicate error pages
             if content_length and int(content_length) < 1000: # Check if length is suspiciously small (e.g. < 1KB)
                 print(f"    [HEAD Check Warning] URL {final_url} is PDF type but Content-Length ({content_length}) is very small. May not be valid.")
                 # Decide whether to proceed or reject based on this warning
                 # return False, final_url # Option to reject small PDFs

             print(f"    [HEAD Check OK] URL Content-Type is PDF: {final_url}")
             return True, final_url
        print(f"    [HEAD Check Failed] URL: {final_url}, Status: {response.status_code}, Content-Type: {content_type}")
        return False, final_url
    except requests.exceptions.RequestException as e:
        print(f"    [HEAD Check Error] Failed to check URL {url}: {e}")
        return False, url # Return original url on error

def find_pdf_in_html(html_content, base_url):
    """Parses HTML content to find links ending in .pdf."""
    soup = BeautifulSoup(html_content, 'lxml')
    pdf_links = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        # Check if href ends with .pdf (case-insensitive) or contains /pdf/
        if href.lower().endswith('.pdf') or '/pdf/' in href.lower():
            # Construct absolute URL if relative
            absolute_url = urllib.parse.urljoin(base_url, href)
            pdf_links.append(absolute_url)

    if pdf_links:
        # Prioritize links ending directly in .pdf
        direct_pdfs = [l for l in pdf_links if l.lower().endswith('.pdf')]
        if direct_pdfs:
             print(f"      Found direct PDF link(s) in HTML: {direct_pdfs[0]}")
             return direct_pdfs[0] # Return the first direct link found
        else:
             print(f"      Found potential PDF link(s) in HTML (via /pdf/): {pdf_links[0]}")
             return pdf_links[0] # Return the first plausible link
    return None

def download_pdf(url, title, author_id):
    """Downloads a PDF from a URL, potentially following redirects and checking HTML."""
    print(f"  Attempting download for: '{title}' from URL: {url}")
    author_dir = os.path.join(output_dir, author_id)
    os.makedirs(author_dir, exist_ok=True)
    filename = f"{sanitize_filename(title)}.pdf"
    filepath = os.path.join(author_dir, filename)

    if os.path.exists(filepath):
        print(f"    Already downloaded: {filename}")
        return True

    current_url = url
    retries = MAX_RETRIES
    attempt = 0
    while attempt <= retries:
        attempt += 1
        pdf_check_ok, final_url = check_url_is_pdf(current_url)
        current_url = final_url # Update URL based on redirects from HEAD check

        if pdf_check_ok:
            # Direct PDF link found via HEAD, attempt download
            try:
                print(f"    Verified PDF Content-Type. Attempting GET...: {current_url}")
                response = requests.get(current_url, stream=True, headers=REQUESTS_HEADERS, timeout=30)
                response.raise_for_status()

                # Double check content type after GET, sometimes HEAD differs
                get_content_type = response.headers.get('Content-Type', '').lower()
                if not ('application/pdf' in get_content_type or 'application/x-pdf' in get_content_type):
                    print(f"    [GET Check Failed] Content-Type was not PDF after GET ({get_content_type}). Trying HTML parse if possible.")
                    # Fall through to HTML parsing logic if status was 200
                    if response.status_code == 200 and 'text/html' in get_content_type:
                        pass # Let the HTML parsing logic handle it below
                    else:
                        return False # Give up if GET shows non-PDF/non-HTML
                else:
                    # Confirmed PDF via GET, save it
                    with open(filepath, 'wb') as f:
                        for chunk in response.iter_content(8192):
                            f.write(chunk)
                    print(f"    SUCCESS: Downloaded {filename}")
                    return True

            except requests.exceptions.RequestException as e:
                print(f"    Error downloading verified PDF from {current_url}: {e}")
                if attempt > retries:
                     return False # Give up after retries
                print(f"    Retrying download ({attempt}/{retries})... after {SLEEP_TIME_RETRY}s")
                time.sleep(SLEEP_TIME_RETRY)
                continue # Retry the HEAD/GET cycle for this URL
            except Exception as e:
                print(f"    Unexpected error during verified PDF download of '{title}': {e}")
                return False
        else:
            # HEAD check failed or showed non-PDF type. Try GETting content to parse HTML.
            print(f"    HEAD check failed/not PDF. Attempting GET for HTML parsing...: {current_url}")
            try:
                response = requests.get(current_url, headers=REQUESTS_HEADERS, timeout=20, allow_redirects=True)
                response.raise_for_status()
                final_get_url = response.url # URL after GET redirects
                get_content_type = response.headers.get('Content-Type', '').lower()

                if 'text/html' in get_content_type:
                    print("      Content is HTML. Parsing for PDF links...")
                    pdf_link_in_html = find_pdf_in_html(response.text, final_get_url)
                    if pdf_link_in_html:
                        print(f"      Found potential PDF link in HTML: {pdf_link_in_html}. Trying this new URL.")
                        # IMPORTANT: Now we need to try downloading *this* new URL
                        # Set current_url and loop again (or call download_pdf recursively, careful with depth)
                        current_url = pdf_link_in_html
                        # Reset attempt counter for the new URL?
                        attempt = 0 # Start fresh for the new link
                        continue # Go back to start of while loop with the new URL
                    else:
                        print("      No PDF links found within HTML.")
                        return False # Give up if HTML has no PDF links
                elif 'application/pdf' in get_content_type or 'application/x-pdf' in get_content_type:
                    # Sometimes GET reveals PDF when HEAD didn't (e.g. redirects)
                    print(f"    [GET Check OK] Content-Type is PDF after GET ({get_content_type}). Saving...")
                    with open(filepath, 'wb') as f:
                        f.write(response.content) # Save content directly
                    print(f"    SUCCESS: Downloaded {filename} (identified via GET)")
                    return True
                else:
                    print(f"      Content-Type after GET is neither HTML nor PDF ({get_content_type}). Giving up.")
                    return False # Unrecognized content type

            except requests.exceptions.RequestException as e:
                print(f"    Error during GET request for {current_url}: {e}")
                if attempt > retries:
                     return False # Give up after retries
                print(f"    Retrying GET ({attempt}/{retries})... after {SLEEP_TIME_RETRY}s")
                time.sleep(SLEEP_TIME_RETRY)
                continue # Retry the GET for this URL
            except Exception as e:
                print(f"    Unexpected error during GET/HTML parse of '{title}': {e}")
                return False

    # Should only reach here if all retries failed
    print(f"    FAILED to download '{title}' after {attempt} attempts.")
    # Clean up potentially incomplete file
    if os.path.exists(filepath):
        try: os.remove(filepath) 
        except OSError: pass
    return False

def extract_doi_from_url(url):
    """Extracts DOI from various URL patterns using regex."""
    if not url:
        return None
    doi_match = re.search(r'(10\.\d{4,}(\.\d+)*\/[-._;()/:A-Z0-9]+)', url, re.IGNORECASE)
    if doi_match:
        return doi_match.group(1).strip().rstrip('/')
    return None

def get_pdf_from_unpaywall(doi):
    """Queries Unpaywall API for the best Open Access PDF URL with retries."""
    if not doi: return None
    print(f"    Querying Unpaywall for DOI: {doi}")
    api_url = f"https://api.unpaywall.org/v2/{doi}?email={UNPAYWALL_EMAIL}"
    retries = MAX_RETRIES
    for attempt in range(retries + 1):
        try:
            resp = requests.get(api_url, headers=REQUESTS_HEADERS, timeout=15)
            if resp.status_code == 429: # Rate limited
                print(f"      Unpaywall rate limit hit (attempt {attempt+1}). Retrying after {SLEEP_TIME_RETRY}s...")
                if attempt < retries: time.sleep(SLEEP_TIME_RETRY); continue
                else: resp.raise_for_status() # Raise after final retry
            resp.raise_for_status()
            data = resp.json()
            oa_location = data.get("best_oa_location")
            if oa_location and oa_location.get("url_for_pdf"):
                pdf_url = oa_location.get("url_for_pdf")
                print(f"      Unpaywall found PDF: {pdf_url}")
                return pdf_url
            else:
                print("      Unpaywall: No OA PDF found.")
                return None # Success, but no PDF
        except requests.exceptions.RequestException as e:
            print(f"      Unpaywall API error (attempt {attempt+1}) for DOI {doi}: {e}")
            if attempt >= retries: return None # Give up after retries
            time.sleep(SLEEP_TIME_RETRY)
        except json.JSONDecodeError:
            print(f"      Unpaywall returned non-JSON response for DOI {doi}")
            return None # Don't retry JSON errors
        except Exception as e:
            print(f"      Unexpected Unpaywall error for DOI {doi}: {e}")
            return None # Don't retry unknown errors
    return None

def search_arxiv(title, author_names=None):
    """Searches arXiv API by title and optionally authors with retries."""
    print(f"    Querying arXiv for title: '{title}'")
    # Try precise title search first
    query_url = f'http://export.arxiv.org/api/query?search_query=ti:"{title.replace(" ", "+")}"&start=0&max_results=1'
    pdf_url = _fetch_arxiv_results(query_url, title)
    if pdf_url:
        return pdf_url

    # Fallback: broader title search (remove quotes) and add authors if available
    print(f"    Querying arXiv (fallback) for title: {title}")
    # Sanitize title for URL query
    sanitized_title = re.sub(r'[\W_]+', '+', title.split(':')[0].strip())
    search_terms = f"ti:{sanitized_title}"
    if author_names:
         last_names = [re.sub(r'[\W_]+', '', name.split()[-1]) for name in author_names if name.strip()]
         if last_names:
             author_query = "+AND+".join([f'au:{name}' for name in last_names])
             search_terms += f"+AND+({author_query})"

    query_url = f"http://export.arxiv.org/api/query?search_query={search_terms}&start=0&max_results=3" # Get a few results
    return _fetch_arxiv_results(query_url, title)

def _fetch_arxiv_results(query_url, original_title):
    """Helper function to fetch and parse arXiv results with retries."""
    retries = MAX_RETRIES
    for attempt in range(retries + 1):
        try:
            print(f"      Querying arXiv API: {query_url}")
            resp = requests.get(query_url, headers=REQUESTS_HEADERS, timeout=20)
            if resp.status_code == 503: # Service Unavailable (common for arXiv)
                 print(f"      arXiv API unavailable (503). Retrying after {SLEEP_TIME_RETRY}s...")
                 if attempt < retries: time.sleep(SLEEP_TIME_RETRY); continue
                 else: resp.raise_for_status()
            resp.raise_for_status()

            # --- Improved XML Parsing --- #
            # Define the Atom namespace
            namespaces = {'atom': 'http://www.w3.org/2005/Atom'}
            try:
                 root = ET.fromstring(resp.content)
                 entries = root.findall('.//atom:entry', namespaces)
            except ET.ParseError as pe:
                 print(f"      arXiv returned invalid XML (attempt {attempt+1}). Error: {pe}")
                 # Log response content for debugging
                 # print(f"Response content:\n{resp.text[:500]}...")
                 if attempt < retries: time.sleep(SLEEP_TIME_RETRY); continue
                 else: return None # Give up on parse error
            # --- End Improved XML Parsing --- #

            if not entries:
                print("      arXiv: No matching entries found in feed.")
                return None

            for entry in entries: # Check all returned entries
                entry_title_elem = entry.find('atom:title', namespaces)
                entry_title = entry_title_elem.text.strip() if entry_title_elem is not None else ""

                # Simple title check (can be improved with fuzzy matching)
                # Normalize whitespace for comparison
                norm_orig_title = ' '.join(original_title.lower().split())
                norm_entry_title = ' '.join(entry_title.lower().split())
                if norm_orig_title not in norm_entry_title and norm_entry_title not in norm_orig_title:
                    print(f"      arXiv: Entry title '{entry_title}' mismatch. Skipping.")
                    continue

                print(f"      arXiv: Found matching entry: '{entry_title}'")
                for link in entry.findall('atom:link', namespaces):
                    if link.get("title") == "pdf" and link.get("href"):
                         pdf_url = link.get("href")
                         print(f"      arXiv found potential PDF link: {pdf_url}")
                         # Ensure .pdf extension ONLY if it seems missing (handles version numbers)
                         parsed_url = urllib.parse.urlparse(pdf_url)
                         if not parsed_url.path.lower().endswith('.pdf'):
                             pdf_url += ".pdf"
                             print(f"        (Appended .pdf extension: {pdf_url})")
                         return pdf_url

            print("      arXiv: Found entries but no PDF links.")
            return None # No PDF link found in relevant entries

        except requests.exceptions.RequestException as e:
            print(f"      arXiv API search error (attempt {attempt+1}): {e}")
            if attempt >= retries: return None
            time.sleep(SLEEP_TIME_RETRY)
        except Exception as e:
            print(f"      Unexpected arXiv search error: {e}")
            return None # Don't retry unknown errors
    return None

def search_semantic_scholar(title, doi=None):
    """Queries Semantic Scholar (S2) API with retries."""
    print(f"    Querying Semantic Scholar for: '{title}'")
    base_url = "https://api.semanticscholar.org/graph/v1"
    if doi:
        query = f"/paper/DOI:{doi}"
        params = {'fields': 'isOpenAccess,openAccessPdf,title'}
    else:
        query = "/paper/search"
        params = {'query': title, 'limit': 1, 'fields': 'isOpenAccess,openAccessPdf,title'}

    api_url = base_url + query
    s2_headers = {**REQUESTS_HEADERS}
    if SEMANTIC_SCHOLAR_API_KEY:
        s2_headers['x-api-key'] = SEMANTIC_SCHOLAR_API_KEY

    retries = MAX_RETRIES
    for attempt in range(retries + 1):
        try:
            resp = requests.get(api_url, headers=s2_headers, params=params, timeout=20)
            if resp.status_code == 429:
                 print(f"      S2 rate limit hit (attempt {attempt+1}). Retrying after {SLEEP_TIME_RETRY}s...")
                 if attempt < retries: time.sleep(SLEEP_TIME_RETRY); continue
                 else: resp.raise_for_status()
            resp.raise_for_status()
            data = resp.json()
            paper_data = None
            if doi: paper_data = data
            elif data.get('total', 0) > 0 and data.get('data'):
                found_title = data['data'][0].get('title','').lower()
                norm_orig_title = ' '.join(title.lower().split())
                norm_found_title = ' '.join(found_title.split())
                # Allow partial match for robustness
                if norm_orig_title in norm_found_title or norm_found_title in norm_orig_title:
                     paper_data = data['data'][0]
                else:
                     print(f"      Semantic Scholar: Title mismatch ('{found_title}' vs '{title}').")

            if paper_data:
                if paper_data.get("isOpenAccess") and paper_data.get("openAccessPdf", {}):
                    pdf_url = paper_data["openAccessPdf"].get("url")
                    if pdf_url:
                        print(f"      Semantic Scholar found OA PDF: {pdf_url}")
                        return pdf_url
                else:
                    print(f"      Semantic Scholar: No Open Access PDF link available.")
            else:
                print(f"      Semantic Scholar: No definitive match found.")
            return None # Success, but no match/PDF found

        except requests.exceptions.RequestException as e:
            print(f"      Semantic Scholar API error (attempt {attempt+1}): {e}")
            if attempt >= retries: return None
            time.sleep(SLEEP_TIME_RETRY)
        except json.JSONDecodeError:
            print(f"      Semantic Scholar returned non-JSON response.")
            return None
        except Exception as e:
            print(f"      Unexpected Semantic Scholar error: {e}")
            return None
    return None

def safe_year(p):
    """Safely extracts publication year, returns 0 on failure."""
    year_str = p.get("bib", {}).get("pub_year", "0")
    try:
        return int(year_str)
    except (ValueError, TypeError):
        return 0

# --- Main Execution (Updated Error Handling) ---
if not user_ids:
    print("No user IDs loaded. Exiting.")
    sys.exit(1)

for user_id in user_ids:
    print(f"\nProcessing researcher: {user_id}")
    try:
        # Use proxy if needed, configure scholarly for robustness
        # scholarly.use_proxy(http="your_proxy", https="your_proxy")
        scholarly.set_retries(5)
        scholarly.set_sleep_interval(SLEEP_TIME_SCHOLARLY)

        author = scholarly.search_author_id(user_id)
        print(f"  Fetching publications for: {author.get('name', 'Unknown Name')}")
        author = scholarly.fill(author, sections=["publications"], sortby="year", publication_limit=PUBLICATION_LIMIT)
        time.sleep(SLEEP_TIME_SCHOLARLY)

        publications = author.get("publications", [])[:PUBLICATION_LIMIT]
        print(f"  Checking {len(publications)} most recent publications (limit: {PUBLICATION_LIMIT}).")

        for i, pub_summary in enumerate(publications):
            pdf_found = False
            print(f"\n  Checking publication {i+1}/{len(publications)}...")

            try:
                # Attempt to fill the publication summary
                print(f"    Fetching details for: {pub_summary.get('bib',{}).get('title','[title unavailable]')}")
                pub_filled = scholarly.fill(pub_summary)
                time.sleep(SLEEP_TIME_SCHOLARLY) # Pause after filling pub details

                bib = pub_filled.get("bib", {})
                title = bib.get("title", f"untitled_{i}")
                print(f"    Title: {title}")
                authors = bib.get("author", "").split(' and ')

                # --- PDF Finding Strategy (Retry logic within helpers now) ---
                sources_tried = []
                pdf_url = None

                # 1. Check eprint_url
                sources_tried.append("eprint_url")
                eprint_url = pub_filled.get("eprint_url")
                if eprint_url:
                    print(f"    Found eprint_url: {eprint_url}")
                    if download_pdf(eprint_url, title, user_id):
                        pdf_found = True

                # 2. Try Unpaywall via DOI
                doi = None
                if not pdf_found:
                    sources_tried.append("DOI/Unpaywall")
                    pub_url_gs = pub_filled.get("pub_url", "") # URL from Google Scholar
                    doi = extract_doi_from_url(pub_url_gs)
                    if doi:
                        pdf_url = get_pdf_from_unpaywall(doi)
                        time.sleep(SLEEP_TIME_API)
                        if pdf_url and download_pdf(pdf_url, title, user_id):
                             pdf_found = True
                    else:
                        print("      No DOI found in Google Scholar pub_url.")

                # 3. Try Semantic Scholar
                if not pdf_found:
                    sources_tried.append("Semantic Scholar")
                    pdf_url = search_semantic_scholar(title, doi) # Use DOI if found earlier
                    time.sleep(SLEEP_TIME_API)
                    if pdf_url and download_pdf(pdf_url, title, user_id):
                        pdf_found = True

                # 4. Try arXiv Search
                if not pdf_found:
                     sources_tried.append("arXiv")
                     pdf_url = search_arxiv(title, authors)
                     time.sleep(SLEEP_TIME_API)
                     if pdf_url and download_pdf(pdf_url, title, user_id):
                         pdf_found = True

                # --- End PDF Finding ---

                if not pdf_found:
                    print(f"    ----> No verified PDF found for '{title}' after trying: {', '.join(sources_tried)}")

            except MaxTriesExceededException:
                print(f"    ERROR: Scholarly max tries exceeded for publication {i+1}. Skipping.")
                time.sleep(SLEEP_TIME_SCHOLARLY * 5) # Longer sleep after hitting limit
            except TimeoutException:
                 print(f"    ERROR: Scholarly timed out for publication {i+1}. Skipping.")
                 time.sleep(SLEEP_TIME_SCHOLARLY * 2)
            except Exception as e:
                print(f"    ERROR processing publication {i+1}: {e}")
                time.sleep(SLEEP_TIME_SCHOLARLY * 2)

    except MaxTriesExceededException:
         print(f"  FAILED: Scholarly max tries exceeded for author {user_id}. Skipping author.")
         time.sleep(SLEEP_TIME_SCHOLARLY * 10) # Long sleep
    except TimeoutException:
         print(f"  FAILED: Scholarly timed out for author {user_id}. Skipping author.")
         time.sleep(SLEEP_TIME_SCHOLARLY * 5)
    except Exception as e:
        print(f"  FAILED to process user {user_id}: {e}")
        time.sleep(SLEEP_TIME_SCHOLARLY * 5)

print("\nScript finished.")