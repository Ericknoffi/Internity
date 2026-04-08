import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.google.com/",
}

# Sites that reliably block scraping — tell user upfront
BLOCKED_SITES = ["linkedin.com", "glassdoor.com", "monster.com"]

# ── URL validation ────────────────────────────────────────────────────────────

def is_valid_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False

def get_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""

# ── Site-specific extractors ──────────────────────────────────────────────────

def scrape_indeed(soup: BeautifulSoup) -> str:
    # Indeed often injects content via JS — try multiple known containers
    selectors = [
        {"id": "jobDescriptionText"},
        {"class": "jobsearch-jobDescriptionText"},
        {"class": "job-description"},
        {"class": "jobsearch-JobComponent-description"},
    ]
    for sel in selectors:
        tag = soup.find("div", sel)
        if tag and len(tag.get_text(strip=True)) > 100:
            return tag.get_text(separator=" ", strip=True)
    return ""

def scrape_naukri(soup: BeautifulSoup) -> str:
    selectors = [
        {"class": "job-desc"},
        {"class": "dang-inner-html"},
        {"id": "job-description"},
        {"class": "jobDescriptionLabel"},
    ]
    for sel in selectors:
        tag = soup.find("div", sel)
        if tag and len(tag.get_text(strip=True)) > 100:
            return tag.get_text(separator=" ", strip=True)
    return ""

def scrape_simplyhired(soup: BeautifulSoup) -> str:
    tag = soup.find("div", {"class": "viewjob-description"})
    if tag:
        return tag.get_text(separator=" ", strip=True)
    return ""

def scrape_generic(soup: BeautifulSoup) -> str:
    """
    Tries common job description containers in order.
    Falls back to <main> or <article> if nothing specific found.
    """
    generic_selectors = [
        {"class": "job-description"},
        {"class": "jobDescription"},
        {"class": "description"},
        {"class": "posting-description"},
        {"class": "job-details"},
        {"id": "job-description"},
        {"id": "description"},
        {"id": "job-details"},
    ]
    for sel in generic_selectors:
        tag = soup.find("div", sel)
        if tag and len(tag.get_text(strip=True)) > 100:
            return tag.get_text(separator=" ", strip=True)

    # Try semantic HTML tags
    for tag_name in ["main", "article"]:
        tag = soup.find(tag_name)
        if tag and len(tag.get_text(strip=True)) > 100:
            return tag.get_text(separator=" ", strip=True)

    # Last resort: largest <div> by text length
    all_divs = soup.find_all("div")
    if all_divs:
        biggest = max(all_divs, key=lambda d: len(d.get_text(strip=True)))
        text = biggest.get_text(separator=" ", strip=True)
        if len(text) > 100:
            return text

    return ""

# ── Main scrape function ──────────────────────────────────────────────────────

def scrape_job_url(url: str) -> dict:
    """
    Scrapes job text from a URL.
    Returns:
        success  (bool)
        text     (str)
        source   (str)
        error    (str)  — only on failure
    """

    # Step 1: validate URL format
    if not is_valid_url(url):
        return {
            "success": False,
            "error": "Invalid URL. Please enter a full URL starting with http:// or https://"
        }

    # Step 2: block known anti-scrape sites early
    domain = get_domain(url)
    for blocked in BLOCKED_SITES:
        if blocked in domain:
            return {
                "success": False,
                "error": (
                    f"{blocked.split('.')[0].capitalize()} blocks automated access. "
                    "Please copy and paste the job description text manually."
                )
            }

    # Step 3: fetch page
    try:
        response = requests.get(url, headers=HEADERS, timeout=12)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Request timed out after 12 seconds. The site may be slow or blocking requests."
        }
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code
        if code == 403:
            return {"success": False, "error": "Access denied (403). This site blocks scraping. Please paste the text manually."}
        elif code == 404:
            return {"success": False, "error": "Job posting not found (404). The listing may have been removed."}
        else:
            return {"success": False, "error": f"HTTP error {code}. Please paste the text manually."}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Could not connect. Check the URL or your internet connection."}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Request failed: {str(e)}"}

    # Step 4: parse — try lxml first, fall back to html.parser
    try:
        soup = BeautifulSoup(response.text, "lxml")
    except Exception:
        soup = BeautifulSoup(response.text, "html.parser")

    # Remove noise
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
        tag.decompose()

    # Step 5: site-specific extraction
    if "indeed.com" in domain:
        text   = scrape_indeed(soup)
        source = "Indeed"
    elif "naukri.com" in domain:
        text   = scrape_naukri(soup)
        source = "Naukri"
    elif "simplyhired.com" in domain:
        text   = scrape_simplyhired(soup)
        source = "SimplyHired"
    else:
        text   = scrape_generic(soup)
        source = "Generic"

    # Step 6: check if JS-rendered (empty shell)
    if not text or len(text.strip()) < 100:
        return {
            "success": False,
            "error": (
                f"Could not extract job text from this {source} page. "
                "The site likely loads content via JavaScript. "
                "Please paste the job description text manually."
            )
        }

    return {
        "success": True,
        "text": text[:8000],
        "source": source
    }