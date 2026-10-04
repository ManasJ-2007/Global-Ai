"""arXiv search tool for the Research Agent.

Follows the arXiv API Terms of Use:
  * at most one request every 3 seconds, one connection at a time
  * metadata only (title, abstract, authors, ids); never stores or serves PDFs
  * link to the abstract page, and acknowledge arXiv in the final report

Every returned paper has a real arXiv id and URL, so the agent can only cite
what was actually retrieved.
"""
import json
import os
import threading
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

API_URL = "https://export.arxiv.org/api/query"
MIN_INTERVAL_S = 3.0  # arXiv rate limit
CACHE_PATH = os.environ.get("ARXIV_CACHE", "lab_data/arxiv_cache.json")
ACK = "Thank you to arXiv for use of its open access interoperability."

_NS = {"a": "http://www.w3.org/2005/Atom"}
_lock = threading.Lock()  # one connection at a time, even if agents run in parallel
_last_call = 0.0


def _load_cache():
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_PATH) or ".", exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=1)


def parse_feed(xml_text):
    """Parse an arXiv Atom feed into a list of paper dicts."""
    root = ET.fromstring(xml_text)
    papers = []
    for e in root.findall("a:entry", _NS):
        url = (e.findtext("a:id", "", _NS) or "").strip()
        if not url or "/api/errors" in url:  # arXiv reports errors as an entry
            continue
        papers.append({
            "arxiv_id": url.rsplit("/abs/", 1)[-1],
            "title": " ".join((e.findtext("a:title", "", _NS) or "").split()),
            "authors": [a.findtext("a:name", "", _NS) for a in e.findall("a:author", _NS)],
            "published": (e.findtext("a:published", "", _NS) or "")[:10],
            "abstract": " ".join((e.findtext("a:summary", "", _NS) or "").split()),
            "url": url,  # abstract page
        })
    return papers


def search_arxiv(query: str, max_results: int = 5, sort_by: str = "relevance") -> dict:
    """Search arXiv metadata. `query` uses arXiv syntax, e.g.
    'all:"bin packing" AND all:heuristic' or 'ti:FunSearch'.
    Returns {"query", "papers":[...], "acknowledgement"} or {"error": ...}."""
    global _last_call
    max_results = max(1, min(int(max_results), 10))
    key = f"{query}|{max_results}|{sort_by}"

    cache = _load_cache()
    if key in cache:
        return {"query": query, "papers": cache[key], "cached": True, "acknowledgement": ACK}

    params = urllib.parse.urlencode({
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": sort_by,  # relevance | lastUpdatedDate | submittedDate
        "sortOrder": "descending",
    })
    with _lock:
        wait = MIN_INTERVAL_S - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        try:
            req = urllib.request.Request(
                f"{API_URL}?{params}", headers={"User-Agent": "bin-packing-lab/0.1 (research)"}
            )
            with urllib.request.urlopen(req, timeout=30) as r:
                xml_text = r.read().decode("utf-8")
        except Exception as exc:  # network down, rate limited, etc.
            return {"error": f"arXiv request failed: {exc}", "papers": []}
        finally:
            _last_call = time.monotonic()

    try:
        papers = parse_feed(xml_text)
    except ET.ParseError as exc:
        return {"error": f"could not parse arXiv response: {exc}", "papers": []}

    cache[key] = papers
    _save_cache(cache)
    return {"query": query, "papers": papers, "cached": False, "acknowledgement": ACK}


def get_arxiv_paper(arxiv_id: str) -> dict:
    """Fetch metadata for one paper by id (e.g. '2106.07998')."""
    global _last_call
    cache = _load_cache()
    key = f"id:{arxiv_id}"
    if key in cache:
        return {"papers": cache[key], "cached": True, "acknowledgement": ACK}
    with _lock:
        wait = MIN_INTERVAL_S - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        try:
            url = f"{API_URL}?{urllib.parse.urlencode({'id_list': arxiv_id})}"
            with urllib.request.urlopen(url, timeout=30) as r:
                papers = parse_feed(r.read().decode("utf-8"))
        except Exception as exc:
            return {"error": f"arXiv request failed: {exc}", "papers": []}
        finally:
            _last_call = time.monotonic()
    cache[key] = papers
    _save_cache(cache)
    return {"papers": papers, "cached": False, "acknowledgement": ACK}
