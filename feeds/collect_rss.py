import os
import time
import xml.etree.ElementTree as ET
from urllib.parse import urlparse, urljoin
from utils import io
import httpx

def run(cfg, ctx):
    """
    Collect RSS/Atom feeds from configured URLs.
    
    Configuration:
    feeds:
      urls: ["http://example.com/feed.xml", "http://example.com/rss"]
      max_entries: 50
      timeout: 30
    """
    
    feeds_config = cfg.get("feeds", {})
    feed_urls = feeds_config.get("urls", [])
    max_entries = feeds_config.get("max_entries", 50)
    timeout = feeds_config.get("timeout", 30)
    
    if not feed_urls:
        return {"status": "SKIPPED", "reason": "NO_FEED_URLS"}
    
    out = os.path.join(ctx["outdir"], "feeds")
    io.ensure_dir(out)
    
    collected_feeds = []
    total_entries = 0
    
    logger = ctx.get("logger")
    
    for feed_url in feed_urls[:10]:  # Limit to 10 feeds for safety
        try:
            if logger:
                logger.info(f"Collecting feed: {feed_url}")
            
            feed_data = _fetch_feed(feed_url, timeout)
            if feed_data:
                # Save raw feed XML
                domain = urlparse(feed_url).netloc or "unknown"
                filename = f"{domain}_feed.xml"
                file_path = os.path.join(out, filename)
                io.write_text(file_path, feed_data["raw_content"])
                
                # Parse and extract structured data
                parsed_data = _parse_feed(feed_data["raw_content"], feed_url, max_entries)
                if parsed_data:
                    # Save parsed data
                    json_filename = f"{domain}_feed.json"
                    json_path = os.path.join(out, json_filename)
                    io.write_json(json_path, parsed_data)
                    
                    collected_feeds.append({
                        "url": feed_url,
                        "domain": domain,
                        "xml_file": filename,
                        "json_file": json_filename,
                        "entries": len(parsed_data.get("entries", []))
                    })
                    total_entries += len(parsed_data.get("entries", []))
            
            # Rate limiting
            time.sleep(1)
            
        except Exception as e:
            if logger:
                logger.warning(f"Failed to collect feed {feed_url}: {e}")
            continue
    
    return {
        "status": "OK",
        "feeds_collected": len(collected_feeds),
        "total_entries": total_entries,
        "feeds": collected_feeds,
        "output_dir": out
    }

def _fetch_feed(url, timeout):
    """Fetch RSS/Atom feed content."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; FeedBot/1.0)"
        }
        
        with httpx.Client(timeout=timeout) as client:
            response = client.get(url, headers=headers, follow_redirects=True)
            response.raise_for_status()
            
            content_type = response.headers.get("content-type", "").lower()
            if not any(t in content_type for t in ["xml", "rss", "atom", "feed"]):
                # Try anyway, might be misconfigured server
                pass
            
            return {
                "raw_content": response.text,
                "headers": dict(response.headers)
            }
            
    except Exception as e:
        return None

def _parse_feed(content, feed_url, max_entries):
    """Parse RSS/Atom feed content."""
    try:
        root = ET.fromstring(content)
        
        # Detect feed type
        if root.tag == "rss":
            return _parse_rss(root, feed_url, max_entries)
        elif root.tag.endswith("feed"):  # Atom feeds
            return _parse_atom(root, feed_url, max_entries)
        else:
            # Try to find channel or feed elements
            channel = root.find(".//channel")
            if channel is not None:
                return _parse_rss_channel(channel, feed_url, max_entries)
            
            # Check for Atom feed elements
            if root.tag.endswith("feed") or root.find(".//{http://www.w3.org/2005/Atom}feed") is not None:
                return _parse_atom(root, feed_url, max_entries)
        
        return None
        
    except Exception as e:
        return None

def _parse_rss(root, feed_url, max_entries):
    """Parse RSS format feed."""
    channel = root.find("channel")
    if channel is None:
        return None
    
    return _parse_rss_channel(channel, feed_url, max_entries)

def _parse_rss_channel(channel, feed_url, max_entries):
    """Parse RSS channel element."""
    feed_data = {
        "type": "RSS",
        "source_url": feed_url,
        "title": _get_text(channel, "title"),
        "description": _get_text(channel, "description"),
        "link": _get_text(channel, "link"),
        "entries": []
    }
    
    items = channel.findall("item")[:max_entries]
    
    for item in items:
        entry = {
            "title": _get_text(item, "title"),
            "link": _get_text(item, "link"),
            "description": _get_text(item, "description"),
            "pub_date": _get_text(item, "pubDate"),
            "guid": _get_text(item, "guid"),
            "categories": [cat.text for cat in item.findall("category") if cat.text]
        }
        
        # Extract any email addresses or contact info from description
        if entry["description"]:
            entry["extracted_emails"] = _extract_emails(entry["description"])
            entry["extracted_phones"] = _extract_phones(entry["description"])
        
        feed_data["entries"].append(entry)
    
    return feed_data

def _parse_atom(root, feed_url, max_entries):
    """Parse Atom format feed."""
    # Handle namespace
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    
    feed_data = {
        "type": "Atom",
        "source_url": feed_url,
        "title": _get_text_ns(root, "atom:title", ns),
        "description": _get_text_ns(root, "atom:subtitle", ns),
        "link": None,
        "entries": []
    }
    
    # Get feed link
    link_elem = root.find("atom:link[@rel='alternate']", ns)
    if link_elem is not None:
        feed_data["link"] = link_elem.get("href")
    
    entries = root.findall("atom:entry", ns)[:max_entries]
    
    for entry in entries:
        entry_data = {
            "title": _get_text_ns(entry, "atom:title", ns),
            "link": None,
            "description": _get_text_ns(entry, "atom:content", ns) or _get_text_ns(entry, "atom:summary", ns),
            "pub_date": _get_text_ns(entry, "atom:published", ns) or _get_text_ns(entry, "atom:updated", ns),
            "guid": _get_text_ns(entry, "atom:id", ns),
            "categories": []
        }
        
        # Get entry link
        link_elem = entry.find("atom:link[@rel='alternate']", ns)
        if link_elem is not None:
            entry_data["link"] = link_elem.get("href")
        
        # Get categories
        category_elems = entry.findall("atom:category", ns)
        for cat in category_elems:
            term = cat.get("term")
            if term:
                entry_data["categories"].append(term)
        
        # Extract contact info from content
        if entry_data["description"]:
            entry_data["extracted_emails"] = _extract_emails(entry_data["description"])
            entry_data["extracted_phones"] = _extract_phones(entry_data["description"])
        
        feed_data["entries"].append(entry_data)
    
    return feed_data

def _get_text(element, tag):
    """Get text content from XML element."""
    child = element.find(tag)
    return child.text if child is not None and child.text else ""

def _get_text_ns(element, xpath, namespaces):
    """Get text content using namespace."""
    child = element.find(xpath, namespaces)
    return child.text if child is not None and child.text else ""

def _extract_emails(text):
    """Extract email addresses from text."""
    import re
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.findall(email_pattern, text)

def _extract_phones(text):
    """Extract phone numbers from text."""
    import re
    phone_pattern = r'(?:\+33|0)[1-9](?:[.\-\s]?\d{2}){4}'
    return re.findall(phone_pattern, text)
