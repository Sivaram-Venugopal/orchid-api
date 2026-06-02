import os
import json
import logging
import requests
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tle_cache.json")

ACTIVE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
DEBRIS_URL = "https://celestrak.org/NORAD/elements/gp.php?INTDES=1999-025&FORMAT=tle"

# Default presets to fall back on if cache doesn't exist and network fails
DEFAULT_PRESETS = {
    "25544": {
        "name": "ISS (ZARYA)",
        "line1": "1 25544U 98067A   24143.57865230  .00014290  00000-0  25464-3 0  9993",
        "line2": "2 25544  51.6397 122.3325 0004543  89.8708 349.5041 15.49969245455207",
        "type": "active",
        "timestamp": datetime.now(timezone.utc).isoformat()
    },
    "36248": {
        "name": "COSMOS 2251 DEBRIS",
        "line1": "1 36248U 93036AE  24143.20814986  .00010996  00000-0  18195-3 0  9993",
        "line2": "2 36248  74.0322  91.8023 0019253 234.3411 125.6425 15.00681340798224",
        "type": "debris",
        "timestamp": datetime.now(timezone.utc).isoformat()
    },
    "36249": {
        "name": "COSMOS 2251 DEBRIS",
        "line1": "1 36249U 93036AF  24143.48625902  .00012224  00000-0  21703-3 0  9999",
        "line2": "2 36249  74.0411  86.2912 0017112 219.0345 140.9234 15.04123518798993",
        "type": "debris",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
}

def parse_tle_text(text: str, catalog_type: str) -> dict:
    """Parses bulk 3-line TLE format text into structured dict."""
    catalog = {}
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    i = 0
    timestamp = datetime.now(timezone.utc).isoformat()
    while i + 2 < len(lines):
        name = lines[i]
        line1 = lines[i+1]
        line2 = lines[i+2]
        
        # Validate that lines look like TLE lines
        if line1.startswith("1 ") and line2.startswith("2 "):
            norad_id = line1[2:7].strip()
            catalog[norad_id] = {
                "name": name,
                "line1": line1,
                "line2": line2,
                "type": catalog_type,
                "timestamp": timestamp
            }
        i += 3
    return catalog

def fetch_and_cache_tle() -> dict:
    """Fetches TLE catalogs from CelesTrak and stores them in local cache."""
    logger.info("Attempting to fetch active and debris catalogs from CelesTrak...")
    try:
        # Fetch Active
        r_active = requests.get(ACTIVE_URL, timeout=15)
        r_active.raise_for_status()
        active_catalog = parse_tle_text(r_active.text, "active")
        
        # Fetch Debris
        r_debris = requests.get(DEBRIS_URL, timeout=15)
        r_debris.raise_for_status()
        debris_catalog = parse_tle_text(r_debris.text, "debris")
        
        # Combine
        combined = {**active_catalog, **debris_catalog}
        
        # Save to cache
        with open(CACHE_FILE, "w") as f:
            json.dump(combined, f, indent=2)
            
        logger.info(f"Successfully cached {len(combined)} TLE entries (Active: {len(active_catalog)}, Debris: {len(debris_catalog)}).")
        return combined
    except Exception as e:
        logger.error(f"Failed to fetch TLE catalogs: {e}")
        # If cache file exists, load and return it (even if expired)
        if os.path.exists(CACHE_FILE):
            logger.info("Loading expired TLE cache as fallback.")
            try:
                with open(CACHE_FILE, "r") as f:
                    return json.load(f)
            except Exception as read_err:
                logger.error(f"Failed to read local cache file: {read_err}")
        
        # If no cache exists, use hardcoded defaults
        logger.warning("No cache file found. Falling back to default presets.")
        return DEFAULT_PRESETS

def load_tle_catalog(force_refresh: bool = False) -> dict:
    """Loads the TLE catalog, fetching from CelesTrak if expired/missing."""
    if not force_refresh and os.path.exists(CACHE_FILE):
        # Check modification time
        mtime = os.path.getmtime(CACHE_FILE)
        cache_age = datetime.now().timestamp() - mtime
        if cache_age < 86400:  # 24 hours
            logger.info("Loading TLE catalog from local cache (under 24 hours old).")
            try:
                with open(CACHE_FILE, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading TLE cache: {e}. Re-fetching...")
                
    return fetch_and_cache_tle()
