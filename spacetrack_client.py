import os
import requests
import logging

logger = logging.getLogger(__name__)

class SpaceTrackClient:
    """
    Authenticated client to query the official JSpOC Space-Track.org catalog.
    Falls back to CelesTrak General Perturbations (GP) API if credentials are not set.
    """
    def __init__(self):
        self.session = requests.Session()
        self.identity = os.getenv("SPACETRACK_IDENTITY")
        self.password = os.getenv("SPACETRACK_PASSWORD")
        self.authenticated = False
        
    def authenticate(self) -> bool:
        if not self.identity or not self.password:
            logger.warning("[Space-Track API] Credentials not configured. Using CelesTrak fallback.")
            return False
            
        try:
            url = "https://www.space-track.org/ajaxauth/login"
            r = self.session.post(url, data={
                "identity": self.identity,
                "password": self.password
            }, timeout=15)
            r.raise_for_status()
            
            if "Incorrect" in r.text or "Error" in r.text:
                logger.error("[Space-Track API] Authentication rejected: incorrect identity or password.")
                return False
                
            self.authenticated = True
            logger.info("[Space-Track API] Authenticated successfully with JSpOC servers.")
            return True
        except Exception as e:
            logger.error(f"[Space-Track API] Authentication network error: {e}")
            return False
            
    def fetch_tle(self, norad_id: str) -> dict:
        """Queries Space-Track for TLE lines, falling back to CelesTrak if unauthenticated."""
        if not self.authenticated:
            self.authenticate()
            
        if self.authenticated:
            try:
                # Query latest TLE epoch for the NORAD ID
                url = f"https://www.space-track.org/basicquery/class/tle/NORAD_CAT_ID/{norad_id}/orderby/EPOCH%20desc/limit/1/format/tle"
                r = self.session.get(url, timeout=12)
                r.raise_for_status()
                
                lines = [line.strip() for line in r.text.splitlines() if line.strip()]
                if len(lines) >= 2:
                    # Space-track raw TLE does not include the name line in standard 'format/tle', 
                    # so we fetch name from basic query or use defaults
                    return {
                        "name": f"NORAD {norad_id}",
                        "line1": lines[0],
                        "line2": lines[1]
                    }
            except Exception as e:
                logger.error(f"[Space-Track API] Query failed for NORAD {norad_id}: {e}. Falling back.")
                
        # CelesTrak fallback
        return self.fetch_celestrak_fallback(norad_id)

    def fetch_celestrak_fallback(self, norad_id: str) -> dict:
        url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=tle"
        try:
            r = requests.get(url, timeout=12)
            if r.status_code == 200:
                lines = [line.strip() for line in r.text.splitlines() if line.strip()]
                if len(lines) >= 3:
                    return {
                        "name": lines[0],
                        "line1": lines[1],
                        "line2": lines[2]
                    }
        except Exception as e:
            logger.error(f"[Space-Track API] Failed to fetch TLE fallback from CelesTrak for {norad_id}: {e}")
        return None
