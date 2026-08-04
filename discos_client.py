import os
import requests
import logging

logger = logging.getLogger(__name__)

_dimensions_cache = {}
_logged_token_warning = False

class DiscosClient:
    """
    Client for the ESA DISCOS (Database and Information System Characterising Objects in Space) API.
    Retrieves satellite structural configurations for 3D projected CAD mesh calculations.
    """
    def __init__(self):
        global _logged_token_warning
        self.api_token = os.getenv("DISCOS_API_TOKEN")
        self.headers = {
            "Accept": "application/vnd.api+json"
        }
        if self.api_token:
            self.headers["Authorization"] = f"Bearer {self.api_token}"
        elif not _logged_token_warning:
            logger.info("[ESA DISCOS API] DISCOS_API_TOKEN environment variable not configured. Using default structural dimension presets.")
            _logged_token_warning = True
        
    def fetch_satellite_dimensions(self, norad_id: str) -> dict:
        """
        Queries ESA DISCOS database for spacecraft dry mass, shape, area, and array spans.
        Falls back to default parameters if unauthenticated or error occurs.
        """
        norad_id_str = str(norad_id)
        if norad_id_str in _dimensions_cache:
            return _dimensions_cache[norad_id_str]

        default_specs = {
            "norad_id": norad_id_str,
            "dry_mass_kg": 500.0,
            "cross_section_m2": 12.0,
            "shape": "box_with_panels",
            "bus_dimensions": [3.0, 2.0, 2.0],  # [length, width, height] meters
            "panel_dimensions": [1.0, 5.0],      # [width, span] meters
            "source": "ORCHID_Default_Presets"
        }
        
        if not self.api_token:
            _dimensions_cache[norad_id_str] = default_specs
            return default_specs
            
        try:
            # Query ESA DISCOS objects endpoint using the correct hostname and filter syntax
            url = f"https://discosweb.esoc.esa.int/api/objects?filter=eq(satno,{norad_id_str})&include=initial_orbit,reentry"
            r = requests.get(url, headers=self.headers, timeout=12)
            
            if r.status_code == 200:
                data = r.json()
                if "data" in data and len(data["data"]) > 0:
                    attributes = data["data"][0].get("attributes", {})
                    
                    dry_mass = attributes.get("mass", 500.0)
                    area = attributes.get("crossSection", 12.0)
                    span = attributes.get("span", 5.0)
                    
                    logger.info(f"[ESA DISCOS API] Successfully retrieved structural specs for NORAD {norad_id_str}.")
                    res = {
                        "norad_id": norad_id_str,
                        "dry_mass_kg": float(dry_mass) if dry_mass else 500.0,
                        "cross_section_m2": float(area) if area else 12.0,
                        "shape": "box_with_panels",
                        "bus_dimensions": [3.0, 2.0, 2.0],
                        "panel_dimensions": [1.0, float(span) / 2.0 if span else 2.5],
                        "source": "ESA_DISCOS_API"
                    }
                    _dimensions_cache[norad_id_str] = res
                    return res
            else:
                logger.warning(f"[ESA DISCOS API] Query for NORAD {norad_id_str} returned HTTP {r.status_code}. Using default presets.")
        except Exception as e:
            logger.error(f"[ESA DISCOS API] Query failed for NORAD {norad_id_str}: {e}")
            
        _dimensions_cache[norad_id_str] = default_specs
        return default_specs
