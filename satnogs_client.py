import os
import requests
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class SatNOGSClient:
    """
    Client for the SatNOGS Network API.
    Fetches real-time ground station directory data and satellite tracking observations.
    Falls back to high-fidelity simulation mode if credentials or API is offline.
    """
    def __init__(self):
        self.api_url = "https://network.satnogs.org/api/v2"
        # Optional token, can run unauthenticated for basic read operations
        self.api_token = os.getenv("SATNOGS_API_TOKEN")
        self.headers = {}
        if self.api_token:
            self.headers["Authorization"] = f"Token {self.api_token}"

    def get_ground_stations(self) -> List[Dict[str, Any]]:
        """
        Fetches the active SatNOGS ground station network list.
        Returns station locations, statuses, and coordinates.
        """
        try:
            url = f"{self.api_url}/stations/"
            # Limit page size to prevent downloading the entire database (thousands of stations)
            r = requests.get(url, headers=self.headers, params={"status": "online", "limit": 50}, timeout=10)
            if r.status_code == 200:
                stations = r.json()
                logger.info(f"[SatNOGS API] Retrieved {len(stations)} online ground stations.")
                return [
                    {
                        "station_id": str(s.get("id")),
                        "name": s.get("name", f"Station {s.get('id')}")[:30],
                        "lat": float(s.get("lat", 0.0)) if s.get("lat") else 0.0,
                        "lng": float(s.get("lng", 0.0)) if s.get("lng") else 0.0,
                        "alt_m": float(s.get("altitude", 100.0)) if s.get("altitude") else 100.0,
                        "status": "ONLINE"
                    }
                    for s in stations if s.get("lat") and s.get("lng")
                ]
        except Exception as e:
            logger.error(f"[SatNOGS API] Failed to fetch ground stations: {e}. Loading default global grid.")
            
        # Return default global preset grid if API fails (diverse geographic locations)
        return [
            {"station_id": "ST-1", "name": "SatNOGS Tokyo Station", "lat": 35.6762, "lng": 139.6503, "alt_m": 40.0, "status": "ONLINE"},
            {"station_id": "ST-2", "name": "SatNOGS Munich Station", "lat": 48.1351, "lng": 11.5820, "alt_m": 520.0, "status": "ONLINE"},
            {"station_id": "ST-3", "name": "SatNOGS Boston Station", "lat": 42.3601, "lng": -71.0589, "alt_m": 10.0, "status": "ONLINE"},
            {"station_id": "ST-4", "name": "SatNOGS Sydney Station", "lat": -33.8688, "lng": 151.2093, "alt_m": 25.0, "status": "ONLINE"},
            {"station_id": "ST-5", "name": "SatNOGS Cape Town Station", "lat": -33.9249, "lng": 18.4241, "alt_m": 15.0, "status": "ONLINE"}
        ]

    def get_observations(self, norad_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves recent telemetry observations for a satellite from SatNOGS.
        Generates simulated measurements if none exist in the last 24h.
        """
        try:
            url = f"{self.api_url}/observations/"
            params = {
                "satellite__norad_cat_id": norad_id,
                "status": "good",
                "limit": 10
            }
            r = requests.get(url, headers=self.headers, params=params, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if data:
                    logger.info(f"[SatNOGS API] Fetched {len(data)} observations for NORAD {norad_id}.")
                    return [
                        {
                            "observation_id": str(obs.get("id")),
                            "station_id": str(obs.get("ground_station")),
                            "timestamp": obs.get("start", datetime.now(timezone.utc).isoformat()),
                            "frequency_hz": float(obs.get("frequency", 437.5e6)), # Default UHF amateur band
                            "doppler_shift_hz": 0.0 # Will be populated by Doppler corrector engine
                        }
                        for obs in data
                    ]
        except Exception as e:
            logger.error(f"[SatNOGS API] Failed to query observations for {norad_id}: {e}")
            
        # Fallback simulated observations generator
        now = datetime.now(timezone.utc)
        simulated = []
        for i in range(3):
            obs_time = now - timedelta(hours=i * 4)
            simulated.append({
                "observation_id": f"OBS-SIM-{norad_id}-{i}",
                "station_id": f"ST-{(i%5)+1}",
                "timestamp": obs_time.isoformat(),
                "frequency_hz": 437.5e6, # 437.5 MHz UHF downlink
                "doppler_shift_hz": 0.0 # To be calculated by the Doppler Engine
            })
        return simulated
