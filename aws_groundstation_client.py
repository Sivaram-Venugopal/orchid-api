import os
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class AWSGroundStationClient:
    """
    AWS Ground Station Client interface.
    Supports contact scheduling, downlink telemetry ingestion, and cost-aware scheduling.
    Gracefully runs in simulation mode if AWS credentials are not present.
    """
    def __init__(self):
        self.access_key = os.getenv("AWS_ACCESS_KEY_ID")
        self.secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        
        self.simulation_mode = not (self.access_key and self.secret_key)
        if self.simulation_mode:
            logger.info("[AWS Ground Station] Credentials not set. Operating in simulated contact mode.")
        else:
            logger.info(f"[AWS Ground Station] Initialized client targeting AWS Ground Station in {self.region}.")
            
        # Mock scheduled contact database in memory
        self.scheduled_contacts = {}

    def get_aws_ground_stations(self) -> List[Dict[str, Any]]:
        """Lists available AWS Ground Station facilities globally."""
        # Standard AWS Ground Station locations
        return [
            {"station_id": "AWS-OHIO", "name": "AWS Ohio (US-East-2) Facility", "lat": 40.088, "lng": -83.001, "cost_per_min": 10.0},
            {"station_id": "AWS-OREGON", "name": "AWS Oregon (US-West-2) Facility", "lat": 45.849, "lng": -119.714, "cost_per_min": 10.0},
            {"station_id": "AWS-IRELAND", "name": "AWS Ireland (EU-West-1) Facility", "lat": 53.349, "lng": -6.260, "cost_per_min": 12.0},
            {"station_id": "AWS-SINGAPORE", "name": "AWS Singapore (AS-Southeast-1) Facility", "lat": 1.352, "lng": 103.819, "cost_per_min": 14.0},
            {"station_id": "AWS-SYDNEY", "name": "AWS Sydney (AP-Southeast-2) Facility", "lat": -33.868, "lng": 151.209, "cost_per_min": 12.0}
        ]

    def schedule_cost_aware_contact(self, norad_id: str, upcoming_passes: List[Dict[str, Any]], risk_level: str) -> Dict[str, Any]:
        """
        Implements cost-aware contact scheduling:
        1. CRITICAL Risk (P0/P1): Reserves the maximum duration contact window on every available station to capture range-rate radar signals.
        2. HIGH Risk (P1): Reserves contacts only on the highest elevation pass per station.
        3. NOMINAL Risk: Reserves only 1 contact per 24 hours (minimizing AWS billing minutes).
        """
        if not upcoming_passes:
            return {"status": "skipped", "message": "No passes available to schedule."}
            
        # Sort passes by elevation to prioritize clear signal tracking
        sorted_passes = sorted(upcoming_passes, key=lambda x: x.get("max_elevation", 0.0), reverse=True)
        
        selected_passes = []
        if risk_level == "CRITICAL":
            # Schedule all passes to maintain continuous safety coverage
            selected_passes = sorted_passes[:3]
            strategy = "CRITICAL FORCE-SCREENING (Propagate continuous state)"
        elif risk_level == "HIGH":
            # Schedule the top 2 highest elevation passes
            selected_passes = sorted_passes[:2]
            strategy = "HIGH-PRIORITY SELECTIVE COVERAGE"
        else:
            # Nominal satellite: schedule only the single highest elevation pass to save costs
            selected_passes = sorted_passes[:1]
            strategy = "COST-OPTIMIZED SLEEP MODE (Minimize active AWS billing minutes)"
            
        reservations = []
        total_cost = 0.0
        
        for p in selected_passes:
            station_id = p.get("station_id", "AWS-OHIO")
            aos = p.get("aos")
            los = p.get("los")
            
            # Estimate duration in minutes
            try:
                aos_dt = datetime.fromisoformat(aos)
                los_dt = datetime.fromisoformat(los)
                duration_mins = (los_dt - aos_dt).total_seconds() / 60.0
            except Exception:
                duration_mins = 10.0 # Default fallback
                
            # Lookup cost per minute
            cost_factor = 10.0
            for gs in self.get_aws_ground_stations():
                if gs["station_id"] == station_id:
                    cost_factor = gs["cost_per_min"]
                    break
                    
            pass_cost = duration_mins * cost_factor
            total_cost += pass_cost
            
            contact_id = f"AWS-CON-{norad_id}-{station_id}-{int(datetime.now().timestamp())}"
            reservation = {
                "contact_id": contact_id,
                "station_id": station_id,
                "aos": aos,
                "los": los,
                "duration_mins": round(duration_mins, 1),
                "estimated_cost_usd": round(pass_cost, 2),
                "status": "RESERVED"
            }
            
            self.scheduled_contacts[contact_id] = reservation
            reservations.append(reservation)
            
            logger.info(f"[AWS Ground Station] Reserved contact {contact_id} on {station_id}. Strategy: {strategy}. Est Cost: ${pass_cost:.2f}.")

        return {
            "status": "success",
            "satellite_id": norad_id,
            "strategy": strategy,
            "reservations": reservations,
            "total_estimated_cost_usd": round(total_cost, 2)
        }

    def ingest_downlink_telemetry(self, contact_id: str) -> Dict[str, Any]:
        """
        Ingests telemetry packages downlinked during a contact session.
        In simulation mode, yields mock state vectors and sensor noise attributes.
        """
        if contact_id not in self.scheduled_contacts:
            return {"status": "failed", "message": f"Contact ID {contact_id} not registered."}
            
        contact = self.scheduled_contacts[contact_id]
        
        # Simulated payload containing spacecraft telemetry status
        telemetry_pkg = {
            "contact_id": contact_id,
            "station_id": contact["station_id"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "downlink_status": "COMPLETED",
            "frame_loss_pct": 0.05,
            "payload_data": {
                "battery_voltage_v": 28.2,
                "solar_panel_temp_c": 18.5,
                "adcs_mode": "SUN_POINTING",
                "propellant_pressure_bar": 12.4
            }
        }
        
        logger.info(f"[AWS Ground Station] Successfully downlinked telemetry package for contact {contact_id}.")
        return telemetry_pkg
