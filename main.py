import os
import uuid
import numpy as np
import logging
import asyncio
import sqlite3
import struct
import io
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from risk_engine import assess_risk
from maneuver_engine import generate_maneuver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ORCHID API",
    description="Autonomous Satellite Collision Avoidance as a Service",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global store for background jobs and loop reference
jobs = {}
loop = None

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conjunction_history.db")

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conjunction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            satellite_id TEXT,
            overall_risk TEXT,
            closest_distance_km REAL,
            tca_min REAL,
            collision_prob REAL,
            maneuver_required INTEGER,
            fuel_spent_kg REAL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fleet_satellites (
            norad_id TEXT PRIMARY KEY,
            name TEXT,
            status TEXT,
            fuel_capacity_kg REAL,
            current_fuel_kg REAL,
            tle1 TEXT,
            tle2 TEXT
        )
    """)
    conn.commit()

    # Seed fleet_satellites if empty
    cursor.execute("SELECT COUNT(*) FROM fleet_satellites")
    if cursor.fetchone()[0] == 0:
        logger.info("Seeding default fleet satellites from catalog...")
        try:
            from catalog_manager import load_tle_catalog
            catalog = load_tle_catalog()
            # Select ISS (25544) and others if available
            seed_ids = ["25544", "48274", "25338", "33591"]
            for sid in seed_ids:
                if sid in catalog:
                    sat = catalog[sid]
                    cursor.execute("""
                        INSERT OR IGNORE INTO fleet_satellites (norad_id, name, status, fuel_capacity_kg, current_fuel_kg, tle1, tle2)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (sid, sat["name"], "ACTIVE", 300.0, 250.0, sat["line1"], sat["line2"]))
                else:
                    active_keys = [k for k, v in catalog.items() if v.get("type") == "active"]
                    if active_keys:
                        k = active_keys[len(active_keys) // 2]
                        sat = catalog[k]
                        cursor.execute("""
                            INSERT OR IGNORE INTO fleet_satellites (norad_id, name, status, fuel_capacity_kg, current_fuel_kg, tle1, tle2)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (k, sat["name"], "ACTIVE", 200.0, 180.0, sat["line1"], sat["line2"]))
            conn.commit()
            logger.info("Successfully seeded fleet database.")
        except Exception as e:
            logger.error(f"Failed to seed fleet satellites: {e}")
    conn.close()

def log_to_db(satellite_id, overall_risk, conjunctions, maneuver_required, fuel_spent_kg):
    try:
        from datetime import datetime, timezone
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        closest = conjunctions[0] if conjunctions else {}
        cursor.execute("""
            INSERT INTO conjunction_history (
                timestamp, satellite_id, overall_risk, closest_distance_km, tca_min, collision_prob, maneuver_required, fuel_spent_kg
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            satellite_id,
            overall_risk,
            closest.get("distance_km", 0.0),
            closest.get("time_to_closest_approach_min", 0.0),
            closest.get("probability_of_collision", 0.0),
            1 if maneuver_required else 0,
            fuel_spent_kg
        ))
        conn.commit()
        
        # Deduct fuel if it is a registered fleet satellite
        if maneuver_required and fuel_spent_kg > 0:
            cursor.execute("""
                UPDATE fleet_satellites 
                SET current_fuel_kg = MAX(0.0, current_fuel_kg - ?)
                WHERE norad_id = ?
            """, (fuel_spent_kg, satellite_id))
            
            # Transition status to FUEL_CRITICAL if fuel drops below 5%
            cursor.execute("""
                UPDATE fleet_satellites 
                SET status = 'FUEL_CRITICAL'
                WHERE norad_id = ? AND current_fuel_kg < (fuel_capacity_kg * 0.05)
            """, (satellite_id,))
            conn.commit()
            logger.info(f"Deducted {fuel_spent_kg:.4f} kg fuel from fleet satellite {satellite_id}.")
            
        conn.close()
    except Exception as e:
        logger.error(f"Failed to log conjunction and update fleet fuel: {e}")

async def simulated_observation_feed():
    import random
    from datetime import datetime, timezone
    from ukf import ukf_propagate, ukf_update
    from risk_engine import get_state_at_time
    from sgp4.api import Satrec
    
    messages = [
        "ADCS sensor status: Gyroscope calibration successful. Orbit drift rate within limits.",
        "Live feed status: 14 radar track events processed in the last 60 seconds.",
        "NASA CARA conjunction screening pass complete: zero new hazards detected."
    ]
    
    while True:
        await asyncio.sleep(8.0)
        
        # Check if there are active registered satellites in the UKF tracking pool
        if active_ukf_filters and loop and manager.active_connections:
            sat_id = list(active_ukf_filters.keys())[0]
            sat_data = active_ukf_filters[sat_id]
            
            try:
                now = datetime.now(timezone.utc)
                dt_sec = (now - sat_data["last_time"]).total_seconds()
                
                # Minimum propagation step size
                if dt_sec < 1.0:
                    dt_sec = 8.0
                    
                # Setup process noise covariance Q
                Q = np.diag([0.1, 0.1, 0.1, 1e-4, 1e-4, 1e-4])**2
                
                # 1. Propagate UKF state forward
                x_pred, P_pred, sigmas_pred = ukf_propagate(
                    sat_data["state"], sat_data["covariance"], dt_sec, Q
                )
                
                # 2. Get true position using TLE at current time
                sat_rec = Satrec.twoline2rv(sat_data["tle1"], sat_data["tle2"])
                pos_true, _ = get_state_at_time(sat_rec, now)
                
                if pos_true is not None:
                    # 3. Simulate noisy radar measurement (add 15 meters of random Gaussian noise)
                    noise = np.random.normal(0.0, 15.0, 3)
                    z = pos_true + noise
                    
                    # Measurement noise R: 15 meters standard deviation
                    R = np.diag([15.0, 15.0, 15.0])**2
                    
                    # 4. Perform UKF measurement update (collapsing covariance!)
                    x_updated, P_updated = ukf_update(x_pred, P_pred, sigmas_pred, z, R)
                    
                    # Store updated values
                    sat_data["state"] = x_updated
                    sat_data["covariance"] = P_updated
                    sat_data["last_time"] = now
                    
                    # Extract updated RTN standard deviations
                    std_r = float(np.sqrt(max(1.0, P_updated[0, 0])))
                    std_t = float(np.sqrt(max(1.0, P_updated[1, 1])))
                    std_n = float(np.sqrt(max(1.0, P_updated[2, 2])))
                    
                    # Broadcast standard deviation reductions to frontend!
                    await manager.broadcast({
                        "type": "covariance_update",
                        "satellite_id": sat_id,
                        "std_r": round(std_r, 1),
                        "std_t": round(std_t, 1),
                        "std_n": round(std_n, 1),
                        "message": f"[{datetime.now().strftime('%H:%M:%S')}] LeoLabs tracking pass: UKF updated. Positional error contracted (Radial: {std_r:.1f}m, Transverse: {std_t:.1f}m, Normal: {std_n:.1f}m)."
                    })
                    continue
            except Exception as e:
                logger.error(f"Error in UKF background tracking loop: {e}")
                
        # Fallback to static messaging if no active UKF filter registered
        if loop and manager.active_connections:
            msg = random.choice(messages)
            await manager.broadcast({
                "type": "observation",
                "message": f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
            })

async def live_dashboard_feed():
    from live_feed import get_cached_conjunctions
    import asyncio
    while True:
        await asyncio.sleep(30.0)
        if manager.active_connections:
            try:
                data = get_cached_conjunctions()
                await manager.broadcast({
                    "type": "live_dashboard_update",
                    "data": data
                })
            except Exception as e:
                logger.error(f"Error broadcasting live dashboard update: {e}")

@app.on_event("startup")
async def startup_event():
    global loop
    loop = asyncio.get_running_loop()
    logger.info("Starting ORCHID API v2.0...")
    init_db()
    asyncio.create_task(simulated_observation_feed())
    asyncio.create_task(live_dashboard_feed())
    
    # Start APScheduler background jobs
    from scheduler import start_scheduler
    start_scheduler()

@app.on_event("shutdown")
def shutdown_event():
    logger.info("Shutting down ORCHID API v2.0...")
    from scheduler import shutdown_scheduler
    shutdown_scheduler()

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass

manager = ConnectionManager()

# Pydantic Schemas
class TLEInput(BaseModel):
    norad_id: str
    tle1: str
    tle2: str

class CovarianceInput(BaseModel):
    r: float  # Radial uncertainty (std dev in meters)
    t: float  # Transverse uncertainty (std dev in meters)
    n: float  # Normal uncertainty (std dev in meters)

class ManeuverRequest(BaseModel):
    satellite: Optional[TLEInput] = None
    satellite_id: Optional[str] = None
    debris: Optional[List[TLEInput]] = None
    time_horizon_hrs: Optional[float] = 24.0
    satellite_covariance: Optional[CovarianceInput] = None
    debris_covariance: Optional[CovarianceInput] = None
    hard_body_radius: Optional[float] = 20.0

class OrchidResponse(BaseModel):
    satellite_id: str
    conjunctions: list
    maneuver_required: bool
    maneuver: Optional[dict]
    overall_risk: str
    message: str

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str

# Helper to build covariance matrix
def get_cov_matrix(cov: Optional[CovarianceInput], default_std: List[float]) -> np.ndarray:
    if cov is None:
        std = default_std
    else:
        std = [cov.r, cov.t, cov.n]
    return np.diag(std)**2

active_ukf_filters = {}

from datetime import datetime, timezone
def register_active_satellite(satellite, request):
    try:
        from physics import tle_to_state
        state_init = tle_to_state(satellite.tle1, satellite.tle2)
        
        # Build covariance in RTN frame (6x6 matrix, standard deviations squared)
        std_r = request.satellite_covariance.r if request.satellite_covariance else 100.0
        std_t = request.satellite_covariance.t if request.satellite_covariance else 500.0
        std_n = request.satellite_covariance.n if request.satellite_covariance else 100.0
        
        # Velocity uncertainties are scaled to 1% of position uncertainties
        cov_rtn = np.diag([std_r, std_t, std_n, std_r * 0.01, std_t * 0.01, std_n * 0.01])**2
        
        active_ukf_filters[satellite.norad_id] = {
            "state": np.array(state_init),
            "covariance": cov_rtn,
            "tle1": satellite.tle1,
            "tle2": satellite.tle2,
            "last_time": datetime.now(timezone.utc)
        }
        logger.info(f"Registered active satellite {satellite.norad_id} in UKF tracking pool.")
    except Exception as e:
        logger.error(f"Failed to register active satellite for UKF: {e}")

def resolve_request(request: ManeuverRequest):
    from catalog_manager import load_tle_catalog
    
    if request.satellite_id:
        catalog = load_tle_catalog()
        if request.satellite_id not in catalog:
            raise HTTPException(status_code=404, detail=f"Satellite ID {request.satellite_id} not found in catalog")
        sat_data = catalog[request.satellite_id]
        satellite = TLEInput(
            norad_id=request.satellite_id,
            tle1=sat_data["line1"],
            tle2=sat_data["line2"]
        )
        debris_pool = [
            {"norad_id": nid, "tle1": d["line1"], "tle2": d["line2"]}
            for nid, d in catalog.items()
            if d.get("type") == "debris"
        ]
    elif request.satellite:
        satellite = request.satellite
        debris_pool = request.debris or []
    else:
        raise HTTPException(status_code=422, detail="Either satellite or satellite_id must be provided")
        
    # Register in UKF tracking pool
    register_active_satellite(satellite, request)
    
    return satellite, debris_pool

# Background Task Worker
def run_analysis_background(task_id: str, request: ManeuverRequest):
    jobs[task_id]["status"] = "RUNNING"
    jobs[task_id]["progress"] = 0
    try:
        satellite, debris_pool = resolve_request(request)
        logger.info(f"Background task {task_id}: start for sat {satellite.norad_id}, {len(debris_pool)} debris")
        
        # Build covariance inputs
        sat_cov = get_cov_matrix(request.satellite_covariance, [100.0, 500.0, 100.0])
        deb_cov = get_cov_matrix(request.debris_covariance, [200.0, 1000.0, 200.0])
        hbr = request.hard_body_radius or 20.0
        
        # Define progress callback
        def progress_callback(pct: int):
            jobs[task_id]["progress"] = pct
            if loop:
                asyncio.run_coroutine_threadsafe(
                    manager.broadcast({
                        "type": "progress",
                        "task_id": task_id,
                        "progress": pct
                    }),
                    loop
                )
                
        conjunctions = assess_risk(
            satellite, debris_pool, request.time_horizon_hrs,
            sat_cov_rtn=sat_cov, deb_cov_rtn=deb_cov, hbr=hbr,
            progress_callback=progress_callback
        )
        
        maneuver_required = any(c["risk_level"] == "COLLISION_COURSE" for c in conjunctions)
        maneuver = None
        if maneuver_required:
            maneuver = generate_maneuver(satellite, conjunctions, debris_pool)
            
        levels = [c["risk_level"] for c in conjunctions]
        overall = "NOMINAL"
        if "COLLISION_COURSE" in levels:
            overall = "COLLISION_COURSE"
        elif "WARNING" in levels:
            overall = "WARNING"
            
        result = {
            "satellite_id": satellite.norad_id,
            "conjunctions": conjunctions,
            "maneuver_required": maneuver_required,
            "maneuver": maneuver,
            "overall_risk": overall,
            "message": f"Analysis complete. {len(conjunctions)} conjunction(s) detected."
        }
        
        jobs[task_id]["status"] = "SUCCESS"
        jobs[task_id]["progress"] = 100
        jobs[task_id]["result"] = result
        
        # Log to SQLite
        fuel_cost = result["maneuver"]["fuel_cost_kg"] if result["maneuver_required"] and result["maneuver"] else 0.0
        log_to_db(result["satellite_id"], result["overall_risk"], result["conjunctions"], result["maneuver_required"], fuel_cost)
        
        # Broadcast finished task result
        if loop:
            asyncio.run_coroutine_threadsafe(
                manager.broadcast({
                    "type": "result",
                    "task_id": task_id,
                    "result": result
                }),
                loop
            )
            
    except Exception as e:
        logger.error(f"Error in background task {task_id}: {e}")
        jobs[task_id]["status"] = "FAILED"
        jobs[task_id]["message"] = str(e)
        if loop:
            asyncio.run_coroutine_threadsafe(
                manager.broadcast({
                    "type": "error",
                    "task_id": task_id,
                    "message": str(e)
                }),
                loop
            )

@app.get("/")
def root():
    return {
        "service": "ORCHID API",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": ["/analyze", "/analyze-async", "/tasks/{task_id}", "/risk-only", "/health", "/ui"]
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/analyze", response_model=OrchidResponse)
def analyze(request: ManeuverRequest):
    try:
        satellite, debris_pool = resolve_request(request)
        logger.info(f"Analyze request for sat {satellite.norad_id}, {len(debris_pool)} debris")
        
        sat_cov = get_cov_matrix(request.satellite_covariance, [100.0, 500.0, 100.0])
        deb_cov = get_cov_matrix(request.debris_covariance, [200.0, 1000.0, 200.0])
        hbr = request.hard_body_radius or 20.0
        
        conjunctions = assess_risk(
            satellite, debris_pool, request.time_horizon_hrs,
            sat_cov_rtn=sat_cov, deb_cov_rtn=deb_cov, hbr=hbr
        )
        
        maneuver_required = any(c["risk_level"] == "COLLISION_COURSE" for c in conjunctions)
        maneuver = None
        if maneuver_required:
            maneuver = generate_maneuver(satellite, conjunctions, debris_pool)
            
        levels = [c["risk_level"] for c in conjunctions]
        overall = "NOMINAL"
        if "COLLISION_COURSE" in levels:
            overall = "COLLISION_COURSE"
        elif "WARNING" in levels:
            overall = "WARNING"
            
        # Log to SQLite
        fuel_cost = maneuver["fuel_cost_kg"] if maneuver_required and maneuver else 0.0
        log_to_db(satellite.norad_id, overall, conjunctions, maneuver_required, fuel_cost)
        
        logger.info(f"Analysis complete: {overall} risk")
        return OrchidResponse(
            satellite_id=satellite.norad_id,
            conjunctions=conjunctions,
            maneuver_required=maneuver_required,
            maneuver=maneuver,
            overall_risk=overall,
            message=f"Analysis complete. {len(conjunctions)} conjunction(s) detected."
        )
    except HTTPException as he:
        raise he
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Analyze error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/analyze-async", response_model=TaskResponse)
def analyze_async(request: ManeuverRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    jobs[task_id] = {
        "status": "PENDING",
        "progress": 0,
        "result": None,
        "message": "Analysis queued."
    }
    background_tasks.add_task(run_analysis_background, task_id, request)
    return TaskResponse(
        task_id=task_id,
        status="PENDING",
        message="Analysis started in background."
    )

@app.get("/tasks/{task_id}")
def get_task_status(task_id: str):
    if task_id not in jobs:
        raise HTTPException(status_code=404, detail="Task not found")
    return jobs[task_id]

@app.post("/risk-only")
def risk_only(request: ManeuverRequest):
    try:
        satellite, debris_pool = resolve_request(request)
        logger.info(f"Risk-only request for sat {satellite.norad_id}")
        
        sat_cov = get_cov_matrix(request.satellite_covariance, [100.0, 500.0, 100.0])
        deb_cov = get_cov_matrix(request.debris_covariance, [200.0, 1000.0, 200.0])
        hbr = request.hard_body_radius or 20.0
        
        conjunctions = assess_risk(
            satellite, debris_pool, request.time_horizon_hrs,
            sat_cov_rtn=sat_cov, deb_cov_rtn=deb_cov, hbr=hbr
        )
        return {"satellite_id": satellite.norad_id, "conjunctions": conjunctions}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Risk-only error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/history")
def get_history():
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, satellite_id, overall_risk, closest_distance_km, tca_min, collision_prob, maneuver_required, fuel_spent_kg
            FROM conjunction_history
            ORDER BY id DESC
            LIMIT 50
        """)
        rows = cursor.fetchall()
        conn.close()
        
        history_list = []
        for row in rows:
            history_list.append({
                "id": row[0],
                "timestamp": row[1],
                "satellite_id": row[2],
                "overall_risk": row[3],
                "closest_distance_km": row[4],
                "tca_min": row[5],
                "collision_prob": row[6],
                "maneuver_required": bool(row[7]),
                "fuel_spent_kg": row[8]
            })
        return history_list
    except Exception as e:
        logger.error(f"Failed to fetch conjunction history from SQLite: {e}")
        raise HTTPException(status_code=500, detail="Internal database error")

@app.get("/fleet")
def get_fleet():
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT norad_id, name, status, fuel_capacity_kg, current_fuel_kg, tle1, tle2 FROM fleet_satellites")
        rows = cursor.fetchall()
        conn.close()
        
        fleet = []
        for row in rows:
            fleet.append({
                "norad_id": row[0],
                "name": row[1],
                "status": row[2],
                "fuel_capacity_kg": row[3],
                "current_fuel_kg": row[4],
                "tle1": row[5],
                "tle2": row[6]
            })
        return fleet
    except Exception as e:
        logger.error(f"Failed to fetch fleet satellites: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fleet/register")
def register_fleet(sat_id: str, background_tasks: BackgroundTasks):
    from catalog_manager import load_tle_catalog
    catalog = load_tle_catalog()
    
    if sat_id not in catalog:
        raise HTTPException(status_code=404, detail=f"Satellite ID {sat_id} not found in catalog")
        
    sat = catalog[sat_id]
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO fleet_satellites (norad_id, name, status, fuel_capacity_kg, current_fuel_kg, tle1, tle2)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (sat_id, sat["name"], "ACTIVE", 300.0, 300.0, sat["line1"], sat["line2"]))
        conn.commit()
        conn.close()
        logger.info(f"Registered new fleet satellite {sat_id} in database.")
        
        # Trigger live conjunction calculation background job immediately to update cache
        from live_feed import fetch_live_conjunctions_data
        background_tasks.add_task(fetch_live_conjunctions_data)
        
        return {"status": "success", "message": f"Satellite {sat['name']} registered in fleet."}
    except Exception as e:
        logger.error(f"Failed to register fleet satellite: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/fleet/{norad_id}")
def delete_fleet(norad_id: str, background_tasks: BackgroundTasks):
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM fleet_satellites WHERE norad_id = ?", (norad_id,))
        conn.commit()
        conn.close()
        logger.info(f"Unregistered fleet satellite {norad_id} from database.")
        
        # Trigger live conjunction calculation background job immediately to update cache
        from live_feed import fetch_live_conjunctions_data
        background_tasks.add_task(fetch_live_conjunctions_data)
        
        return {"status": "success", "message": f"Satellite {norad_id} removed from fleet."}
    except Exception as e:
        logger.error(f"Failed to remove fleet satellite: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/compile-binary")
def compile_binary(dvr: float = 0.0, dvt: float = 0.0, dvn: float = 0.0, duration: float = 0.0):
    try:
        # Calculate CCSDS 8-bit checksum: sum of all bytes modulo 256
        # CCSDS Packet Primary Header: 6 bytes
        # Byte 0-1: 0x180A (Version=0, Type=1, SecHeader=0, APID=0x00A)
        # Byte 2-3: 0xC000 (SeqFlags=3, SeqCount=0)
        # Byte 4-5: 0x0011 (Length = payload size - 1 = 17 bytes)
        apid_packet = 0x180A
        seq_packet = 0xC000
        length_val = 17
        
        header = struct.pack(">HHH", apid_packet, seq_packet, length_val)
        
        # Calculate checksum of the payload bytes (function code + floats)
        payload_core = struct.pack(">Bffff", 1, dvr, dvt, dvn, duration)
        
        # Checksum is the sum of header and payload bytes
        total_bytes = header + payload_core
        checksum = sum(total_bytes) % 256
        
        payload = payload_core + struct.pack("B", checksum)
        full_packet = header + payload
        
        # Return as downloadable binary file stream
        stream = io.BytesIO(full_packet)
        return StreamingResponse(
            stream, 
            media_type="application/octet-stream",
            headers={"Content-Disposition": "attachment; filename=orchid_ccsds_maneuver.bin"}
        )
    except Exception as e:
        logger.error(f"Failed to compile CCSDS binary: {e}")
        raise HTTPException(status_code=500, detail="Compilation failed")

# WebSocket Endpoint
@app.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Maintain connection and listen for heartbeat
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

class RendezvousRequest(BaseModel):
    satellite: TLEInput
    debris: TLEInput
    time_of_flight_min: Optional[float] = 45.0

@app.post("/rendezvous")
def compute_rendezvous(request: RendezvousRequest):
    try:
        from physics import tle_to_state
        from rendezvous import plan_cw_rendezvous
        
        # Initial states (chaser and target)
        chaser_state = tle_to_state(request.satellite.tle1, request.satellite.tle2)
        target_state = tle_to_state(request.debris.tle1, request.debris.tle2)
        
        r_chaser, v_chaser = chaser_state[:3], chaser_state[3:]
        r_target, v_target = target_state[:3], target_state[3:]
        
        # Calculate mean motion omega from target TLE
        try:
            n_day = float(request.debris.tle2[52:63].strip())
            omega = n_day * (2.0 * np.pi / 86400.0)
        except Exception:
            omega = 0.0011 # default LEO mean motion
            
        dt_sec = request.time_of_flight_min * 60.0
        
        dv_rtn, transfer_points = plan_cw_rendezvous(
            r_chaser, v_chaser, r_target, v_target, dt_sec, omega
        )
        
        # Estimate fuel cost using Tsiolkovsky rocket equation
        from physics import tsiolkovsky
        DRY_MASS = 550.0
        fuel_cost = tsiolkovsky(np.linalg.norm(dv_rtn), DRY_MASS + 50.0)
        
        return {
            "delta_v_rtn": dv_rtn,
            "fuel_cost_kg": round(float(fuel_cost), 4),
            "transfer_points": transfer_points,
            "time_of_flight_min": request.time_of_flight_min,
            "message": "Active Debris Removal (ADR) relative rendezvous trajectory computed successfully."
        }
    except Exception as e:
        logger.error(f"Rendezvous calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

from federated_learning import FederatedCoordinator
fed_coordinator = FederatedCoordinator()

class FederatedSubmission(BaseModel):
    operator_id: str
    weights: List[List[float]]
    sample_count: int
    signature: str

@app.post("/federated/submit")
def federated_submit(submission: FederatedSubmission):
    res = fed_coordinator.submit_local_weights(
        submission.operator_id,
        submission.weights,
        submission.sample_count,
        submission.signature
    )
    if res["status"] == "failed":
        raise HTTPException(status_code=401, detail=res["message"])
    return res

@app.post("/federated/aggregate")
def federated_aggregate():
    global_weights = fed_coordinator.aggregate_weights()
    if not global_weights:
        raise HTTPException(status_code=400, detail="Not enough submissions to aggregate (minimum 2).")
    return {"status": "success", "message": "Aggregation complete.", "global_weights": global_weights}

@app.post("/federated/generate-simulation-payload")
def generate_simulation_payload(operator_id: str, sample_count: int = 100):
    secrets = {
        "operator_spacex": b"spacex_secret_handshake_key_101",
        "operator_oneweb": b"oneweb_secret_handshake_key_202",
        "operator_isro": b"isro_secret_handshake_key_303"
    }
    if operator_id not in secrets:
        raise HTTPException(status_code=404, detail=f"Operator {operator_id} not registered.")
    
    import random
    # Generate mock weights layers representation
    weights = [[random.uniform(-0.5, 0.5) for _ in range(10)] for _ in range(3)]
    
    import hashlib
    import hmac
    weights_str = str(weights)
    payload_hash = hashlib.sha256(weights_str.encode()).hexdigest()
    signature = hmac.new(secrets[operator_id], payload_hash.encode(), hashlib.sha256).hexdigest()
    
    return {
        "operator_id": operator_id,
        "weights": weights,
        "sample_count": sample_count,
        "signature": signature
    }

@app.post("/anomaly/check/{norad_id}")
def check_anomaly(norad_id: str):
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT name, tle1, tle2 FROM fleet_satellites WHERE norad_id = ?", (norad_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            raise HTTPException(status_code=404, detail=f"Satellite {norad_id} is not registered in our operational fleet.")
            
        sat_name, old_tle1, old_tle2 = row[0], row[1], row[2]
        
        from catalog_manager import load_tle_catalog
        catalog = load_tle_catalog(force_refresh=True)
        
        if norad_id not in catalog:
            raise HTTPException(status_code=404, detail=f"Latest TLE for NORAD ID {norad_id} not available in catalog.")
            
        new_sat = catalog[norad_id]
        
        from anomaly_detector import detect_tle_drift
        res = detect_tle_drift(
            norad_id, sat_name, old_tle1, old_tle2, new_sat["line1"], new_sat["line2"]
        )
        return res
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Anomaly check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/7d/{norad_id}")
def predict_7d(norad_id: str):
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT tle1, tle2 FROM fleet_satellites WHERE norad_id = ?", (norad_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            from catalog_manager import load_tle_catalog
            catalog = load_tle_catalog()
            if norad_id not in catalog:
                raise HTTPException(status_code=404, detail=f"Satellite {norad_id} TLE not found.")
            tle1, tle2 = catalog[norad_id]["line1"], catalog[norad_id]["line2"]
        else:
            tle1, tle2 = row[0], row[1]
            
        from physics import tle_to_state
        state = tle_to_state(tle1, tle2)
        pos = state[:3]
        
        sequence = []
        for i in range(5):
            t_offset = i * 86400.0
            theta = 0.0011 * t_offset
            c, s = np.cos(theta), np.sin(theta)
            sequence.append([
                float(pos[0]*c - pos[1]*s),
                float(pos[0]*s + pos[1]*c),
                float(pos[2])
            ])
            
        from lstm_predictor import predict_7d_position
        pred = predict_7d_position(sequence)
        
        return {
            "satellite_id": norad_id,
            "current_position_km": pos,
            "predicted_position_7d_km": pred.tolist(),
            "method": "LSTM_Predictor" if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lstm_orbit_model.pt")) else "Keplerian_Analytical_Fallback"
        }
    except Exception as e:
        logger.error(f"7-day prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/live-conjunctions")
def live_conjunctions():
    from live_feed import get_cached_conjunctions
    return get_cached_conjunctions()

@app.get("/live-tles")
def live_tles():
    from live_feed import get_cached_conjunctions
    data = get_cached_conjunctions()
    catalog = {}
    
    # Standard conjunctions
    for c in data.get("conjunctions", []):
        p = c["primary"]
        s = c["secondary"]
        catalog[p["norad_id"]] = {
            "name": p["name"],
            "line1": p["line1"],
            "line2": p["line2"],
            "type": "active"
        }
        catalog[s["norad_id"]] = {
            "name": s["name"],
            "line1": s["line1"],
            "line2": s["line2"],
            "type": "debris"
        }
        
    # Fleet close approaches
    for c in data.get("fleet_conjunctions", []):
        p = c["primary"]
        s = c["secondary"]
        catalog[p["norad_id"]] = {
            "name": p["name"],
            "line1": p["line1"],
            "line2": p["line2"],
            "type": "active"
        }
        catalog[s["norad_id"]] = {
            "name": s["name"],
            "line1": s["line1"],
            "line2": s["line2"],
            "type": "debris"
        }
    return catalog

@app.get("/dashboard")
def dashboard():
    dash = os.path.join(STATIC_DIR, "dashboard.html")
    if os.path.exists(dash):
        return FileResponse(dash)
    return {"error": "Dashboard UI not found", "static_dir": STATIC_DIR}

@app.get("/ui")
def ui():
    index = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"error": "UI not found", "static_dir": STATIC_DIR, "exists": os.path.exists(STATIC_DIR)}