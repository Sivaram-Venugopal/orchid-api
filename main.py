import os
import uuid
import numpy as np
import logging
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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

@app.on_event("startup")
async def startup_event():
    global loop
    loop = asyncio.get_running_loop()
    logger.info("Starting ORCHID API v2.0...")

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
            TLEInput(norad_id=nid, tle1=d["line1"], tle2=d["line2"])
            for nid, d in catalog.items()
            if d.get("type") == "debris"
        ]
    elif request.satellite:
        satellite = request.satellite
        debris_pool = request.debris or []
    else:
        raise HTTPException(status_code=422, detail="Either satellite or satellite_id must be provided")
        
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/ui")
def ui():
    index = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"error": "UI not found", "static_dir": STATIC_DIR, "exists": os.path.exists(STATIC_DIR)}