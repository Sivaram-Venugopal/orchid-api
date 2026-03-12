from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from risk_engine import assess_risk
from maneuver_engine import generate_maneuver
from doc_generator import generate_docs, generate_contract

app = FastAPI(
    title="ORCHID API",
    description="Autonomous Satellite Collision Avoidance as a Service + AI Documentation Generator",
    version="2.0.0"
)

class TLEInput(BaseModel):
    norad_id: str
    tle1: str
    tle2: str

class ManeuverRequest(BaseModel):
    satellite: TLEInput
    debris: List[TLEInput]
    time_horizon_hrs: Optional[float] = 24.0

class OrchidResponse(BaseModel):
    satellite_id: str
    conjunctions: list
    maneuver_required: bool
    maneuver: Optional[dict]
    overall_risk: str
    message: str

class CodeInput(BaseModel):
    code: str

@app.get("/")
def root():
    return {
        "service": "ORCHID API",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": ["/analyze", "/risk-only", "/health", "/generate-docs", "/generate-contract", "/ui"]
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/analyze", response_model=OrchidResponse)
def analyze(request: ManeuverRequest):
    try:
        conjunctions = assess_risk(request.satellite, request.debris, request.time_horizon_hrs)
        maneuver_required = any(c["risk_level"] in ["HIGH", "CRITICAL"] for c in conjunctions)
        maneuver = None
        if maneuver_required:
            maneuver = generate_maneuver(request.satellite, conjunctions)
        levels = [c["risk_level"] for c in conjunctions]
        if "CRITICAL" in levels:    overall = "CRITICAL"
        elif "HIGH" in levels:      overall = "HIGH"
        elif "MEDIUM" in levels:    overall = "MEDIUM"
        else:                       overall = "LOW"
        return OrchidResponse(
            satellite_id=request.satellite.norad_id,
            conjunctions=conjunctions,
            maneuver_required=maneuver_required,
            maneuver=maneuver,
            overall_risk=overall,
            message=f"Analysis complete. {len(conjunctions)} conjunction(s) detected."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/risk-only")
def risk_only(request: ManeuverRequest):
    try:
        conjunctions = assess_risk(request.satellite, request.debris, request.time_horizon_hrs)
        return {"satellite_id": request.satellite.norad_id, "conjunctions": conjunctions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-docs")
def gen_docs(input: CodeInput):
    try:
        docs = generate_docs(input.code)
        return {"documentation": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-contract")
def gen_contract(input: CodeInput):
    try:
        contract = generate_contract(input.code)
        return contract
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/ui")
def ui():
    return FileResponse("static/index.html")
    