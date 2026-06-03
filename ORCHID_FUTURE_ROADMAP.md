# ORCHID: Future Scope & Roadmap (v4.0.0-draft)

This document outlines the next engineering steps, structural design specifications, and integrations proposed for ORCHID. It provides the next development sessions with the conceptual and mathematical foundation to continue scaling the platform.

---

## A. Deployment Strategy: Free/Low-Cost Alternatives to Railway

Since the Railway trial period has ended, ORCHID needs a cost-effective cloud hosting solution capable of running a FastAPI application, a continuous background scheduler, and machine learning inference (stable-baselines3/PyTorch).

### 1. Oracle Cloud "Always Free" (ARM Ampere VMs)
* **Resource Profile**: 4 ARM Ampere vCPUs, 24 GB RAM, 200 GB Storage, Always-On.
* **Pros**: 
  - Massive memory capacity, easily holding PyTorch models and the entire 20,000+ public debris TLE catalog.
  - Always-on: the background scheduler will run continuously.
  - Root access: full control over Docker, WebSockets, and reverse-proxy firewalls.
* **Cons**: Signup registration is notoriously selective (frequent credit card rejections).
* **Verdict**: **Recommended** if signup is successful.

### 2. Hugging Face Spaces (Docker SDK)
* **Resource Profile**: 2 vCPUs, 16 GB RAM, 50 GB Storage, Always-On (free tier).
* **Pros**: 
  - Generous RAM allocation (16GB) is ideal for machine learning pipelines.
  - Native Docker SDK: reads our `Dockerfile` directly from the Git repository.
* **Cons**:
  - WebSockets can experience proxy timeouts; might require falling back to the 15s HTTP polling loop.
  - Storage is ephemeral (cleared on space restart).
* **Verdict**: Best option for hosting model testing environments for free.

### 3. Render.com (Free Tier)
* **Resource Profile**: 512 MB RAM, 0.1 Shared CPU.
* **Pros**: 
  - Git integration is simple and automatic.
* **Cons**:
  - **Inactivity Spin-down**: Container goes to sleep after 15 minutes of zero traffic. This **halts the background APScheduler scheduler**, preventing the 6-hour CelesTrak scrape from running automatically.
  - 512MB RAM is too small; PyTorch models will trigger Out-of-Memory (OOM) crashes.
* **Verdict**: Not suitable for ORCHID's operational scheduling, but good for hosting static UI templates.

### 4. Fly.io (Free Tier)
* **Resource Profile**: 256MB to 512MB RAM, 1 Shared CPU.
* **Pros**:
  - Always-on, low-latency edge deployments.
  - Custom Docker container support.
* **Cons**:
  - Strict 256MB RAM limit. PyTorch/Gymnasium imports will crash on startup due to OOM.
* **Verdict**: Too resource-constrained for ORCHID's numerical physics and ML dependencies.

---

## B. Phase 7: Multi-Satellite Fleet Management

Transitioning ORCHID from a single-conjunction assessor to an enterprise fleet coordinator.

```
       +-------------------------------------------------------------+
       |                  Fleet Dashboard Control                    |
       +-------------------------------------------------------------+
       |   SAT-101 (LEO)  -  [Active]   - Conjunction risk: LOW      |
       |   SAT-102 (LEO)  -  [Alert]    - Conjunction risk: HIGH ⚠️  |
       |   SAT-201 (GEO)  -  [Maneuver] - Executing Burn... ⚡        |
       +-------------------------------------------------------------+
```

### 1. Vectorized Constellation Ingestion
* Rather than looping through satellites one by one, ORCHID's screening pipeline should be upgraded to process constellation states (e.g., 100+ operational satellites) in parallel.
* A Shared Debris Search Grid filters the entire catalog against the fleet's apogee/perigee windows in a single pass, cutting screening overhead by 90%.

### 2. Fleet Status Heatmap
* Serves a constellation monitoring board showing fuel capacity, drift metrics, and active threats.
* An interactive dashboard display maps the fleet against incoming conjunction vectors, sorted by Time to Closest Approach (TCA) and probability of collision ($P_c$).

### 3. Automated Alert Queue & Priority Dispatch
* **Urgency Classification**: Conjunctions are prioritized into a tri-stage queue:
  - **P0 (Critical)**: $P_c > 10^{-4}$ or range $< 150\text{m}$ occurring within 12 hours. Triggers SMS alert.
  - **P1 (High)**: Conjunction occurring within 24 hours. Triggers urgent email digest.
  - **P2 (Medium)**: Conjunction within 72 hours. Visual flag on the dashboard.
* **Twilio/SendGrid SMS & Email Integrations**:
  - Twilio SMS: `[CRITICAL ALERT] ORCHID Fleet: Sat-12 conjoins with Debris-402 in 3.4h. Distance: 92m. Assess avoidance at: http://localhost:8080/ui`
  - SendGrid: Sends automated reports containing 3D orbit plots, risk calculations, and CCSDS telecommand suggestions.

### 4. Constellation Fuel Optimization
* When planning evasive maneuvers, the algorithm optimizes fuel consumption across the entire fleet to prevent a single satellite from draining its reserves.
* Evaluates relative orbital drift differences caused by Earth's Equatorial Bulge ($J_2$) to perform fuel-free phasing maneuvers (using gravity perturbations to drift apart) instead of burning thruster fuel immediately.

---

## C. Phase 8: Machine Learning Enhancements

### 1. Regime-Specific Reinforcement Learning Policies
* **impulsive LEO Model**: Train a Proximal Policy Optimization (PPO) agent using chemical thruster bounds (high thrust, instant change in velocity vector) optimizing for separation distance.
* **Continuous GEO Model**: Train a continuous-thrust PPO agent simulating electric propulsion (ion engine, low thrust over days) optimizing for fuel burn over long drift windows.

### 2. Federated Learning Across Operators
* Operators (e.g., SpaceX, OneWeb, NASA) train local RL models using their proprietary satellite positions.
* A secure coordinator aggregates model weights (using FedAvg) without transferring raw orbital tracking data, preventing coordinate leakage.

### 3. Long-Short Term Memory (LSTM) Conjunction Predictor
* Train an LSTM neural network on historical TLE drift rates to predict a satellite's position 7 days in advance.
* Captures orbital deviations caused by space weather, atmospheric density swells, and geomagnetism, raising flags early before they appear in public catalogs.

---

## D. Phase 9: Space Agency & Catalog Integrations

### 1. Authenticated Space-Track.org API
* Transition from CelesTrak scraping to Space-Track's query API:
  `https://www.space-track.org/basicquery/class/tle/NORAD_CAT_ID/{ids}/orderby/EPOCH%20desc/limit/1`
* Downloads JSpOC (Joint Space Operations Center) Conjunction Data Messages (CDMs) automatically.

### 2. ESA DISCOS Database Integration
* Connect to the European Space Agency's Database and Information System Characterising Objects in Space (DISCOS).
* Automatically extracts spacecraft dimensions (bus volume, solar array spans, dry mass, drag coefficient $C_d$) to dynamically build the 3D projected CAD mesh check.

### 3. CCSDS Message Formats
* **CDM (Conjunction Data Message - CCSDS 508.0-B-1)**: Import/Export raw CDMs containing covariance matrices, approach ranges, and probabilities.
* **OEM (Orbit Ephemeris Message - CCSDS 502.0-B-2)**: Export planned avoidance paths as orbital ephemeris matrices ready for space agency submission.
