# ORCHID: Space Conjunction Assessment & Avoidance Service (v2.0.0)

ORCHID is an operational-grade autonomous satellite collision avoidance service. It provides high-performance debris catalog screening, 3D covariance-based collision probability calculations, dynamic policy routing for different orbital regimes, and real-time interactive telemetry visualization.

---

## 🛠️ Architecture & System Design

The system is organized into a modular FastAPI application:

```
orchid-api/
├── main.py                   # FastAPI application, WebSocket server & Async Task manager
├── risk_engine.py            # Parallelized screening & Foster's 2D b-plane collision math
├── maneuver_engine.py        # Regime-specific orbital routing & RL maneuver generation
├── physics.py                # Orbital state conversions and rocket dynamics (Tsiolkovsky)
├── catalog_manager.py        # TLE debris catalog parser and manager
├── static/
│   └── index.html            # Premium UI Dashboard, SVG Covariance Radar, WebSocket Client
└── README.md                 # Project documentation
```

---

## 🚀 Key Features Implemented in v2.0

### 1. Parallelized & High-Performance Screening
* **Multithreaded Propagation**: Leveraging `concurrent.futures.ThreadPoolExecutor`, ORCHID propagates and screens the active debris catalog concurrently. Because the SGP4 library's core computations run in GIL-free C++ wrappers, thread-level concurrency achieves multi-core speedups on both Windows and Linux without multiprocessing overhead.
* **Three-Stage Grid Filtering**:
  1. *Grid Filter*: Discards pairs whose apogee/perigee altitude ranges do not overlap within a 100 km safety corridor.
  2. *Coarse Sweep*: Propagates at a 10-minute resolution over a 24-hour horizon, filtering pairs whose absolute minimum distance exceeds 100 km.
  3. *Fine Sweep*: Performs a high-fidelity 10-second sweep around the minimum approach window to resolve the exact Time of Closest Approach (TCA).

### 2. Analytical 2D Covariance Collision Probability ($P_c$)
Rather than using simple distance-based risk metrics, ORCHID projects the relative 3D position and covariance of the encounter onto the **encounter plane (b-plane)** at TCA:
1. **ECI Covariance Transformation**: Converts the satellite and debris covariances from their respective 3D Radial-Transverse-Normal (RTN) frames to Earth-Centered Inertial (ECI) coordinate frames.
2. **b-Plane Projection**: Defines a 2D coordinate system perpendicular to the relative velocity vector at TCA and projects the combined relative ECI covariance onto this plane:
   $$C_{2D} = P^T (C_{\text{sat, ECI}} + C_{\text{deb, ECI}}) P$$
3. **Foster's Probability**: Computes the collision probability ($P_c$) by integrating the 2D Gaussian density function over the Hard Body Radius (HBR) of the satellite using a fast, highly stable analytical infinitesimal-area approximation:
   $$P_c \approx \frac{R_{HBR}^2}{2\sqrt{\det(C_{2D})}} \exp\left( - \frac{d^2}{2} C_{2D}^{-1}[0,0] \right)$$
   where $d$ is the relative miss distance.

### 3. Real-Time Telemetry & Asynchronous Workers
* **FastAPI BackgroundTasks**: Relieves request blocking by executing long-horizon runs asynchronously. Initiating a scan via `/analyze-async` immediately returns a `task_id` and starts background processing.
* **Status Tracking**: The state (`PENDING`, `RUNNING`, `SUCCESS`, `FAILED`) and percentage progress of background tasks can be polled via the `/tasks/{task_id}` endpoint.
* **WebSocket Streams**: Active client connections to `/ws/telemetry` receive real-time, event-driven broadcasts of scanning progress percentages and final completed risk metrics.

### 4. Dynamic RL Policy Routing
* **Orbital Regime Classifier**: Categorizes the satellite's orbital environment by calculating its semi-major axis from TLE Line 2 data:
  * **LEO**: Altitude $< 2,000$ km
  * **GEO**: Altitude between $35,000$ and $36,500$ km
  * **MEO**: All other altitudes
* **Policy Binding**: Dynamically selects and loads the optimal Stable-Baselines3 RL model:
  * `orchid_leo_impulsive` for LEO (impulsive thrust maneuvers).
  * `orchid_geo_electric` for GEO (continuous electric propulsion).
  * `orchid_v4_best` as a fallback or MEO profile.
* **Dynamic Burn Timing & Fuel Optimization**: Computes the optimal burn timing based on the Time to Closest Approach (TCA). For early detections (TCA > 180 min), it schedules the burn 120 minutes prior to TCA, allowing drift time to build up separation and scaling down the required delta-V to as low as $20\%$ of nominal. Critical encounters schedule an immediate burn with higher delta-V scaling to ensure safety.

### 5. Premium SVG Conjunction Radar UI
* **Real-time Progress Dashboard**: Displays a progress bar driven by WebSocket telemetry with a graceful HTTP polling fallback.
* **Advanced Covariance Overrides**: Includes collapsible configuration panels for inputting customized RTN standard deviations (in meters) for both the satellite and debris.
* **3-Sigma Covariance Ellipse Visualizer**: Centers and draws the SVG `<ellipse>` representing the 3-sigma uncertainty boundary on the interactive radar.

---

## 🏃 Getting Started

### 📋 Prerequisites
Ensure you have Python 3.11+ installed. (We recommend using [uv](https://github.com/astral-sh/uv) for fast environment setup).

### ⚙️ Installation
1. Clone the repository and navigate to the project directory:
   ```bash
   cd orchid-api
   ```
2. Set up the virtual environment and install dependencies:
   ```bash
   uv venv
   .venv\Scripts\activate      # Windows
   source .venv/bin/activate    # macOS/Linux
   uv pip install -r requirements.txt
   ```

### 🖥️ Running the Application
Launch the Uvicorn web server:
```bash
.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8080
```
Open your browser and navigate to:
* **Interactive UI**: [http://127.0.0.1:8080/ui](http://127.0.0.1:8080/ui)
* **REST API Documentation**: [http://127.0.0.1:8080/docs](http://127.0.0.1:8080/docs)

---

## 🧪 Running Verification Tests

Run the mathematical validation suite to verify Foster's probability calculations and regime classification:
```bash
.venv\Scripts\python.exe .gemini/antigravity-cli/brain/3eda25bd-0ea0-41ca-9b4d-f95b88629411/scratch/verify_all.py
```

Run the end-to-end integration tests to verify API endpoints:
```bash
.venv\Scripts\python.exe .gemini/antigravity-cli/brain/3eda25bd-0ea0-41ca-9b4d-f95b88629411/scratch/test_analyze.py
```
