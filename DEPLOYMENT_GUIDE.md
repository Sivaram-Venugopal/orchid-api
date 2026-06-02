# ORCHID: Live Dashboard & Conjunctions Feed Deployment Guide

This guide outlines the step-by-step process to deploy the newly implemented Live TLE Conjunctions Feed, APScheduler Background Jobs, and Live HTML5 Canvas Dashboard to your existing Railway deployment.

---

## 🛠️ Step 1: Git Staging & Local Commit

You need to commit the new files (`live_feed.py`, `scheduler.py`, `static/dashboard.html`) and updated configuration files (`main.py`, `requirements.txt`) to your repository.

1. Open your terminal in the project root (`C:/Users/LAKSHMI/orchid-api`).
2. Verify all modified and untracked files are recognized:
   ```bash
   git status
   ```
3. Stage all modified and new files:
   ```bash
   git add requirements.txt main.py live_feed.py scheduler.py static/dashboard.html
   ```
4. Commit the changes locally:
   ```bash
   git commit -m "Implement Live TLE Feed, APScheduler Background Jobs, and Live Dashboard"
   ```

---

## 🚀 Step 2: Push to GitHub (Triggers Railway Build)

Railway is configured with continuous deployment (CD), meaning any push to the connected GitHub repository's main branch triggers a new cloud build.

1. Push your commit to GitHub:
   ```bash
   git push origin master
   ```
2. Navigate to your repository page (e.g., `https://github.com/Sivaram-Venugopal/orchid-api`) to verify that the commit is visible on the remote server.

---

## 🏗️ Step 3: Monitor the Build on Railway

1. Open your browser and go to your **Railway Dashboard**: [https://railway.app/](https://railway.app/)
2. Log in and open your **ORCHID project**.
3. Under the **Deployments** tab of your service, you will see a new deployment marked as `BUILDING` or `QUEUED`.
4. Click on the active deployment to view the build logs. Railway will:
   * Load the `Dockerfile`.
   * Compile and install the updated Python dependencies (including the newly added `apscheduler`).
   * Compile C++ SGP4 wrappers.
   * Start the uvicorn API gateway process binding to the dynamic `$PORT`.
5. Once the build log finishes successfully, the status will turn green (`ACTIVE`).

---

## 🧪 Step 4: Verification of Live Services

Once the deployment completes, verify the active endpoints:

1. **Verify Health**:
   Open your browser and navigate to:
   `https://[your-railway-app].up.railway.app/health`
   It should return:
   ```json
   {"status": "healthy"}
   ```
2. **Verify live conjunctions list**:
   Open:
   `https://[your-railway-app].up.railway.app/live-conjunctions`
   It should return the cached JSON list of the top 20 real-time SOCRATES conjunctions assessed against our numerical $J_2$+drag dynamics.
3. **Verify live TLE list**:
   Open:
   `https://[your-railway-app].up.railway.app/live-tles`
   It should return the compiled TLE lines for the catalog elements currently under monitoring.
4. **Open the Live Dashboard**:
   Open:
   `https://[your-railway-app].up.railway.app/dashboard`
   This is the new page displaying the interactive b-plane conjunction radar, animated sweep line, telemetry indicators, and color-coded risk tables.
