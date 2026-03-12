# Railway Deployment Fixes TODO
Current working directory: c:/Users/LAKSHMI/orchid-api

## Planned Steps (Approved)
1. [x] Edit Dockerfile: Fix CMD to use $PORT for Railway.
2. [x] Edit main.py: Add logging, improve error handling, add /docs endpoint, startup event for model check.
3. [x] Edit maneuver_engine.py: Add try/except for PPO.load() with fallback.
4. [x] Edit doc_generator.py: Add ANTHROPIC_API_KEY env check.
5. [x] Edit requirements.txt: Pin scipy version.
6. [x] Create .railway.toml: Service config.
7. [x] Restore policy_pool.py (user feedback - keep as-is).
8. [x] Fix Railway build: Dockerfile single pip + pytorch index; torch commented.
9. [ ] Test locally: docker build . && docker run -p 8080:8080 -e PORT=8080 <image>
10. [ ] Deploy to Railway: git push, set ANTHROPIC_API_KEY, monitor.

**Next**: Complete step 1 (Dockerfile).
