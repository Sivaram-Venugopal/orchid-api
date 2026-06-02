import logging
from apscheduler.schedulers.background import BackgroundScheduler
from live_feed import fetch_live_conjunctions_data

logger = logging.getLogger(__name__)
scheduler = BackgroundScheduler()

def start_scheduler():
    logger.info("Initializing APScheduler Background Task Scheduler...")
    
    # Add periodic 6-hour fetch job
    scheduler.add_job(
        fetch_live_conjunctions_data,
        'interval',
        hours=6,
        id='socrates_feed_job',
        replace_existing=True
    )
    
    # Add immediate execution job on startup
    scheduler.add_job(
        fetch_live_conjunctions_data,
        'date',
        id='socrates_feed_initial',
        replace_existing=True
    )
    
    scheduler.start()
    logger.info("Background Scheduler started successfully.")

def shutdown_scheduler():
    logger.info("Shutting down APScheduler Background Task Scheduler...")
    try:
        scheduler.shutdown(wait=False)
        logger.info("Background Scheduler shut down.")
    except Exception as e:
        logger.error(f"Error shutting down scheduler: {e}")
