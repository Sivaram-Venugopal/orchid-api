import os
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

ALERT_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alerts.log")

try:
    from twilio.rest import Client as TwilioClient
except ImportError:
    TwilioClient = None

try:
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail
except ImportError:
    SendGridAPIClient = None
    Mail = None

def trigger_alert(level: str, title: str, text: str):
    """
    Triggers an alert based on priority level:
    - P0: Critical risk. Sends SMS (Twilio) + Email (SendGrid) + Local log.
    - P1: High risk. Sends Email (SendGrid) + Local log.
    - P2: Medium/Low warning. Appends to local log only.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    log_entry = f"[{timestamp}] [{level}] {title}\n{text}\n{'='*60}\n"
    
    # 1. Log locally to alerts.log
    try:
        with open(ALERT_LOG_FILE, "a") as f:
            f.write(log_entry)
        logger.info(f"Local alert logged for level {level}: {title}")
    except Exception as e:
        logger.error(f"Failed to write to local alerts.log: {e}")
        
    # 2. Dispatch SMS (P0 only)
    if level == "P0":
        dispatch_sms(title, text)
        
    # 3. Dispatch Email (P0 and P1)
    if level in ["P0", "P1"]:
        dispatch_email(title, text)

def dispatch_sms(title: str, text: str):
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_FROM_NUMBER")
    to_number = os.getenv("TWILIO_TO_NUMBER")
    
    if not (account_sid and auth_token and from_number and to_number):
        logger.warning("[Twilio SMS Alert] Credentials not set. SMS skipped. (Available in alerts.log)")
        return
        
    if TwilioClient is None:
        logger.error("[Twilio SMS Alert] twilio package not installed. SMS skipped.")
        return
        
    try:
        client = TwilioClient(account_sid, auth_token)
        message = client.messages.create(
            body=f"{title}\n{text}",
            from_=from_number,
            to=to_number
        )
        logger.info(f"Twilio SMS sent. SID: {message.sid}")
    except Exception as e:
        logger.error(f"Twilio SMS sending failed: {e}")

def dispatch_email(title: str, text: str):
    api_key = os.getenv("SENDGRID_API_KEY")
    from_email = os.getenv("SENDGRID_FROM_EMAIL")
    to_email = os.getenv("SENDGRID_TO_EMAIL")
    
    if not (api_key and from_email and to_email):
        logger.warning("[SendGrid Email Alert] Credentials not set. Email skipped. (Available in alerts.log)")
        return
        
    if SendGridAPIClient is None:
        logger.error("[SendGrid Email Alert] sendgrid package not installed. Email skipped.")
        return
        
    try:
        html_text = text.replace('\n', '<br>')
        message = Mail(
            from_email=from_email,
            to_emails=to_email,
            subject=title,
            html_content=f"<p>{html_text}</p>"
        )
        sg = SendGridAPIClient(api_key)
        response = sg.send(message)
        logger.info(f"SendGrid email sent. Status Code: {response.status_code}")
    except Exception as e:
        logger.error(f"SendGrid email sending failed: {e}")
