# app/services/emailer.py
import os
import smtplib
import ssl
from email.message import EmailMessage
from urllib.parse import urlencode, urljoin

from app.security import create_access_token
from app.settings import settings

VERIFY_EXPIRE_MIN = 60   # minutes
RESET_EXPIRE_MIN  = 30   # minutes


def _bool(v) -> bool:
    if isinstance(v, bool):
        return v
    s = (str(v or "")).strip().lower()
    return s in {"1", "true", "yes", "y", "on"}


def _send_console(to: str, subject: str, html: str):
    print(f"[EMAIL:CONSOLE] to={to} subject={subject}\n{html}\n")


def _send_smtp(to: str, subject: str, html: str):
    if not settings.SMTP_HOST:
        raise RuntimeError("SMTP_HOST not set")
    msg = EmailMessage()
    msg["From"] = settings.EMAIL_FROM
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content("HTML email. Please view in an HTML-capable client.")
    msg.add_alternative(html, subtype="html")

    context = ssl.create_default_context()
    with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=30) as server:
        if _bool(settings.SMTP_USE_TLS):
            server.starttls(context=context)
        if settings.SMTP_USER and settings.SMTP_PASS:
            server.login(settings.SMTP_USER, settings.SMTP_PASS)
        server.send_message(msg)


def _send_acs(to: str, subject: str, html: str):
    try:
        from azure.communication.email import EmailClient
    except Exception as e:
        raise RuntimeError("azure-communication-email not installed") from e

    conn = os.getenv("ACS_CONNECTION_STRING")
    sender = os.getenv("ACS_SENDER")
    if not conn or not sender:
        raise RuntimeError("Set ACS_CONNECTION_STRING and ACS_SENDER for ACS provider")

    client = EmailClient.from_connection_string(conn)
    message = {
        "senderAddress": sender,
        "content": {"subject": subject, "html": html},
        "recipients": {"to": [{"address": to}]},
    }
    client.begin_send(message).result(30)


def send_email(to: str, subject: str, html: str) -> bool:
    provider = (settings.EMAIL_PROVIDER or "console").lower()
    try:
        if provider == "smtp":
            _send_smtp(to, subject, html)
        elif provider == "acs":
            _send_acs(to, subject, html)
        else:
            _send_console(to, subject, html)
        return True
    except Exception as e:
        print(f"[EMAIL:ERROR] provider={provider} err={e!r}")
        _send_console(to, subject, html)
        return False


# -------- Verification --------
def make_verify_token(email: str) -> str:
    return create_access_token({"sub": email}, minutes=VERIFY_EXPIRE_MIN, token_type="verify")


def send_verification_email(to: str, frontend_origin: str):
    token = make_verify_token(to)
    link = urljoin(frontend_origin.rstrip("/") + "/", "verify") + "?" + urlencode({"token": token})
    html = (
        "<p>Welcome to MedRAG!</p>"
        f"<p>Click to verify: <a href='{link}'>Verify email</a></p>"
        f"<p>This link expires in {VERIFY_EXPIRE_MIN} minutes.</p>"
    )
    send_email(to, "Verify your MedRAG account", html)
    return token


# -------- Password reset --------
def make_reset_token(email: str) -> str:
    return create_access_token({"sub": email}, minutes=RESET_EXPIRE_MIN, token_type="reset")


def send_reset_email(to: str, frontend_origin: str):
    token = make_reset_token(to)
    # IMPORTANT: your page lives at /reset because (auth) is a route group.
    reset_path = "/reset"
    link = urljoin(frontend_origin.rstrip("/") + "/", reset_path.lstrip("/")) + "?" + urlencode({"token": token})
    html = (
        "<p>Password reset requested.</p>"
        f"<p>Click to reset: <a href='{link}'>Reset password</a></p>"
        f"<p>This link expires in {RESET_EXPIRE_MIN} minutes.</p>"
    )
    send_email(to, "Reset your MedRAG password", html)
    return token
