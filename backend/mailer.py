"""Email for FarmScan's weekly alerts, through the same email service as Geohub and the other GeoEstate apps: set the
same variables Geohub has — Resend (RESEND_API_KEY) or SMTP (SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD), with
MAIL_FROM and MAIL_REPLY_TO. With neither set, nothing is sent (and the alerts
can still be read in FarmScan)."""
import os
import smtplib
from email.message import EmailMessage

import httpx

OUTBOX: list[dict] = []          # tests read what would have been sent


def configured() -> bool:
    return bool(os.environ.get("RESEND_API_KEY") or os.environ.get("SMTP_HOST") or os.environ.get("MAIL_TEST_OUTBOX"))


def send(to: str, subject: str, text: str, html: str | None = None) -> None:
    sender = os.environ.get("MAIL_FROM", "FarmScan <no-reply@example.org>")
    if os.environ.get("MAIL_TEST_OUTBOX"):
        OUTBOX.append({"to": to, "subject": subject, "text": text, "html": html})
        return
    if key := os.environ.get("RESEND_API_KEY"):
        r = httpx.post("https://api.resend.com/emails", headers={"Authorization": f"Bearer {key}"}, timeout=20,
                       json={"from": sender, "to": [to], "subject": subject, "text": text, **({"html": html} if html else {}),
                             **({"reply_to": os.environ["MAIL_REPLY_TO"]} if os.environ.get("MAIL_REPLY_TO") else {})})
        r.raise_for_status()
        return
    host = os.environ.get("SMTP_HOST")
    if not host:
        raise RuntimeError("No email provider is set (RESEND_API_KEY or SMTP_HOST)")
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = sender, to, subject
    if os.environ.get("MAIL_REPLY_TO"):
        msg["Reply-To"] = os.environ["MAIL_REPLY_TO"]
    msg.set_content(text)
    if html:
        msg.add_alternative(html, subtype="html")
    port = int(os.environ.get("SMTP_PORT", "587"))
    with smtplib.SMTP(host, port, timeout=30) as s:
        if port != 25:
            s.starttls()
        if os.environ.get("SMTP_USER"):
            s.login(os.environ["SMTP_USER"], os.environ.get("SMTP_PASSWORD", ""))
        s.send_message(msg)
