"""Email for FarmScan's weekly alerts, through the same email service as Geohub and the other GeoEstate apps, with the
same variables and the same order of preference: Brevo (BREVO_API_KEY), then Resend (RESEND_API_KEY), then SMTP
(SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD); from MAIL_FROM, replies to MAIL_REPLY_TO. With none set, nothing is
sent (the alerts can still be read in FarmScan)."""
import os
import smtplib
import time
from email.message import EmailMessage
from email.utils import parseaddr

import httpx

BREVO_URL = "https://api.brevo.com/v3/smtp/email"
RETRY = (429, 500, 502, 503, 504)
OUTBOX: list[dict] = []          # tests read what would have been sent


def backend() -> str:
    if os.environ.get("MAIL_TEST_OUTBOX"):
        return "test"
    return ("brevo" if os.environ.get("BREVO_API_KEY") else "resend" if os.environ.get("RESEND_API_KEY")
            else "smtp" if os.environ.get("SMTP_HOST") else "none")


def configured() -> bool:
    return backend() != "none"


def _sender() -> tuple[str, str]:
    name, email = parseaddr(os.environ.get("MAIL_FROM", "FarmScan <no-reply@example.org>"))
    return name or "GeoEstate FarmScan", email


def send(to: str, subject: str, text: str, html: str | None = None) -> None:
    how = backend()
    reply = os.environ.get("MAIL_REPLY_TO", "")
    name, email = _sender()
    if how == "test":
        OUTBOX.append({"to": to, "subject": subject, "text": text, "html": html, "from": email})
        return
    if how == "brevo":
        payload = {"sender": {"name": name, "email": email}, "to": [{"email": to}], "subject": subject, "textContent": text,
                   **({"htmlContent": html} if html else {}), **({"replyTo": {"email": parseaddr(reply)[1]}} if reply else {})}
        for attempt in range(3):
            r = httpx.post(BREVO_URL, json=payload, timeout=20,
                           headers={"api-key": os.environ["BREVO_API_KEY"], "accept": "application/json"})
            if r.status_code < 400:
                return
            if r.status_code in RETRY and attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise RuntimeError(f"Brevo refused the email ({r.status_code}): {r.text[:300]}")
    if how == "resend":
        r = httpx.post("https://api.resend.com/emails", headers={"Authorization": f"Bearer {os.environ['RESEND_API_KEY']}"},
                       timeout=20, json={"from": f"{name} <{email}>", "to": [to], "subject": subject, "text": text,
                                         **({"html": html} if html else {}), **({"reply_to": reply} if reply else {})})
        r.raise_for_status()
        return
    if how == "smtp":
        msg = EmailMessage()
        msg["From"], msg["To"], msg["Subject"] = f"{name} <{email}>", to, subject
        if reply:
            msg["Reply-To"] = reply
        msg.set_content(text)
        if html:
            msg.add_alternative(html, subtype="html")
        port = int(os.environ.get("SMTP_PORT", "587"))
        smtp = smtplib.SMTP_SSL(os.environ["SMTP_HOST"], 465, timeout=30) if port == 465 else smtplib.SMTP(os.environ["SMTP_HOST"], port, timeout=30)
        with smtp as s:
            if port not in (25, 465):
                s.starttls()
            if os.environ.get("SMTP_USER"):
                s.login(os.environ["SMTP_USER"], os.environ.get("SMTP_PASSWORD", ""))
            s.send_message(msg)
        return
    raise RuntimeError("No email service is set (BREVO_API_KEY, RESEND_API_KEY or SMTP_HOST)")
