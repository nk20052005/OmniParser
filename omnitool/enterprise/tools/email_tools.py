"""
Email operation tools.

Handles reading, sending, replying, forwarding, and searching emails
via IMAP/SMTP.
"""

import email
import imaplib
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class _EmailBase(BaseTool):
    """Base class for email tools with shared connection config."""

    def __init__(
        self,
        smtp_server: str = "",
        smtp_port: int = 587,
        imap_server: str = "",
        username: str = "",
        password: str = "",
    ):
        self._smtp_server = smtp_server
        self._smtp_port = smtp_port
        self._imap_server = imap_server
        self._username = username
        self._password = password


class EmailReadTool(_EmailBase):
    name = "email_read"
    description = "Read emails from mailbox"
    required_params: list[str] = []
    optional_params = ["folder", "filter", "count"]

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        folder = params.get("folder", "INBOX")
        mail_filter = params.get("filter", "UNSEEN")
        count = int(params.get("count", 10))

        try:
            imap = imaplib.IMAP4_SSL(self._imap_server)
            imap.login(self._username, self._password)
            imap.select(folder)

            criterion = "(UNSEEN)" if mail_filter == "unread" else "ALL"
            _, data = imap.search(None, criterion)
            mail_ids = data[0].split()

            emails = []
            for mid in mail_ids[-count:]:
                _, msg_data = imap.fetch(mid, "(RFC822)")
                raw = msg_data[0][1]
                msg = email.message_from_bytes(raw)
                emails.append({
                    "id": mid.decode(),
                    "from": msg.get("From", ""),
                    "to": msg.get("To", ""),
                    "subject": msg.get("Subject", ""),
                    "date": msg.get("Date", ""),
                    "snippet": self._get_body(msg)[:200],
                })

            imap.logout()
            return ToolResult(
                success=True,
                data={"emails": emails, "count": len(emails), "folder": folder},
            )
        except Exception as e:
            logger.exception("Failed to read emails")
            return ToolResult(success=False, error=str(e))

    @staticmethod
    def _get_body(msg: email.message.Message) -> str:
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True)
                    return payload.decode(errors="replace") if payload else ""
        else:
            payload = msg.get_payload(decode=True)
            return payload.decode(errors="replace") if payload else ""
        return ""


class EmailSendTool(_EmailBase):
    name = "email_send"
    description = "Send an email"
    required_params = ["recipient", "subject", "body"]
    optional_params = ["cc"]

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        try:
            msg = MIMEMultipart()
            msg["From"] = self._username
            msg["To"] = params["recipient"]
            msg["Subject"] = params["subject"]
            if params.get("cc"):
                msg["Cc"] = params["cc"]
            msg.attach(MIMEText(params["body"], "plain"))

            with smtplib.SMTP(self._smtp_server, self._smtp_port) as server:
                server.starttls()
                server.login(self._username, self._password)
                recipients = [params["recipient"]]
                if params.get("cc"):
                    recipients.extend(params["cc"].split(","))
                server.sendmail(self._username, recipients, msg.as_string())

            return ToolResult(
                success=True,
                data={
                    "recipient": params["recipient"],
                    "subject": params["subject"],
                },
                message=f"Email sent to {params['recipient']}.",
            )
        except Exception as e:
            logger.exception("Failed to send email")
            return ToolResult(success=False, error=str(e))


class EmailReplyTool(_EmailBase):
    name = "email_reply"
    description = "Reply to an email"
    required_params = ["email_id", "body"]

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        try:
            # Fetch original email to get reply-to info
            imap = imaplib.IMAP4_SSL(self._imap_server)
            imap.login(self._username, self._password)
            imap.select("INBOX")
            _, msg_data = imap.fetch(params["email_id"].encode(), "(RFC822)")
            raw = msg_data[0][1]
            original = email.message_from_bytes(raw)
            imap.logout()

            reply_to = original.get("Reply-To") or original.get("From", "")
            subject = original.get("Subject", "")
            if not subject.lower().startswith("re:"):
                subject = f"Re: {subject}"

            msg = MIMEMultipart()
            msg["From"] = self._username
            msg["To"] = reply_to
            msg["Subject"] = subject
            msg["In-Reply-To"] = original.get("Message-ID", "")
            msg.attach(MIMEText(params["body"], "plain"))

            with smtplib.SMTP(self._smtp_server, self._smtp_port) as server:
                server.starttls()
                server.login(self._username, self._password)
                server.sendmail(self._username, [reply_to], msg.as_string())

            return ToolResult(
                success=True,
                data={"replied_to": reply_to, "subject": subject},
                message=f"Reply sent to {reply_to}.",
            )
        except Exception as e:
            logger.exception("Failed to reply to email")
            return ToolResult(success=False, error=str(e))


class EmailForwardTool(_EmailBase):
    name = "email_forward"
    description = "Forward an email"
    required_params = ["email_id", "recipient"]
    optional_params = ["body"]

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        try:
            imap = imaplib.IMAP4_SSL(self._imap_server)
            imap.login(self._username, self._password)
            imap.select("INBOX")
            _, msg_data = imap.fetch(params["email_id"].encode(), "(RFC822)")
            raw = msg_data[0][1]
            original = email.message_from_bytes(raw)
            imap.logout()

            subject = original.get("Subject", "")
            if not subject.lower().startswith("fwd:"):
                subject = f"Fwd: {subject}"

            body = params.get("body", "")
            original_body = EmailReadTool._get_body(original)
            full_body = f"{body}\n\n--- Forwarded Message ---\n{original_body}"

            msg = MIMEMultipart()
            msg["From"] = self._username
            msg["To"] = params["recipient"]
            msg["Subject"] = subject
            msg.attach(MIMEText(full_body, "plain"))

            with smtplib.SMTP(self._smtp_server, self._smtp_port) as server:
                server.starttls()
                server.login(self._username, self._password)
                server.sendmail(self._username, [params["recipient"]], msg.as_string())

            return ToolResult(
                success=True,
                data={"forwarded_to": params["recipient"], "subject": subject},
                message=f"Email forwarded to {params['recipient']}.",
            )
        except Exception as e:
            logger.exception("Failed to forward email")
            return ToolResult(success=False, error=str(e))


class EmailSearchTool(_EmailBase):
    name = "email_search"
    description = "Search emails in mailbox"
    required_params = ["query"]
    optional_params = ["folder"]

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        folder = params.get("folder", "INBOX")
        query = params["query"]
        try:
            imap = imaplib.IMAP4_SSL(self._imap_server)
            imap.login(self._username, self._password)
            imap.select(folder)

            # Search in subject and from
            _, data = imap.search(None, f'(OR SUBJECT "{query}" FROM "{query}")')
            mail_ids = data[0].split()

            emails = []
            for mid in mail_ids[-20:]:
                _, msg_data = imap.fetch(mid, "(RFC822)")
                raw = msg_data[0][1]
                msg = email.message_from_bytes(raw)
                emails.append({
                    "id": mid.decode(),
                    "from": msg.get("From", ""),
                    "subject": msg.get("Subject", ""),
                    "date": msg.get("Date", ""),
                })

            imap.logout()
            return ToolResult(
                success=True,
                data={"emails": emails, "count": len(emails), "query": query},
            )
        except Exception as e:
            logger.exception("Failed to search emails")
            return ToolResult(success=False, error=str(e))


def create_email_tools(
    smtp_server: str = "",
    smtp_port: int = 587,
    imap_server: str = "",
    username: str = "",
    password: str = "",
) -> list[BaseTool]:
    """Factory function to create all email tools."""
    args = (smtp_server, smtp_port, imap_server, username, password)
    return [
        EmailReadTool(*args),
        EmailSendTool(*args),
        EmailReplyTool(*args),
        EmailForwardTool(*args),
        EmailSearchTool(*args),
    ]
