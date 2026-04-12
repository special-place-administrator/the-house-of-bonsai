from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from pathlib import Path
import os

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
SCRIPT_DIR = Path(__file__).resolve().parent
TOKEN_FILE = Path(os.environ.get("GMAIL_TOKEN_FILE", SCRIPT_DIR / "bcba/secure/gmail-token.json"))

creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
service = build("gmail", "v1", credentials=creds)

results = service.users().messages().list(userId="me", maxResults=5, labelIds=["INBOX"]).execute()
messages = results.get("messages", [])

if not messages:
    print("No inbox messages found.")
else:
    for idx, msg in enumerate(messages, start=1):
        full = service.users().messages().get(userId="me", id=msg["id"], format="metadata", metadataHeaders=["Subject", "From", "Date"]).execute()
        headers = {h["name"]: h["value"] for h in full.get("payload", {}).get("headers", [])}
        subject = headers.get("Subject", "(No Subject)")
        sender = headers.get("From", "(Unknown Sender)")
        date = headers.get("Date", "(Unknown Date)")
        print(f"{idx}. Subject: {subject}")
        print(f"   From: {sender}")
        print(f"   Date: {date}")
