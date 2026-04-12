from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from pathlib import Path
import os

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
SCRIPT_DIR = Path(__file__).resolve().parent
CREDENTIALS_FILE = Path(os.environ.get("GMAIL_OAUTH_CLIENT_FILE", SCRIPT_DIR / "bcba/secure/google-oauth-client.json"))
TOKEN_FILE = Path(os.environ.get("GMAIL_TOKEN_FILE", SCRIPT_DIR / "bcba/secure/gmail-token.json"))

creds = None
if TOKEN_FILE.exists():
    creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
        creds = flow.run_local_server(port=0)
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TOKEN_FILE, "w") as token:
        token.write(creds.to_json())

service = build("gmail", "v1", credentials=creds)
profile = service.users().getProfile(userId="me").execute()
print("Authorized Gmail:", profile["emailAddress"])
