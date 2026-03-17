import datetime
import os.path

import pandas as pd

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/calendar"]


def get_calendar_service():

    creds = None

    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:

        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())

        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)

        with open("token.json", "w") as token:
            token.write(creds.to_json())

    service = build("calendar", "v3", credentials=creds)

    return service


def export_to_calendar(timeline_df):

    service = get_calendar_service()

    for _, row in timeline_df.iterrows():

        event = {
            "summary": row["task"],
            "start": {
                "dateTime": row["start_time"].isoformat(),
                "timeZone": "Europe/Warsaw",
            },
            "end": {
                "dateTime": row["end_time"].isoformat(),
                "timeZone": "Europe/Warsaw",
            },
        }

        service.events().insert(
            calendarId="primary",
            body=event
        ).execute()

        print("Event added:", row["task"])

import pandas as pd
from calendar_export import export_to_calendar

if __name__ == "__main__":
    data = {
        "task": ["Gym", "Study AI", "Read"],
        "start_time": [
            pd.Timestamp("2026-03-14 08:00"),
            pd.Timestamp("2026-03-14 10:00"),
            pd.Timestamp("2026-03-14 20:00")
        ],
        "end_time": [
            pd.Timestamp("2026-03-14 09:00"),
            pd.Timestamp("2026-03-14 12:00"),
            pd.Timestamp("2026-03-14 21:00")
        ],
    }

    timeline_df = pd.DataFrame(data)

    export_to_calendar(timeline_df)
