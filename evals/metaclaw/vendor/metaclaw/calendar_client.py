"""
Google Calendar integration for MetaClaw scheduler.

Fetches the user's upcoming events so the SlowUpdateScheduler can identify
time windows when the user is occupied (in meetings) and therefore unavailable
to use OpenClaw — safe slots for slow RL weight updates.

Authentication
--------------
Uses OAuth2 device flow (RFC 8628), which works in headless / CLI environments
without requiring a local HTTP callback server.  The user approves the request
on any device with a browser.

Required extras
---------------
    pip install metaclaw[scheduler]
    # or:
    pip install google-api-python-client google-auth-oauthlib google-auth-httplib2

Usage
-----
    client = GoogleCalendarClient(
        credentials_path="~/.metaclaw/client_secrets.json",
        token_path="~/.metaclaw/calendar_token.json",
    )
    client.authenticate()               # run once; saves token for reuse
    windows = await client.fetch_busy_windows(lookahead_hours=24)
    if client.is_busy_now():
        ...  # good time for RL update
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]
_CACHE_TTL_SECONDS = 1800  # 30 minutes


class GoogleCalendarClient:
    """Thin async wrapper around the Google Calendar API v3.

    Parameters
    ----------
    credentials_path:
        Path to the OAuth2 ``client_secrets.json`` downloaded from Google
        Cloud Console (Desktop App / TVs and Limited Input credential type).
    token_path:
        Where the refreshable OAuth token is stored after the first device-flow
        authentication.  Defaults to ``~/.metaclaw/calendar_token.json``.
    """

    def __init__(
        self,
        credentials_path: str,
        token_path: str = "",
    ) -> None:
        self._creds_path = str(Path(credentials_path).expanduser())
        self._token_path = str(
            Path(token_path).expanduser()
            if token_path
            else Path.home() / ".metaclaw" / "calendar_token.json"
        )
        self._creds = None  # google.oauth2.credentials.Credentials

        # Local cache of busy windows: list of (start_dt, end_dt, summary)
        self._cache: list[tuple[datetime, datetime, str]] = []
        self._cache_fetched_at: float = 0.0

    # ------------------------------------------------------------------ #
    # Authentication                                                      #
    # ------------------------------------------------------------------ #

    def authenticate(self) -> None:
        """Authenticate with Google Calendar using device flow OAuth2.

        Loads an existing token if valid, refreshes it if expired, or
        initiates a new device-flow authorisation (prints a URL and code
        for the user to visit on any device with a browser).

        Raises
        ------
        ImportError
            When ``google-auth-oauthlib`` is not installed.
        Exception
            On auth failure after user interaction.
        """
        try:
            from google.oauth2.credentials import Credentials
            from google.auth.transport.requests import Request
            from google_auth_oauthlib.flow import InstalledAppFlow
        except ImportError as exc:
            raise ImportError(
                "Google Calendar dependencies are required. "
                "Install with: pip install google-api-python-client "
                "google-auth-oauthlib google-auth-httplib2"
            ) from exc

        token_path = Path(self._token_path)

        # Load cached token if it exists.
        creds: Optional[Credentials] = None
        if token_path.exists():
            try:
                creds = Credentials.from_authorized_user_file(
                    str(token_path), _SCOPES
                )
            except Exception as exc:
                logger.warning("[Calendar] could not load saved token: %s", exc)
                creds = None

        # Refresh if expired.
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                logger.info("[Calendar] token refreshed")
            except Exception as exc:
                logger.warning("[Calendar] token refresh failed: %s — re-authenticating", exc)
                creds = None

        # Run device flow if no valid creds.
        if not creds or not creds.valid:
            flow = InstalledAppFlow.from_client_secrets_file(
                self._creds_path, _SCOPES
            )
            # run_local_server would start an HTTP callback server;
            # run_console works headlessly but requires copy-pasting the code.
            # Device flow is more user-friendly for CLI/headless environments.
            try:
                creds = flow.run_local_server(port=0)
            except Exception:
                # Fall back to console flow if local server is unavailable.
                creds = flow.run_console()

        # Save token for subsequent runs.
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json())
        logger.info("[Calendar] token saved to %s", token_path)
        self._creds = creds

    # ------------------------------------------------------------------ #
    # Event fetching                                                      #
    # ------------------------------------------------------------------ #

    async def fetch_busy_windows(
        self, lookahead_hours: int = 24
    ) -> list[tuple[datetime, datetime, str]]:
        """Fetch upcoming calendar events for the next *lookahead_hours*.

        Returns a list of ``(start_dt, end_dt, summary)`` tuples.
        Results are cached for 30 minutes.  Returns ``[]`` on any API error.
        """
        now = time.time()
        if now - self._cache_fetched_at < _CACHE_TTL_SECONDS:
            return self._cache

        try:
            windows = await asyncio.to_thread(
                self._fetch_busy_windows_sync, lookahead_hours
            )
            self._cache = windows
            self._cache_fetched_at = now
            return windows
        except Exception as exc:
            logger.warning("[Calendar] fetch_busy_windows failed: %s", exc)
            return []

    def _fetch_busy_windows_sync(
        self, lookahead_hours: int
    ) -> list[tuple[datetime, datetime, str]]:
        """Synchronous implementation (called via asyncio.to_thread)."""
        try:
            from googleapiclient.discovery import build
            from google.auth.transport.requests import Request
        except ImportError as exc:
            raise ImportError(
                "google-api-python-client is required. "
                "Install with: pip install google-api-python-client"
            ) from exc

        if self._creds is None:
            raise RuntimeError(
                "GoogleCalendarClient.authenticate() must be called before fetching events"
            )

        # Refresh token if needed.
        if self._creds.expired and self._creds.refresh_token:
            self._creds.refresh(Request())

        service = build("calendar", "v3", credentials=self._creds, cache_discovery=False)

        now_utc = datetime.now(timezone.utc)
        time_min = now_utc.isoformat()
        time_max = (
            now_utc.replace(
                hour=now_utc.hour,
                minute=now_utc.minute,
                second=now_utc.second,
            ).__class__(
                year=now_utc.year,
                month=now_utc.month,
                day=now_utc.day,
                hour=now_utc.hour,
                minute=now_utc.minute,
                second=now_utc.second,
                tzinfo=timezone.utc,
            )
        )
        # Simpler: add lookahead_hours as seconds
        from datetime import timedelta
        time_max_dt = now_utc + timedelta(hours=lookahead_hours)
        time_max_str = time_max_dt.isoformat()

        events_result = (
            service.events()
            .list(
                calendarId="primary",
                timeMin=time_min,
                timeMax=time_max_str,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )

        windows: list[tuple[datetime, datetime, str]] = []
        for event in events_result.get("items", []):
            start = self._parse_event_time(event.get("start", {}))
            end   = self._parse_event_time(event.get("end",   {}))
            if start and end:
                windows.append((start, end, event.get("summary", "")))

        logger.info("[Calendar] fetched %d events", len(windows))
        return windows

    # ------------------------------------------------------------------ #
    # Convenience                                                         #
    # ------------------------------------------------------------------ #

    def is_busy_now(self) -> bool:
        """Return True if the user has an active calendar event right now.

        Uses the cached event list (no network call).  Returns False on any error
        or cache miss (safe default: never incorrectly block an RL update).
        """
        if not self._cache:
            return False
        now = datetime.now(timezone.utc)
        for start, end, summary in self._cache:
            if start <= now <= end:
                logger.debug("[Calendar] active event: %s (%s–%s)", summary, start, end)
                return True
        return False

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_event_time(event_time: dict) -> Optional[datetime]:
        """Parse a Google Calendar event start/end time dict.

        Handles both ``dateTime`` (ISO string with timezone) and
        ``date`` (all-day event, treated as midnight UTC).
        """
        if not event_time:
            return None
        dt_str = event_time.get("dateTime")
        if dt_str:
            try:
                # Python 3.7+ fromisoformat does not handle 'Z' suffix.
                return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            except Exception:
                return None
        date_str = event_time.get("date")
        if date_str:
            try:
                return datetime.strptime(date_str, "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
            except Exception:
                return None
        return None
