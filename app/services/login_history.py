"""Per-user login history for the perimeter risk engine (version 2 model).

The version 2 model compares each login with the same user's earlier SUCCESSFUL logins, exactly as
it was trained (ml/train_risk_model_v2.py): is the country, network (autonomous system), app
signature, operating system or device type new for this user, how many logins came before, and
how long since the last one.

Where the values come from at run time:
  * client IP   - Cloudflare's CF-Connecting-IP header, else the first X-Forwarded-For address,
                  else the socket address
  * country     - Cloudflare's CF-IPCountry header (free on every plan); X-Simulate-Country
                  overrides it for demonstrations
  * network     - the client IP looked up in the IPtoASN table (app/ml/ip2asn-combined.tsv.gz,
                  public domain, https://iptoasn.com); missing table -> network unknown
  * app signature, OS, device type - from the device details the app already sends

Unknown values never count as "new": a login without a country (local development, or an
address Cloudflare cannot place) and logins recorded before these columns existed carry no
information, so they neither raise nor lower the risk.
"""
from __future__ import annotations

import bisect
import gzip
import ipaddress
import logging
import math
import os
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import case, func, select

from app.models.entities import ContextualLogin

logger = logging.getLogger(__name__)

ASN_TABLE_PATH = os.environ.get(
    "ASN_TABLE_PATH",
    os.path.join(os.path.dirname(__file__), "..", "ml", "ip2asn-combined.tsv.gz"),
)
HISTORY_FEATURES = ["first_login", "log_prior_logins", "log_hours_since_last", "new_country",
                    "new_asn", "new_user_agent", "new_os", "new_device_type"]
_COUNTRY = re.compile(r"^[A-Z0-9]{2}$")


@dataclass
class LoginContext:
    ip: str | None
    country: str | None
    asn: int | None
    user_agent: str | None
    os_family: str | None
    device_type: str = "mobile"
    history: dict | None = None
    simulated_country: bool = False

    def columns(self) -> dict:
        """Values stored on the contextual_logins row so later logins can compare against them."""
        return {"country": self.country, "asn": self.asn, "user_agent": self.user_agent,
                "os_family": self.os_family, "device_type": self.device_type}


class AsnLookup:
    """IP address -> autonomous system number, from the IPtoASN combined table. Loaded on first use."""

    def __init__(self, path: str = ASN_TABLE_PATH):
        self.path = path
        self._lock = threading.Lock()
        self._loaded = False
        self._tables: dict[int, tuple[list[int], list[int], list[int]]] = {}

    def _load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            tables = {4: ([], [], []), 6: ([], [], [])}
            if os.path.exists(self.path):
                try:
                    with gzip.open(self.path, "rt", encoding="utf-8", errors="replace") as handle:
                        for line in handle:
                            parts = line.split("\t", 3)
                            if len(parts) < 3 or parts[2] == "0":
                                continue          # AS 0 means "not routed"
                            start, end = ipaddress.ip_address(parts[0]), ipaddress.ip_address(parts[1])
                            starts, ends, numbers = tables[start.version]
                            starts.append(int(start)); ends.append(int(end)); numbers.append(int(parts[2]))
                    logger.info("ASN table loaded: %s IPv4 and %s IPv6 ranges",
                                len(tables[4][0]), len(tables[6][0]))
                except Exception as exc:          # a damaged file must not stop sign-in
                    logger.warning("ASN table could not be read (%s); network history disabled", exc)
                    tables = {4: ([], [], []), 6: ([], [], [])}
            else:
                logger.warning("ASN table not found at %s; network history disabled", self.path)
            self._tables = tables
            self._loaded = True

    def lookup(self, ip: str | None) -> int | None:
        if not ip:
            return None
        try:
            address = ipaddress.ip_address(ip.strip())
        except ValueError:
            return None
        if not address.is_global:
            return None
        if not self._loaded:
            self._load()
        starts, ends, numbers = self._tables.get(address.version, ([], [], []))
        value = int(address)
        index = bisect.bisect_right(starts, value) - 1
        if index >= 0 and value <= ends[index]:
            return numbers[index]
        return None


asn_lookup = AsnLookup()


def client_ip(request, reported: str | None = None) -> str | None:
    """The address that reached Cloudflare, falling back to proxies, the socket and the app's report."""
    if request is not None:
        cf_ip = (request.headers.get("cf-connecting-ip") or "").strip()
        if cf_ip:
            return cf_ip[:45]
        forwarded = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
        if forwarded:
            return forwarded[:45]
        if request.client and request.client.host:
            return request.client.host[:45]
    return (reported or None) and reported[:45]


def client_country(request) -> tuple[str | None, bool]:
    """Two-letter country code from Cloudflare, or from X-Simulate-Country for demonstrations."""
    if request is None:
        return None, False
    simulated = (request.headers.get("x-simulate-country") or "").strip().upper()
    if _COUNTRY.match(simulated):
        return simulated, True
    country = (request.headers.get("cf-ipcountry") or "").strip().upper()
    return (country if _COUNTRY.match(country) and country != "XX" else None), False


def app_signature(raw: dict | None) -> str | None:
    """The app's equivalent of a browser user-agent string: device and OS version."""
    raw = raw or {}
    brand = " ".join(str(raw.get("device_brand") or "").split())
    model = " ".join(str(raw.get("device_model") or "").split())
    os_name = " ".join(str(raw.get("os_name") or "").split())
    if "release-keys" in os_name.lower() or "test-keys" in os_name.lower():
        os_name = "Android"
    version = " ".join(str(raw.get("os_version") or "").split())
    if not (brand or model or os_name):
        return None
    return f"MzaziCare ({brand} {model}; {os_name} {version})".replace("  ", " ")[:200]


def build_context(request, raw: dict | None, *, risk_ip: str | None, os_family: str | None,
                  reported_ip: str | None = None) -> LoginContext:
    ip = client_ip(request, reported_ip)
    country, simulated = client_country(request)
    # A simulated IP (demonstration header) is looked up too, so the demo exercises the same path.
    asn = asn_lookup.lookup(risk_ip) or asn_lookup.lookup(ip)
    return LoginContext(ip=ip, country=country, asn=asn, user_agent=app_signature(raw),
                        os_family=os_family, simulated_country=simulated)


def features_from_summary(summary: dict, ctx: LoginContext, now: datetime | None = None) -> dict:
    """Turn the aggregate of a user's earlier successful logins into the model's history features."""
    prior = int(summary.get("prior") or 0)
    if prior == 0:
        # As in training: a user's first login is new in every respect (unknown values excepted).
        return {"first_login": 1.0, "log_prior_logins": 0.0, "log_hours_since_last": 0.0,
                "new_country": 1.0 if ctx.country else 0.0, "new_asn": 1.0 if ctx.asn else 0.0,
                "new_user_agent": 1.0 if ctx.user_agent else 0.0,
                "new_os": 1.0 if ctx.os_family else 0.0, "new_device_type": 1.0}
    now = now or datetime.now(timezone.utc)
    last = summary.get("last_login")
    hours = 0.0
    if last is not None:
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        hours = max((now - last).total_seconds() / 3600.0, 0.0)

    def novel(value, recorded_key, matched_key) -> float:
        # New only if the value is known, the user has recorded values, and none of them match.
        if value is None or not summary.get(recorded_key):
            return 0.0
        return 0.0 if summary.get(matched_key) else 1.0

    return {"first_login": 0.0, "log_prior_logins": math.log1p(prior),
            "log_hours_since_last": math.log1p(hours),
            "new_country": novel(ctx.country, "countries_recorded", "country_seen"),
            "new_asn": novel(ctx.asn, "asns_recorded", "asn_seen"),
            "new_user_agent": novel(ctx.user_agent, "agents_recorded", "agent_seen"),
            "new_os": novel(ctx.os_family, "os_recorded", "os_seen"),
            "new_device_type": novel(ctx.device_type, "devices_recorded", "device_seen")}


def _seen(column, value):
    if value is None:
        return func.max(0)
    return func.max(case((column == value, 1), else_=0))


async def lookup_history(db, user_id, ctx: LoginContext) -> dict | None:
    """History features for this login from the user's earlier successful logins (one query)."""
    try:
        C = ContextualLogin
        statement = select(
            func.count(C.login_id), func.max(C.login_time),
            func.count(C.country), _seen(C.country, ctx.country),
            func.count(C.asn), _seen(C.asn, ctx.asn),
            func.count(C.user_agent), _seen(C.user_agent, ctx.user_agent),
            func.count(C.os_family), _seen(C.os_family, ctx.os_family),
            func.count(C.device_type), _seen(C.device_type, ctx.device_type),
        ).where(C.user_id == user_id, C.is_successful.is_(True))
        row = (await db.execute(statement)).one()
    except Exception as exc:
        logger.warning("Login history unavailable (%s); scoring without history", exc)
        return None
    summary = dict(zip(["prior", "last_login", "countries_recorded", "country_seen",
                        "asns_recorded", "asn_seen", "agents_recorded", "agent_seen",
                        "os_recorded", "os_seen", "devices_recorded", "device_seen"], row))
    return features_from_summary(summary, ctx)


async def latest_challenge_columns(db, user_id, fingerprint: str) -> dict | None:
    """Context of the login that triggered the pending email challenge, so a verified login
    becomes part of the user's history (a genuine traveller is not challenged every time)."""
    try:
        C = ContextualLogin
        row = (await db.execute(
            select(C.ip_address, C.country, C.asn, C.user_agent, C.os_family, C.device_type)
            .where(C.user_id == user_id, C.device_fingerprint == fingerprint, C.is_successful.is_(False))
            .order_by(C.login_time.desc()).limit(1)
        )).first()
    except Exception as exc:
        logger.warning("Challenged login context unavailable (%s)", exc)
        return None
    if row is None:
        return None
    return {"ip": row[0], "country": row[1], "asn": row[2], "user_agent": row[3],
            "os_family": row[4], "device_type": row[5]}
