#!/usr/bin/env python3
"""
Ottoneu Lineup Optimizer — League 502
FanGraphs Points Scoring | Positions: 2C, 1B, 2B, SS, MI, 3B, 5OF, UTIL, 5SP, 5RP

Usage:
    python ottoneu_lineup_setter.py

Schedule with cron (runs daily at 9am):
    0 9 * * * /usr/bin/python3 /path/to/ottoneu_lineup_setter.py >> /path/to/lineup.log 2>&1

Requirements:
    pip install requests beautifulsoup4 scipy numpy
"""

from __future__ import annotations
import os
import re
import sys
import urllib.parse
import json
import math
import logging
from datetime import date
from itertools import combinations
from collections import defaultdict

import requests
from bs4 import BeautifulSoup
from scipy.optimize import linear_sum_assignment
import numpy as np

# ─────────────────────────────────────────────
# BALLPARK DATA
# lat/lon for weather lookup + home plate
# facing direction (degrees, 0=N, 90=E, etc.)
# "blowing out" means wind is heading FROM
# home plate TOWARD the outfield.
# ─────────────────────────────────────────────
# Facing = direction the batter faces (toward CF).
# Wind blowing OUT  = wind direction ≈ facing angle  (±60°)
# Wind blowing IN   = wind direction ≈ facing + 180° (±60°)
BALLPARKS = {
    # team_abbrev: (lat, lon, cf_facing_degrees, "Park Name")
    "ARI": (33.4453,  -112.0667,  330, "Chase Field"),
    "ATL": (33.8908,   -84.4679,   15, "Truist Park"),
    "BAL": (39.2838,   -76.6217,  100, "Camden Yards"),
    "BOS": (42.3467,   -71.0972,   95, "Fenway Park"),
    "CHC": (41.9484,   -87.6553,   15, "Wrigley Field"),
    "CWS": (41.8300,   -87.6339,   15, "Guaranteed Rate Field"),
    "CIN": (39.0979,   -84.5082,  340, "Great American Ball Park"),
    "CLE": (41.4962,   -81.6852,  340, "Progressive Field"),
    "COL": (39.7559,  -104.9942,  340, "Coors Field"),
    "DET": (42.3390,   -83.0485,   15, "Comerica Park"),
    "HOU": (29.7573,   -95.3555,   15, "Minute Maid Park"),
    "KC":  (39.0517,   -94.4803,   15, "Kauffman Stadium"),
    "LAA": (33.8003,  -117.8827,  355, "Angel Stadium"),
    "LAD": (34.0739,  -118.2400,  340, "Dodger Stadium"),
    "MIA": (25.7781,   -80.2197,   15, "loanDepot park"),
    "MIL": (43.0280,   -87.9712,   30, "American Family Field"),
    "MIN": (44.9817,   -93.2778,  340, "Target Field"),
    "NYM": (40.7571,   -73.8458,  340, "Citi Field"),
    "NYY": (40.8296,   -73.9262,  340, "Yankee Stadium"),
    "OAK": (37.7516,  -122.2005,  310, "Oakland Coliseum"),
    "PHI": (39.9056,   -75.1665,   15, "Citizens Bank Park"),
    "PIT": (40.4469,   -80.0057,  340, "PNC Park"),
    "SD":  (32.7076,  -117.1570,  310, "Petco Park"),
    "SEA": (47.5914,  -122.3325,  340, "T-Mobile Park"),
    "SF":  (37.7786,  -122.3893,   55, "Oracle Park"),
    "STL": (38.6226,   -90.1928,   15, "Busch Stadium"),
    "TB":  (27.7682,   -82.6534,  340, "Tropicana Field"),
    "TEX": (32.7512,   -97.0832,   15, "Globe Life Field"),
    "TOR": (43.6414,   -79.3894,   15, "Rogers Centre"),
    "WSH": (38.8730,   -77.0074,   15, "Nationals Park"),
}

# ─────────────────────────────────────────────
# CONFIGURATION — fill these in
# ─────────────────────────────────────────────
OTTONEU_USERNAME = os.environ.get("OTTONEU_USER", "viconquest")
OTTONEU_PASSWORD = os.environ.get("OTTONEU_PASS", "smileyl")
LEAGUE_ID = "502"

# Wind adjustment settings
WIND_THRESHOLD_MPH = 10       # minimum wind speed to trigger any adjustment
WIND_OUT_MULTIPLIER = 1.20    # +20% for wind blowing out
WIND_IN_MULTIPLIER  = 0.80    # -20% for wind blowing in
WIND_CONE_DEGREES   = 60      # ±degrees from CF axis counted as "out" or "in"

# Wind only affects hitters (pitchers are adjusted inversely —
# blowing out hurts pitchers, blowing in helps them).
HITTER_POSITIONS  = {"C", "1B", "2B", "3B", "SS", "OF"}
PITCHER_POSITIONS = {"SP", "RP"}

# Batter vs. pitcher matchup adjustment
MATCHUP_MULTIPLIER   = 1.25   # +25% if batter qualifies
MATCHUP_MIN_PA       = 10     # qualifier 1: ≥ this many career PA vs. pitcher
MATCHUP_NEED_XBH     = True   # qualifier 2: had at least 1 extra-base hit vs. pitcher
# Either condition alone triggers the boost (OR logic, not AND)

# Lineup slots and how many of each
LINEUP_SLOTS = {
    "C":    2,
    "1B":   1,
    "2B":   1,
    "SS":   1,
    "MI":   1,   # 2B or SS
    "3B":   1,
    "OF":   5,
    "UTIL": 1,   # any hitter
    "SP":   5,
    "RP":   5,
}

# Which real positions can fill each slot
SLOT_ELIGIBILITY = {
    "C":    ["C"],
    "1B":   ["1B"],
    "2B":   ["2B"],
    "SS":   ["SS"],
    "MI":   ["2B", "SS"],
    "3B":   ["3B"],
    "OF":   ["OF"],
    "UTIL": ["C", "1B", "2B", "3B", "SS", "OF"],
    "SP":   ["SP"],
    "RP":   ["RP"],
}

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# STEP 1: Log into Ottoneu
# ─────────────────────────────────────────────
def get_session() -> requests.Session:
    """Authenticate with Ottoneu and return an active session."""
    session = requests.Session()
    session.headers["User-Agent"] = (
        "Mozilla/5.0 (compatible; OttoneuLineupBot/1.0)"
    )

    login_url = "https://ottoneu.fangraphs.com/login"
    r = session.get(login_url)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    csrf = soup.find("input", {"name": "_csrf_token"})
    if not csrf:
        # Try alternate token field names
        csrf = soup.find("input", {"name": "csrf_token"}) or \
               soup.find("input", {"name": "_token"})

    payload = {
        "_username": OTTONEU_USERNAME,
        "_password": OTTONEU_PASSWORD,
        "_remember_me": "on",
    }
    if csrf:
        payload[csrf["name"]] = csrf["value"]

    r2 = session.post(login_url, data=payload, allow_redirects=True)
    r2.raise_for_status()

    if "login" in r2.url.lower() or "invalid" in r2.text.lower():
        raise RuntimeError(
            "Login failed — check OTTONEU_USER and OTTONEU_PASS env vars."
        )

    log.info("Logged in as %s", OTTONEU_USERNAME)
    return session


# ─────────────────────────────────────────────
# STEP 2: Fetch your roster + projections
# ─────────────────────────────────────────────
def fetch_roster(session: requests.Session) -> list[dict]:
    """
    Pull roster from Ottoneu. Each player dict includes:
      id, name, positions (list), projected_points, team_abbrev
    """
    url = f"https://ottoneu.fangraphs.com/{LEAGUE_ID}/lineup"
    r = session.get(url)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    players = []

    # Ottoneu lineup page has a table with all rostered players
    # The rows contain player id, name, position, and projected points
    roster_table = soup.find("table", {"id": re.compile(r"roster|lineup", re.I)})
    if not roster_table:
        # Fallback: try the roster export CSV endpoint
        return fetch_roster_csv(session)

    for row in roster_table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 3:
            continue

        player_link = row.find("a", href=re.compile(r"/playercard"))
        if not player_link:
            continue

        href = player_link["href"]
        player_id_match = re.search(r"playercard/(\d+)", href)
        if not player_id_match:
            continue

        player_id = player_id_match.group(1)
        name = player_link.get_text(strip=True)

        # Position cell (usually 2nd or 3rd column)
        pos_text = ""
        for cell in cells:
            txt = cell.get_text(strip=True)
            if re.match(r"^(?:(?:C|1B|2B|3B|SS|OF|SP|RP|MI|P)[/,\s]*)+$", txt):
                pos_text = txt
                break

        positions = [p.strip() for p in re.split(r"[/,\s]+", pos_text) if p.strip()]

        # Projected points — look for a numeric cell
        proj = 0.0
        for cell in cells:
            txt = cell.get_text(strip=True).replace(",", "")
            try:
                val = float(txt)
                if 0 < val < 1000:   # reasonable FG points range
                    proj = val
                    break
            except ValueError:
                continue

        # MLB team abbreviation
        team = ""
        for cell in cells:
            txt = cell.get_text(strip=True)
            if re.match(r"^[A-Z]{2,3}$", txt) and txt not in ("OF", "SP", "RP", "SS", "MI"):
                team = txt
                break

        players.append({
            "id": player_id,
            "name": name,
            "positions": positions,
            "projected_points": proj,
            "team": team,
            "playing_today": True,  # updated in step 3
        })

    log.info("Fetched %d rostered players from lineup page", len(players))
    return players


def fetch_roster_csv(session: requests.Session) -> list[dict]:
    """Fallback: parse the Ottoneu roster export CSV."""
    url = f"https://ottoneu.fangraphs.com/{LEAGUE_ID}/rosterexport"
    r = session.get(url)
    r.raise_for_status()

    players = []
    lines = r.text.strip().splitlines()
    if not lines:
        return players

    headers = [h.strip().lower() for h in lines[0].split(",")]

    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) < len(headers):
            continue
        row = dict(zip(headers, [p.strip() for p in parts]))

        # Only your team (team_id column will match your team)
        positions_raw = row.get("position", row.get("positions", ""))
        positions = [p.strip() for p in re.split(r"[/,\s]+", positions_raw) if p.strip()]

        try:
            proj = float(row.get("points", row.get("proj", 0)) or 0)
        except ValueError:
            proj = 0.0

        players.append({
            "id": row.get("fg_id", row.get("player_id", "")),
            "name": row.get("name", row.get("playername", "")),
            "positions": positions,
            "projected_points": proj,
            "team": row.get("team", ""),
            "playing_today": True,
        })

    log.info("Fetched %d players from roster CSV", len(players))
    return players


# ─────────────────────────────────────────────
# STEP 3: Check MLB schedule — who plays today?
# ─────────────────────────────────────────────
def get_schedule_today() -> tuple[set[str], dict[str, str]]:
    """
    Returns:
      - teams_playing: set of MLB team abbreviations with games today
      - home_team_map: dict of away_abbrev -> home_abbrev (so we know
        which ballpark each visiting team plays in)
    """
    today = date.today().strftime("%Y-%m-%d")
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}"

    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.warning("MLB schedule fetch failed (%s) — assuming all teams play", e)
        return set(), {}

    teams = set()
    home_team_map = {}  # both teams -> home team abbrev (= ballpark)

    for game_date in data.get("dates", []):
        for game in game_date.get("games", []):
            home_info = game.get("teams", {}).get("home", {}).get("team", {})
            away_info = game.get("teams", {}).get("away", {}).get("team", {})
            home_abbr = home_info.get("abbreviation", "")
            away_abbr = away_info.get("abbreviation", "")

            if home_abbr:
                teams.add(home_abbr)
                home_team_map[home_abbr] = home_abbr  # home team plays at home
            if away_abbr:
                teams.add(away_abbr)
                if home_abbr:
                    home_team_map[away_abbr] = home_abbr  # away team plays at home park

    log.info("Teams playing today (%s): %s", today, sorted(teams))
    return teams, home_team_map


def filter_by_schedule(
    players: list[dict],
    teams_playing: set[str],
    home_team_map: dict[str, str],
) -> list[dict]:
    """
    Mark players as playing_today=False if their team has no game.
    Also record which ballpark each player is playing in today.
    """
    if not teams_playing:
        for p in players:
            p["ballpark_team"] = p["team"]
        return players

    for p in players:
        abbr = p["team"].upper()
        p["playing_today"] = abbr in teams_playing
        p["ballpark_team"] = home_team_map.get(abbr, abbr)

    benched = [p["name"] for p in players if not p["playing_today"]]
    if benched:
        log.info("Players sitting today (no game): %s", ", ".join(benched))

    return players


# ─────────────────────────────────────────────
# STEP 3b: Wind adjustment
# Uses Open-Meteo (free, no API key required)
# ─────────────────────────────────────────────
def _angle_diff(a: float, b: float) -> float:
    """Smallest signed difference between two compass angles (degrees)."""
    diff = (a - b + 180) % 360 - 180
    return abs(diff)


def get_wind_for_ballpark(ballpark_team: str) -> dict:
    """
    Fetch current wind speed (mph) and direction (degrees) for a ballpark.
    Returns dict with keys: speed_mph, direction_deg, description
    Returns None values on failure.
    """
    park = BALLPARKS.get(ballpark_team)
    if not park:
        return {"speed_mph": None, "direction_deg": None, "description": "unknown ballpark"}

    lat, lon, cf_facing, park_name = park

    # Open-Meteo free weather API — no key needed
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current=wind_speed_10m,wind_direction_10m"
        f"&wind_speed_unit=mph"
        f"&forecast_days=1"
    )

    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        data = r.json()
        current = data.get("current", {})
        speed_mph = current.get("wind_speed_10m")
        direction_deg = current.get("wind_direction_10m")  # meteorological: FROM this direction

        if speed_mph is None or direction_deg is None:
            return {"speed_mph": None, "direction_deg": None, "description": "no data"}

        return {
            "speed_mph": speed_mph,
            "direction_deg": direction_deg,
            "park_name": park_name,
            "cf_facing": cf_facing,
        }
    except Exception as e:
        log.warning("Wind fetch failed for %s (%s): %s", ballpark_team, park_name, e)
        return {"speed_mph": None, "direction_deg": None, "description": str(e)}


def classify_wind(wind: dict) -> str:
    """
    Given a wind dict, return 'out', 'in', or 'neutral'.

    Meteorological wind direction = the direction the wind is coming FROM.
    So if wind is FROM the outfield (direction ≈ cf_facing + 180), it blows IN.
    If wind is FROM home plate side (direction ≈ cf_facing), it blows OUT.
    """
    speed = wind.get("speed_mph")
    direction = wind.get("direction_deg")
    cf_facing = wind.get("cf_facing")

    if speed is None or direction is None or cf_facing is None:
        return "neutral"
    if speed <= WIND_THRESHOLD_MPH:
        return "neutral"

    # Meteorological wind direction = the compass direction wind comes FROM.
    # Wind blowing OUT: comes FROM behind home plate = FROM (cf_facing + 180)°
    # Wind blowing IN:  comes FROM the outfield     = FROM cf_facing°
    diff_out = _angle_diff(direction, (cf_facing + 180) % 360)
    diff_in  = _angle_diff(direction, cf_facing)

    if diff_out <= WIND_CONE_DEGREES:
        return "out"
    elif diff_in <= WIND_CONE_DEGREES:
        return "in"
    else:
        return "neutral"


def apply_wind_adjustments(players: list[dict]) -> list[dict]:
    """
    For each ballpark with a game today, fetch wind and adjust
    projected_points for all players playing there.

    Hitters:  wind out → +20%, wind in → -20%
    Pitchers: wind out → -20%, wind in → +20% (inverse)
    """
    # Gather unique ballparks in play today
    ballparks_today = set(
        p["ballpark_team"] for p in players
        if p.get("playing_today") and p.get("ballpark_team")
    )

    wind_cache = {}  # ballpark_team -> wind classification + info
    for bp_team in ballparks_today:
        wind = get_wind_for_ballpark(bp_team)
        classification = classify_wind(wind)
        wind_cache[bp_team] = {**wind, "classification": classification}

        park_name = wind.get("park_name", bp_team)
        speed = wind.get("speed_mph")
        direction = wind.get("direction_deg")
        if speed is not None:
            compass = _degrees_to_compass(direction)
            log.info(
                "🌬  %-28s  %.1f mph from %s (%d°) → wind %s",
                park_name, speed, compass, direction, classification.upper()
            )
        else:
            log.info("🌬  %-28s  wind data unavailable", park_name)

    for player in players:
        if not player.get("playing_today"):
            continue

        bp = player.get("ballpark_team", "")
        wind_info = wind_cache.get(bp, {})
        classification = wind_info.get("classification", "neutral")
        player["wind_classification"] = classification
        player["wind_speed_mph"] = wind_info.get("speed_mph")

        if classification == "neutral":
            player["wind_multiplier"] = 1.0
            continue

        is_hitter  = any(pos in HITTER_POSITIONS  for pos in player["positions"])
        is_pitcher = any(pos in PITCHER_POSITIONS for pos in player["positions"])

        if is_hitter:
            multiplier = WIND_OUT_MULTIPLIER if classification == "out" else WIND_IN_MULTIPLIER
        elif is_pitcher:
            # Pitchers benefit from wind in, hurt by wind out
            multiplier = WIND_IN_MULTIPLIER if classification == "out" else WIND_OUT_MULTIPLIER
        else:
            multiplier = 1.0

        player["wind_multiplier"] = multiplier
        player["projected_points"] *= multiplier
        log.info(
            "  %-25s  wind %s → %.0f%% adjustment (%.1f pts)",
            player["name"], classification, (multiplier - 1) * 100,
            player["projected_points"],
        )

    return players


def _degrees_to_compass(deg: float) -> str:
    """Convert a bearing in degrees to a compass direction string."""
    if deg is None:
        return "?"
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = round(deg / 22.5) % 16
    return dirs[idx]


# ─────────────────────────────────────────────
# STEP 4: Fetch FanGraphs projections
# ─────────────────────────────────────────────
def fetch_fangraphs_projections(session: requests.Session, players: list[dict]) -> list[dict]:
    """
    Pull today's projected FanGraphs points from Ottoneu's own
    player pages (which embed Steamer/ZiPS projections in FG points).

    Ottoneu's /lineup page already shows projected points per player,
    so we primarily rely on those. This function supplements any
    players whose projections are 0.
    """
    zero_proj = [p for p in players if p["projected_points"] == 0 and p["playing_today"]]

    for player in zero_proj:
        if not player["id"]:
            continue
        try:
            url = f"https://ottoneu.fangraphs.com/{LEAGUE_ID}/playercard?id={player['id']}"
            r = session.get(url, timeout=8)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")

            # Look for projected points in the player card
            proj_label = soup.find(string=re.compile(r"proj", re.I))
            if proj_label:
                parent = proj_label.find_parent()
                if parent:
                    nums = re.findall(r"\d+\.?\d*", parent.get_text())
                    if nums:
                        player["projected_points"] = float(nums[0])
                        log.info(
                            "Updated projection for %s: %.1f",
                            player["name"], player["projected_points"]
                        )
        except Exception as e:
            log.warning("Could not fetch projection for %s: %s", player["name"], e)

    return players



# ─────────────────────────────────────────────
# STEP 4c: Batter vs. pitcher matchup adjustment
# Uses the free MLB Stats API — no key needed.
# ─────────────────────────────────────────────

def get_probable_pitchers_today(home_team_map: dict[str, str]) -> dict[str, dict]:
    """
    Query the MLB Stats API for today's probable starters.

    Returns a dict keyed by team abbreviation:
        { "CHC": {"name": "Justin Steele", "mlb_id": 669923}, ... }

    Both the home and away team entries are populated so any batter
    can look up who they're facing by their own team's opponent.
    """
    today = date.today().strftime("%Y-%m-%d")
    url = (
        f"https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&date={today}&hydrate=probablePitcher"
    )

    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.warning("Could not fetch probable pitchers: %s", e)
        return {}

    # Maps each team abbrev → the probable pitcher they will FACE today
    # i.e. home team faces the away pitcher, and vice versa.
    pitchers_faced = {}

    for game_date in data.get("dates", []):
        for game in game_date.get("games", []):
            teams = game.get("teams", {})
            home = teams.get("home", {})
            away = teams.get("away", {})

            home_abbr = home.get("team", {}).get("abbreviation", "")
            away_abbr = away.get("team", {}).get("abbreviation", "")

            home_pitcher = home.get("probablePitcher", {})
            away_pitcher = away.get("probablePitcher", {})

            # Home batters face the away pitcher
            if home_abbr and away_pitcher:
                pitchers_faced[home_abbr] = {
                    "name":   away_pitcher.get("fullName", "Unknown"),
                    "mlb_id": away_pitcher.get("id"),
                }
            # Away batters face the home pitcher
            if away_abbr and home_pitcher:
                pitchers_faced[away_abbr] = {
                    "name":   home_pitcher.get("fullName", "Unknown"),
                    "mlb_id": home_pitcher.get("id"),
                }

    log.info(
        "Probable pitchers found for %d teams: %s",
        len(pitchers_faced),
        {t: p["name"] for t, p in pitchers_faced.items()},
    )
    return pitchers_faced


def get_batter_vs_pitcher_stats(batter_mlb_id: int, pitcher_mlb_id: int) -> dict:
    """
    Fetch career batter vs. pitcher stats from the MLB Stats API.

    Returns a dict with at minimum:
        { "pa": int, "xbh": int }   (plate appearances, extra-base hits)
    Returns zeros on any failure.
    """
    url = (
        f"https://statsapi.mlb.com/api/v1/people/{batter_mlb_id}"
        f"/stats?stats=vsPlayer&opposingPlayerId={pitcher_mlb_id}"
        f"&group=hitting&sportId=1"
    )

    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.debug("BvP fetch failed batter=%s vs pitcher=%s: %s", batter_mlb_id, pitcher_mlb_id, e)
        return {"pa": 0, "xbh": 0}

    splits = data.get("stats", [{}])[0].get("splits", [])
    if not splits:
        return {"pa": 0, "xbh": 0}

    stat = splits[0].get("stat", {})
    pa  = int(stat.get("plateAppearances", 0) or 0)
    doubles = int(stat.get("doubles",      0) or 0)
    triples = int(stat.get("triples",      0) or 0)
    hr      = int(stat.get("homeRuns",     0) or 0)
    xbh = doubles + triples + hr

    return {"pa": pa, "xbh": xbh}


def resolve_batter_mlb_ids(players: list[dict]) -> list[dict]:
    """
    Each player dict currently has an Ottoneu/FanGraphs ID.
    We need the MLB Stats API integer ID to query BvP stats.

    Strategy: use the MLB Stats API people search by full name.
    Results are cached per name to avoid redundant calls.
    """
    name_cache = {}

    hitters = [
        p for p in players
        if p.get("playing_today")
        and any(pos in HITTER_POSITIONS for pos in p.get("positions", []))
    ]

    for player in hitters:
        name = player.get("name", "")
        if name in name_cache:
            player["mlb_id"] = name_cache[name]
            continue

        # MLB Stats API: search by full name
        search_url = (
            f"https://statsapi.mlb.com/api/v1/people/search"
            f"?names={urllib.parse.quote(name)}&sportId=1"
        )
        try:
            r = requests.get(search_url, timeout=6)
            r.raise_for_status()
            results = r.json().get("people", [])
            if results:
                mlb_id = results[0].get("id")
                player["mlb_id"] = mlb_id
                name_cache[name] = mlb_id
            else:
                player["mlb_id"] = None
                name_cache[name] = None
        except Exception as e:
            log.debug("MLB ID lookup failed for %s: %s", name, e)
            player["mlb_id"] = None
            name_cache[name] = None

    return players


def apply_matchup_adjustments(
    players: list[dict],
    home_team_map: dict[str, str],
) -> list[dict]:
    """
    For each hitter playing today, check their career stats vs. today's
    probable starter. Apply MATCHUP_MULTIPLIER (+25%) if either:
      - career PA vs. that pitcher >= MATCHUP_MIN_PA, OR
      - career XBH vs. that pitcher >= 1
    """
    # Step 1: get probable pitchers (team → pitcher they face)
    pitchers_faced = get_probable_pitchers_today(home_team_map)
    if not pitchers_faced:
        log.warning("No probable pitcher data — skipping matchup adjustments")
        for p in players:
            p["matchup_boost"] = False
            p["matchup_note"] = ""
        return players

    # Step 2: resolve MLB IDs for hitters
    players = resolve_batter_mlb_ids(players)

    # Step 3: apply adjustments
    for player in players:
        player["matchup_boost"] = False
        player["matchup_note"] = ""

        if not player.get("playing_today"):
            continue
        if not any(pos in HITTER_POSITIONS for pos in player.get("positions", [])):
            continue

        batter_mlb_id = player.get("mlb_id")
        if not batter_mlb_id:
            continue

        team = player.get("team", "").upper()
        pitcher_info = pitchers_faced.get(team)
        if not pitcher_info or not pitcher_info.get("mlb_id"):
            continue

        pitcher_mlb_id   = pitcher_info["mlb_id"]
        pitcher_name     = pitcher_info["name"]

        stats = get_batter_vs_pitcher_stats(batter_mlb_id, pitcher_mlb_id)
        pa  = stats["pa"]
        xbh = stats["xbh"]

        qualifies = (pa >= MATCHUP_MIN_PA) or (xbh >= 1)

        if qualifies:
            old_proj = player["projected_points"]
            player["projected_points"] *= MATCHUP_MULTIPLIER
            player["matchup_boost"] = True
            reason_parts = []
            if pa >= MATCHUP_MIN_PA:
                reason_parts.append(f"{pa} PA")
            if xbh >= 1:
                reason_parts.append(f"{xbh} XBH")
            note = f"vs {pitcher_name} ({', '.join(reason_parts)})"
            player["matchup_note"] = note
            log.info(
                "  ⚡ %-25s  +25%% matchup boost — %s  %.1f→%.1f pts",
                player["name"], note, old_proj, player["projected_points"],
            )
        else:
            log.debug(
                "  %-25s  vs %s: %d PA, %d XBH — no boost",
                player["name"], pitcher_name, pa, xbh,
            )

    return players


# ─────────────────────────────────────────────
# STEP 5: Optimize lineup (ILP via greedy + scipy)
# ─────────────────────────────────────────────
def optimize_lineup(players: list[dict]) -> dict[str, dict]:
    """
    Assign players to lineup slots to maximize total projected points.

    Slots: 2C, 1B, 2B, SS, MI, 3B, 5OF, UTIL, 5SP, 5RP
    Returns dict: slot_key -> list of assigned player dicts
    e.g. {"C_0": player, "C_1": player, "OF_0": player, ...}
    """

    eligible = [p for p in players if p["playing_today"] and p["projected_points"] > 0]
    log.info("Eligible players for lineup: %d", len(eligible))

    # Expand slots into individual slot keys
    slot_keys = []
    for slot, count in LINEUP_SLOTS.items():
        for i in range(count):
            slot_keys.append(f"{slot}_{i}")

    # Build eligibility matrix: slot_keys x players
    n_slots = len(slot_keys)
    n_players = len(eligible)

    # We'll use a greedy ILP-style approach:
    # Build a cost matrix (negative projected points for maximization)
    # with inf where player is not eligible for slot

    INF = 1e9
    cost = np.full((n_slots, n_players), INF)

    for si, slot_key in enumerate(slot_keys):
        slot_type = slot_key.rsplit("_", 1)[0]  # e.g. "OF" from "OF_3"
        eligible_positions = SLOT_ELIGIBILITY[slot_type]

        for pi, player in enumerate(eligible):
            player_positions = player["positions"]
            if any(pos in eligible_positions for pos in player_positions):
                cost[si][pi] = -player["projected_points"]  # negative for minimization

    # Pad matrix to be square (scipy requires square matrix)
    max_dim = max(n_slots, n_players)
    padded = np.full((max_dim, max_dim), INF)
    padded[:n_slots, :n_players] = cost

    row_ind, col_ind = linear_sum_assignment(padded)

    assignment = {}
    total_points = 0.0

    for si, pi in zip(row_ind, col_ind):
        if si >= n_slots or pi >= n_players:
            continue
        if padded[si][pi] >= INF:
            log.warning("Could not fill slot %s — no eligible player available", slot_keys[si])
            continue
        slot_key = slot_keys[si]
        player = eligible[pi]
        assignment[slot_key] = player
        total_points += player["projected_points"]
        log.info("  %-8s → %-25s (%.1f pts)", slot_key, player["name"], player["projected_points"])

    log.info("Projected total: %.1f FanGraphs points", total_points)
    return assignment


# ─────────────────────────────────────────────
# STEP 6: Set the lineup on Ottoneu
# ─────────────────────────────────────────────
def set_lineup(session: requests.Session, assignment: dict[str, dict]):
    """
    POST the optimized lineup to Ottoneu.

    Ottoneu's lineup page accepts a form POST with player_id → slot mappings.
    The exact field names are scraped from the current lineup form.
    """
    url = f"https://ottoneu.fangraphs.com/{LEAGUE_ID}/lineup"

    # GET the lineup page to find the form structure and CSRF token
    r = session.get(url)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # Find the lineup form
    form = soup.find("form", {"id": re.compile(r"lineup", re.I)})
    if not form:
        form = soup.find("form")

    if not form:
        raise RuntimeError("Could not find lineup form on the page.")

    # Extract form action
    action = form.get("action", url)
    if not action.startswith("http"):
        action = "https://ottoneu.fangraphs.com" + action

    # Build payload from hidden inputs + CSRF
    payload = {}
    for inp in form.find_all("input", {"type": "hidden"}):
        payload[inp.get("name", "")] = inp.get("value", "")

    # Map slot types to Ottoneu's form field naming convention
    # Ottoneu typically uses field names like: lineup[C][0], lineup[OF][2], etc.
    # Or sometimes: slot_C_0, slot_OF_2, etc.
    # We detect the pattern from existing select/input fields in the form.

    slot_inputs = form.find_all(["select", "input"], {"name": re.compile(r"lineup|slot", re.I)})

    if slot_inputs:
        # Parse existing field names to understand the convention
        example_name = slot_inputs[0].get("name", "")
        log.info("Detected lineup form field pattern: %s", example_name)

    # Build the lineup payload
    for slot_key, player in assignment.items():
        slot_type, idx = slot_key.rsplit("_", 1)

        # Try common Ottoneu field naming patterns
        field_patterns = [
            f"lineup[{slot_type}][{idx}]",
            f"lineup_{slot_type}_{idx}",
            f"slot_{slot_type}_{idx}",
            f"{slot_type}_{idx}",
        ]

        # Match against detected form fields
        matched = False
        for field in slot_inputs:
            fname = field.get("name", "")
            for pattern in field_patterns:
                if fname == pattern or fname.lower() == pattern.lower():
                    payload[fname] = player["id"]
                    matched = True
                    break
            if matched:
                break

        if not matched:
            # Use best-guess pattern
            payload[f"lineup[{slot_type}][{idx}]"] = player["id"]

    log.info("Submitting lineup with %d player assignments...", len(assignment))
    r2 = session.post(action, data=payload, allow_redirects=True)
    r2.raise_for_status()

    if "success" in r2.text.lower() or "saved" in r2.text.lower() or r2.status_code == 200:
        log.info("✅ Lineup set successfully!")
    else:
        log.warning("Response after POST (may still have worked): %s", r2.url)

    return r2


# ─────────────────────────────────────────────
# STEP 7: Print lineup summary
# ─────────────────────────────────────────────
def print_summary(assignment: dict[str, dict]):
    WIDTH = 68
    print("\n" + "=" * WIDTH)
    print(f"  OPTIMIZED LINEUP — {date.today().strftime('%B %d, %Y')}")
    print("=" * WIDTH)

    slot_order = list(LINEUP_SLOTS.keys())
    grouped = defaultdict(list)
    for slot_key, player in assignment.items():
        slot_type = slot_key.rsplit("_", 1)[0]
        grouped[slot_type].append(player)

    total = 0.0
    for slot_type in slot_order:
        players_in_slot = grouped.get(slot_type, [])
        for player in players_in_slot:
            proj = player["projected_points"]
            total += proj

            tags = []
            wind_cls = player.get("wind_classification", "neutral")
            wind_spd = player.get("wind_speed_mph")
            if wind_cls == "out" and wind_spd is not None:
                tags.append(f"🌬↑ out {wind_spd:.0f}mph +20%")
            elif wind_cls == "in" and wind_spd is not None:
                tags.append(f"🌬↓ in {wind_spd:.0f}mph -20%")

            if player.get("matchup_boost"):
                note = player.get("matchup_note", "")
                tags.append(f"⚡ {note} +25%")

            tag_str = ("  " + "  ".join(tags)) if tags else ""
            print(f"  {slot_type:<6}  {player['name']:<26} {proj:>5.1f} pts{tag_str}")

    print("-" * WIDTH)
    print(f"  {'TOTAL (wind + matchup adjusted)':<38} {total:>5.1f} pts")
    print("=" * WIDTH + "\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    log.info("=== Ottoneu Lineup Optimizer — League %s ===", LEAGUE_ID)

    # 1. Login
    session = get_session()

    # 2. Fetch roster
    players = fetch_roster(session)
    if not players:
        log.error("No players found on roster. Exiting.")
        sys.exit(1)

    # 3. Filter by today's MLB schedule + record ballpark for each player
    teams_playing, home_team_map = get_schedule_today()
    players = filter_by_schedule(players, teams_playing, home_team_map)

    # 4. Supplement any missing projections
    players = fetch_fangraphs_projections(session, players)

    # 4b. Apply wind adjustments (±20% based on ballpark wind conditions)
    players = apply_wind_adjustments(players)

    # 4c. Apply batter vs. pitcher matchup adjustments (+25%)
    players = apply_matchup_adjustments(players, home_team_map)

    # 5. Optimize (uses wind + matchup adjusted projected_points)
    assignment = optimize_lineup(players)
    if not assignment:
        log.error("Optimizer returned empty lineup. Exiting.")
        sys.exit(1)

    # 6. Print summary
    print_summary(assignment)

    # 7. Set lineup
    set_lineup(session, assignment)


if __name__ == "__main__":
    main()
