"""
scrape.py
---------
Collects NHL skater data for the 2022-23 through 2025-26 regular seasons
from multiple sources and produces a merged, qualification-filtered dataset.

Sources:
  1. Hockey Reference  — core stats (G, A, SOG, BLK, HIT, PIM, +/-) + TOI splits
  2. NHL Stats API     — player height / weight / nationality
  3. HockeyFights.com  — fights per season
  4. NHL Edge API      — skating speed, shot speed (best-effort)

Output: players_raw.csv
Intermediates saved to raw/ so individual steps can be re-run without full re-scrape.

Run:  python scrape.py
"""

import re
import time
import unicodedata
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

# =============================================================================
# Config
# =============================================================================

SEASONS     = [2023, 2024, 2025, 2026]   # HR convention: season end year
SEASON_GP   = {2023: 82, 2024: 82, 2025: 82, 2026: 82}
QUALIFY_PCT = 0.70
HR_DELAY    = 4.0   # seconds between Hockey Reference requests

HERE    = Path(__file__).parent
RAW_DIR = HERE / "raw"
RAW_DIR.mkdir(exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
})

# =============================================================================
# Helpers
# =============================================================================

def normalise_name(name: str) -> str:
    """Strip accents, lowercase, remove non-alpha for cross-source key matching."""
    name = unicodedata.normalize("NFD", str(name))
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = re.sub(r"[^a-z ]", "", name.lower()).strip()
    return re.sub(r"\s+", " ", name)


def mmss_to_min(val) -> float:
    """Convert 'MM:SS' string to decimal minutes."""
    try:
        m, s = str(val).split(":")
        return int(m) + int(s) / 60
    except Exception:
        return np.nan


def load_or_scrape(cache_path: Path, scrape_fn, *args, **kwargs) -> pd.DataFrame:
    """Return cached CSV if it exists, otherwise run scrape_fn and cache result."""
    if cache_path.exists():
        print(f"  [cache] {cache_path.name}")
        return pd.read_csv(cache_path)
    df = scrape_fn(*args, **kwargs)
    if not df.empty:
        df.to_csv(cache_path, index=False)
    return df


def hr_fetch(url: str) -> BeautifulSoup:
    """GET a Hockey Reference page with polite delay."""
    time.sleep(HR_DELAY)
    r = SESSION.get(url, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def hr_parse_table(soup: BeautifulSoup, table_id: str) -> pd.DataFrame:
    """
    Find a table by ID on an HR page (handles tables wrapped in HTML comments).
    Drops repeated header rows. Returns a flat DataFrame.
    """
    tbl = soup.find("table", {"id": table_id})
    if tbl is None:
        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            inner = BeautifulSoup(comment, "html.parser")
            tbl   = inner.find("table", {"id": table_id})
            if tbl:
                break
    if tbl is None:
        raise ValueError(f"Table id='{table_id}' not found")

    df = pd.read_html(StringIO(str(tbl)), header=[0, 1])[0]

    # Flatten multi-level columns: drop generic "Unnamed" top-level labels
    flat = []
    for top, sub in df.columns:
        top_clean = "" if str(top).startswith("Unnamed") else str(top).strip()
        sub_clean = str(sub).strip()
        flat.append(f"{top_clean} {sub_clean}".strip() if top_clean else sub_clean)
    df.columns = flat

    # Drop repeated header rows and rank rows
    rk_col = df.columns[0]
    df = df[df[rk_col] != rk_col].copy()
    df = df[df[rk_col] != "Rk"].reset_index(drop=True)

    return df


def keep_totals(df: pd.DataFrame) -> pd.DataFrame:
    """
    For players traded mid-season HR lists multiple rows; '2TM'/'3TM' etc. is the season total.
    Keep the multi-team total row for traded players, single rows for everyone else.
    """
    team_col = "Tm" if "Tm" in df.columns else "Team"
    multi    = df[team_col].str.match(r"^\d+TM$", na=False)
    traded   = df.loc[multi, "Player"].unique()
    return df[~(df["Player"].isin(traded) & ~multi)].reset_index(drop=True)


# =============================================================================
# 1. Hockey Reference — skater stats
# =============================================================================

def scrape_hr_stats(season: int) -> pd.DataFrame:
    url  = f"https://www.hockey-reference.com/leagues/NHL_{season}_skaters.html"
    print(f"  HR stats {season-1}-{str(season)[-2:]}: {url}")
    soup = hr_fetch(url)
    df   = hr_parse_table(soup, "player_stats")

    # Debug: show what columns arrived so we can adjust if HR changes layout
    print(f"    columns: {list(df.columns)}")

    # Actual HR column names (confirmed from live page):
    # 'Scoring G', 'Scoring A', 'Scoring PTS', 'Goals PPG', 'Assists PP',
    # 'Shots SOG', 'Ice Time ATOI', 'BLK', 'HIT', 'Team'
    out          = pd.DataFrame()
    out["Player"] = df["Player"].str.replace(r"\*$", "", regex=True).str.strip()
    out["Tm"]     = df["Team"]
    out["Pos"]    = df["Pos"]
    out["Age"]    = pd.to_numeric(df.get("Age"), errors="coerce")

    for src, dst in [
        ("GP",          "GP"),
        ("Scoring G",   "G"),
        ("Scoring A",   "A"),
        ("Scoring PTS", "PTS"),
        ("+/-",         "PlusMinus"),
        ("PIM",         "PIM"),
        ("Shots SOG",   "SOG"),
        ("BLK",         "BLK"),
        ("HIT",         "HIT"),
    ]:
        out[dst] = pd.to_numeric(df.get(src), errors="coerce")

    pp_g = pd.to_numeric(df.get("Goals PPG"), errors="coerce").fillna(0)
    pp_a = pd.to_numeric(df.get("Assists PP"), errors="coerce").fillna(0)
    out["PP_PTS"] = pp_g + pp_a

    out["ATOI"]   = df.get("Ice Time ATOI", pd.Series(dtype=str))
    out["Season"]  = season

    out = keep_totals(out)
    out = out[(out["Pos"] != "G") & (out["GP"] > 0)].dropna(subset=["GP"])
    return out.reset_index(drop=True)


# =============================================================================
# 2. Hockey Reference — TOI splits
# =============================================================================

def scrape_hr_toi(season: int) -> pd.DataFrame:
    url  = f"https://www.hockey-reference.com/leagues/NHL_{season}_skaters-time-on-ice.html"
    print(f"  HR TOI  {season-1}-{str(season)[-2:]}: {url}")
    soup = hr_fetch(url)
    df   = hr_parse_table(soup, "stats_toi")

    print(f"    columns: {list(df.columns)}")

    def get_toi_col(df, *candidates):
        for c in candidates:
            if c in df.columns:
                return df[c].apply(mmss_to_min)
        return pd.Series(np.nan, index=df.index)

    # Confirmed HR column names (live page 2023-26):
    # ES/PP/SH columns contain per-game averages in MM:SS format.
    # Overall ATOI is in the unnamed column at position 6 which renders as NaN
    # in the parsed table — we derive it as ES + PP + SH instead.
    out              = pd.DataFrame()
    out["Player"]    = df["Player"].str.replace(r"\*$", "", regex=True).str.strip()
    out["Tm"]        = df.get("Tm", df.get("Team", pd.Series(dtype=str)))
    out["GP"]        = pd.to_numeric(df.get("GP"), errors="coerce")
    out["ES_TOI_GP"] = get_toi_col(df, "Even Strength TOI")
    out["PP_TOI_GP"] = get_toi_col(df, "Power Play TOI")
    out["SH_TOI_GP"] = get_toi_col(df, "Short Handed TOI")
    out["TOI_GP"]    = out["ES_TOI_GP"].fillna(0) + out["PP_TOI_GP"].fillna(0) + out["SH_TOI_GP"].fillna(0)
    out["TOI_GP"]    = out["TOI_GP"].where(out["ES_TOI_GP"].notna())  # NaN if all splits missing
    out["Season"]    = season

    out = keep_totals(out)
    return out.reset_index(drop=True)


# =============================================================================
# 3. NHL Stats API — player bios (height / weight)
# =============================================================================

def fetch_nhl_bios() -> pd.DataFrame:
    """Pull bios for all seasons and deduplicate — physicals don't change."""
    frames = []
    for season in SEASONS:
        sid = f"{season-1}{season}"
        url = f"https://api.nhle.com/stats/rest/en/skater/bios?limit=-1&cayenneExp=seasonId={sid}"
        print(f"  NHL bios {season-1}-{str(season)[-2:]}: {url}")
        time.sleep(1.5)
        try:
            r = SESSION.get(url, timeout=20)
            r.raise_for_status()
            data = r.json().get("data", [])
            if data:
                frames.append(pd.DataFrame(data))
        except Exception as e:
            print(f"    WARNING: {e}")

    if not frames:
        return pd.DataFrame(columns=["Player","Height_in","Weight_lbs","Nationality"])

    raw = pd.concat(frames, ignore_index=True)
    print(f"    bio columns: {list(raw.columns)}")

    # Field names from NHL API (verified against public docs)
    name_col   = next((c for c in raw.columns if "fullname" in c.lower() or c == "skaterFullName"), None)
    height_col = next((c for c in raw.columns if "height" in c.lower()), None)
    weight_col = next((c for c in raw.columns if "weight" in c.lower()), None)
    nat_col    = next((c for c in raw.columns if "nationality" in c.lower()), None)

    out             = pd.DataFrame()
    out["Player"]   = raw[name_col]   if name_col   else ""
    out["Height_in"]  = pd.to_numeric(raw[height_col] if height_col else np.nan, errors="coerce")
    out["Weight_lbs"] = pd.to_numeric(raw[weight_col] if weight_col else np.nan, errors="coerce")
    out["Nationality"]= raw[nat_col]  if nat_col    else ""

    return out.drop_duplicates("Player").reset_index(drop=True)


# =============================================================================
# 4. NHL API — fighting majors (proxy for fights)
# HockeyFights.com is JS-rendered and can't be scraped with requests.
# The NHL penalties endpoint includes majorPenalties which captures
# fighting majors (5-min fighting penalties) alongside other majors.
# =============================================================================

def fetch_fights_season(season: int) -> pd.DataFrame:
    sid = f"{season-1}{season}"
    url = (f"https://api.nhle.com/stats/rest/en/skater/penalties"
           f"?limit=-1&cayenneExp=seasonId={sid}")
    print(f"  NHL penalties {season-1}-{str(season)[-2:]}: {url}")
    time.sleep(1.5)
    try:
        r = SESSION.get(url, timeout=20)
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            print(f"    WARNING: empty response")
            return pd.DataFrame(columns=["Player", "Fights", "Season"])
        df = pd.DataFrame(data)
        out = pd.DataFrame()
        out["Player"] = df["skaterFullName"]
        out["Fights"] = pd.to_numeric(df.get("majorPenalties", 0), errors="coerce").fillna(0).astype(int)
        out["Season"] = season
        print(f"    -> {len(out)} players, total majors: {out['Fights'].sum()}")
        return out
    except Exception as e:
        print(f"    WARNING: {e}")
        return pd.DataFrame(columns=["Player", "Fights", "Season"])


# =============================================================================
# 5. MoneyPuck — 5v5 advanced stats (CF%, xGF%, individual xG)
# Direct CSV download; no HTML parsing needed.
# URL: https://moneypuck.com/moneypuck/playerData/seasonSummary/{year}/regular/skaters.csv
# =============================================================================

def fetch_moneypuck_season(season: int) -> pd.DataFrame:
    """Download MoneyPuck skater CSV for one season and extract 5v5 rows."""
    url = (f"https://moneypuck.com/moneypuck/playerData/seasonSummary"
           f"/{season}/regular/skaters.csv")
    print(f"  MoneyPuck {season-1}-{str(season)[-2:]}: {url}")
    time.sleep(1.5)
    try:
        r = SESSION.get(url, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        df = df[df["situation"] == "5on5"].copy()
        if df.empty:
            print(f"    WARNING: no 5v5 rows for {season}")
            return pd.DataFrame()

        name_col = next(
            (c for c in df.columns if c.lower() in ("name", "playername", "player")),
            None
        )
        if name_col is None:
            print(f"    WARNING: name column not found. Cols: {list(df.columns)}")
            return pd.DataFrame()

        out = pd.DataFrame()
        out["Player"]  = df[name_col]
        out["Season"]  = season
        out["icetime"] = pd.to_numeric(df.get("icetime", 0), errors="coerce").fillna(0)
        for col in ["OnIce_F_shotAttempts", "OnIce_A_shotAttempts",
                    "OnIce_F_xGoals", "OnIce_A_xGoals", "I_F_xGoals"]:
            out[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)

        print(f"    -> {len(out)} players (5v5)")
        return out.reset_index(drop=True)
    except Exception as e:
        print(f"    WARNING: {e}")
        return pd.DataFrame()


# =============================================================================
# 6. NHL Edge API — skating / shot speed (best-effort)
# =============================================================================

EDGE_REPORTS = {
    "skating":  "skater/skating",
    "shooting": "skater/shooting",
}

def fetch_edge_report(report_key: str) -> pd.DataFrame:
    frames = []
    for season in SEASONS:
        sid = f"{season-1}{season}"
        url = (f"https://api.nhle.com/stats/rest/en/{EDGE_REPORTS[report_key]}"
               f"?limit=-1&cayenneExp=seasonId={sid}")
        print(f"  NHL Edge {report_key} {season-1}-{str(season)[-2:]}: {url}")
        time.sleep(1.5)
        try:
            r = SESSION.get(url, timeout=20)
            r.raise_for_status()
            data = r.json().get("data", [])
            if data:
                df           = pd.DataFrame(data)
                df["Season"] = season
                frames.append(df)
                print(f"    -> {len(df)} rows, cols: {list(df.columns)}")
            else:
                print(f"    -> empty response")
        except Exception as e:
            print(f"    WARNING: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# =============================================================================
# Main collection
# =============================================================================

print("=" * 65)
print("STEP 1 — Hockey Reference stats")
print("=" * 65)
stats_frames, toi_frames = [], []
for season in SEASONS:
    sf = load_or_scrape(RAW_DIR / f"hr_stats_{season}.csv", scrape_hr_stats, season)
    tf = load_or_scrape(RAW_DIR / f"hr_toi_{season}.csv",   scrape_hr_toi,   season)
    if not sf.empty: stats_frames.append(sf)
    if not tf.empty: toi_frames.append(tf)

print("\n" + "=" * 65)
print("STEP 2 — NHL API bios")
print("=" * 65)
bios = load_or_scrape(RAW_DIR / "nhl_bios.csv", fetch_nhl_bios)

print("\n" + "=" * 65)
print("STEP 3 — NHL Penalty API (major penalties / fights)")
print("=" * 65)
fight_frames = []
for season in SEASONS:
    ff = load_or_scrape(RAW_DIR / f"fights_{season}.csv", fetch_fights_season, season)
    if not ff.empty: fight_frames.append(ff)

print("\n" + "=" * 65)
print("STEP 4 — MoneyPuck advanced stats (5v5)")
print("=" * 65)
mp_frames = []
for season in SEASONS:
    mf = load_or_scrape(RAW_DIR / f"moneypuck_{season}.csv", fetch_moneypuck_season, season)
    if not mf.empty:
        mp_frames.append(mf)

print("\n" + "=" * 65)
print("STEP 5 — NHL Edge tracking (best-effort)")
print("=" * 65)
edge_skate  = load_or_scrape(RAW_DIR / "edge_skating.csv",  fetch_edge_report, "skating")
edge_shoot  = load_or_scrape(RAW_DIR / "edge_shooting.csv", fetch_edge_report, "shooting")


# =============================================================================
# Merge + qualify + derive stats
# =============================================================================

print("\n" + "=" * 65)
print("STEP 5 — Merge, qualify, derive per-60")
print("=" * 65)

stats_all = pd.concat(stats_frames, ignore_index=True)
toi_all   = pd.concat(toi_frames,   ignore_index=True) if toi_frames else pd.DataFrame()

stats_all["key"] = stats_all["Player"].apply(normalise_name)
if not toi_all.empty:
    toi_all["key"] = toi_all["Player"].apply(normalise_name)
    stats_all = stats_all.merge(
        toi_all[["key","Season","TOI_GP","ES_TOI_GP","PP_TOI_GP","SH_TOI_GP"]],
        on=["key","Season"], how="left"
    )

# ── aggregate across seasons ─────────────────────────────────────────────────
COUNT_COLS = ["GP","G","A","PTS","PIM","SOG","BLK","HIT","PP_PTS","PlusMinus"]

agg = (
    stats_all.groupby("key", as_index=False)
    .agg(
        Name         = ("Player",  "first"),
        Pos          = ("Pos",     "first"),
        First_Season = ("Season",  "min"),
        **{c: (c, "sum") for c in COUNT_COLS if c in stats_all.columns},
    )
)

# Weighted-average TOI per game (by GP)
if "TOI_GP" in stats_all.columns:
    def wavg_toi(grp, col):
        gp  = grp["GP"].fillna(0)
        toi = grp[col].fillna(0)
        w   = gp.sum()
        return (toi * gp).sum() / w if w > 0 else np.nan

    toi_agg = (
        stats_all.groupby("key")
        .apply(lambda g: pd.Series({
            "TOI_GP":    wavg_toi(g, "TOI_GP"),
            "ES_TOI_GP": wavg_toi(g, "ES_TOI_GP"),
            "PP_TOI_GP": wavg_toi(g, "PP_TOI_GP"),
            "SH_TOI_GP": wavg_toi(g, "SH_TOI_GP"),
        }), include_groups=False)
        .reset_index()
    )
    agg = agg.merge(toi_agg, on="key", how="left")

# ── qualification filter ──────────────────────────────────────────────────────
def possible_gp(first_season):
    return sum(v for k, v in SEASON_GP.items() if k >= int(first_season))

agg["Possible_GP"]  = agg["First_Season"].apply(possible_gp)
agg["Qualify_Pct"]  = (agg["GP"] / agg["Possible_GP"]).round(4)
qualified           = agg[agg["Qualify_Pct"] >= QUALIFY_PCT].copy()
print(f"  Total players across seasons : {len(stats_all['key'].unique())}")
print(f"  Qualified (>={QUALIFY_PCT*100:.0f}% of possible GP): {len(qualified)}")

# ── per-60 ────────────────────────────────────────────────────────────────────
if "TOI_GP" in qualified.columns:
    qualified["Total_TOI_min"] = qualified["TOI_GP"] * qualified["GP"]
    def p60(col):
        return (qualified[col] / qualified["Total_TOI_min"] * 60).round(3)
    for src, dst in [("G","G_60"),("A","A_60"),("PTS","PTS_60"),
                     ("SOG","SOG_60"),("BLK","BLK_60"),("HIT","HIT_60"),("PP_PTS","PP_PTS_60")]:
        if src in qualified.columns:
            qualified[dst] = p60(src)
else:
    print("  WARNING: TOI data missing — using per-game stats instead of per-60")
    for src, dst in [("G","G_60"),("A","A_60"),("PTS","PTS_60"),
                     ("SOG","SOG_60"),("BLK","BLK_60"),("HIT","HIT_60"),("PP_PTS","PP_PTS_60")]:
        if src in qualified.columns:
            qualified[dst] = (qualified[src] / qualified["GP"]).round(3)

qualified["PIM_GP"]       = (qualified["PIM"]       / qualified["GP"]).round(3)
qualified["PlusMinus_GP"] = (qualified["PlusMinus"] / qualified["GP"]).round(3)

# ── physicals ─────────────────────────────────────────────────────────────────
if not bios.empty:
    bios["key"] = bios["Player"].apply(normalise_name)
    qualified   = qualified.merge(
        bios[["key","Height_in","Weight_lbs","Nationality"]],
        on="key", how="left"
    )
    qualified["BMI"] = (
        qualified["Weight_lbs"] * 703 / (qualified["Height_in"] ** 2)
    ).round(2)

# ── fights ────────────────────────────────────────────────────────────────────
if fight_frames:
    fights_all      = pd.concat(fight_frames, ignore_index=True)
    fights_all["key"] = fights_all["Player"].apply(normalise_name)
    fights_agg      = fights_all.groupby("key")["Fights"].sum().reset_index()
    qualified        = qualified.merge(fights_agg, on="key", how="left")
    qualified["Fights"] = qualified["Fights"].fillna(0).astype(int)
else:
    qualified["Fights"] = 0

# ── MoneyPuck 5v5 advanced stats ──────────────────────────────────────────────
if mp_frames:
    mp_all      = pd.concat(mp_frames, ignore_index=True)
    mp_all["key"] = mp_all["Player"].apply(normalise_name)
    mp_agg = mp_all.groupby("key", as_index=False).agg(
        icetime_5v5 = ("icetime",                "sum"),
        CF_F        = ("OnIce_F_shotAttempts",   "sum"),
        CF_A        = ("OnIce_A_shotAttempts",   "sum"),
        xG_F        = ("OnIce_F_xGoals",         "sum"),
        xG_A        = ("OnIce_A_xGoals",         "sum"),
        ixG_raw     = ("I_F_xGoals",             "sum"),
    )
    total_cf = mp_agg["CF_F"] + mp_agg["CF_A"]
    total_xg = mp_agg["xG_F"] + mp_agg["xG_A"]
    mp_agg["CF_pct_5v5"]  = (mp_agg["CF_F"] / total_cf.replace(0, np.nan) * 100).round(2)
    mp_agg["xGF_pct_5v5"] = (mp_agg["xG_F"] / total_xg.replace(0, np.nan) * 100).round(2)
    mp_agg["ixG_60_5v5"]  = (mp_agg["ixG_raw"] / mp_agg["icetime_5v5"].replace(0, np.nan) * 3600).round(3)
    qualified = qualified.merge(
        mp_agg[["key", "CF_pct_5v5", "xGF_pct_5v5", "ixG_60_5v5"]],
        on="key", how="left"
    )
    n = qualified["CF_pct_5v5"].notna().sum()
    print(f"  MoneyPuck matched {n}/{len(qualified)} players")
else:
    print("  WARNING: no MoneyPuck data — CF_pct_5v5/xGF_pct_5v5/ixG_60_5v5 will be missing")

# ── NHL Edge tracking ─────────────────────────────────────────────────────────
def attach_edge(qualified, edge_df, speed_col_hint, out_col):
    if edge_df.empty:
        return qualified
    edge_df.to_csv(RAW_DIR / f"edge_{out_col}_raw.csv", index=False)
    name_col  = next((c for c in edge_df.columns if "name" in c.lower()), None)
    speed_col = next((c for c in edge_df.columns if speed_col_hint in c.lower()), None)
    if name_col and speed_col:
        edge_df["key"] = edge_df[name_col].apply(normalise_name)
        agg_edge = edge_df.groupby("key")[speed_col].mean().round(3).reset_index()
        agg_edge.rename(columns={speed_col: out_col}, inplace=True)
        qualified = qualified.merge(agg_edge, on="key", how="left")
        n = qualified[out_col].notna().sum()
        print(f"  {out_col}: matched {n}/{len(qualified)} players")
    else:
        print(f"  WARNING: could not find name/speed columns for {out_col}")
        print(f"    Available: {list(edge_df.columns)}")
    return qualified

qualified = attach_edge(qualified, edge_skate, "avgspeed", "Avg_Skate_Speed")
qualified = attach_edge(qualified, edge_skate, "topspeed", "Top_Skate_Speed")
qualified = attach_edge(qualified, edge_shoot, "speed",    "Avg_Shot_Speed")

# =============================================================================
# Export
# =============================================================================

FINAL_COLS = [
    "Name", "Pos", "Nationality",
    "GP", "Possible_GP", "Qualify_Pct",
    "TOI_GP", "ES_TOI_GP", "PP_TOI_GP", "SH_TOI_GP",
    "G_60", "A_60", "PTS_60", "SOG_60", "BLK_60", "HIT_60", "PP_PTS_60",
    "PIM_GP", "PlusMinus_GP",
    "Height_in", "Weight_lbs", "BMI",
    "Fights",
    "CF_pct_5v5", "xGF_pct_5v5", "ixG_60_5v5",
    "Avg_Skate_Speed", "Top_Skate_Speed", "Avg_Shot_Speed",
]

out = qualified[[c for c in FINAL_COLS if c in qualified.columns]]
out = out.sort_values("Name").reset_index(drop=True)

out_path = HERE / "players_raw.csv"
out.to_csv(out_path, index=False)

print(f"\nSaved {len(out)} players -> {out_path}")
print("\nColumn coverage (% non-null):")
print(out.notna().mean().mul(100).round(1).to_string())
