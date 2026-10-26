"""
Sample data generator for the Turnover & Care Continuity Impact tool.

Sector: child welfare / youth services, New York City.

EVERYTHING HERE IS FICTIONAL:

- The organization, Kinbridge Family Services, does not exist. It is
  modeled on the general structure of NYC child welfare agencies (family
  foster care and prevention offices, youth residences, a clinic and a
  central office) and is not based on any specific agency.
- Its sites are fictional too. Each sits in a real neighborhood (so the
  borough mix and map placement look right), but the street names are
  invented and every site name ends in "(SYN)".
- Every staff member, child and assignment is synthetic. Staff IDs start
  SYN-S-, child IDs start SYN-C-, and every row carries is_synthetic=True,
  so the data marks itself as fake even when a file is separated from this
  script. Children have no names, ages or case details.

The workforce data is shaped like a real child welfare agency, not flat
random numbers, so the calculator has something meaningful to find:

- Each role has its own annual turnover rate. Direct-care and caseworker
  roles churn several times faster than admin and supervisors.
- Tenure is Weibull-distributed with shape < 1, so the hazard is highest
  early on: many departures in the first year, a long-tenured core after
  that. Retention curves bend the way real ones do.
- Every departure leaves the seat vacant for a role-specific number of
  days before it is backfilled (log-normal), so vacancy rates and
  days-to-fill come out of the same history as turnover.
- Children are assigned to positions (a caseworker, often a clinician, and
  a key youth care worker in the residences). When the person in that
  position leaves, the child gets whoever covers the gap and then the new
  hire, so caregiver churn follows staff turnover.

Positions are simulated from 2004-2013 (each seat starts on a random day) so the staff in seat on 2023-01-01 have
realistic tenure. Only records active in the three-year history window
(2023-01-01 to 2025-12-31) are written.

Deterministic: the same seed always produces byte-identical files.

Usage:
    python generate_sample_data.py [--seed 42] [--out-dir .]
"""

import argparse
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_SEED = 42

WARMUP_START = pd.Timestamp("2004-01-01")
WARMUP_STAGGER_DAYS = 3652   # first incumbents start on a random day in 2004-2013
HISTORY_START = pd.Timestamp("2023-01-01")
HISTORY_END = pd.Timestamp("2026-01-01")   # exclusive: data is as of 2025-12-31

SITES_FILE = "sites.csv"
POSITIONS_FILE = "synthetic_positions.csv"
STAFF_FILE = "synthetic_staff.csv"
CHILDREN_FILE = "synthetic_children.csv"
ASSIGNMENTS_FILE = "synthetic_assignments.csv"

ORG_NAME = "Kinbridge Family Services"

# Fictional sites in real neighborhoods. Street names are invented;
# coordinates are approximate neighborhood centers, for the MVP 2 map.
SITES = [
    {"site_id": "SITE-01", "site_name": "Kinbridge Central Office (SYN)",
     "site_type": "Central Office", "borough": "Manhattan", "neighborhood": "Central Harlem",
     "address": "2150 Wexley Avenue, New York, NY 10027", "latitude": 40.8090, "longitude": -73.9480},
    {"site_id": "SITE-02", "site_name": "Kinbridge East Harlem Family Center (SYN)",
     "site_type": "Family Services", "borough": "Manhattan", "neighborhood": "East Harlem",
     "address": "318 Calloway Street, New York, NY 10029", "latitude": 40.7960, "longitude": -73.9390},
    {"site_id": "SITE-03", "site_name": "Kinbridge Washington Heights Family Center (SYN)",
     "site_type": "Family Services", "borough": "Manhattan", "neighborhood": "Washington Heights",
     "address": "4420 Brindle Place, New York, NY 10033", "latitude": 40.8480, "longitude": -73.9360},
    {"site_id": "SITE-04", "site_name": "Kinbridge Bedford-Stuyvesant Family Center (SYN)",
     "site_type": "Family Services", "borough": "Brooklyn", "neighborhood": "Bedford-Stuyvesant",
     "address": "761 Marlowe Avenue, Brooklyn, NY 11216", "latitude": 40.6870, "longitude": -73.9420},
    {"site_id": "SITE-05", "site_name": "Kinbridge Brownsville Youth Residence (SYN)",
     "site_type": "Youth Residence", "borough": "Brooklyn", "neighborhood": "Brownsville",
     "address": "92 Ashgrove Street, Brooklyn, NY 11212", "latitude": 40.6630, "longitude": -73.9100},
    {"site_id": "SITE-06", "site_name": "Kinbridge Mott Haven Family Center (SYN)",
     "site_type": "Family Services", "borough": "Bronx", "neighborhood": "Mott Haven",
     "address": "530 Tolliver Avenue, Bronx, NY 10454", "latitude": 40.8090, "longitude": -73.9230},
    {"site_id": "SITE-07", "site_name": "Kinbridge Tremont Youth Residence (SYN)",
     "site_type": "Youth Residence", "borough": "Bronx", "neighborhood": "Tremont",
     "address": "1875 Pennick Place, Bronx, NY 10457", "latitude": 40.8460, "longitude": -73.8990},
    {"site_id": "SITE-08", "site_name": "Kinbridge Fordham Clinical Services (SYN)",
     "site_type": "Clinic", "borough": "Bronx", "neighborhood": "Fordham",
     "address": "2604 Sorrel Street, Bronx, NY 10458", "latitude": 40.8610, "longitude": -73.8900},
]

# Generation parameters per role: the "true" rates the calculator should
# recover. annual_turnover is separations per filled seat per year.
ROLES = {
    "Youth Care Worker":        {"program_area": "Direct Care",     "annual_turnover": 0.55, "voluntary_share": 0.70, "days_to_fill": 35},
    "Residential Counselor":    {"program_area": "Direct Care",     "annual_turnover": 0.45, "voluntary_share": 0.72, "days_to_fill": 40},
    "Caseworker":               {"program_area": "Case Management", "annual_turnover": 0.42, "voluntary_share": 0.80, "days_to_fill": 55},
    "Case Supervisor":          {"program_area": "Case Management", "annual_turnover": 0.20, "voluntary_share": 0.85, "days_to_fill": 70},
    "Clinician":                {"program_area": "Clinical",        "annual_turnover": 0.25, "voluntary_share": 0.88, "days_to_fill": 85},
    "Clinical Supervisor":      {"program_area": "Clinical",        "annual_turnover": 0.15, "voluntary_share": 0.90, "days_to_fill": 100},
    "Administrative Assistant": {"program_area": "Admin",           "annual_turnover": 0.14, "voluntary_share": 0.70, "days_to_fill": 30},
    "Finance Associate":        {"program_area": "Admin",           "annual_turnover": 0.10, "voluntary_share": 0.75, "days_to_fill": 45},
}

FAMILY_SERVICES_STAFFING = {"Caseworker": 16, "Case Supervisor": 3, "Clinician": 4,
                            "Clinical Supervisor": 1, "Administrative Assistant": 2}
RESIDENCE_STAFFING = {"Youth Care Worker": 20, "Residential Counselor": 8, "Clinician": 2,
                      "Administrative Assistant": 1}
SITE_STAFFING = {
    "SITE-01": {"Administrative Assistant": 8, "Finance Associate": 10, "Clinical Supervisor": 1},
    "SITE-02": FAMILY_SERVICES_STAFFING,
    "SITE-03": FAMILY_SERVICES_STAFFING,
    "SITE-04": FAMILY_SERVICES_STAFFING,
    "SITE-05": RESIDENCE_STAFFING,
    "SITE-06": FAMILY_SERVICES_STAFFING,
    "SITE-07": RESIDENCE_STAFFING,
    "SITE-08": {"Clinician": 10, "Clinical Supervisor": 2, "Administrative Assistant": 2},
}

WEIBULL_SHAPE = 0.8        # < 1: departures concentrate early in tenure
MIN_TENURE_DAYS = 7
DAYS_TO_FILL_SIGMA = 0.5

# Children
CASELOAD_PER_CASEWORKER = 10
BEDS_PER_RESIDENCE = 20
FOSTER_CARE_MEDIAN_STAY_DAYS = 420
RESIDENTIAL_MEDIAN_STAY_DAYS = 240
STAY_SIGMA = 0.6
MIN_STAY_DAYS = 30
MAX_GAP_BETWEEN_PLACEMENTS_DAYS = 30
CLINICIAN_SHARE_FOSTER_CARE = 0.5          # residential children always have one
# Where residential children's caseworkers sit, and which clinicians serve each site.
RESIDENCE_CASEWORKER_SITE = {"SITE-05": "SITE-04", "SITE-07": "SITE-06"}
CLINICIAN_POOL = {"SITE-06": ["SITE-06", "SITE-08"]}


def _days(ts: pd.Timestamp) -> int:
    return (ts - WARMUP_START).days


def _date(day: Optional[int]) -> Optional[pd.Timestamp]:
    return None if day is None else WARMUP_START + pd.Timedelta(days=int(day))


def build_positions() -> pd.DataFrame:
    rows = []
    for site in SITES:
        for role, count in SITE_STAFFING[site["site_id"]].items():
            for _ in range(count):
                rows.append({"position_id": f"POS-{len(rows) + 1:03d}", "site_id": site["site_id"],
                             "role": role, "program_area": ROLES[role]["program_area"]})
    return pd.DataFrame(rows)


def _days_to_fill(rng: np.random.Generator, mean_days: float) -> int:
    mu = math.log(mean_days) - DAYS_TO_FILL_SIGMA ** 2 / 2   # keeps the mean at mean_days
    return max(1, int(round(rng.lognormal(mu, DAYS_TO_FILL_SIGMA))))


def simulate_position(rng: np.random.Generator, role: str) -> List[Tuple[int, Optional[int], Optional[bool]]]:
    """Successive incumbents of one seat: (hire_day, termination_day, voluntary)."""
    params = ROLES[role]
    mean_tenure_days = 365.25 / params["annual_turnover"]
    scale = mean_tenure_days / math.gamma(1 + 1 / WEIBULL_SHAPE)
    end = _days(HISTORY_END)
    day, out = int(rng.integers(0, WARMUP_STAGGER_DAYS)), []
    while day < end:
        tenure = max(MIN_TENURE_DAYS, int(round(scale * rng.weibull(WEIBULL_SHAPE))))
        term = day + tenure
        if term >= end:
            out.append((day, None, None))
            break
        out.append((day, term, bool(rng.random() < params["voluntary_share"])))
        day = term + _days_to_fill(rng, params["days_to_fill"])
    return out


def build_staff(rng: np.random.Generator, positions: pd.DataFrame) -> pd.DataFrame:
    start, end = _days(HISTORY_START), _days(HISTORY_END)
    rows = []
    for pos in positions.itertuples(index=False):
        for hire, term, voluntary in simulate_position(rng, pos.role):
            if term is not None and term < start:
                continue   # left before the history window
            rows.append({
                "position_id": pos.position_id, "site_id": pos.site_id,
                "role": pos.role, "program_area": pos.program_area,
                "hire_date": _date(hire), "termination_date": _date(term),
                "termination_type": "" if term is None else ("Voluntary" if voluntary else "Involuntary"),
                "status": "Active" if term is None else "Terminated",
                "tenure_days": (end if term is None else term) - hire,
            })
    staff = pd.DataFrame(rows).sort_values(["hire_date", "position_id"]).reset_index(drop=True)
    staff.insert(0, "staff_id", [f"SYN-S-{i + 1:04d}" for i in range(len(staff))])
    staff["tenure_months"] = (staff["tenure_days"] / (365.25 / 12)).round(1)
    staff["is_synthetic"] = True
    return staff


def _placements(rng: np.random.Generator, median_days: int) -> List[Tuple[int, Optional[int]]]:
    """Successive placements in one caseload slot or bed: (start_day, end_day)."""
    start, end = _days(HISTORY_START), _days(HISTORY_END)
    day = start - int(rng.integers(0, median_days))   # first child is mid-placement
    out = []
    while day < end:
        length = max(MIN_STAY_DAYS, int(round(rng.lognormal(math.log(median_days), STAY_SIGMA))))
        stop = day + length
        if stop > start:
            out.append((day, stop if stop < end else None))
        day = stop + int(rng.integers(1, MAX_GAP_BETWEEN_PLACEMENTS_DAYS + 1))
    return out


def _segments(incumbents, a: int, b: int):
    """Split [a, b) into (staff_id or None, seg_start, seg_end) by who held the seat."""
    cursor = a
    for staff_id, hire, term in incumbents:
        s, e = max(hire, a), min(term, b)
        if s >= e:
            continue
        if s > cursor:
            yield None, cursor, s
        yield staff_id, s, e
        cursor = e
    if cursor < b:
        yield None, cursor, b


def build_children_and_assignments(rng: np.random.Generator, positions: pd.DataFrame,
                                   staff: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    end = _days(HISTORY_END)
    held = {}   # position_id -> [(staff_id, hire_day, term_day)]
    for s in staff.itertuples(index=False):
        term = end if pd.isna(s.termination_date) else _days(s.termination_date)
        held.setdefault(s.position_id, []).append((s.staff_id, _days(s.hire_date), term))
    pos_by = {k: list(g["position_id"]) for k, g in positions.groupby(["site_id", "role"], sort=False)}
    site_of = dict(zip(positions["position_id"], positions["site_id"]))
    role_of = dict(zip(positions["position_id"], positions["role"]))

    def clinician_pool(site_id):
        return [p for s in CLINICIAN_POOL.get(site_id, [site_id]) for p in pos_by.get((s, "Clinician"), [])]

    plans = []   # (site_id, program, placement_start, placement_end, [position_ids])
    for site in SITES:
        site_id = site["site_id"]
        if site["site_type"] == "Family Services":
            pool = clinician_pool(site_id)
            for cw in pos_by[(site_id, "Caseworker")]:
                for _ in range(CASELOAD_PER_CASEWORKER):
                    for p_start, p_end in _placements(rng, FOSTER_CARE_MEDIAN_STAY_DAYS):
                        links = [cw]
                        if rng.random() < CLINICIAN_SHARE_FOSTER_CARE:
                            links.append(pool[int(rng.integers(len(pool)))])
                        plans.append((site_id, "Foster Care", p_start, p_end, links))
        elif site["site_type"] == "Youth Residence":
            ycw = pos_by[(site_id, "Youth Care Worker")]
            pool = clinician_pool(site_id)
            cws = pos_by[(RESIDENCE_CASEWORKER_SITE[site_id], "Caseworker")]
            for _ in range(BEDS_PER_RESIDENCE):
                for p_start, p_end in _placements(rng, RESIDENTIAL_MEDIAN_STAY_DAYS):
                    links = [cws[int(rng.integers(len(cws)))],
                             ycw[int(rng.integers(len(ycw)))],
                             pool[int(rng.integers(len(pool)))]]
                    plans.append((site_id, "Residential", p_start, p_end, links))

    start = _days(HISTORY_START)
    children, assignments = [], []
    for i, (site_id, program, p_start, p_end, links) in enumerate(plans):
        child_id = f"SYN-C-{i + 1:04d}"
        children.append({"child_id": child_id, "site_id": site_id, "program": program,
                         "placement_start": _date(p_start), "placement_end": _date(p_end),
                         "is_synthetic": True})
        a, b = max(p_start, start), end if p_end is None else p_end
        for pos_id in links:
            for staff_id, s, e in _segments(held.get(pos_id, []), a, b):
                is_coverage = staff_id is None
                if is_coverage:
                    # a coworker in the same role and site covers the vacant seat
                    cover = [(sid, h, t) for p in pos_by[(site_of[pos_id], role_of[pos_id])] if p != pos_id
                             for sid, h, t in held.get(p, []) if h <= s < t]
                    if not cover:
                        continue
                    staff_id, _, t = cover[int(rng.integers(len(cover)))]
                    e = min(e, t)
                assignments.append({"child_id": child_id, "site_id": site_id, "role": role_of[pos_id],
                                    "position_id": pos_id, "staff_id": staff_id,
                                    "start_date": _date(s), "end_date": _date(e),
                                    "is_coverage": is_coverage, "is_synthetic": True})
    return pd.DataFrame(children), pd.DataFrame(assignments)


def generate(seed: int = DEFAULT_SEED) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    sites = pd.DataFrame(SITES)
    sites.insert(1, "organization", ORG_NAME)
    sites["is_synthetic"] = True
    positions = build_positions()
    staff = build_staff(rng, positions)
    children, assignments = build_children_and_assignments(rng, positions, staff)
    return {SITES_FILE: sites, POSITIONS_FILE: positions, STAFF_FILE: staff,
            CHILDREN_FILE: children, ASSIGNMENTS_FILE: assignments}


def write(frames: Dict[str, pd.DataFrame], out_dir: str = ".") -> None:
    os.makedirs(out_dir, exist_ok=True)
    for name, df in frames.items():
        df.to_csv(os.path.join(out_dir, name), index=False, date_format="%Y-%m-%d")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()
    frames = generate(args.seed)
    write(frames, args.out_dir)
    for name, df in frames.items():
        print(f"{name}: {len(df):,} rows")


if __name__ == "__main__":
    main()
