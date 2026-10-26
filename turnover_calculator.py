"""
Turnover & Care Continuity calculator (MVP 1): deterministic, pandas only.

Takes a staff roster (one row per person in a seat), the seats themselves,
and child-caregiver assignments, and computes:

- turnover rate by role / program area, annualized, voluntary vs. involuntary
- vacancy rate and average days-to-fill by role
- retention cohort curves
- fully-loaded cost of turnover per departure, from per-role assumptions
- break-even tenure by role, with a configurable ramp-up discount and
  vacancy-cost basis
- vacancy cost crossover: months of coverage spend that equal the cost of
  recruiting and onboarding a replacement
- a care-continuity risk score: distinct caregivers per child, weighted by
  role criticality
- a continuity gap: family-days without an assigned caseworker or
  clinician (vacant caseload-carrying field seat-days x typical caseload)

Conventions used everywhere:

- Periods are half-open: [start, end). A three-year history is
  start=2023-01-01, end=2026-01-01.
- A person is employed from hire_date up to (not including)
  termination_date; the termination date is the first day they are gone.
  Assignments work the same way with start_date / end_date.
- Turnover rate = separations / average headcount, annualized. Average
  headcount is filled seat-days in the period / days in the period.

Usage:
    python turnover_calculator.py [--data-dir .] [--start 2023-01-01] [--end 2026-01-01]
"""

import argparse
import math
import os
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

from generate_sample_data import (
    ASSIGNMENTS_FILE,
    CHILDREN_FILE,
    HISTORY_END,
    HISTORY_START,
    POSITIONS_FILE,
    STAFF_FILE,
)


DAYS_PER_YEAR = 365.25
DAYS_PER_MONTH = DAYS_PER_YEAR / 12

COST_ASSUMPTIONS_FILE = "role_cost_assumptions.csv"
COST_COLUMNS = ["annual_salary", "fringe_rate", "recruiting_cost", "training_cost",
                "days_to_fill", "coverage_premium"]

# Ramp-up: a new hire's productivity in each of their first months. Linear
# from RAMP_START_PRODUCTIVITY in month 1 to 100% by RAMP_MONTHS_TO_FULL
# (defaults: 50% / 75% / 100%). Full productivity after the schedule ends.
RAMP_START_PRODUCTIVITY = 0.50
RAMP_MONTHS_TO_FULL = 3

# What a filled seat saves each month, for break-even tenure:
#   "coverage_premium" - the overtime / temp premium above normal pay
#                        (loaded monthly cost x coverage_premium)
#   "full_salary"      - the seat's full loaded monthly cost
VACANCY_COST_BASES = ("coverage_premium", "full_salary")
VACANCY_COST_BASIS = "coverage_premium"

# staff_type on every staff and seat record: "field" roles have direct
# family / child contact; "central_admin" roles are internal only. Scope
# by this column rather than by role name.
FIELD, CENTRAL_ADMIN = "field", "central_admin"
STAFF_TYPES = (FIELD, CENTRAL_ADMIN)

# Field roles that carry an individual family caseload, and the typical
# caseload of one seat. Only vacancies in these roles produce family-days
# in the continuity gap. Direct care is shift-based (no individual cases)
# and central admin has no family contact, so neither is listed.
TYPICAL_CASELOAD = {"Caseworker": 10, "Clinician": 20}

# How much a change of person in each child-facing role disrupts a child's
# care. Roles not listed here are not counted in the continuity score.
ROLE_CRITICALITY = {
    "Caseworker": 1.0,
    "Youth Care Worker": 0.9,
    "Clinician": 0.8,
}

RETENTION_MONTHS = (3, 6, 12, 18, 24)

Schedule = Optional[Sequence[float]]
By = Union[None, str, List[str]]


def linear_ramp(start_productivity: float = RAMP_START_PRODUCTIVITY,
                months_to_full: int = RAMP_MONTHS_TO_FULL) -> Tuple[float, ...]:
    """Monthly productivity rising evenly from start_productivity to 1.0."""
    if not 0 < start_productivity <= 1:
        raise ValueError("start_productivity must be in (0, 1]")
    if months_to_full < 1:
        raise ValueError("months_to_full must be at least 1")
    if months_to_full == 1:
        return (1.0,)
    step = (1 - start_productivity) / (months_to_full - 1)
    return tuple(round(start_productivity + step * i, 6) for i in range(months_to_full))


RAMP_UP_SCHEDULE = linear_ramp()


# ---------------------------------------------------------------------------
# Loading and shared helpers
# ---------------------------------------------------------------------------

def load_data(data_dir: str = ".") -> Dict[str, pd.DataFrame]:
    def read(name, dates):
        return pd.read_csv(os.path.join(data_dir, name), parse_dates=dates)

    return {
        "positions": read(POSITIONS_FILE, []),
        "staff": read(STAFF_FILE, ["hire_date", "termination_date"]).fillna({"termination_type": ""}),
        "children": read(CHILDREN_FILE, ["placement_start", "placement_end"]),
        "assignments": read(ASSIGNMENTS_FILE, ["start_date", "end_date"]),
    }


def load_cost_assumptions(path: str = COST_ASSUMPTIONS_FILE) -> pd.DataFrame:
    """Per-role cost assumptions, indexed by role."""
    df = pd.read_csv(path).set_index("role")
    missing = [c for c in COST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"cost assumptions missing columns: {missing}")
    if (df[COST_COLUMNS] < 0).any().any():
        raise ValueError("cost assumptions must be non-negative")
    return df


def _period(start, end) -> Tuple[pd.Timestamp, pd.Timestamp, int]:
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if end <= start:
        raise ValueError("period end must be after start")
    return start, end, (end - start).days


def _keys(df: pd.DataFrame, by: By) -> Tuple[pd.DataFrame, List[str]]:
    if by is None:
        return df.assign(scope="All"), ["scope"]
    return df, [by] if isinstance(by, str) else list(by)


def exposure_days(staff: pd.DataFrame, start, end) -> pd.Series:
    """Days each person was employed inside [start, end)."""
    start, end, _ = _period(start, end)
    first = staff["hire_date"].clip(lower=start)
    last = staff["termination_date"].fillna(end).clip(upper=end)
    return (last - first).dt.days.clip(lower=0)


# ---------------------------------------------------------------------------
# Turnover, vacancy, retention
# ---------------------------------------------------------------------------

def turnover_rates(staff: pd.DataFrame, start, end, by: By = "role") -> pd.DataFrame:
    """Annualized turnover, split voluntary vs. involuntary."""
    start, end, days = _period(start, end)
    term = staff["termination_date"]
    left = term.notna() & (term >= start) & (term < end)
    df = staff.assign(
        _exposure=exposure_days(staff, start, end),
        _sep=left,
        _vol=left & (staff["termination_type"] == "Voluntary"),
        _invol=left & (staff["termination_type"] == "Involuntary"),
    )
    df, keys = _keys(df, by)
    out = df.groupby(keys).agg(
        exposure_days=("_exposure", "sum"),
        separations=("_sep", "sum"),
        voluntary_separations=("_vol", "sum"),
        involuntary_separations=("_invol", "sum"),
    )
    out["avg_headcount"] = out["exposure_days"] / days
    per_year = (DAYS_PER_YEAR / days) / out["avg_headcount"].where(out["avg_headcount"] > 0)
    out["turnover_rate"] = out["separations"] * per_year
    out["voluntary_rate"] = out["voluntary_separations"] * per_year
    out["involuntary_rate"] = out["involuntary_separations"] * per_year
    return out.drop(columns="exposure_days")


def fill_events(staff: pd.DataFrame, start, end) -> pd.DataFrame:
    """Backfill hires in [start, end): days from the prior holder leaving to the new hire."""
    start, end, _ = _period(start, end)
    df = staff.sort_values(["position_id", "hire_date"]).copy()
    df["prior_termination"] = df.groupby("position_id")["termination_date"].shift()
    df = df[df["prior_termination"].notna() & (df["hire_date"] >= start) & (df["hire_date"] < end)]
    return df.assign(days_to_fill=(df["hire_date"] - df["prior_termination"]).dt.days)


def vacancy_report(staff: pd.DataFrame, positions: pd.DataFrame, start, end,
                   by: By = "role") -> pd.DataFrame:
    """Vacancy rate (vacant seat-days / total seat-days), days-to-fill and open seats."""
    start, end, days = _period(start, end)
    pos, keys = _keys(positions, by)
    st, _ = _keys(staff.assign(_exposure=exposure_days(staff, start, end)), by)
    out = pos.groupby(keys).size().to_frame("positions")
    filled = st.groupby(keys)["_exposure"].sum()
    out["vacancy_rate"] = 1 - filled.reindex(out.index, fill_value=0) / (out["positions"] * days)

    fills, _ = _keys(fill_events(staff, start, end), by)
    g = fills.groupby(keys)["days_to_fill"]
    out["fills"] = g.size().reindex(out.index, fill_value=0)
    out["avg_days_to_fill"] = g.mean().reindex(out.index)

    term = staff["termination_date"]
    in_seat = set(staff.loc[(staff["hire_date"] < end) & (term.isna() | (term >= end)), "position_id"])
    open_seats = pos[~pos["position_id"].isin(in_seat)].groupby(keys).size()
    out["open_at_end"] = open_seats.reindex(out.index, fill_value=0)
    return out


def vacancy_spells(staff: pd.DataFrame, positions: pd.DataFrame, start, end,
                   role: Optional[str] = None) -> pd.DataFrame:
    """Every stretch a seat sat empty, clipped to [start, end).

    A seat whose first known holder starts after `start` counts as empty
    from `start`; a seat with no one in it at the end is still_open.
    """
    start, end, _ = _period(start, end)
    pos = positions if role is None else positions[positions["role"] == role]
    held = staff[staff["position_id"].isin(pos["position_id"])].sort_values(["position_id", "hire_date"])
    by_seat = {pid: list(zip(g["hire_date"], g["termination_date"])) for pid, g in held.groupby("position_id")}
    rows = []
    for p in pos.itertuples(index=False):
        vacant_from = start
        for hire, term in by_seat.get(p.position_id, []):
            if hire > vacant_from:
                rows.append((p.position_id, p.site_id, vacant_from, hire, False))
            if pd.isna(term):
                vacant_from = None
                break
            vacant_from = term
        if vacant_from is not None:
            rows.append((p.position_id, p.site_id, vacant_from, end, True))
    out = pd.DataFrame(rows, columns=["position_id", "site_id", "vacancy_start", "vacancy_end", "still_open"])
    out["vacancy_start"] = out["vacancy_start"].clip(lower=start)
    out["vacancy_end"] = out["vacancy_end"].clip(upper=end)
    out["days_vacant"] = (out["vacancy_end"] - out["vacancy_start"]).dt.days
    return out[out["days_vacant"] > 0].reset_index(drop=True)


def caseload_vacancy_family_days(staff: pd.DataFrame, positions: pd.DataFrame, start, end,
                                 caseload: Mapping[str, int] = TYPICAL_CASELOAD,
                                 seat_caseload: Optional[Mapping[str, int]] = None) -> pd.DataFrame:
    """CONTINUITY GAP: family-days without an assigned caseworker (or
    clinician), per vacancy.

    Only seats with staff_type "field" in a role listed in `caseload`
    count; any other vacancy has no caseload attached and produces no
    family-days. family_days = days the seat was empty in [start, end) x
    its typical caseload (`caseload` by role, overridden per seat by
    `seat_caseload`, position_id -> caseload). This measures how long
    families go without a worker of record; it is separate from the
    care-continuity score, which counts changes of caregiver.
    """
    listed = positions["role"].isin(list(caseload))
    not_field = sorted(set(positions.loc[listed & (positions["staff_type"] != FIELD), "role"]))
    if not_field:
        raise ValueError(f"caseload roles must be field roles: {not_field}")
    seats = positions[listed & (positions["staff_type"] == FIELD)]
    spells = vacancy_spells(staff, seats, start, end).merge(
        seats[["position_id", "role", "staff_type"]], on="position_id")
    spells["caseload"] = spells["role"].map(caseload)
    if seat_caseload:
        spells["caseload"] = spells["position_id"].map(seat_caseload).fillna(spells["caseload"])
    spells["family_days"] = spells["days_vacant"] * spells["caseload"]
    return spells


def family_days_without_assigned_worker(staff: pd.DataFrame, positions: pd.DataFrame, start, end,
                                        caseload: Mapping[str, int] = TYPICAL_CASELOAD,
                                        seat_caseload: Optional[Mapping[str, int]] = None,
                                        by: By = "role") -> pd.DataFrame:
    """CONTINUITY GAP summed over all caseload-carrying vacancies in the period."""
    spells, keys = _keys(caseload_vacancy_family_days(staff, positions, start, end, caseload, seat_caseload), by)
    return spells.groupby(keys).agg(vacancies=("position_id", "size"),
                                    still_open=("still_open", "sum"),
                                    vacant_days=("days_vacant", "sum"),
                                    family_days=("family_days", "sum"))


def retention_curves(staff: pd.DataFrame, as_of, months: Iterable[int] = RETENTION_MONTHS,
                     by: By = "hire_year", hired_from=None) -> pd.DataFrame:
    """Share of each hire cohort still employed m months after hire.

    Only people hired at least m months before as_of count toward month m
    (as_of is exclusive, like a period end). Long format: one row per
    cohort and month.
    """
    as_of = pd.Timestamp(as_of)
    df = staff.assign(hire_year=staff["hire_date"].dt.year)
    df = df[df["hire_date"] < as_of]
    if hired_from is not None:
        df = df[df["hire_date"] >= pd.Timestamp(hired_from)]
    df, keys = _keys(df, by)
    frames = []
    for m in months:
        milestone = df["hire_date"] + pd.DateOffset(months=m)
        eligible = milestone < as_of
        retained = eligible & (df["termination_date"].isna() | (df["termination_date"] > milestone))
        g = df.assign(eligible=eligible, retained=retained).groupby(keys)[["eligible", "retained"]].sum()
        frames.append(g.assign(months=m))
    out = pd.concat(frames).reset_index().set_index(keys + ["months"]).sort_index()
    out["retention"] = out["retained"] / out["eligible"].where(out["eligible"] > 0)
    return out


# ---------------------------------------------------------------------------
# Cost of turnover and break-even tenure
# ---------------------------------------------------------------------------

def _check_schedule(schedule: Schedule) -> Tuple[float, ...]:
    schedule = tuple(schedule or ())
    if any(not 0 < p <= 1 for p in schedule):
        raise ValueError("ramp-up productivity values must be in (0, 1]")
    return schedule


def monthly_loaded_cost(assumptions: pd.DataFrame) -> pd.Series:
    return assumptions["annual_salary"] * (1 + assumptions["fringe_rate"]) / 12


def ramp_up_lost_months(schedule: Schedule = RAMP_UP_SCHEDULE) -> float:
    """Months of a full salary paid for work not done while ramping up."""
    return sum(1 - p for p in _check_schedule(schedule))


def cost_per_departure(assumptions: pd.DataFrame,
                       days_to_fill: Optional[Mapping[str, float]] = None,
                       schedule: Schedule = RAMP_UP_SCHEDULE) -> pd.DataFrame:
    """Fully-loaded cost of one departure, per role.

    recruiting + training + vacancy coverage + ramp-up productivity loss.
    Coverage is the overtime / temp premium paid to cover the seat while it
    is empty: days_to_fill x loaded monthly cost x coverage_premium.
    days_to_fill overrides the planning assumption per role (e.g. with the
    observed average from vacancy_report).
    """
    out = assumptions[["recruiting_cost", "training_cost"]].copy()
    dtf = assumptions["days_to_fill"].astype(float)
    if days_to_fill is not None:
        observed = pd.Series(days_to_fill, dtype=float).dropna()
        dtf.update(observed[observed.index.isin(dtf.index)])
    monthly = monthly_loaded_cost(assumptions)
    out.insert(0, "days_to_fill", dtf)
    out["coverage_cost"] = dtf / DAYS_PER_MONTH * monthly * assumptions["coverage_premium"]
    out["ramp_up_cost"] = monthly * ramp_up_lost_months(schedule)
    out["cost_per_departure"] = out[["recruiting_cost", "training_cost", "coverage_cost", "ramp_up_cost"]].sum(axis=1)
    return out


def annual_turnover_cost(rates: pd.DataFrame, costs: pd.DataFrame) -> pd.DataFrame:
    """Expected departures per year x cost per departure, per role."""
    out = rates[["avg_headcount", "turnover_rate"]].join(costs["cost_per_departure"], how="inner")
    out["departures_per_year"] = out["avg_headcount"] * out["turnover_rate"]
    out["annual_cost"] = out["departures_per_year"] * out["cost_per_departure"]
    return out


def breakeven_months(upfront_cost: float, monthly_cost_avoided: float,
                     schedule: Schedule = RAMP_UP_SCHEDULE) -> float:
    """Months a new hire must stay before the seat being filled pays back
    the cost of recruiting and onboarding them.

    In month m the filled seat avoids monthly_cost_avoided x productivity(m);
    schedule=None means full productivity from day one. Fractional months
    are interpolated within the month the cost is recovered.
    """
    if upfront_cost < 0:
        raise ValueError("upfront_cost must be non-negative")
    if monthly_cost_avoided <= 0:
        return math.inf
    recovered = 0.0
    for month, productivity in enumerate(_check_schedule(schedule), start=1):
        gain = monthly_cost_avoided * productivity
        if recovered + gain >= upfront_cost:
            return month - 1 + (upfront_cost - recovered) / gain
        recovered += gain
    return len(_check_schedule(schedule)) + (upfront_cost - recovered) / monthly_cost_avoided


def monthly_coverage_premium(assumptions: pd.DataFrame) -> pd.Series:
    """Overtime / temp spend above normal pay while a seat is empty, per month."""
    return monthly_loaded_cost(assumptions) * assumptions["coverage_premium"]


def monthly_vacancy_cost(assumptions: pd.DataFrame, basis: str = VACANCY_COST_BASIS) -> pd.Series:
    """What a filled seat saves each month, on the chosen VACANCY_COST_BASES basis."""
    if basis == "coverage_premium":
        return monthly_coverage_premium(assumptions)
    if basis == "full_salary":
        return monthly_loaded_cost(assumptions)
    raise ValueError(f"basis must be one of {VACANCY_COST_BASES}, not {basis!r}")


def breakeven_tenure(assumptions: pd.DataFrame, schedule: Schedule = RAMP_UP_SCHEDULE,
                     staff: Optional[pd.DataFrame] = None, start=None, end=None,
                     basis: str = VACANCY_COST_BASIS) -> pd.DataFrame:
    """Break-even tenure per role, with and without the ramp-up discount.

    Upfront cost = recruiting + training. Monthly cost avoided = what a
    filled seat saves, per `basis` (see VACANCY_COST_BASIS). With staff and
    a period, also reports the share of departures in that period who left
    before reaching break-even.
    """
    out = pd.DataFrame(index=assumptions.index)
    out["upfront_cost"] = assumptions["recruiting_cost"] + assumptions["training_cost"]
    out["monthly_cost_avoided"] = monthly_vacancy_cost(assumptions, basis)
    pairs = list(zip(out["upfront_cost"], out["monthly_cost_avoided"]))
    out["breakeven_months_no_ramp"] = [breakeven_months(u, m, None) for u, m in pairs]
    out["breakeven_months"] = [breakeven_months(u, m, schedule) for u, m in pairs]
    out["ramp_delay_months"] = out["breakeven_months"] - out["breakeven_months_no_ramp"]
    if staff is not None:
        start, end, _ = _period(start, end)
        term = staff["termination_date"]
        left = staff[term.notna() & (term >= start) & (term < end)]
        tenure = left["tenure_days"] / DAYS_PER_MONTH
        early = tenure < left["role"].map(out["breakeven_months"])
        out["departures"] = left.groupby("role").size().reindex(out.index, fill_value=0)
        out["left_before_breakeven"] = early.groupby(left["role"]).sum().reindex(out.index, fill_value=0)
        out["share_left_before_breakeven"] = out["left_before_breakeven"] / out["departures"].where(out["departures"] > 0)
    return out


def vacancy_cost_crossover(assumptions: pd.DataFrame) -> pd.DataFrame:
    """Months of overtime / temp coverage spend that add up to the cost of
    recruiting and onboarding a replacement, per role.

    crossover_months = (recruiting + training) / monthly coverage premium.
    An empty seat covered for longer than this has cost more than hiring.
    Independent of VACANCY_COST_BASIS and the ramp-up.
    """
    out = pd.DataFrame(index=assumptions.index)
    out["recruiting_and_onboarding_cost"] = assumptions["recruiting_cost"] + assumptions["training_cost"]
    out["monthly_coverage_premium"] = monthly_coverage_premium(assumptions)
    premium = out["monthly_coverage_premium"]
    out["crossover_months"] = (out["recruiting_and_onboarding_cost"] / premium.where(premium > 0)).fillna(math.inf)
    return out


# ---------------------------------------------------------------------------
# Care continuity
# ---------------------------------------------------------------------------

def care_continuity(assignments: pd.DataFrame, start, end,
                    weights: Mapping[str, float] = ROLE_CRITICALITY) -> pd.DataFrame:
    """Distinct caregivers per child and role in [start, end), and a
    criticality-weighted score per child.

    continuity_score = sum(weight x distinct people in role) / sum(weight)
    over the roles the child had. 1.0 is perfect continuity (one person in
    every role); 2.0 means two people per role on average. transitions is
    distinct people minus one, summed over roles.
    """
    start, end, _ = _period(start, end)
    if any(w <= 0 for w in weights.values()):
        raise ValueError("criticality weights must be positive")
    a = assignments[(assignments["start_date"] < end) & (assignments["end_date"] > start)
                    & assignments["role"].isin(list(weights))]
    counts = a.groupby(["child_id", "role"])["staff_id"].nunique().unstack(fill_value=0)
    w = pd.Series(weights, dtype=float).reindex(counts.columns)
    present = counts > 0
    out = counts.copy()
    out["transitions"] = (counts - present.astype(int)).sum(axis=1)
    out["continuity_score"] = (counts * w).sum(axis=1) / (present * w).sum(axis=1)
    out["site_id"] = a.groupby("child_id")["site_id"].first()
    return out


def care_continuity_score(assignments: pd.DataFrame, start, end,
                          weights: Mapping[str, float] = ROLE_CRITICALITY,
                          by: Optional[str] = None) -> Union[float, pd.Series]:
    """Average continuity score across children (optionally per site)."""
    per_child = care_continuity(assignments, start, end, weights)
    if by is None:
        return float(per_child["continuity_score"].mean())
    return per_child.groupby(by)["continuity_score"].mean()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_report(data_dir: str = ".", start=HISTORY_START, end=HISTORY_END,
               cost_path: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    data = load_data(data_dir)
    staff, positions = data["staff"], data["positions"]
    assumptions = load_cost_assumptions(cost_path or os.path.join(data_dir, COST_ASSUMPTIONS_FILE))
    vacancy = vacancy_report(staff, positions, start, end, by="role")
    costs = cost_per_departure(assumptions, days_to_fill=vacancy["avg_days_to_fill"])
    by_role = turnover_rates(staff, start, end, by="role")
    last_year = pd.Timestamp(end) - pd.DateOffset(years=1)
    return {
        "turnover_by_staff_type": turnover_rates(staff, start, end, by="staff_type"),
        "turnover_by_program_area": turnover_rates(staff, start, end, by="program_area"),
        "turnover_by_role": by_role,
        "vacancy_by_role": vacancy,
        "retention_by_hire_year": retention_curves(staff, end, hired_from=start)["retention"].unstack("months"),
        "cost_per_departure": costs,
        "annual_turnover_cost": annual_turnover_cost(by_role, costs),
        "breakeven_tenure": breakeven_tenure(assumptions, staff=staff, start=start, end=end),
        "vacancy_cost_crossover": vacancy_cost_crossover(assumptions),
        "continuity_by_site_last_12_months": care_continuity_score(
            data["assignments"], last_year, end, by="site_id").to_frame("continuity_score"),
        "continuity_gap_family_days_without_assigned_worker": family_days_without_assigned_worker(
            staff, positions, start, end, by=["site_id", "role"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Turnover & care continuity report")
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--start", default=str(HISTORY_START.date()))
    parser.add_argument("--end", default=str(HISTORY_END.date()), help="exclusive")
    args = parser.parse_args()
    report = run_report(args.data_dir, args.start, args.end)
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", "{:,.2f}".format):
        for name, df in report.items():
            print(f"\n== {name.replace('_', ' ')} ==")
            print(df)


if __name__ == "__main__":
    main()
