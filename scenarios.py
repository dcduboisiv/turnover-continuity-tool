"""
Scenario engine (MVP 3): the views and what-if changes the AI layer asks
for, done deterministically on top of turnover_calculator.py. There is no
AI in this module; the dashboard and the AI layer both use it, so an
answer's "before" numbers are the same numbers the dashboard shows.

- View / Settings:       what slice of the data to look at, and the engine
                         settings (costs, ramp-up, vacancy-cost basis,
                         caseloads) the sidebar controls.
- filter_view():         the staff, seats and children in a view.
- headline(), role_table(), site_table(): the dashboard's numbers.
- avoid_departures():    "what if turnover dropped X%": a share of the
                         period's departures don't happen. Those people
                         stay in their seats instead of being replaced, so
                         children keep their caregiver and the seat never
                         goes vacant. Everything downstream is recomputed
                         from the adjusted data.
- apply_changes():       applies a list of what-if changes to the data and
                         settings.
"""

import copy
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from generate_sample_data import HISTORY_END, HISTORY_START
from turnover_calculator import (
    RAMP_MONTHS_TO_FULL,
    RAMP_START_PRODUCTIVITY,
    ROLE_CRITICALITY,
    TYPICAL_CASELOAD,
    VACANCY_COST_BASES,
    VACANCY_COST_BASIS,
    annual_turnover_cost,
    breakeven_tenure,
    care_continuity_score,
    caseload_vacancy_family_days,
    cost_per_departure,
    linear_ramp,
    open_seats,
    turnover_rates,
    vacancy_cost_crossover,
    vacancy_report,
)


@dataclass(frozen=True)
class View:
    """A slice of the data. None means no filter."""
    site_id: Optional[str] = None
    program_areas: Optional[Tuple[str, ...]] = None
    staff_type: Optional[str] = None
    roles: Optional[Tuple[str, ...]] = None
    start: pd.Timestamp = HISTORY_START
    end: pd.Timestamp = HISTORY_END   # exclusive


@dataclass
class Settings:
    assumptions: pd.DataFrame
    ramp_on: bool = True
    ramp_start: float = RAMP_START_PRODUCTIVITY
    ramp_months: int = RAMP_MONTHS_TO_FULL
    basis: str = VACANCY_COST_BASIS
    caseload: Dict[str, int] = field(default_factory=lambda: dict(TYPICAL_CASELOAD))

    @property
    def schedule(self) -> Tuple[float, ...]:
        return linear_ramp(self.ramp_start, self.ramp_months)

    @property
    def applied_schedule(self) -> Optional[Tuple[float, ...]]:
        return self.schedule if self.ramp_on else None


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

def view_roles(positions: pd.DataFrame, view: View, role_order: Sequence[str]) -> List[str]:
    """Roles available in a view's site / program areas / staff type, narrowed by view.roles."""
    pos = positions
    if view.program_areas is not None:
        pos = pos[pos["program_area"].isin(view.program_areas)]
    if view.site_id:
        pos = pos[pos["site_id"] == view.site_id]
    if view.staff_type:
        pos = pos[pos["staff_type"] == view.staff_type]
    present = set(pos["role"])
    return [r for r in role_order if r in present and (view.roles is None or r in view.roles)]


def filter_view(data: Dict[str, pd.DataFrame], view: View, role_order: Sequence[str]) -> Dict[str, object]:
    """Staff, seats and child assignments in a view.

    Continuity follows the children a site serves: those placed there, or
    assigned a worker from there, through every in-view role they have.
    """
    roles = view_roles(data["positions"], view, role_order)

    def in_scope(df):
        mask = df["role"].isin(roles)
        if view.program_areas is not None:
            mask &= df["program_area"].isin(view.program_areas)
        if view.site_id:
            mask &= df["site_id"] == view.site_id
        if view.staff_type:
            mask &= df["staff_type"] == view.staff_type
        return df[mask]

    assignments = data["assignments"][data["assignments"]["role"].isin(roles)]
    if view.site_id:
        worker_site = assignments["position_id"].map(data["positions"].set_index("position_id")["site_id"])
        served = set(assignments.loc[(assignments["site_id"] == view.site_id) | (worker_site == view.site_id),
                                     "child_id"])
        assignments = assignments[assignments["child_id"].isin(served)]
    return {"staff": in_scope(data["staff"]), "positions": in_scope(data["positions"]),
            "assignments": assignments, "roles": roles}


# ---------------------------------------------------------------------------
# The dashboard's numbers
# ---------------------------------------------------------------------------

def _role_costs(f, view: View, settings: Settings, schedule):
    roles = f["roles"]
    vacancy = vacancy_report(f["staff"], f["positions"], view.start, view.end, by="role").reindex(roles)
    costs = cost_per_departure(settings.assumptions.loc[roles], vacancy["avg_days_to_fill"], schedule)
    return vacancy, costs


def headline(data, view: View, settings: Settings, role_order: Sequence[str]) -> Dict[str, Optional[float]]:
    """The dashboard's headline numbers for one view. None where a number doesn't apply."""
    f = filter_view(data, view, role_order)
    staff, positions, assignments = f["staff"], f["positions"], f["assignments"]
    if positions.empty:
        return {}
    start, end = view.start, view.end
    overall = turnover_rates(staff, start, end, by=None).loc["All"]
    vacancy = vacancy_report(staff, positions, start, end, by=None).loc["All"]
    by_role = turnover_rates(staff, start, end, by="role").reindex(f["roles"])
    _, costs = _role_costs(f, view, settings, settings.applied_schedule)
    weights = {r: w for r, w in ROLE_CRITICALITY.items() if r in f["roles"]}
    in_period = assignments[(assignments["start_date"] < end) & (assignments["end_date"] > start)]
    continuity = (care_continuity_score(assignments, start, end, weights)
                  if weights and not in_period.empty else None)
    gap = caseload_vacancy_family_days(staff, positions, start, end, settings.caseload)
    return {
        "turnover_rate": overall["turnover_rate"],
        "voluntary_rate": overall["voluntary_rate"],
        "involuntary_rate": overall["involuntary_rate"],
        "departures": int(overall["separations"]),
        "avg_headcount": overall["avg_headcount"],
        "vacancy_rate": vacancy["vacancy_rate"],
        "seats": int(vacancy["positions"]),
        "open_seats_at_end": int(vacancy["open_at_end"]),
        "continuity_score": continuity,
        "children_served": int(in_period["child_id"].nunique()) if weights else None,
        "annual_turnover_cost": float(annual_turnover_cost(by_role.dropna(subset=["turnover_rate"]),
                                                           costs)["annual_cost"].sum()),
        "family_days_without_worker": float(gap["family_days"].sum()),
        "caseload_vacancies": int(len(gap)),
    }


def role_table(data, view: View, settings: Settings, role_order: Sequence[str]) -> pd.DataFrame:
    """Per-role numbers for one view: turnover, vacancy, cost, break-even, crossover, family-days."""
    f = filter_view(data, view, role_order)
    roles = f["roles"]
    if not roles:
        return pd.DataFrame()
    start, end = view.start, view.end
    rates = turnover_rates(f["staff"], start, end, by="role").reindex(roles)
    vacancy, costs = _role_costs(f, view, settings, settings.applied_schedule)
    a = settings.assumptions.loc[roles]
    be = breakeven_tenure(a, settings.applied_schedule, staff=f["staff"], start=start, end=end, basis=settings.basis)
    be_ramp = breakeven_tenure(a, settings.schedule, basis=settings.basis)
    gap = caseload_vacancy_family_days(f["staff"], f["positions"], start, end, settings.caseload)
    out = pd.DataFrame(index=pd.Index(roles, name="role"))
    out["turnover_rate"] = rates["turnover_rate"]
    out["departures"] = rates["separations"]
    out["vacancy_rate"] = vacancy["vacancy_rate"]
    out["avg_days_to_fill"] = vacancy["avg_days_to_fill"]
    out["cost_per_departure"] = costs["cost_per_departure"]
    out["breakeven_months_no_ramp"] = be_ramp["breakeven_months_no_ramp"]
    out["breakeven_months_with_ramp"] = be_ramp["breakeven_months"]
    out["share_left_before_breakeven"] = be["share_left_before_breakeven"]
    out["crossover_months"] = vacancy_cost_crossover(a)["crossover_months"]
    out["family_days_without_worker"] = gap.groupby("role")["family_days"].sum().reindex(roles)
    return out


def site_table(data, view: View, settings: Settings, role_order: Sequence[str],
               sites: Sequence[str]) -> pd.DataFrame:
    """The headline numbers for each site, with the view's other filters. With a
    single role in view, that role's per-role numbers are added too."""
    rows = {}
    for site_id in sites:
        site_view = replace(view, site_id=site_id)
        h = headline(data, site_view, settings, role_order)
        if not h:
            continue
        roles = filter_view(data, site_view, role_order)["roles"]
        if len(roles) == 1:
            h.update(role_table(data, site_view, settings, role_order).iloc[0].drop(
                ["turnover_rate", "departures", "vacancy_rate", "family_days_without_worker"]).to_dict())
        rows[site_id] = h
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("site_id")


# ---------------------------------------------------------------------------
# What-ifs
# ---------------------------------------------------------------------------

def pick_evenly(n_items: int, fraction: float) -> List[int]:
    """Indices of round(fraction x n) items spread evenly through 0..n-1."""
    n = int(round(fraction * n_items))
    return [int((i + 0.5) * n_items / n) for i in range(n)] if n else []


def avoid_departures(data: Dict[str, pd.DataFrame], fraction: float, start, end,
                     roles: Optional[Sequence[str]] = None,
                     sites: Optional[Sequence[str]] = None) -> Dict[str, pd.DataFrame]:
    """Model `fraction` fewer departures among the given roles / sites in [start, end).

    Departures are taken in date order and every k-th one is avoided (an
    even spread, so the result is deterministic). An avoided leaver stays in
    the seat: they take over their successor's stint (and its eventual
    departure, if any), the successor's record is removed, and children who
    had the successor, or a coworker covering the vacancy, keep the leaver.
    """
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between 0 and 1")
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    staff = data["staff"].set_index("staff_id").copy()
    assignments = data["assignments"].copy()
    term = staff["termination_date"]
    scope = term.notna() & (term >= start) & (term < end)
    if roles is not None:
        scope &= staff["role"].isin(roles)
    if sites is not None:
        scope &= staff["site_id"].isin(sites)
    leavers = staff[scope].sort_values(["termination_date"], kind="mergesort").index
    chosen = [leavers[i] for i in pick_evenly(len(leavers), fraction)]

    merged_into = {}
    for staff_id in chosen:
        while staff_id in merged_into:
            staff_id = merged_into[staff_id]
        left = staff.at[staff_id, "termination_date"]
        if pd.isna(left):
            continue
        pos = staff.at[staff_id, "position_id"]
        later = staff[(staff["position_id"] == pos) & (staff["hire_date"] >= left)]
        if later.empty:
            gap_end = None
            staff.loc[staff_id, ["termination_date", "termination_type"]] = [pd.NaT, ""]
        else:
            successor = later["hire_date"].idxmin()
            gap_end = staff.at[successor, "hire_date"]
            staff.loc[staff_id, ["termination_date", "termination_type"]] = \
                staff.loc[successor, ["termination_date", "termination_type"]].tolist()
            assignments.loc[assignments["staff_id"] == successor, "staff_id"] = staff_id
            staff = staff.drop(index=successor)
            merged_into[successor] = staff_id
        covered = ((assignments["position_id"] == pos) & assignments["is_coverage"]
                   & (assignments["start_date"] >= left))
        if gap_end is not None:
            covered &= assignments["start_date"] < gap_end
        assignments.loc[covered, ["staff_id", "is_coverage"]] = [staff_id, False]

    staff["status"] = staff["termination_date"].isna().map({True: "Active", False: "Terminated"})
    staff["tenure_days"] = (staff["termination_date"].fillna(HISTORY_END) - staff["hire_date"]).dt.days
    staff["tenure_months"] = (staff["tenure_days"] / (365.25 / 12)).round(1)
    out = dict(data)
    out["staff"] = staff.reset_index()
    out["assignments"] = assignments
    return out


# Change levers. role / site: "" means every role / site the lever allows.
TURNOVER_REDUCTION = "turnover_reduction"   # value: fraction (relative) or rate points
CASELOAD = "caseload"                       # value: families per seat
COST_FIELDS = ("recruiting_cost", "training_cost", "annual_salary", "coverage_premium")  # relative or absolute
RAMP_START = "ramp_start_productivity"
RAMP_MONTHS = "ramp_months_to_full"
RAMP_ENABLED = "ramp_enabled"               # 1 = on, 0 = off
VACANCY_BASIS = "vacancy_cost_basis_full_salary"   # 1 = full_salary, 0 = coverage_premium
LEVERS = (TURNOVER_REDUCTION, CASELOAD) + COST_FIELDS + (RAMP_START, RAMP_MONTHS, RAMP_ENABLED, VACANCY_BASIS)


def turnover_fraction(data, change: dict, view: View) -> float:
    """A turnover_reduction change as the fraction of departures avoided."""
    if change["unit"] != "points":
        return change["value"]
    scope = data["staff"]
    if change["role"]:
        scope = scope[scope["role"] == change["role"]]
    if change["site"]:
        scope = scope[scope["site_id"] == change["site"]]
    rate = turnover_rates(scope, view.start, view.end, by=None)["turnover_rate"]
    if rate.empty or not rate.iloc[0] > 0:
        raise ValueError("there is no turnover to reduce in that scope")
    fraction = change["value"] / rate.iloc[0]
    if fraction > 1:
        raise ValueError(f"a {change['value']:.0%} point drop is more than the whole "
                         f"{rate.iloc[0]:.1%} turnover rate")
    return fraction


def apply_changes(data, settings: Settings, view: View, changes: List[dict]):
    """Apply what-if changes. Returns (data, settings) copies; the inputs are unchanged."""
    settings = copy.deepcopy(settings)
    for c in changes:
        lever, value, role = c["lever"], c["value"], c["role"]
        if lever == TURNOVER_REDUCTION:
            data = avoid_departures(data, turnover_fraction(data, c, view), view.start, view.end,
                                    roles=[role] if role else None, sites=[c["site"]] if c["site"] else None)
        elif lever == CASELOAD:
            settings.caseload[role] = int(round(value))
        elif lever in COST_FIELDS:
            rows = [role] if role else list(settings.assumptions.index)
            if c["unit"] == "relative":
                settings.assumptions.loc[rows, lever] = settings.assumptions.loc[rows, lever] * (1 + value)
            else:
                settings.assumptions.loc[rows, lever] = value
        elif lever == RAMP_START:
            settings.ramp_start = float(value)
        elif lever == RAMP_MONTHS:
            settings.ramp_months = int(round(value))
        elif lever == RAMP_ENABLED:
            settings.ramp_on = bool(value)
        elif lever == VACANCY_BASIS:
            settings.basis = VACANCY_COST_BASES[1] if value else VACANCY_COST_BASES[0]
        else:
            raise ValueError(f"unknown lever {lever!r}")
    return data, settings


def describe_change(c: dict, site_names: Dict[str, str]) -> str:
    who = c["role"] or "all roles"
    where = f" at {site_names.get(c['site'], c['site'])}" if c.get("site") else ""
    lever, v = c["lever"], c["value"]
    if lever == TURNOVER_REDUCTION:
        amount = f"{v * 100:g} points" if c["unit"] == "points" else f"{v:.0%}"
        return f"{who} turnover{where} down {amount} (that share of departures avoided; those staff stay)"
    if lever == CASELOAD:
        return f"{who} typical caseload set to {v:g}"
    if lever in COST_FIELDS:
        name = lever.replace("_", " ").replace("coverage premium", "OT / temp premium")
        if c["unit"] == "relative":
            return f"{who} {name} {v:+.0%}"
        return f"{who} {name} set to " + (f"{v:.2f}" if lever == "coverage_premium" else f"${v:,.0f}")
    if lever == RAMP_START:
        return f"month-1 productivity set to {v:.0%}"
    if lever == RAMP_MONTHS:
        return f"ramp-up to full productivity in {v:g} months"
    if lever == RAMP_ENABLED:
        return "ramp-up discount " + ("on" if v else "off")
    if lever == VACANCY_BASIS:
        return "break-even basis: " + ("full salary" if v else "coverage premium")
    return lever
