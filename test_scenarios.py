import pandas as pd
import pytest

import scenarios as sc
from generate_sample_data import HISTORY_END, HISTORY_START
from turnover_calculator import (
    care_continuity,
    care_continuity_score,
    family_days_without_assigned_worker,
    load_cost_assumptions,
    load_data,
    load_program_budgets,
    turnover_rates,
    vacancy_report,
)

DATA = load_data(".")
ASSUMPTIONS = load_cost_assumptions()
ORDER = list(ASSUMPTIONS.index)
SITES = [f"SITE-0{i}" for i in range(1, 9)]
T = pd.Timestamp


def settings(**kw):
    return sc.Settings(ASSUMPTIONS.copy(), **kw)


# ---------------------------------------------------------------------------
# Views and the dashboard's numbers
# ---------------------------------------------------------------------------

def test_headline_matches_the_calculator():
    h = sc.headline(DATA, sc.View(), settings(), ORDER)
    staff, positions = DATA["staff"], DATA["positions"]
    assert h["turnover_rate"] == turnover_rates(staff, HISTORY_START, HISTORY_END, by=None).loc["All", "turnover_rate"]
    assert h["vacancy_rate"] == vacancy_report(staff, positions, HISTORY_START, HISTORY_END, by=None).loc["All", "vacancy_rate"]
    assert h["continuity_score"] == care_continuity_score(DATA["assignments"], HISTORY_START, HISTORY_END)
    gap = family_days_without_assigned_worker(staff, positions, HISTORY_START, HISTORY_END, by=None)
    assert h["family_days_without_worker"] == gap.loc["All", "family_days"]
    assert h["children_served"] == DATA["children"]["child_id"].nunique()


def test_view_with_no_seats_is_empty_and_no_children_view_has_no_score():
    assert sc.headline(DATA, sc.View(site_id="SITE-01", roles=("Caseworker",)), settings(), ORDER) == {}
    admin = sc.headline(DATA, sc.View(staff_type="central_admin"), settings(), ORDER)
    assert admin["continuity_score"] is None and admin["children_served"] is None
    assert admin["family_days_without_worker"] == 0


def test_site_view_follows_the_children_it_serves():
    clinic = sc.filter_view(DATA, sc.View(site_id="SITE-08"), ORDER)
    assert not clinic["assignments"].empty                       # Fordham serves Mott Haven children
    assert set(clinic["assignments"]["site_id"]) == {"SITE-06"}
    assert set(clinic["staff"]["site_id"]) == {"SITE-08"}


def test_site_table_for_one_role_has_identical_breakeven():
    table = sc.site_table(DATA, sc.View(roles=("Caseworker",)), settings(), ORDER, SITES)
    assert set(table.index) == {"SITE-02", "SITE-03", "SITE-04", "SITE-06"}   # sites with caseworkers
    assert table["breakeven_months_with_ramp"].nunique() == 1    # break-even is a role-level number
    assert table["share_left_before_breakeven"].nunique() > 1    # but who leaves before it differs


def test_role_table_matches_settings():
    on = sc.role_table(DATA, sc.View(), settings(), ORDER)
    full = sc.role_table(DATA, sc.View(), settings(basis="full_salary"), ORDER)
    assert (full["breakeven_months_with_ramp"] < on["breakeven_months_with_ramp"]).all()
    assert on.loc["Youth Care Worker", "family_days_without_worker"] != on.loc["Youth Care Worker", "family_days_without_worker"]  # NaN: no caseload


# ---------------------------------------------------------------------------
# avoid_departures: hand-built seat history
# ---------------------------------------------------------------------------

def staff_rows(rows):
    df = pd.DataFrame(rows, columns=["staff_id", "position_id", "hire_date", "termination_date", "termination_type"])
    df["hire_date"] = pd.to_datetime(df["hire_date"])
    df["termination_date"] = pd.to_datetime(df["termination_date"])
    df["termination_type"] = df["termination_type"].fillna("")
    df["site_id"], df["role"], df["program_area"], df["staff_type"] = "S1", "Caseworker", "Case Management", "field"
    return df


# Seat P1 in 2023: A leaves Mar 1 (Z covers to Apr 1); B starts Apr 1, leaves
# Sep 1 (W covers to Oct 1); C starts Oct 1. Z and W are coworkers in P2/P3.
HAND = {
    "staff": staff_rows([
        ("A", "P1", "2020-01-01", "2023-03-01", "Voluntary"),
        ("B", "P1", "2023-04-01", "2023-09-01", "Involuntary"),
        ("C", "P1", "2023-10-01", None, None),
        ("Z", "P2", "2019-01-01", None, None),
        ("W", "P3", "2019-01-01", None, None),
    ]),
    "assignments": pd.DataFrame({
        "child_id": ["X"] * 5, "site_id": ["S1"] * 5, "role": ["Caseworker"] * 5, "position_id": ["P1"] * 5,
        "staff_id": ["A", "Z", "B", "W", "C"],
        "start_date": pd.to_datetime(["2023-01-01", "2023-03-01", "2023-04-01", "2023-09-01", "2023-10-01"]),
        "end_date": pd.to_datetime(["2023-03-01", "2023-04-01", "2023-09-01", "2023-10-01", "2024-01-01"]),
        "is_coverage": [False, True, False, True, False],
    }),
}
Y_START, Y_END = T("2023-01-01"), T("2024-01-01")


def test_pick_evenly():
    assert sc.pick_evenly(10, 0.1) == [5]
    assert sc.pick_evenly(10, 0.3) == [1, 5, 8]
    assert sc.pick_evenly(2, 0.5) == [1]
    assert sc.pick_evenly(4, 0) == []
    assert sc.pick_evenly(3, 1.0) == [0, 1, 2]


def test_avoiding_one_departure_keeps_the_leaver_in_seat():
    out = sc.avoid_departures(HAND, 0.5, Y_START, Y_END)           # 2 departures -> avoids the 2nd (B)
    staff = out["staff"].set_index("staff_id")
    assert "C" not in staff.index                                  # C was never needed
    assert pd.isna(staff.loc["B", "termination_date"]) and staff.loc["B", "status"] == "Active"
    a = out["assignments"]
    assert a["staff_id"].tolist() == ["A", "Z", "B", "B", "B"]     # W's cover and C's stint become B
    assert care_continuity(a, Y_START, Y_END).loc["X", "transitions"] == 2   # A, Z, B (was 4)
    assert care_continuity(HAND["assignments"], Y_START, Y_END).loc["X", "transitions"] == 4
    assert HAND["staff"]["staff_id"].tolist() == ["A", "B", "C", "Z", "W"]   # input unchanged


def test_avoiding_every_departure_chains_through_successors():
    out = sc.avoid_departures(HAND, 1.0, Y_START, Y_END)
    staff = out["staff"].set_index("staff_id")
    assert list(staff.index) == ["A", "Z", "W"]
    assert pd.isna(staff.loc["A", "termination_date"])
    assert set(out["assignments"]["staff_id"]) == {"A"}
    assert not out["assignments"]["is_coverage"].any()
    assert turnover_rates(out["staff"], Y_START, Y_END, by=None).loc["All", "separations"] == 0


def test_avoid_departures_scopes_by_role_and_site():
    out = sc.avoid_departures(HAND, 1.0, Y_START, Y_END, roles=["Clinician"])
    assert len(out["staff"]) == len(HAND["staff"])
    out = sc.avoid_departures(HAND, 1.0, Y_START, Y_END, sites=["S2"])
    assert len(out["staff"]) == len(HAND["staff"])
    with pytest.raises(ValueError):
        sc.avoid_departures(HAND, 1.5, Y_START, Y_END)


# ---------------------------------------------------------------------------
# apply_changes on the sample data
# ---------------------------------------------------------------------------

def change(lever, value, role="", site="", unit="absolute"):
    return {"lever": lever, "role": role, "site": site, "value": value, "unit": unit}


def test_caseworker_turnover_reduction_is_exact_and_improves_continuity():
    view, s = sc.View(), settings()
    before = turnover_rates(DATA["staff"], view.start, view.end).loc["Caseworker"]
    data, _ = sc.apply_changes(DATA, s, view, [change("turnover_reduction", 0.10, "Caseworker", unit="relative")])
    after = turnover_rates(data["staff"], view.start, view.end).loc["Caseworker"]
    assert after["separations"] == before["separations"] - round(0.10 * before["separations"])
    h0, h1 = sc.headline(DATA, view, s, ORDER), sc.headline(data, view, s, ORDER)
    assert h1["continuity_score"] < h0["continuity_score"]
    assert h1["family_days_without_worker"] < h0["family_days_without_worker"]
    # other roles untouched
    other = turnover_rates(data["staff"], view.start, view.end).drop("Caseworker")["separations"]
    assert other.equals(turnover_rates(DATA["staff"], view.start, view.end).drop("Caseworker")["separations"])


def test_turnover_reduction_in_points():
    view = sc.View()
    rate = turnover_rates(DATA["staff"], view.start, view.end).loc["Caseworker", "turnover_rate"]
    c = change("turnover_reduction", 0.10, "Caseworker", unit="points")
    assert sc.turnover_fraction(DATA, c, view) == pytest.approx(0.10 / rate)
    with pytest.raises(ValueError, match="more than the whole"):   # admin assistants turn over ~20%
        sc.turnover_fraction(DATA, change("turnover_reduction", 0.3, "Administrative Assistant", unit="points"), view)
    with pytest.raises(ValueError, match="no turnover"):            # no finance departures in the data
        sc.turnover_fraction(DATA, change("turnover_reduction", 0.05, "Finance Associate", unit="points"), view)


def test_caseload_change_scales_that_roles_family_days():
    view, s = sc.View(), settings()
    _, s2 = sc.apply_changes(DATA, s, view, [change("caseload", 25, "Clinician")])
    assert s2.caseload == {"Caseworker": 10, "Clinician": 25} and s.caseload["Clinician"] == 20
    t0, t1 = sc.role_table(DATA, view, s, ORDER), sc.role_table(DATA, view, s2, ORDER)
    assert t1.loc["Clinician", "family_days_without_worker"] == pytest.approx(
        t0.loc["Clinician", "family_days_without_worker"] * 25 / 20)
    assert t1.loc["Caseworker", "family_days_without_worker"] == t0.loc["Caseworker", "family_days_without_worker"]


def test_cost_ramp_and_basis_changes():
    view, s = sc.View(), settings()
    _, s2 = sc.apply_changes(DATA, s, view, [
        change("recruiting_cost", 0.5, "Caseworker", unit="relative"),
        change("training_cost", 0, unit="absolute"),
        change("ramp_enabled", 0), change("ramp_months_to_full", 4), change("ramp_start_productivity", 0.4),
        change("vacancy_cost_basis_full_salary", 1),
    ])
    assert s2.assumptions.loc["Caseworker", "recruiting_cost"] == 6_000
    assert (s2.assumptions["training_cost"] == 0).all()
    assert (s2.ramp_on, s2.ramp_months, s2.ramp_start, s2.basis) == (False, 4, 0.4, "full_salary")
    assert s.assumptions.loc["Caseworker", "recruiting_cost"] == 4_000 and s.ramp_on   # inputs unchanged
    with pytest.raises(ValueError):
        sc.apply_changes(DATA, s, view, [change("hiring_speed", 1)])


def test_describe_change():
    names = {"SITE-02": "East Harlem Family Center"}
    assert sc.describe_change(change("turnover_reduction", 0.1, "Caseworker", "SITE-02", "relative"), names) == \
        "Caseworker turnover at East Harlem Family Center down 10% (that share of departures avoided; those staff stay)"
    assert sc.describe_change(change("caseload", 25, "Clinician"), names) == "Clinician typical caseload set to 25"
    assert sc.describe_change(change("recruiting_cost", 0.2, unit="relative"), names) == "all roles recruiting cost +20%"


def test_budget_share_for_all_sites_and_one_site():
    budgets = load_program_budgets().to_dict()
    s = sc.Settings(ASSUMPTIONS.copy(), budgets=budgets)
    h = sc.headline(DATA, sc.View(), s, ORDER)
    assert h["program_budget"] == 22_000_000
    assert h["turnover_cost_share_of_budget"] == pytest.approx(h["annual_turnover_cost"] / 22_000_000)
    assert h["monthly_turnover_cost"] == pytest.approx(h["annual_turnover_cost"] / 12)
    assert h["avg_cost_per_departure"] * h["departures"] == pytest.approx(
        h["annual_turnover_cost"] * (HISTORY_END - HISTORY_START).days / 365.25)
    one = sc.headline(DATA, sc.View(site_id="SITE-03"), s, ORDER)
    assert one["program_budget"] == budgets["SITE-03"]
    assert sc.headline(DATA, sc.View(), settings(), ORDER)["turnover_cost_share_of_budget"] is None
