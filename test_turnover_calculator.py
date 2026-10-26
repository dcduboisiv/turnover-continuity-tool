import math

import pandas as pd
import pytest

from turnover_calculator import (
    RAMP_MONTHS_TO_FULL,
    RAMP_START_PRODUCTIVITY,
    RAMP_UP_SCHEDULE,
    ROLE_CRITICALITY,
    breakeven_months,
    breakeven_tenure,
    care_continuity,
    care_continuity_score,
    cost_per_departure,
    fill_events,
    linear_ramp,
    load_cost_assumptions,
    retention_curves,
    turnover_rates,
    vacancy_report,
)

T = pd.Timestamp
START, END = T("2023-01-01"), T("2024-01-01")   # 365 days


def roster(rows):
    df = pd.DataFrame(rows, columns=["staff_id", "position_id", "role", "hire_date",
                                     "termination_date", "termination_type"])
    df["hire_date"] = pd.to_datetime(df["hire_date"])
    df["termination_date"] = pd.to_datetime(df["termination_date"])
    df["termination_type"] = df["termination_type"].fillna("")
    df["program_area"] = df["role"].map({"Caseworker": "Case Management",
                                         "Youth Care Worker": "Direct Care"})
    term = df["termination_date"].fillna(END)
    df["tenure_days"] = (term - df["hire_date"]).dt.days
    return df


# Hand-worked case, 2023 (365 days):
#   A  Caseworker  in seat all year                         365 days
#   B  Caseworker  leaves 2023-03-15 (voluntary)             73 days
#   C  Caseworker  backfills B's seat 2023-04-14             262 days  (30 days vacant)
#   D  Youth Care  hired 2023-01-01, fired 2023-07-02       182 days
#   E  Caseworker  left 2022-12-31: before the period          0 days, not a separation
#   F  Youth Care  leaves 2024-01-01 = period end            365 days, not a separation
STAFF = roster([
    ("A", "P1", "Caseworker", "2020-05-01", None, None),
    ("B", "P2", "Caseworker", "2021-02-01", "2023-03-15", "Voluntary"),
    ("C", "P2", "Caseworker", "2023-04-14", None, None),
    ("D", "P3", "Youth Care Worker", "2023-01-01", "2023-07-02", "Involuntary"),
    ("E", "P4", "Caseworker", "2019-01-01", "2022-12-31", "Voluntary"),
    ("F", "P5", "Youth Care Worker", "2022-06-01", "2024-01-01", "Voluntary"),
])
POSITIONS = pd.DataFrame({
    "position_id": ["P1", "P2", "P3", "P4", "P5"],
    "role": ["Caseworker", "Caseworker", "Youth Care Worker", "Caseworker", "Youth Care Worker"],
})


# ---------------------------------------------------------------------------
# Turnover rate
# ---------------------------------------------------------------------------

def test_turnover_rate_hand_calculated_overall():
    r = turnover_rates(STAFF, START, END, by=None).loc["All"]
    exposure = 365 + 73 + 262 + 182 + 0 + 365          # 1,247 filled seat-days
    assert r["avg_headcount"] == pytest.approx(exposure / 365)
    assert r["separations"] == 2
    # separations / avg headcount, annualized = separations x 365.25 / exposure
    assert r["turnover_rate"] == pytest.approx(2 * 365.25 / exposure)
    assert r["voluntary_rate"] == pytest.approx(1 * 365.25 / exposure)
    assert r["involuntary_rate"] == pytest.approx(1 * 365.25 / exposure)


def test_turnover_rate_hand_calculated_by_role():
    r = turnover_rates(STAFF, START, END, by="role")
    cw, ycw = r.loc["Caseworker"], r.loc["Youth Care Worker"]
    assert (cw["separations"], cw["voluntary_separations"], cw["involuntary_separations"]) == (1, 1, 0)
    assert cw["turnover_rate"] == pytest.approx(365.25 / (365 + 73 + 262))
    assert (ycw["separations"], ycw["involuntary_separations"]) == (1, 1)
    assert ycw["turnover_rate"] == pytest.approx(365.25 / (182 + 365))
    assert ycw["voluntary_rate"] == 0


def test_turnover_rate_annualizes_a_partial_year():
    # Same events over the first half only: one leaver, 2 x 181-day seats.
    staff = roster([("A", "P1", "Caseworker", "2020-01-01", None, None),
                    ("B", "P2", "Caseworker", "2020-01-01", "2023-03-01", "Voluntary"),
                    ("C", "P2", "Caseworker", "2023-03-01", None, None)])
    half = T("2023-07-01")                                 # 181 days
    r = turnover_rates(staff, START, half, by=None).loc["All"]
    assert r["avg_headcount"] == pytest.approx(2.0)
    assert r["turnover_rate"] == pytest.approx(0.5 * 365.25 / 181)


def test_turnover_rate_rejects_empty_period():
    with pytest.raises(ValueError):
        turnover_rates(STAFF, END, START)


# ---------------------------------------------------------------------------
# Vacancy and days-to-fill
# ---------------------------------------------------------------------------

def test_vacancy_and_days_to_fill_hand_calculated():
    v = vacancy_report(STAFF, POSITIONS, START, END, by="role")
    cw = v.loc["Caseworker"]
    # 3 caseworker seats x 365 days; filled 365 (P1) + 335 (P2) + 0 (P4 never refilled)
    assert cw["positions"] == 3
    assert cw["vacancy_rate"] == pytest.approx(1 - (365 + 73 + 262) / (3 * 365))
    assert cw["fills"] == 1
    assert cw["avg_days_to_fill"] == 30                    # 2023-03-15 -> 2023-04-14
    assert cw["open_at_end"] == 1                          # P4
    ycw = v.loc["Youth Care Worker"]
    assert ycw["vacancy_rate"] == pytest.approx(1 - (182 + 365) / (2 * 365))
    assert ycw["fills"] == 0 and math.isnan(ycw["avg_days_to_fill"])
    # D's seat is open; F was still in seat on the last day (leaves on the exclusive end)
    assert ycw["open_at_end"] == 1


def test_fill_events_only_counts_backfills_inside_the_period():
    f = fill_events(STAFF, START, END)
    assert list(f["staff_id"]) == ["C"]
    assert list(f["days_to_fill"]) == [30]
    assert fill_events(STAFF, T("2024-01-01"), T("2025-01-01")).empty


# ---------------------------------------------------------------------------
# Retention cohorts
# ---------------------------------------------------------------------------

def test_retention_cohorts_hand_calculated():
    staff = roster([
        # 2023 cohort, observed to 2024-01-01 (exclusive)
        ("h1", "P1", "Caseworker", "2023-01-01", "2023-02-01", "Voluntary"),   # gone before 3 mo
        ("h2", "P2", "Caseworker", "2023-01-01", "2023-04-01", "Voluntary"),   # leaves ON the 3-mo date
        ("h3", "P3", "Caseworker", "2023-01-01", "2023-08-01", "Voluntary"),   # past 6 mo, not 12
        ("h4", "P4", "Caseworker", "2023-01-01", None, None),
        ("h5", "P5", "Caseworker", "2023-11-01", None, None),                  # too new for 3 mo
        # 2022 cohort
        ("h6", "P6", "Caseworker", "2022-01-01", "2022-10-01", "Voluntary"),
    ])
    c = retention_curves(staff, END, months=(3, 6, 12))
    assert c.loc[(2023, 3), ["eligible", "retained"]].tolist() == [4, 2]
    assert c.loc[(2023, 3), "retention"] == 0.5
    assert c.loc[(2023, 6), "retention"] == 0.5            # h3 still there on 2023-07-01
    assert c.loc[(2023, 12), "eligible"] == 0              # 2024-01-01 not yet observed
    assert math.isnan(c.loc[(2023, 12), "retention"])
    assert c.loc[2022, "retention"].tolist() == [1.0, 1.0, 0.0]


def test_retention_cohorts_by_role_and_hired_from():
    c = retention_curves(STAFF, END, months=(3,), by="role", hired_from=START)
    # only C (Caseworker) and D (Youth Care) were hired in 2023; both past 3 months
    assert c.loc[("Caseworker", 3), "retention"] == 1.0
    assert c.loc[("Youth Care Worker", 3), "retention"] == 1.0


# ---------------------------------------------------------------------------
# Cost of turnover
# ---------------------------------------------------------------------------

KNOWN = pd.DataFrame({
    "annual_salary": [120_000], "fringe_rate": [0.25],     # loaded: $12,500 / month
    "recruiting_cost": [5_000], "training_cost": [7_000],
    "days_to_fill": [2 * 365.25 / 12],                     # exactly two months
    "coverage_premium": [0.40],
}, index=pd.Index(["Test Role"], name="role"))


def test_cost_per_departure_known_input_output():
    c = cost_per_departure(KNOWN).loc["Test Role"]
    assert c["coverage_cost"] == pytest.approx(10_000)     # 2 months x 12,500 x 0.40
    assert c["ramp_up_cost"] == pytest.approx(9_375)       # (0.50 + 0.25 + 0) x 12,500
    assert c["cost_per_departure"] == pytest.approx(31_375)


def test_cost_per_departure_uses_observed_days_to_fill():
    c = cost_per_departure(KNOWN, days_to_fill={"Test Role": 365.25 / 12, "Other": 99}).loc["Test Role"]
    assert c["days_to_fill"] == pytest.approx(365.25 / 12)
    assert c["coverage_cost"] == pytest.approx(5_000)
    # a NaN observation (no fills yet) keeps the planning assumption
    c = cost_per_departure(KNOWN, days_to_fill={"Test Role": float("nan")}).loc["Test Role"]
    assert c["coverage_cost"] == pytest.approx(10_000)


def test_cost_per_departure_without_ramp_drops_ramp_cost():
    c = cost_per_departure(KNOWN, schedule=None).loc["Test Role"]
    assert c["ramp_up_cost"] == 0
    assert c["cost_per_departure"] == pytest.approx(22_000)


def test_shipped_cost_assumptions_cover_every_role():
    a = load_cost_assumptions()
    assert set(a.index) == {
        "Youth Care Worker", "Residential Counselor", "Caseworker", "Case Supervisor",
        "Clinician", "Clinical Supervisor", "Administrative Assistant", "Finance Associate"}
    assert (cost_per_departure(a)["cost_per_departure"] > 0).all()


# ---------------------------------------------------------------------------
# Ramp-up and break-even tenure
# ---------------------------------------------------------------------------

def test_default_ramp_is_linear_50_75_100():
    assert (RAMP_START_PRODUCTIVITY, RAMP_MONTHS_TO_FULL) == (0.50, 3)
    assert RAMP_UP_SCHEDULE == (0.50, 0.75, 1.00)
    assert linear_ramp(0.40, 4) == (0.40, 0.60, 0.80, 1.00)
    assert linear_ramp(0.5, 1) == (1.0,)
    with pytest.raises(ValueError):
        linear_ramp(0, 3)


def test_breakeven_without_ramp():
    assert breakeven_months(10_000, 2_500, schedule=None) == pytest.approx(4.0)
    assert breakeven_months(1_000, 2_500, schedule=None) == pytest.approx(0.4)


def test_breakeven_with_ramp_is_later():
    # Recovered per month at 2,500 x (0.50, 0.75, 1, 1, 1 ...):
    #   1,250 / 3,125 / 5,625 / 8,125 / 10,625  -> crosses 10,000 in month 5
    with_ramp = breakeven_months(10_000, 2_500)
    assert with_ramp == pytest.approx(4 + (10_000 - 8_125) / 2_500)   # 4.75
    assert with_ramp > breakeven_months(10_000, 2_500, schedule=None)
    # inside month 1 the hire only recovers half as fast
    assert breakeven_months(1_000, 2_500) == pytest.approx(0.8)


def test_breakeven_custom_schedule_and_edge_cases():
    assert breakeven_months(10_000, 2_500, schedule=linear_ramp(0.25, 4)) == pytest.approx(
        4 + (10_000 - 2_500 * (0.25 + 0.5 + 0.75 + 1.0)) / 2_500)
    assert breakeven_months(0, 2_500) == 0
    assert breakeven_months(10_000, 0) == math.inf
    with pytest.raises(ValueError):
        breakeven_months(10_000, 2_500, schedule=(0.5, 1.5))


def test_breakeven_tenure_by_role():
    b = breakeven_tenure(KNOWN).loc["Test Role"]
    assert b["upfront_cost"] == 12_000
    assert b["monthly_cost_avoided"] == pytest.approx(5_000)   # 12,500 x 0.40
    assert b["breakeven_months_no_ramp"] == pytest.approx(2.4)
    # 2,500 + 3,750 = 6,250 after month 2; month 3 at full speed -> 2 + 5,750/5,000
    assert b["breakeven_months"] == pytest.approx(3.15)
    assert b["ramp_delay_months"] == pytest.approx(0.75)

    shipped = breakeven_tenure(load_cost_assumptions())
    assert (shipped["breakeven_months"] > shipped["breakeven_months_no_ramp"]).all()


def test_breakeven_tenure_counts_departures_before_breakeven():
    a = KNOWN.rename(index={"Test Role": "Caseworker"})
    # 2023 caseworker leavers: B only (tenure from 2021-02-01: ~25 months, past 3.15)
    b = breakeven_tenure(a, staff=STAFF, start=START, end=END).loc["Caseworker"]
    assert (b["departures"], b["left_before_breakeven"]) == (1, 0)
    quick = roster([("Q", "P1", "Caseworker", "2023-01-01", "2023-03-01", "Voluntary")])  # ~1.9 months
    b = breakeven_tenure(a, staff=quick, start=START, end=END).loc["Caseworker"]
    assert b["share_left_before_breakeven"] == 1.0


# ---------------------------------------------------------------------------
# Care continuity
# ---------------------------------------------------------------------------

def assignments(rows):
    df = pd.DataFrame(rows, columns=["child_id", "site_id", "role", "staff_id", "start_date", "end_date"])
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    return df


# Period 2024. Countable caregiver changes:
#   C1  caseworker A all year; clinician X -> Y                    1 transition
#   C2  caseworker A -> B (coverage) -> D; youth care worker P     2 transitions
#   C3  caseworker A; an older 2023 caseworker Z and a supervisor
#       outside the weighted roles must not count                  0 transitions
#   C4  caseworker A -> B -> A (two distinct people)               1 transition
ROSTER = assignments([
    ("C1", "S1", "Caseworker", "A", "2024-01-01", "2025-01-01"),
    ("C1", "S1", "Clinician", "X", "2024-01-01", "2024-06-01"),
    ("C1", "S1", "Clinician", "Y", "2024-06-01", "2025-01-01"),
    ("C2", "S1", "Caseworker", "A", "2024-01-01", "2024-04-01"),
    ("C2", "S1", "Caseworker", "B", "2024-04-01", "2024-05-15"),
    ("C2", "S1", "Caseworker", "D", "2024-05-15", "2025-01-01"),
    ("C2", "S1", "Youth Care Worker", "P", "2024-01-01", "2025-01-01"),
    ("C3", "S2", "Caseworker", "Z", "2023-01-01", "2024-01-01"),     # ends as the period starts
    ("C3", "S2", "Caseworker", "A", "2024-01-01", "2025-01-01"),
    ("C3", "S2", "Case Supervisor", "K", "2024-01-01", "2025-01-01"),
    ("C4", "S2", "Caseworker", "A", "2024-01-01", "2024-03-01"),
    ("C4", "S2", "Caseworker", "B", "2024-03-01", "2024-04-01"),
    ("C4", "S2", "Caseworker", "A", "2024-04-01", "2025-01-01"),
])
P_START, P_END = T("2024-01-01"), T("2025-01-01")


def test_continuity_counts_known_transitions():
    c = care_continuity(ROSTER, P_START, P_END)
    assert c["transitions"].to_dict() == {"C1": 1, "C2": 2, "C3": 0, "C4": 1}
    assert c["transitions"].sum() == 4
    assert c.loc["C2", "Caseworker"] == 3
    assert "Case Supervisor" not in c.columns


def test_continuity_score_weighted_by_role_criticality():
    w = ROLE_CRITICALITY
    c = care_continuity(ROSTER, P_START, P_END)["continuity_score"]
    expected = {
        "C1": (1 * w["Caseworker"] + 2 * w["Clinician"]) / (w["Caseworker"] + w["Clinician"]),
        "C2": (3 * w["Caseworker"] + 1 * w["Youth Care Worker"]) / (w["Caseworker"] + w["Youth Care Worker"]),
        "C3": 1.0,
        "C4": 2.0,
    }
    assert c.to_dict() == pytest.approx(expected)
    assert care_continuity_score(ROSTER, P_START, P_END) == pytest.approx(sum(expected.values()) / 4)
    by_site = care_continuity_score(ROSTER, P_START, P_END, by="site_id")
    assert by_site["S1"] == pytest.approx((expected["C1"] + expected["C2"]) / 2)
    assert by_site["S2"] == pytest.approx(1.5)


def test_continuity_weights_change_the_score_and_period_clips():
    flat = {"Caseworker": 1.0, "Clinician": 1.0}
    c = care_continuity(ROSTER, P_START, P_END, weights=flat)
    assert c.loc["C1", "continuity_score"] == pytest.approx(1.5)
    assert "C2" in c.index and "Youth Care Worker" not in c.columns
    # first quarter only: C2 has just A, C4 has A then B
    q1 = care_continuity(ROSTER, P_START, T("2024-04-01"))
    assert q1["transitions"].to_dict() == {"C1": 0, "C2": 0, "C3": 0, "C4": 1}
    with pytest.raises(ValueError):
        care_continuity(ROSTER, P_START, P_END, weights={"Caseworker": 0})
