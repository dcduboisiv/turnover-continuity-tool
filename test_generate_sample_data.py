import filecmp

import pandas as pd
import pytest

import generate_sample_data as gen
from turnover_calculator import (
    care_continuity,
    load_data,
    retention_curves,
    turnover_rates,
    vacancy_report,
)


@pytest.fixture(scope="module")
def data():
    return load_data(".")


@pytest.fixture(scope="module")
def sites():
    return pd.read_csv(gen.SITES_FILE)


def test_generation_is_deterministic(tmp_path):
    gen.write(gen.generate(7), tmp_path / "a")
    gen.write(gen.generate(7), tmp_path / "b")
    for name in gen.generate(7):
        assert filecmp.cmp(tmp_path / "a" / name, tmp_path / "b" / name, shallow=False)


def test_committed_files_match_the_default_seed(tmp_path):
    gen.write(gen.generate(gen.DEFAULT_SEED), tmp_path)
    for name in gen.generate(gen.DEFAULT_SEED):
        assert filecmp.cmp(tmp_path / name, name, shallow=False), f"{name} is stale: rerun the generator"


# ---------------------------------------------------------------------------
# Fictional org, synthetic labels
# ---------------------------------------------------------------------------

def test_single_fictional_org_in_three_boroughs(sites):
    assert set(sites["organization"]) == {"Kinbridge Family Services"}
    assert set(sites["borough"]) == {"Manhattan", "Brooklyn", "Bronx"}   # no Queens / Staten Island
    assert sites["site_name"].str.endswith("(SYN)").all()
    assert sites["is_synthetic"].all()
    assert sites["site_id"].is_unique


def test_no_real_agency_names_anywhere(sites):
    real = ["Children's Aid", "Graham Windham", "SCO", "JCCA", "Good Shepherd", "Foundling",
            "Rising Ground", "Forestdale", "Seamen's", "Cayuga", "St. Dominic", "HeartShare"]
    text = sites.to_csv()
    assert not [name for name in real if name.lower() in text.lower()]


def test_every_person_and_child_is_labeled_synthetic(data):
    staff, children, assignments = data["staff"], data["children"], data["assignments"]
    assert staff["staff_id"].str.startswith("SYN-S-").all() and staff["staff_id"].is_unique
    assert children["child_id"].str.startswith("SYN-C-").all() and children["child_id"].is_unique
    for df in (staff, children, assignments):
        assert df["is_synthetic"].all()


# ---------------------------------------------------------------------------
# Roster shape
# ---------------------------------------------------------------------------

def test_roster_has_required_fields_and_program_areas(data):
    staff = data["staff"]
    for col in ["hire_date", "termination_date", "role", "program_area", "staff_type", "site_id",
                "termination_type", "tenure_days", "tenure_months"]:
        assert col in staff.columns
    assert set(staff["program_area"]) == {"Direct Care", "Case Management", "Clinical", "Admin"}


def test_three_years_of_history(data):
    staff = data["staff"]
    term = staff["termination_date"].dropna()
    assert term.min() >= gen.HISTORY_START and term.max() < gen.HISTORY_END
    assert set(term.dt.year) == {2023, 2024, 2025}
    assert {2023, 2024, 2025} <= set(staff["hire_date"].dt.year)
    assert staff["hire_date"].max() < gen.HISTORY_END


def test_termination_fields_are_consistent(data):
    staff = data["staff"]
    left = staff["termination_date"].notna()
    assert set(staff.loc[left, "termination_type"]) == {"Voluntary", "Involuntary"}
    assert (staff.loc[~left, "termination_type"] == "").all()
    assert (staff.loc[left, "status"] == "Terminated").all() and (staff.loc[~left, "status"] == "Active").all()
    end = staff["termination_date"].fillna(gen.HISTORY_END)
    assert ((end - staff["hire_date"]).dt.days == staff["tenure_days"]).all()
    assert (staff["tenure_days"] > 0).all()


def test_one_person_per_seat_at_a_time(data):
    staff = data["staff"].sort_values(["position_id", "hire_date"])
    prior_end = staff.groupby("position_id")["termination_date"].shift()
    has_prior = prior_end.notna()
    assert (staff.loc[has_prior, "hire_date"] > prior_end[has_prior]).all()
    # only the last person in each seat can still be active
    assert (staff.groupby("position_id")["termination_date"].apply(lambda s: s.iloc[:-1].notna().all())).all()
    assert set(staff["position_id"]) <= set(data["positions"]["position_id"])


def test_every_staff_and_seat_record_has_a_staff_type(data):
    for df in (data["staff"], data["positions"]):
        assert set(df["staff_type"]) == {"field", "central_admin"}
        # central admin is exactly the internal-only Admin program area
        assert ((df["staff_type"] == "central_admin") == (df["program_area"] == "Admin")).all()
    # supervisors have family contact and count as field
    sup = data["staff"]["role"].isin(["Case Supervisor", "Clinical Supervisor"])
    assert (data["staff"].loc[sup, "staff_type"] == "field").all()


def test_only_field_staff_are_assigned_to_children(data):
    types = data["staff"].set_index("staff_id")["staff_type"]
    assert (data["assignments"]["staff_id"].map(types) == "field").all()


# ---------------------------------------------------------------------------
# Realistic, non-uniform turnover
# ---------------------------------------------------------------------------

def test_turnover_concentrated_in_direct_care_and_casework(data):
    r = turnover_rates(data["staff"], gen.HISTORY_START, gen.HISTORY_END, by="program_area")["turnover_rate"]
    assert r["Direct Care"] >= 2 * r["Admin"]
    assert r["Case Management"] >= 2 * r["Admin"]
    assert r["Direct Care"] > r["Clinical"] > r["Admin"]
    assert r.max() - r.min() > 0.2                         # not flat


def test_most_departures_are_voluntary(data):
    r = turnover_rates(data["staff"], gen.HISTORY_START, gen.HISTORY_END, by=None).loc["All"]
    assert 0.6 < r["voluntary_separations"] / r["separations"] < 0.9


def test_departures_concentrate_early_in_tenure(data):
    curves = retention_curves(data["staff"], gen.HISTORY_END, months=(3, 12, 24),
                              by=None, hired_from=gen.HISTORY_START)["retention"]
    drop_first_year = 1 - curves[("All", 12)]
    drop_second_year = curves[("All", 12)] - curves[("All", 24)]
    assert drop_first_year > 2 * drop_second_year


def test_vacancies_take_longer_to_fill_for_clinical_roles(data):
    v = vacancy_report(data["staff"], data["positions"], gen.HISTORY_START, gen.HISTORY_END, by="role")
    assert v.loc["Clinician", "avg_days_to_fill"] > v.loc["Youth Care Worker", "avg_days_to_fill"]
    assert v["vacancy_rate"].between(0, 0.25).all()


# ---------------------------------------------------------------------------
# Child assignments
# ---------------------------------------------------------------------------

def test_assignments_sit_inside_placements_and_employment(data):
    a = data["assignments"].merge(data["children"], on="child_id", suffixes=("", "_child"))
    assert (a["start_date"] < a["end_date"]).all()
    assert (a["start_date"] >= a["placement_start"]).all()
    assert (a["end_date"] <= a["placement_end"].fillna(gen.HISTORY_END)).all()
    a = a.merge(data["staff"][["staff_id", "role", "hire_date", "termination_date"]],
                on="staff_id", suffixes=("", "_staff"))
    assert (a["role"] == a["role_staff"]).all()           # coverage is by someone in the same role
    assert (a["start_date"] >= a["hire_date"]).all()
    assert (a["end_date"] <= a["termination_date"].fillna(gen.HISTORY_END)).all()


def test_caregiver_churn_tracks_staff_turnover(data):
    c = care_continuity(data["assignments"], gen.HISTORY_START, gen.HISTORY_END)
    assert c["continuity_score"].min() >= 1.0
    assert c["transitions"].sum() > 0
    assert data["assignments"]["is_coverage"].any()
