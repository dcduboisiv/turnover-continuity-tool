import itertools

import pytest
from streamlit.testing.v1 import AppTest

from generate_sample_data import HISTORY_END, HISTORY_START
from turnover_calculator import (
    care_continuity_score,
    family_days_without_assigned_worker,
    load_data,
    open_seats,
    turnover_rates,
    vacancy_report,
)

DATA = load_data(".")
SITES = ["__all__"] + [f"SITE-0{i}" for i in range(1, 9)]


def run_app() -> AppTest:
    at = AppTest.from_file("app.py", default_timeout=60)
    at.run()
    return at


def metrics(at):
    return {m.label: m.value for m in at.metric}


def hints(at):
    return [i.value for i in at.info if "fictional" not in i.value]


def num(s: str) -> float:
    return float(s.replace("$", "").replace(",", "").replace("%", ""))


def test_app_loads_and_labels_data_fictional():
    at = run_app()
    assert not at.exception and not at.error
    assert any("fictional" in i.value for i in at.sidebar.info)


def test_default_view_is_full_and_non_empty():
    at = run_app()
    assert at.sidebar.selectbox(key="site").value == "__all__"
    assert at.sidebar.radio(key="staff_type").value == "All staff"
    m = metrics(at)
    for label in ["Turnover rate", "Vacancy rate", "Care-continuity score", "Cost of turnover / yr",
                  "Family-days without a worker"]:
        assert m[label] not in ("—", "0", "$0", "0.0%"), label
    assert hints(at) == []                                 # no section is empty
    assert len(at.get("arrow_vega_lite_chart")) == 3       # break-even, family-days, retention


def test_headline_metrics_match_the_engine():
    m = metrics(run_app())
    staff, positions = DATA["staff"], DATA["positions"]
    overall = turnover_rates(staff, HISTORY_START, HISTORY_END, by=None).loc["All"]
    vacancy = vacancy_report(staff, positions, HISTORY_START, HISTORY_END, by=None).loc["All"]
    assert m["Turnover rate"] == f"{overall['turnover_rate']:.1%}"
    assert m["Vacancy rate"] == f"{vacancy['vacancy_rate']:.1%}"
    assert m["Care-continuity score"] == f"{care_continuity_score(DATA['assignments'], HISTORY_START, HISTORY_END):.2f}"
    gap = family_days_without_assigned_worker(staff, positions, HISTORY_START, HISTORY_END, by=None)
    assert m["Family-days without a worker"] == f"{gap.loc['All', 'family_days']:,.0f}"


def test_open_seats_table_matches_engine():
    at = run_app()
    table = at.main.dataframe[-1].value
    expected = open_seats(DATA["staff"], DATA["positions"], HISTORY_END)
    assert len(table) == len(expected) > 0
    assert set(table["Seat"]) == set(expected["position_id"])


def test_central_admin_view():
    at = run_app()
    at.sidebar.radio(key="staff_type").set_value("Central admin").run()
    assert not at.exception
    m = metrics(at)
    r = turnover_rates(DATA["staff"], HISTORY_START, HISTORY_END, by="staff_type").loc["central_admin"]
    assert m["Turnover rate"] == f"{r['turnover_rate']:.1%}"
    assert m["Care-continuity score"] == "—"
    assert any("No caseload-carrying roles" in h for h in hints(at))
    roles = at.sidebar.multiselect[1].options
    assert set(roles) == {"Administrative Assistant", "Finance Associate"}


def test_site_filter_limits_roles_and_scores_children_served():
    at = run_app()
    at.sidebar.selectbox(key="site").select("SITE-08").run()   # clinic: no children placed there
    assert not at.exception
    assert "Caseworker" not in at.sidebar.multiselect[1].options
    assert metrics(at)["Care-continuity score"] != "—"         # its clinicians serve Mott Haven children


def test_ramp_toggle_changes_cost_but_table_keeps_both():
    at = run_app()
    with_ramp = num(metrics(at)["Cost of turnover / yr"])
    table_on = at.main.dataframe[0].value
    at.sidebar.toggle(key="ramp_on").set_value(False).run()
    assert num(metrics(at)["Cost of turnover / yr"]) < with_ramp
    table_off = at.main.dataframe[0].value
    for col in ["Break-even months (no ramp)", "Break-even months (with ramp)"]:
        assert table_off[col].tolist() == table_on[col].tolist()
    assert (table_on["Break-even months (with ramp)"] > table_on["Break-even months (no ramp)"]).all()


def test_ramp_settings_flow_through():
    at = run_app()
    at.sidebar.slider(key="ramp_months").set_value(1).run()  # full productivity in month 1 = no ramp
    t = at.main.dataframe[0].value
    assert t["Break-even months (with ramp)"].tolist() == pytest.approx(t["Break-even months (no ramp)"].tolist())


def test_vacancy_cost_basis_toggle():
    at = run_app()
    before = at.main.dataframe[0].value["Break-even months (no ramp)"]
    at.sidebar.radio(key="basis").set_value("full_salary").run()
    after = at.main.dataframe[0].value["Break-even months (no ramp)"]
    assert (after < before).all()
    # crossover always uses the coverage premium, whatever the basis
    assert at.main.dataframe[1].value["Crossover months"].tolist() == pytest.approx(before.tolist())


def test_caseload_inputs_scale_family_days():
    at = run_app()
    base = num(metrics(at)["Family-days without a worker"])
    at.sidebar.number_input(key="caseload_Caseworker").set_value(20)
    at.sidebar.number_input(key="caseload_Clinician").set_value(40).run()
    assert num(metrics(at)["Family-days without a worker"]) == pytest.approx(2 * base)


def test_single_year_period():
    at = run_app()
    at.sidebar.select_slider(key="period").set_value((2025, 2025)).run()
    assert not at.exception
    r = turnover_rates(DATA["staff"], "2025-01-01", "2026-01-01", by=None).loc["All"]
    assert metrics(at)["Turnover rate"] == f"{r['turnover_rate']:.1%}"


@pytest.mark.parametrize("site,staff_type", list(itertools.product(SITES, ["All staff", "Field", "Central admin"])))
def test_every_site_and_staff_type_renders(site, staff_type):
    at = run_app()
    at.sidebar.selectbox(key="site").select(site)
    at.sidebar.radio(key="staff_type").set_value(staff_type).run()
    assert not at.exception and not at.error
