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


GAP = "Days without a worker, per affected family"


def run_app() -> AppTest:
    at = AppTest.from_file("app.py", default_timeout=60)
    at.run()
    return at


def metrics(at):
    return {m.label: m.value for m in at.metric}


def hints(at):
    """Empty-section notes; ignores the sidebar label and the AI section's no-key note."""
    return [i.value for i in at.info if "fictional" not in i.value and "API key" not in i.value]


def num(s: str) -> float:
    return float(s.replace("$", "").replace(",", "").replace("%", ""))


def exact_cost(at) -> float:
    """The exact annual cost from the cost card's tooltip (the card shows $X.XXM)."""
    card = next(m for m in at.metric if m.label == "Cost of turnover / yr")
    return num(card.help.split("Exactly ")[1].split(" a year")[0])


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
                  GAP, "Families affected"]:
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
    g = gap.loc["All"]
    assert m[GAP] == f"{g['avg_days_per_affected_family']:.1f} days"
    assert m["Families affected"] == f"{g['affected_families']:,.0f}"


def test_gap_headline_tells_families_times_days_with_the_total_as_detail():
    at = run_app()
    m = metrics(at)
    families, days = num(m["Families affected"]), num(m[GAP].replace(" days", ""))
    total = family_days_without_assigned_worker(DATA["staff"], DATA["positions"], HISTORY_START, HISTORY_END,
                                                by=None).loc["All", "family_days"]
    assert families * days == pytest.approx(total, rel=1e-3)
    assert any(f"{families:,.0f} families experienced a gap" in c.value and f"{total:,.0f} family-days" in c.value
               for c in at.caption)


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
    with_ramp = exact_cost(at)
    table_on = at.main.dataframe[0].value
    at.sidebar.toggle(key="ramp_on").set_value(False).run()
    assert exact_cost(at) < with_ramp
    table_off = at.main.dataframe[0].value
    for col in ["Break-even, no ramp", "Break-even, with ramp"]:
        assert table_off[col].tolist() == table_on[col].tolist()
    assert (table_on["Break-even, with ramp"] > table_on["Break-even, no ramp"]).all()


def test_ramp_settings_flow_through():
    at = run_app()
    at.sidebar.slider(key="ramp_months").set_value(1).run()  # full productivity in month 1 = no ramp
    t = at.main.dataframe[0].value
    assert t["Break-even, with ramp"].tolist() == pytest.approx(t["Break-even, no ramp"].tolist())


def test_vacancy_cost_basis_toggle():
    at = run_app()
    before = at.main.dataframe[0].value["Break-even, no ramp"]
    at.sidebar.radio(key="basis").set_value("full_salary").run()
    after = at.main.dataframe[0].value["Break-even, no ramp"]
    assert (after < before).all()
    # crossover always uses the coverage premium, whatever the basis
    assert at.main.dataframe[1].value["Crossover months"].tolist() == pytest.approx(before.tolist())


def test_caseload_inputs_scale_family_days():
    at = run_app()
    base_days, base_families = num(metrics(at)[GAP].replace(" days", "")), num(metrics(at)["Families affected"])
    at.sidebar.number_input(key="caseload_Caseworker").set_value(20)
    at.sidebar.number_input(key="caseload_Clinician").set_value(40).run()
    # twice the families on each seat, each waiting just as long
    assert num(metrics(at)["Families affected"]) == pytest.approx(2 * base_families)
    assert num(metrics(at)[GAP].replace(" days", "")) == pytest.approx(base_days)


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


# ---------------------------------------------------------------------------
# AI assistant (Claude mocked: no API key or spend needed)
# ---------------------------------------------------------------------------

import json

import turnover_agent as agent_ai
from test_turnover_agent import FakeClaude, change, question_json


def app_with_key(monkeypatch, *replies):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    fake = FakeClaude(*replies)
    monkeypatch.setattr(agent_ai, "call_claude", fake)
    at = AppTest.from_file("app.py", default_timeout=60)
    at.secrets["ANTHROPIC_API_KEY"] = "test-key"
    at.run()
    return at, fake


def ask(at, question):
    at.text_area[0].input(question)
    at.button[0].click()
    return at.run()


def test_question_box_is_first_on_the_page(monkeypatch):
    at, _ = app_with_key(monkeypatch)
    order = [type(node).__name__ for node in at.main]
    assert order.index("TextArea") < order.index("Metric")
    assert at.main.subheader[0].value == "Ask about turnover and continuity"


def test_ai_without_key_explains_setup_and_dashboard_still_works(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    at = run_app()
    assert not at.exception and not at.error
    assert any("needs an Anthropic API key" in i.value for i in at.info)
    assert len(at.main.text_area) == 0
    assert metrics(at)["Turnover rate"] != "—"


def test_ask_what_if_shows_answer_and_engine_deltas(monkeypatch):
    at, fake = app_with_key(
        monkeypatch,
        question_json(changes=[change("turnover_reduction", 0.10, "Caseworker", unit="relative")]),
        "Continuity improves.",
    )
    at = ask(at, "What happens to the care-continuity score if caseworker turnover drops 10%?")
    assert not at.exception and not at.error
    assert len(fake.calls) == 2
    assert "Kinbridge" in fake.calls[0]["system"]
    assert any("Continuity improves." in i.value for i in at.info)
    assert any("Caseworker turnover down 10%" in c.value for c in at.caption)
    context = json.loads(fake.calls[1]["user_prompt"])
    what_if = next(x for x in at.metric if x.label == "Care-continuity score")   # what-if row comes first
    assert what_if.value == context["after"]["continuity_score"]
    assert what_if.delta.startswith("-")


def test_ask_uses_the_sidebar_view_and_settings(monkeypatch):
    at, fake = app_with_key(monkeypatch, question_json(), "Answer.")
    at.sidebar.selectbox(key="site").select("SITE-03")
    at.sidebar.number_input(key="caseload_Clinician").set_value(25)
    at = ask(at.run(), "How is this site doing?")
    context = json.loads(fake.calls[1]["user_prompt"])
    assert context["view"]["site"] == "Kinbridge Washington Heights Family Center"
    assert context["settings"]["typical_caseload"]["Clinician"] == "25"


def test_ask_invalid_question_is_a_warning_not_a_crash(monkeypatch):
    at, fake = app_with_key(monkeypatch, question_json(changes=[change("caseload", 25, "Youth Care Worker")]))
    at = ask(at, "Raise youth care worker caseloads to 25")
    assert not at.exception
    assert any("Couldn't answer that" in w.value for w in at.warning)
    assert len(fake.calls) == 1


def test_ask_flags_invented_figures(monkeypatch):
    at, _ = app_with_key(monkeypatch, question_json(), "Turnover costs roughly $1.2M.")
    at = ask(at, "What does turnover cost?")
    assert any("$1.2M" in w.value for w in at.warning)


def test_breakdown_table_is_shown(monkeypatch):
    at, _ = app_with_key(monkeypatch, question_json(roles=["Caseworker"], breakdown="site",
                                                    focus_metric="breakeven_months_with_ramp"), "Same everywhere.")
    at = ask(at, "Which site has the worst break-even tenure for caseworkers?")
    table = at.main.dataframe[0].value
    assert len(table) == 4 and "site" in table.columns


def test_answer_survives_reruns_without_calling_ai_again(monkeypatch):
    at, fake = app_with_key(monkeypatch, question_json(), "Answer text.")
    at = ask(at, "How are we doing?")
    at.sidebar.toggle(key="ramp_on").set_value(False).run()
    assert any("Answer text." in i.value for i in at.info)
    assert len(fake.calls) == 2


def test_funder_summary_for_the_sidebar_view(monkeypatch):
    at, fake = app_with_key(monkeypatch, "Kinbridge served children. Turnover was 99%.")
    at.sidebar.selectbox(key="site").select("SITE-05")
    at.run()
    at.button(key="funder_report").click().run()
    assert not at.exception and not at.error
    context = json.loads(fake.calls[0]["user_prompt"])
    assert context["view"]["site"] == "Kinbridge Brownsville Youth Residence"
    assert any(t.label.startswith("Funder summary") and "Kinbridge served" in t.value for t in at.text_area)
    assert any("99%" in w.value for w in at.warning)


def test_cost_card_supporting_figures_match_the_engine():
    import scenarios as sc
    from turnover_calculator import load_cost_assumptions, load_program_budgets
    at = run_app()
    h = sc.headline(DATA, sc.View(), sc.Settings(load_cost_assumptions(),
                                                 budgets=load_program_budgets().to_dict()),
                    list(load_cost_assumptions().index))
    assert metrics(at)["Cost of turnover / yr"] == f"${h['annual_turnover_cost'] / 1e6:,.2f}M"   # $1.17M
    assert metrics(at)["Cost of turnover / yr"] == "$1.17M"
    assert exact_cost(at) == round(h["annual_turnover_cost"])          # exact figure on hover
    card = next(c.value for c in at.caption if "per departure" in c.value)
    assert f"${h['avg_cost_per_departure']:,.0f} per departure" in card
    assert f"${h['monthly_turnover_cost']:,.0f} per month" in card
    assert f"{h['turnover_cost_share_of_budget']:.1%} of the $22.00M program budget" in card
    assert h["departures"] == 177


def test_site_view_uses_that_sites_budget():
    at = run_app()
    at.sidebar.selectbox(key="site").select("SITE-03").run()
    card = next(c.value for c in at.caption if "per departure" in c.value)
    assert "of the $3.10M program budget" in card
