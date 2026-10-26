"""
Tests for the AI layer. Every Claude call is mocked: call_claude is
replaced with FakeClaude, which returns queued replies and records what it
was sent, so no API key or API spend is needed. The tests that exercise
call_claude itself use a fake client object to check the request it builds.
"""

import json
from types import SimpleNamespace

import pandas as pd
import pytest

import scenarios as sc
import turnover_agent as agent

ORG = agent.load_org(".")
VIEW = sc.View()


def settings():
    return sc.Settings(ORG_ASSUMPTIONS.copy())


ORG_ASSUMPTIONS = agent.load_cost_assumptions()


class FakeClaude:
    """Stands in for call_claude: returns queued replies and records each call."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, client, system, user_prompt, json_schema=None):
        self.calls.append({"system": system, "user_prompt": user_prompt, "json_schema": json_schema})
        if not self.replies:
            raise AssertionError("call_claude was called more times than the test expected")
        return self.replies.pop(0)


def question_json(**overrides):
    body = {
        "site_id": "", "staff_type": "", "program_areas": [], "roles": [], "start_year": 0, "end_year": 0,
        "changes": [], "breakdown": "none", "focus_metric": "", "assumption_notes": "", "unsupported": "",
    }
    body.update(overrides)
    return json.dumps(body)


def change(lever, value, role="", site="", unit="absolute"):
    return {"lever": lever, "role": role, "site": site, "value": value, "unit": unit}


@pytest.fixture
def fake(monkeypatch):
    def install(*replies):
        f = FakeClaude(*replies)
        monkeypatch.setattr(agent, "call_claude", f)
        return f
    return install


# --- client and request -------------------------------------------------------

def test_get_client_requires_a_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        agent.get_client()


class FakeClient:
    def __init__(self, stop_reason="end_turn", text="ok"):
        self.sent = None
        reply = SimpleNamespace(stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)])

        def create(**kwargs):
            self.sent = kwargs
            return reply
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=create))


def test_call_claude_builds_the_request():
    client = FakeClient(text='{"a": 1}')
    schema = {"type": "object"}
    assert agent.call_claude(client, "sys", "hello", json_schema=schema) == '{"a": 1}'
    sent = client.sent
    assert sent["model"] == "claude-opus-5-5"
    assert sent["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": schema}}
    assert sent["betas"] == ["server-side-fallback-2026-07-01"] and sent["fallbacks"] == "default"
    assert sent["messages"] == [{"role": "user", "content": "hello"}] and sent["system"] == "sys"
    assert "thinking" not in sent


@pytest.mark.parametrize("stop_reason,text,match", [
    ("refusal", "x", "declined"), ("max_tokens", "x", "cut off"), ("end_turn", "  ", "no text")])
def test_call_claude_failures(stop_reason, text, match):
    with pytest.raises(RuntimeError, match=match):
        agent.call_claude(FakeClient(stop_reason, text), "s", "u")


# --- number check -------------------------------------------------------------

CONTEXT = {"before": {"turnover_rate": "30.9%", "annual_turnover_cost": "$1,165,032",
                      "continuity_score": "1.53", "family_days_without_worker": "63,810"},
           "breakeven": "6.5", "period": "2023–2025"}


def test_unknown_figures_accepts_numbers_from_the_context():
    text = ("Turnover was 30.9% (about 30.9 percent) over 2023–2025, costing $1,165,032; the score was "
            "1.53, with 63,810 family-days and 6.50 months to break even at SITE-02.")
    assert agent.unknown_figures(text, CONTEXT) == []


def test_unknown_figures_flags_every_kind_of_invented_number():
    text = "Turnover was 31% and cost $1.2M; the score fell to 1.4 over 7 months, saving 4,000 family-days."
    assert agent.unknown_figures(text, CONTEXT) == ["31%", "$1.2M", "1.4", "7", "4,000"]


def test_unknown_figures_ignores_ids_and_ordinal_words():
    assert agent.unknown_figures("SITE-07 and SYN-S-0001 ranked 3rd.", CONTEXT) == []


# --- parsing and validation ---------------------------------------------------

def validate(**overrides):
    return agent.validate_question(json.loads(question_json(**overrides)), ORG)


def test_validate_question_accepts_the_three_example_questions():
    a = validate(changes=[change("turnover_reduction", 0.10, "Caseworker", unit="relative")],
                 focus_metric="continuity_score")
    assert a["changes"][0]["value"] == 0.10
    b = validate(changes=[change("caseload", 25, "Clinician")])
    assert b["changes"][0] == change("caseload", 25.0, "Clinician")
    c = validate(roles=["Caseworker"], breakdown="site", focus_metric="breakeven_months_with_ramp")
    assert (c["roles"], c["breakdown"]) == (["Caseworker"], "site")


@pytest.mark.parametrize("overrides,match", [
    ({"site_id": "SITE-99"}, "Unknown site"),
    ({"roles": ["Nurse"]}, "Unknown roles"),
    ({"start_year": 2019}, "Years must be"),
    ({"start_year": 2025, "end_year": 2023}, "after end year"),
    ({"changes": [change("turnover_reduction", 10, "Caseworker", unit="relative")]}, "fractions"),
    ({"changes": [change("turnover_reduction", 0.1, unit="absolute")]}, "relative or points"),
    ({"changes": [change("caseload", 25, "Youth Care Worker")]}, "caseload-carrying"),
    ({"changes": [change("caseload", 12.5, "Clinician")]}, "whole number"),
    ({"changes": [change("caseload", 25, "Clinician", site="SITE-02")]}, "only turnover_reduction"),
    ({"changes": [change("vacancy_cost_basis_full_salary", 2)]}, "outside"),
    ({"changes": [change("ramp_start_productivity", 0.5, "Caseworker")]}, "applies to every role"),
    ({"changes": [change("annual_salary", 5_000_000)]}, "outside"),
    ({"changes": [change("caseload", 25, "Clinician"), change("caseload", 30, "Clinician")]}, "Two changes"),
    ({"focus_metric": "happiness"}, "Unknown metric"),
])
def test_validate_question_rejects_bad_parses(overrides, match):
    with pytest.raises(ValueError, match=match):
        validate(**overrides)


def test_one_year_fills_in_the_other():
    v = validate(end_year=2024)
    assert (v["start_year"], v["end_year"]) == (2024, 2024)


def test_merge_view_keeps_the_dashboard_view_unless_the_question_names_something():
    base = sc.View(site_id="SITE-03", roles=("Caseworker", "Clinician"))
    assert agent.merge_view(base, validate()) == base
    merged = agent.merge_view(base, validate(site_id="SITE-02", start_year=2025))
    assert merged.site_id == "SITE-02" and merged.roles == base.roles
    assert (merged.start, merged.end) == (pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01"))


def test_parse_question_sends_sites_roles_and_current_view(fake):
    f = fake(question_json())
    agent.parse_question(None, "How are we doing?", ORG, sc.View(site_id="SITE-03"))
    system = f.calls[0]["system"]
    assert "SITE-08: Kinbridge Fordham Clinical Services, Bronx, Clinic" in system
    assert "Clinician (Clinical, field)" in system and "Finance Associate (Admin, central_admin)" in system
    assert "site: Kinbridge Washington Heights Family Center" in system
    assert f.calls[0]["json_schema"]["properties"]["site_id"]["enum"][0] == ""


# --- ask end to end -----------------------------------------------------------

def test_ask_turnover_what_if_reruns_the_engine(fake):
    f = fake(question_json(changes=[change("turnover_reduction", 0.10, "Caseworker", unit="relative")],
                           focus_metric="continuity_score"),
             "The score improves from 1.53 to 1.49.")
    r = agent.ask(None, "What happens to the care-continuity score if caseworker turnover drops 10%?",
                  ORG, VIEW, settings())
    data, s = sc.apply_changes(ORG.data, settings(), VIEW, r["parsed"]["changes"])
    assert r["after"] == sc.headline(data, VIEW, s, ORG.role_order)
    assert r["after"]["continuity_score"] < r["before"]["continuity_score"]
    context = json.loads(f.calls[1]["user_prompt"])
    assert context["before"]["continuity_score"] == agent.fmt("continuity_score", r["before"]["continuity_score"])
    assert context["after"]["continuity_score"] == agent.fmt("continuity_score", r["after"]["continuity_score"])
    assert r["changes"] == ["Caseworker turnover down 10% (that share of departures avoided; those staff stay)"]
    assert r["unverified_figures"] == []


def test_ask_caseload_what_if(fake):
    fake(question_json(changes=[change("caseload", 25, "Clinician")]), "Family-days rise.")
    r = agent.ask(None, "What if we raised the clinician caseload cap to 25?", ORG, VIEW, settings())
    assert r["after"]["family_days_without_worker"] > r["before"]["family_days_without_worker"]
    assert r["after"]["turnover_rate"] == r["before"]["turnover_rate"]


def test_ask_site_breakdown_is_ranked_by_the_engine(fake):
    f = fake(question_json(roles=["Caseworker"], breakdown="site", focus_metric="breakeven_months_with_ramp"),
             "Break-even is the same at every site.")
    r = agent.ask(None, "Which site has the worst break-even tenure for caseworkers?", ORG, VIEW, settings())
    b = r["breakdown"]
    assert b["by"] == "site" and len(b["rows"]) == 4
    assert b["all_equal"] is True and b["worse_is"] == "higher"
    assert r["after"] is None
    context = json.loads(f.calls[1]["user_prompt"])
    assert context["after"] == "same as before (no changes)"
    assert context["breakdown"]["rows"][0]["site"].startswith("Kinbridge")


def test_ask_role_breakdown_sorted_highest_first(fake):
    fake(question_json(breakdown="role", focus_metric="turnover_rate"), "Youth care workers turn over most.")
    r = agent.ask(None, "Which roles turn over most?", ORG, VIEW, settings())
    assert r["breakdown"]["highest"] == "Youth Care Worker"
    assert r["breakdown"]["rows"][0]["role"] == "Youth Care Worker"


def test_ask_flags_invented_numbers(fake):
    fake(question_json(), "Turnover is about 31% and costs roughly $1.2M.")
    r = agent.ask(None, "How bad is turnover?", ORG, VIEW, settings())
    assert r["unverified_figures"] == ["31%", "$1.2M"]


def test_ask_invalid_parse_stops_before_narration(fake):
    f = fake(question_json(site_id="SITE-99"))
    with pytest.raises(ValueError, match="Unknown site"):
        agent.ask(None, "q", ORG, VIEW, settings())
    assert len(f.calls) == 1


def test_ask_empty_view_is_an_error(fake):
    fake(question_json(site_id="SITE-01", roles=["Caseworker"]))
    with pytest.raises(ValueError, match="no seats"):
        agent.ask(None, "Caseworkers at the central office?", ORG, VIEW, settings())


# --- funder report ------------------------------------------------------------

def test_funder_report_uses_only_computed_numbers(fake):
    view = sc.View(site_id="SITE-03")
    h = sc.headline(ORG.data, view, settings(), ORG.role_order)
    f = fake(f"In 2023–2025 the center served {h['children_served']:,} children; turnover was "
             f"{agent.fmt('turnover_rate', h['turnover_rate'])} and 90% stayed a year.")
    r = agent.funder_report(None, ORG, view, settings())
    context = json.loads(f.calls[0]["user_prompt"])
    assert context["view"]["site"] == "Kinbridge Washington Heights Family Center"
    assert context["results"]["children_served"] == f"{h['children_served']:,}"
    assert set(context["by_year"]) == {"2023", "2024", "2025"}
    assert "annual_turnover_cost" not in context["results"]        # no costs in a funder report
    assert r["unverified_figures"] == ["90%"]
    assert "do not mention costs" in f.calls[0]["system"]


def test_funder_report_single_year_has_no_by_year(fake):
    f = fake("Summary.")
    agent.funder_report(None, ORG, sc.View(start=pd.Timestamp("2025-01-01")), settings())
    context = json.loads(f.calls[0]["user_prompt"])
    assert "by_year" not in context and context["view"]["period"] == "2025"
