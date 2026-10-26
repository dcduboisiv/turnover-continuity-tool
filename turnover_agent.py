"""
Turnover Agent — MVP 3
AI layer on top of the MVP 1 engine (turnover_calculator.py) and the MVP 3
scenario engine (scenarios.py).

Same discipline as the EDO, RevOps and recruiting agents: the AI never does
the math. It only (a) turns a plain-English question into a structured view
and what-if changes, which plain Python validates and applies by rerunning
the same engine, and (b) writes prose around numbers that were already
computed. Every number the model writes (dollars, percentages, months,
scores, counts) is then checked against the numbers it was given
(unknown_figures), so a mistake shows up as a flagged figure, never as a
silently wrong number.

Two pieces:
- ask():           question -> parse -> validate -> rerun engine -> narrate
- funder_report(): a view's staff-stability and continuity numbers -> a
                   short grant-report paragraph

Requires an Anthropic API key (ANTHROPIC_API_KEY, or passed in by the app).

Usage:
    python turnover_agent.py --ask "What happens to the care-continuity score if caseworker turnover drops 10%?"
    python turnover_agent.py --funder-report --site SITE-03 --start-year 2025 --end-year 2025
"""

import argparse
import json
import math
import os
import re
from dataclasses import replace
from typing import Dict, List, Optional, Sequence

import pandas as pd
from anthropic import Anthropic

import scenarios as sc
from generate_sample_data import HISTORY_END, HISTORY_START, ORG_NAME, SITES_FILE
from turnover_calculator import (
    CENTRAL_ADMIN,
    FIELD,
    TYPICAL_CASELOAD,
    load_cost_assumptions,
    load_data,
    load_program_budgets,
    retention_curves,
    vacancy_report,
)

MODEL = "claude-opus-5-5"
# Parsing and short narration are small, well-specified tasks; low effort
# keeps them fast and cheap without changing models.
EFFORT = "low"
# On a safety-classifier refusal, the API re-runs the request on Anthropic's
# recommended fallback model instead of returning the refusal.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

YEARS = list(range(HISTORY_START.year, HISTORY_END.year))
UNITS = ("relative", "points", "absolute")

# metric: (label, which direction is worse: "higher", "lower" or None)
METRICS = {
    "turnover_rate": ("annualized turnover rate", "higher"),
    "vacancy_rate": ("vacancy rate", "higher"),
    "continuity_score": ("care-continuity score (1.00 = one person in every child-facing role)", "higher"),
    "avg_days_per_affected_family": ("average days an affected family went without its assigned caseworker "
                                     "or clinician", "higher"),
    "affected_families": ("families affected by a caseworker or clinician vacancy", "higher"),
    "family_days_without_worker": ("total family-days without an assigned caseworker or clinician", "higher"),
    "annual_turnover_cost": ("annual cost of turnover", "higher"),
    "avg_cost_per_departure": ("average cost per departure", "higher"),
    "monthly_turnover_cost": ("monthly cost of turnover", "higher"),
    "turnover_cost_share_of_budget": ("annual cost of turnover as a share of the program budget", "higher"),
    "open_seats_at_end": ("open seats at period end", "higher"),
    "avg_days_to_fill": ("average days to fill a seat", "higher"),
    "cost_per_departure": ("fully-loaded cost per departure", "higher"),
    "breakeven_months_with_ramp": ("break-even tenure in months, with ramp-up", "higher"),
    "breakeven_months_no_ramp": ("break-even tenure in months, no ramp-up", "higher"),
    "share_left_before_breakeven": ("share of departures who left before break-even", "higher"),
    "crossover_months": ("vacancy cost crossover in months", None),
}

LEVER_HELP = {
    sc.TURNOVER_REDUCTION: "fewer departures. unit relative: value is the fraction of departures avoided "
                           "('drops 10%' -> 0.10); unit points: value is rate points ('drops 10 points' -> 0.10). "
                           "role and site optional.",
    sc.CASELOAD: "typical caseload per seat for a caseload-carrying role (Caseworker or Clinician); "
                 "value is families per seat, unit absolute. 'Caseload cap' means this.",
    "recruiting_cost": "recruiting cost per hire. unit absolute (dollars) or relative (0.2 = +20%). role optional.",
    "training_cost": "training / onboarding cost per hire. unit absolute or relative. role optional.",
    "annual_salary": "annual salary. unit absolute or relative. role optional.",
    "coverage_premium": "overtime / temp premium over normal pay while a seat is empty (0.5 = time-and-a-half). "
                        "unit absolute or relative. role optional.",
    sc.RAMP_START: "new hire's month-1 productivity, 0.1-1.0, unit absolute.",
    sc.RAMP_MONTHS: "months to reach full productivity, 1-12, unit absolute.",
    sc.RAMP_ENABLED: "ramp-up discount on (1) or off (0), unit absolute.",
    sc.VACANCY_BASIS: "break-even basis: 1 = full salary, 0 = coverage premium, unit absolute.",
}

# Guardrails on parsed values. These catch unit mistakes ("10%" parsed as
# 10 instead of 0.10) before anything is rerun; they are not business limits.
ABSOLUTE_RANGES = {
    sc.CASELOAD: (1, 100),
    "recruiting_cost": (0, 250_000),
    "training_cost": (0, 250_000),
    "annual_salary": (15_000, 500_000),
    "coverage_premium": (0, 2),
    sc.RAMP_START: (0.1, 1.0),
    sc.RAMP_MONTHS: (1, 12),
    sc.RAMP_ENABLED: (0, 1),
    sc.VACANCY_BASIS: (0, 1),
}
RELATIVE_RANGE = (-0.9, 3.0)
TURNOVER_RANGES = {"relative": (0.0, 1.0), "points": (0.0, 0.6)}
COST_UNITS = ("absolute", "relative")


def get_client(api_key: Optional[str] = None) -> Anthropic:
    """
    api_key lets a caller (e.g. the Streamlit app, pulling from st.secrets)
    hand in a key directly. With no argument, this falls back to the
    ANTHROPIC_API_KEY environment variable.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Run: export ANTHROPIC_API_KEY=\"your-key-here\""
        )
    return Anthropic(api_key=key)


def call_claude(client: Anthropic, system: str, user_prompt: str, json_schema: Optional[dict] = None) -> str:
    """
    Thin wrapper around a single-turn Claude call. Isolated like this so tests
    can swap it out and verify the surrounding logic without hitting the real
    API. With json_schema, structured outputs guarantee the reply parses
    against it.
    """
    output_config = {"effort": EFFORT}
    if json_schema is not None:
        output_config["format"] = {"type": "json_schema", "schema": json_schema}
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=system,
        messages=[{"role": "user", "content": user_prompt}],
        output_config=output_config,
        betas=[FALLBACK_BETA],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("The model declined this request (and so did its fallback).")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("The model's reply was cut off at max_tokens.")
    text = "".join(block.text for block in response.content if block.type == "text")
    if not text.strip():
        raise RuntimeError("The model returned no text.")
    return text


def _loads(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON: {raw!r}") from e


# ---------------------------------------------------------------------------
# Number check: every number the model writes must be one it was given
# ---------------------------------------------------------------------------

# $ amounts (optionally K/M), percentages, and plain numbers. Digits inside
# identifiers (SITE-02, SYN-S-0001) and words are not matched.
_NUMBER = re.compile(r"(?<![\w.\-])(\$\s?)?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?:\s?([KkMm])\b)?(\s?%)?(?!\w)")


def _figures(text: str):
    """(as written, kind, value) for each number in text; kind is '$', '%' or ''."""
    out = []
    for m in _NUMBER.finditer(text):
        dollar, digits, suffix, pct = m.groups()
        try:
            value = float(digits.replace(",", ""))
        except ValueError:
            continue
        if suffix:
            value *= 1_000 if suffix.lower() == "k" else 1_000_000
        kind = "$" if dollar else "%" if pct else ""
        out.append((m.group(0).strip(), kind, round(value, 6)))
    return out


def allowed_figures(context) -> set:
    """(kind, value) of every number anywhere in the context sent to the model."""
    return {(kind, value) for _, kind, value in _figures(json.dumps(context, ensure_ascii=False))}


def unknown_figures(text: str, context) -> List[str]:
    """Numbers in `text` that weren't in `context`. A plain number may match a
    $ or % figure of the same value ("30.9 percent"); a $ or % figure must
    match exactly ("$1.2M" does not match "$1,165,032")."""
    allowed = allowed_figures(context)
    values = {value for _, value in allowed}
    seen, unknown = set(), []
    for written, kind, value in _figures(text):
        ok = (kind, value) in allowed or (kind == "" and value in values)
        if not ok and written not in seen:
            seen.add(written)
            unknown.append(written)
    return unknown


def fmt(metric: str, value) -> str:
    """How each metric is written everywhere the model sees it."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if isinstance(value, float) and math.isinf(value):
        return "never"
    if metric in ("turnover_rate", "voluntary_rate", "involuntary_rate", "vacancy_rate",
                  "share_left_before_breakeven", "turnover_cost_share_of_budget"):
        return f"{value:.1%}"
    if metric == "continuity_score":
        return f"{value:.2f}"
    if metric in ("annual_turnover_cost", "cost_per_departure", "avg_cost_per_departure",
                  "monthly_turnover_cost", "program_budget"):
        return f"${value:,.0f}"
    if metric in ("breakeven_months_with_ramp", "breakeven_months_no_ramp", "crossover_months"):
        return f"{value:.1f}"
    if metric in ("avg_headcount", "avg_days_per_affected_family"):
        return f"{value:,.1f}"
    return f"{value:,.0f}"


def format_metrics(h: Dict) -> Dict[str, str]:
    return {k: fmt(k, v) for k, v in h.items()}


# ---------------------------------------------------------------------------
# Shared context helpers
# ---------------------------------------------------------------------------

class Org:
    """Names and lists the AI layer needs, built once from the data."""

    def __init__(self, data: Dict[str, pd.DataFrame], sites: pd.DataFrame, assumptions: pd.DataFrame):
        self.data = data
        self.sites = sites
        self.site_ids = list(sites["site_id"])
        self.site_names = dict(zip(sites["site_id"],
                                   sites["site_name"].str.replace(r"\s*\(SYN\)$", "", regex=True)))
        self.role_order = list(assumptions.index)
        self.areas = list(dict.fromkeys(data["positions"]["program_area"]))
        self.role_area = dict(zip(data["positions"]["role"], data["positions"]["program_area"]))
        self.role_type = dict(zip(data["positions"]["role"], data["positions"]["staff_type"]))


def describe_view(view: sc.View, org: Org) -> Dict[str, str]:
    first, last = view.start.year, (view.end - pd.Timedelta(days=1)).year
    return {
        "site": org.site_names.get(view.site_id, "all sites") if view.site_id else "all sites",
        "staff_type": {FIELD: "field staff", CENTRAL_ADMIN: "central admin"}.get(view.staff_type, "all staff"),
        "program_areas": ", ".join(view.program_areas) if view.program_areas else "all",
        "roles": ", ".join(view.roles) if view.roles else "all",
        "period": str(first) if first == last else f"{first}–{last}",
    }


def describe_settings(settings: sc.Settings) -> Dict[str, object]:
    return {
        "ramp_up": ("on: " + " / ".join(f"{p:.0%}" for p in settings.schedule) + " productivity in months 1-"
                    + str(len(settings.schedule))) if settings.ramp_on else "off",
        "breakeven_basis": settings.basis.replace("_", " "),
        "typical_caseload": {role: str(n) for role, n in settings.caseload.items()},
    }


DEFINITIONS = {
    "turnover_rate": "departures / average headcount, annualized",
    "continuity_score": "distinct caregivers per child per child-facing role, weighted by role; "
                        "1.00 means one person in every role, higher is worse",
    "avg_days_per_affected_family": "for families whose caseworker or clinician seat went vacant: average days "
                                    "they went without that worker (total family-days / affected families)",
    "affected_families": "the typical caseload of every caseworker or clinician seat that had a vacancy, "
                         "counted once per seat",
    "family_days_without_worker": "days a caseworker or clinician seat sat empty x its typical caseload, summed",
    "breakeven_months": "months a new hire must stay before filling the seat pays back recruiting + training",
    "crossover_months": "months of overtime / temp coverage that cost as much as recruiting and onboarding",
    "avg_cost_per_departure": "cost of all the period's departures / number of departures",
    "turnover_cost_share_of_budget": "annual cost of turnover / annual program budget of the sites in view "
                                     "(the budget is a synthetic planning input)",
}


# ---------------------------------------------------------------------------
# Piece 1: scenario questions
# ---------------------------------------------------------------------------

def question_schema(org: Org) -> dict:
    def enum(values):
        return {"type": "string", "enum": values}

    return {
        "type": "object",
        "properties": {
            "site_id": enum([""] + org.site_ids),
            "staff_type": enum(["", FIELD, CENTRAL_ADMIN]),
            "program_areas": {"type": "array", "items": enum(org.areas)},
            "roles": {"type": "array", "items": enum(org.role_order)},
            "start_year": {"type": "integer"},
            "end_year": {"type": "integer"},
            "changes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "lever": enum(list(sc.LEVERS)),
                        "role": enum([""] + org.role_order),
                        "site": enum([""] + org.site_ids),
                        "value": {"type": "number"},
                        "unit": enum(list(UNITS)),
                    },
                    "required": ["lever", "role", "site", "value", "unit"],
                    "additionalProperties": False,
                },
            },
            "breakdown": enum(["none", "site", "role"]),
            "focus_metric": enum([""] + list(METRICS)),
            "assumption_notes": {"type": "string"},
            "unsupported": {"type": "string"},
        },
        "required": ["site_id", "staff_type", "program_areas", "roles", "start_year", "end_year", "changes",
                     "breakdown", "focus_metric", "assumption_notes", "unsupported"],
        "additionalProperties": False,
    }


QUESTION_SYSTEM_PROMPT = """You turn a question from a child welfare nonprofit's leadership \
team into a structured view and what-if changes for a workforce model. You never calculate \
results; you only describe what to look at and what to change.

The organization is {org} (fictional). Sites (site_id: name, borough, type):
{sites}

Roles (program area, staff type; field = family / child contact):
{roles}

The dashboard is currently showing: {current_view}.
View fields: set them only if the question asks for a different slice; otherwise leave them \
empty ("" / [] / 0) and the current view is kept. Years available: {years}.

changes (what-if levers; leave empty for a plain lookup). Each has lever, role ("" = all \
roles the lever applies to), site ("" = all sites; only for turnover_reduction), value, unit:
{levers}
Percentages are fractions: "10%" -> 0.10. "Drops 10%" is relative (unit relative) unless they \
say points.

breakdown: "site" for questions comparing sites ("which site ..."), "role" for comparing \
roles, else "none". focus_metric: the metric the question is about, or "" if none:
{metrics}

assumption_notes: one or two sentences on any interpretation you made.
unsupported: anything the question asks that these views and levers can't express (e.g. \
faster hiring, pay equity, a specific named person), or "" if nothing."""


def _site_lines(org: Org) -> str:
    return "\n".join(f"- {s.site_id}: {org.site_names[s.site_id]}, {s.borough}, {s.site_type}"
                     for s in org.sites.itertuples(index=False))


def parse_question(client: Anthropic, question: str, org: Org, view: sc.View) -> dict:
    """Ask the model for a structured view and changes, then validate them deterministically."""
    cv = describe_view(view, org)
    system = QUESTION_SYSTEM_PROMPT.format(
        org=ORG_NAME,
        sites=_site_lines(org),
        roles="\n".join(f"- {r} ({org.role_area[r]}, {org.role_type[r]})" for r in org.role_order),
        current_view="; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in cv.items()),
        years=", ".join(map(str, YEARS)),
        levers="\n".join(f"- {k}: {v}" for k, v in LEVER_HELP.items()),
        metrics="\n".join(f"- {k}: {label}" for k, (label, _) in METRICS.items()),
    )
    raw = call_claude(client, system, question, json_schema=question_schema(org))
    return validate_question(_loads(raw), org)


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def validate_question(parsed: dict, org: Org) -> dict:
    """
    Check the parsed question before anything is rerun. Structured outputs
    guarantee the shape; this checks the meaning: known sites, roles and
    years, sensible ranges, changes that make sense for their role.
    """
    if not isinstance(parsed, dict):
        raise ValueError(f"Parsed question is not an object: {parsed!r}")

    site_id = parsed.get("site_id") or ""
    if site_id and site_id not in org.site_ids:
        raise ValueError(f"Unknown site {site_id!r}")
    staff_type = parsed.get("staff_type") or ""
    if staff_type not in ("", FIELD, CENTRAL_ADMIN):
        raise ValueError(f"Unknown staff type {staff_type!r}")

    def checked(key, allowed):
        values = list(parsed.get(key) or [])
        bad = [v for v in values if v not in allowed]
        if bad:
            raise ValueError(f"Unknown {key.replace('_', ' ')}: {bad}")
        return values

    roles = checked("roles", org.role_order)
    areas = checked("program_areas", org.areas)

    start_year, end_year = parsed.get("start_year") or 0, parsed.get("end_year") or 0
    for y in (start_year, end_year):
        if isinstance(y, bool) or not isinstance(y, int) or (y and y not in YEARS):
            raise ValueError(f"Years must be in {YEARS[0]}–{YEARS[-1]}, got {y!r}")
    start_year, end_year = start_year or end_year, end_year or start_year
    if start_year > end_year:
        raise ValueError(f"Start year {start_year} is after end year {end_year}")

    changes, seen = [], set()
    for c in parsed.get("changes") or []:
        lever, role, site, unit = c.get("lever"), c.get("role") or "", c.get("site") or "", c.get("unit")
        if lever not in sc.LEVERS:
            raise ValueError(f"Unknown lever {lever!r}; expected one of {list(sc.LEVERS)}")
        if role and role not in org.role_order:
            raise ValueError(f"Unknown role {role!r}")
        if site and site not in org.site_ids:
            raise ValueError(f"Unknown site {site!r}")
        if site and lever != sc.TURNOVER_REDUCTION:
            raise ValueError(f"{lever} applies organization-wide; only turnover_reduction takes a site")
        value = _number(c.get("value"), lever)

        if lever == sc.TURNOVER_REDUCTION:
            if unit not in TURNOVER_RANGES:
                raise ValueError("turnover_reduction needs unit relative or points")
            low, high = TURNOVER_RANGES[unit]
            if not low < value <= high:
                raise ValueError(f"turnover_reduction = {value} is outside {low}..{high}; percentages "
                                 "are fractions (10% = 0.10)")
        elif lever in sc.COST_FIELDS:
            if unit not in COST_UNITS:
                raise ValueError(f"{lever} needs unit absolute or relative")
            low, high = RELATIVE_RANGE if unit == "relative" else ABSOLUTE_RANGES[lever]
            if not low <= value <= high:
                raise ValueError(f"{lever} = {value} ({unit}) is outside {low}..{high}")
        else:
            if unit != "absolute":
                raise ValueError(f"{lever} takes an absolute value")
            low, high = ABSOLUTE_RANGES[lever]
            if not low <= value <= high:
                raise ValueError(f"{lever} = {value} is outside {low}..{high}")
            if lever in (sc.CASELOAD, sc.RAMP_MONTHS) and value != int(value):
                raise ValueError(f"{lever} must be a whole number, got {value}")
            if lever in (sc.RAMP_ENABLED, sc.VACANCY_BASIS) and value not in (0, 1):
                raise ValueError(f"{lever} must be 0 or 1, got {value}")
            if lever == sc.CASELOAD and role not in TYPICAL_CASELOAD:
                raise ValueError(f"Only caseload-carrying roles have a caseload ({', '.join(TYPICAL_CASELOAD)}); "
                                 f"got {role or 'no role'!r}")
            if lever != sc.CASELOAD and role:
                raise ValueError(f"{lever} applies to every role")
        key = (lever, role, site)
        if key in seen:
            raise ValueError(f"Two changes to {lever} for the same role and site; combine them into one")
        seen.add(key)
        changes.append({"lever": lever, "role": role, "site": site, "value": value, "unit": unit})

    breakdown = parsed.get("breakdown") or "none"
    if breakdown not in ("none", "site", "role"):
        raise ValueError(f"Unknown breakdown {breakdown!r}")
    focus = parsed.get("focus_metric") or ""
    if focus and focus not in METRICS:
        raise ValueError(f"Unknown metric {focus!r}")

    return {
        "site_id": site_id, "staff_type": staff_type, "program_areas": areas, "roles": roles,
        "start_year": start_year, "end_year": end_year, "changes": changes,
        "breakdown": breakdown, "focus_metric": focus,
        "assumption_notes": str(parsed.get("assumption_notes") or "").strip(),
        "unsupported": str(parsed.get("unsupported") or "").strip(),
    }


# --- applying it: deterministic, no AI past this point ----------------------

def merge_view(view: sc.View, parsed: dict) -> sc.View:
    """The question's view: the dashboard's view with anything the question named replaced."""
    out = view
    if parsed["site_id"]:
        out = replace(out, site_id=parsed["site_id"])
    if parsed["staff_type"]:
        out = replace(out, staff_type=parsed["staff_type"])
    if parsed["program_areas"]:
        out = replace(out, program_areas=tuple(parsed["program_areas"]))
    if parsed["roles"]:
        out = replace(out, roles=tuple(parsed["roles"]))
    if parsed["start_year"]:
        out = replace(out, start=pd.Timestamp(f"{parsed['start_year']}-01-01"),
                      end=pd.Timestamp(f"{parsed['end_year'] + 1}-01-01"))
    return out


def breakdown_rows(data, view: sc.View, settings: sc.Settings, org: Org, by: str, focus: str) -> dict:
    """The per-site or per-role table, ranked on the focus metric by the engine (not the model)."""
    if by == "site":
        table = sc.site_table(data, view, settings, org.role_order, org.site_ids)
        names = {k: org.site_names[k] for k in table.index}
    else:
        table = sc.role_table(data, view, settings, org.role_order)
        names = {k: k for k in table.index}
    if table.empty:
        return {"by": by, "rows": []}
    metric = focus if focus in table.columns else None
    if metric:
        table = table.sort_values(metric, ascending=False, kind="mergesort", na_position="last")
    rows = [{by: names[idx], **format_metrics(r.dropna().to_dict())} for idx, r in table.iterrows()]
    out = {"by": by, "rows": rows}
    if metric:
        col = table[metric].dropna()
        label, worse = METRICS[metric]
        out.update({
            "ranked_by": label,
            "sorted": "highest first",
            "worse_is": worse or "neither (context-dependent)",
            "all_equal": bool(len(col) > 1 and col.max() == col.min()),
        })
        if len(col):
            out["highest"] = names[col.idxmax()]
            out["lowest"] = names[col.idxmin()]
    return out


NARRATE_SYSTEM_PROMPT = """You answer a question from a child welfare nonprofit's leadership \
team using results that were already computed for you. Write 2-4 sentences of plain \
language, no bullet points, no headings. Lead with the direct answer.

If what-if changes were modeled, say what moved between "before" and "after". If there is a \
breakdown, use its ranking (highest / lowest, worse_is) rather than comparing numbers \
yourself; if all_equal is true, say the values are the same everywhere and point to a \
metric in the rows that does differ, if one does. For vacancy gaps, lead with affected \
families and the average days per affected family; the family-days total is a detail.

Use only numbers that appear in the data, written exactly as they appear there. Do not \
calculate, round, add up, subtract or estimate anything yourself, so do not state a \
difference between two numbers unless the difference itself is in the data. The \
organization is fictional and its data synthetic. If something in the question couldn't \
be modeled, say what was left out."""


def explain(client: Anthropic, context: dict) -> str:
    return call_claude(client, NARRATE_SYSTEM_PROMPT, json.dumps(context, ensure_ascii=False)).strip()


def ask(client: Anthropic, question: str, org: Org, view: sc.View, settings: sc.Settings) -> dict:
    """Parse -> validate -> rerun the engine (before and after) -> narrate -> check the numbers."""
    parsed = parse_question(client, question, org, view)
    q_view = merge_view(view, parsed)
    before = sc.headline(org.data, q_view, settings, org.role_order)
    if not before:
        raise ValueError("no seats match that view")
    data_after, settings_after = sc.apply_changes(org.data, settings, q_view, parsed["changes"])
    after = sc.headline(data_after, q_view, settings_after, org.role_order) if parsed["changes"] else None
    breakdown = None
    if parsed["breakdown"] != "none":
        breakdown = breakdown_rows(data_after, q_view, settings_after, org, parsed["breakdown"],
                                   parsed["focus_metric"])

    changes = [sc.describe_change(c, org.site_names) for c in parsed["changes"]]
    context = {
        "question": question,
        "organization": f"{ORG_NAME} (fictional)",
        "view": describe_view(q_view, org),
        "settings": describe_settings(settings_after),
        "changes_modeled": changes,
        "assumption_notes": parsed["assumption_notes"],
        "not_modeled": parsed["unsupported"],
        "definitions": DEFINITIONS,
        "focus_metric": METRICS[parsed["focus_metric"]][0] if parsed["focus_metric"] else "",
        "before": format_metrics(before),
        "after": format_metrics(after) if after else "same as before (no changes)",
    }
    if breakdown:
        context["breakdown" + (" (after changes)" if parsed["changes"] else "")] = breakdown
    explanation = explain(client, context)
    return {
        "parsed": parsed,
        "view": q_view,
        "changes": changes,
        "before": before,
        "after": after,
        "breakdown": breakdown,
        "explanation": explanation,
        "unverified_figures": unknown_figures(explanation, context),
    }


# ---------------------------------------------------------------------------
# Piece 2: funder-report narrative
# ---------------------------------------------------------------------------

FUNDER_SYSTEM_PROMPT = """You write the staff stability and care continuity paragraph of a \
grant report for a child welfare nonprofit, from numbers that were already computed for \
you. Write 120-180 words of plain, factual prose for a program officer, in one or two \
paragraphs, past tense, no bullet points, no headings.

Cover, in this order: how many children were served, how stable the workforce was \
(turnover, vacancies, early retention), and what that meant for children's continuity of \
care: the continuity score, and how many families went without an assigned caseworker or \
clinician and for how many days on average (lead with those two; the family-days total is \
a secondary detail). Be candid about \
challenges as well as strengths. Do not promise plans, programs or results that are not in \
the data, and do not mention costs.

Use only numbers that appear in the data, written exactly as they appear there. Do not \
calculate, round, add up or estimate anything yourself. Refer to the organization by \
name and the site or sites named in the data."""


def funder_context(org: Org, view: sc.View, settings: sc.Settings) -> dict:
    h = sc.headline(org.data, view, settings, org.role_order)
    if not h:
        raise ValueError("no seats match that view")
    f = sc.filter_view(org.data, view, org.role_order)
    hires = f["staff"][(f["staff"]["hire_date"] >= view.start) & (f["staff"]["hire_date"] < view.end)]
    retained = (retention_curves(hires, view.end, months=(12,), by=None).loc[("All", 12)]
                if len(hires) else {"eligible": 0, "retention": None})
    fill = vacancy_report(f["staff"], f["positions"], view.start, view.end, by="role")
    keep = ["children_served", "turnover_rate", "voluntary_rate", "involuntary_rate", "departures",
            "avg_headcount", "vacancy_rate", "seats", "open_seats_at_end", "continuity_score",
            "affected_families", "avg_days_per_affected_family", "family_days_without_worker",
            "caseload_vacancies"]
    context = {
        "organization": f"{ORG_NAME} (fictional)",
        "view": describe_view(view, org),
        "definitions": {k: v for k, v in DEFINITIONS.items() if k not in ("breakeven_months", "crossover_months")},
        "results": {k: fmt(k, h[k]) for k in keep if h.get(k) is not None},
        "new_hires_measured_at_12_months": f"{int(retained['eligible']):,}",
        "new_hires_still_employed_at_12_months": fmt("turnover_rate", retained["retention"]),
        "avg_days_to_fill_by_role": {r: f"{d:.0f}" for r, d in fill["avg_days_to_fill"].dropna().items()},
    }
    first, last = view.start.year, (view.end - pd.Timedelta(days=1)).year
    if last > first:
        context["by_year"] = {
            str(y): format_metrics({k: v for k, v in sc.headline(
                org.data, replace(view, start=pd.Timestamp(f"{y}-01-01"), end=pd.Timestamp(f"{y + 1}-01-01")),
                settings, org.role_order).items() if k in ("turnover_rate", "vacancy_rate", "continuity_score")})
            for y in range(first, last + 1)}
    return context


def funder_report(client: Anthropic, org: Org, view: sc.View, settings: sc.Settings) -> dict:
    context = funder_context(org, view, settings)
    text = call_claude(client, FUNDER_SYSTEM_PROMPT, json.dumps(context, ensure_ascii=False)).strip()
    return {"text": text, "context": context, "unverified_figures": unknown_figures(text, context)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_org(data_dir: str = ".") -> Org:
    return Org(load_data(data_dir), pd.read_csv(os.path.join(data_dir, SITES_FILE)),
               load_cost_assumptions(os.path.join(data_dir, "role_cost_assumptions.csv")))


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Turnover Agent — AI questions and funder summaries over the calculator.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ask", help="A plain-English question.")
    group.add_argument("--funder-report", action="store_true", help="Draft a funder-report paragraph.")
    parser.add_argument("--site", default=None, help="site_id, e.g. SITE-03 (default: all sites)")
    parser.add_argument("--start-year", type=int, default=YEARS[0])
    parser.add_argument("--end-year", type=int, default=YEARS[-1])
    args = parser.parse_args(argv)

    org = load_org()
    view = sc.View(site_id=args.site, start=pd.Timestamp(f"{args.start_year}-01-01"),
                   end=pd.Timestamp(f"{args.end_year + 1}-01-01"))
    settings = sc.Settings(load_cost_assumptions(), budgets=load_program_budgets().to_dict())
    client = get_client()
    if args.ask:
        result = ask(client, args.ask, org, view, settings)
        print(result["explanation"])
        for change in result["changes"]:
            print(f"  what-if: {change}")
        figures = result["unverified_figures"]
    else:
        result = funder_report(client, org, view, settings)
        print(result["text"])
        figures = result["unverified_figures"]
    if figures:
        print(f"\nCheck these figures; they weren't in the computed data: {', '.join(figures)}")


if __name__ == "__main__":
    main()
