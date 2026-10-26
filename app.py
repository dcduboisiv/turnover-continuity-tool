"""
Turnover & Care Continuity dashboard — MVP 3
Streamlit front end over the MVP 1 engine, with (MVP 3) an AI assistant
for plain-English scenario questions and funder-report summaries at the
top. Every number on the page comes from turnover_calculator.py (through
scenarios.py for views and what-ifs); the app only filters the data,
passes the sidebar settings through, and displays the results. The AI
features go through turnover_agent.py, which parses and narrates but
never calculates.

The organization (Kinbridge Family Services), its sites, staff and
children are all fictional / synthetic.

Run locally:
    streamlit run app.py

Deploy: push to GitHub, connect the repo at share.streamlit.io, main file
app.py. The AI features need ANTHROPIC_API_KEY in the app's Settings >
Secrets (see .streamlit/secrets.toml.example); everything else works
without it.
"""

import os
from typing import Optional

import altair as alt
import anthropic
import pandas as pd
import streamlit as st

import scenarios as sc
import turnover_agent as agent_ai
from generate_sample_data import HISTORY_END, HISTORY_START, ORG_NAME, SITES_FILE
from turnover_calculator import (
    CENTRAL_ADMIN,
    COST_ASSUMPTIONS_FILE,
    FIELD,
    RAMP_MONTHS_TO_FULL,
    RAMP_START_PRODUCTIVITY,
    RETENTION_MONTHS,
    ROLE_CRITICALITY,
    TYPICAL_CASELOAD,
    VACANCY_COST_BASES,
    VACANCY_COST_BASIS,
    annual_turnover_cost,
    breakeven_tenure,
    care_continuity_score,
    family_days_without_assigned_worker,
    cost_per_departure,
    linear_ramp,
    load_cost_assumptions,
    load_data,
    open_seats,
    retention_curves,
    turnover_rates,
    vacancy_cost_crossover,
    vacancy_report,
)

st.set_page_config(page_title="Turnover & Care Continuity", page_icon="🧩", layout="wide")

# Same look as the EDO and recruiting apps: base colors in
# .streamlit/config.toml, Carlito and the teal accent injected here.
TEAL = "#0F766E"
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Carlito:ital,wght@0,400;0,700;1,400;1,700&display=swap');

.stApp, .stApp *:not([data-testid="stIconMaterial"]):not(code):not(pre) {{
    font-family: 'Carlito', Calibri, 'Segoe UI', sans-serif !important;
}}
.stApp h2, .stApp h3, [data-testid="stSidebar"] h2 {{
    color: {TEAL};
}}
</style>
""",
    unsafe_allow_html=True,
)

# Chart colors: categorical slots in fixed order (validated default palette).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
TEXT_SECONDARY = "#52514e"

ALL_SITES = "__all__"
STAFF_TYPE_OPTIONS = {"All staff": None, "Field": FIELD, "Central admin": CENTRAL_ADMIN}
YEARS = list(range(HISTORY_START.year, HISTORY_END.year))
BASIS_LABELS = {"coverage_premium": "Coverage premium (overtime / temp above normal pay)",
                "full_salary": "Full salary (loaded monthly cost)"}
EDITABLE_COSTS = {
    "recruiting_cost": "Recruiting $",
    "training_cost": "Training $",
    "coverage_premium": "OT / temp premium",
    "annual_salary": "Salary $",
}


@st.cache_data
def load():
    data = load_data(".")
    sites = pd.read_csv(SITES_FILE)
    return data, sites, load_cost_assumptions(COST_ASSUMPTIONS_FILE)


data, sites, default_costs = load()
positions_all = data["positions"]
site_names = dict(zip(sites["site_id"], sites["site_name"].str.replace(" (SYN)", "", regex=False)))
site_names_short = {k: v.replace("Kinbridge ", "") for k, v in site_names.items()}
role_order = list(default_costs.index)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("View")
    site = st.selectbox("Site", [ALL_SITES] + list(sites["site_id"]),
                        format_func=lambda s: "All sites" if s == ALL_SITES else site_names_short[s],
                        key="site")
    areas_all = list(dict.fromkeys(positions_all["program_area"]))
    areas = st.multiselect("Program area", areas_all, default=areas_all, key="areas")
    staff_type = STAFF_TYPE_OPTIONS[st.radio("Staff type", list(STAFF_TYPE_OPTIONS), horizontal=True,
                                             key="staff_type")]

    pos_scope = positions_all[positions_all["program_area"].isin(areas)]
    if site != ALL_SITES:
        pos_scope = pos_scope[pos_scope["site_id"] == site]
    if staff_type:
        pos_scope = pos_scope[pos_scope["staff_type"] == staff_type]
    role_options = [r for r in role_order if r in set(pos_scope["role"])]
    # no key: the selection resets to "all" whenever the available roles change
    roles = st.multiselect("Role", role_options, default=role_options)

    first_year, last_year = st.select_slider("Period", options=YEARS, value=(YEARS[0], YEARS[-1]),
                                             key="period")
    start, end = pd.Timestamp(f"{first_year}-01-01"), pd.Timestamp(f"{last_year + 1}-01-01")

    st.header("Assumptions")
    with st.expander("Cost per role", expanded=False):
        st.caption(f"Starting values from {COST_ASSUMPTIONS_FILE}. "
                   "OT / temp premium is the share of loaded pay paid above normal to cover an empty seat.")
        edited = st.data_editor(
            default_costs[list(EDITABLE_COSTS)].reset_index(),
            hide_index=True, disabled=["role"], key="costs",
            column_config={
                "role": st.column_config.TextColumn("Role"),
                "recruiting_cost": st.column_config.NumberColumn("Recruiting $", min_value=0, step=100, format="dollar"),
                "training_cost": st.column_config.NumberColumn("Training $", min_value=0, step=100, format="dollar"),
                "coverage_premium": st.column_config.NumberColumn("OT / temp premium", min_value=0.0, max_value=2.0,
                                                                  step=0.05, format="%.2f"),
                "annual_salary": st.column_config.NumberColumn("Salary $", min_value=0, step=1000, format="dollar"),
            },
        )
    ramp_on = st.toggle("Apply ramp-up discount", value=True, key="ramp_on")
    ramp_start = st.slider("Month-1 productivity (RAMP_START_PRODUCTIVITY)", 0.10, 1.00,
                           RAMP_START_PRODUCTIVITY, 0.05, key="ramp_start")
    ramp_months = st.slider("Months to full productivity (RAMP_MONTHS_TO_FULL)", 1, 12,
                            RAMP_MONTHS_TO_FULL, key="ramp_months")
    basis = st.radio("Vacancy cost basis (break-even)", VACANCY_COST_BASES,
                     index=VACANCY_COST_BASES.index(VACANCY_COST_BASIS),
                     format_func=BASIS_LABELS.get, key="basis")
    st.caption("Typical caseload per seat")
    caseload = {role: st.number_input(role, min_value=1, max_value=100, value=n, step=1, key=f"caseload_{role}")
                for role, n in TYPICAL_CASELOAD.items()}

    st.info(f"**{ORG_NAME}** is fictional, and every person in this data is synthetic.")

schedule = linear_ramp(ramp_start, ramp_months)
applied_schedule = schedule if ramp_on else None
ramp_label = " / ".join(f"{p:.0%}" for p in schedule)

assumptions = default_costs.copy()
for col in EDITABLE_COSTS:
    assumptions[col] = edited.set_index("role")[col].reindex(assumptions.index).fillna(default_costs[col])

# ---------------------------------------------------------------------------
# Filtered data (scenarios.filter_view: the same slice the AI layer uses)
# ---------------------------------------------------------------------------

view = sc.View(site_id=None if site == ALL_SITES else site, program_areas=tuple(areas),
               staff_type=staff_type, roles=tuple(roles), start=start, end=end)
settings = sc.Settings(assumptions=assumptions, ramp_on=ramp_on, ramp_start=ramp_start,
                       ramp_months=ramp_months, basis=basis, caseload=caseload)
scoped = sc.filter_view(data, view, role_order)
staff, positions, assignments = scoped["staff"], scoped["positions"], scoped["assignments"]
weights = {r: w for r, w in ROLE_CRITICALITY.items() if r in roles}
view_roles = scoped["roles"]
period_label = str(first_year) if first_year == last_year else f"{first_year}–{last_year}"

st.title("Turnover & Care Continuity")
st.caption(f"{ORG_NAME} (fictional) · {site_names_short.get(site, 'All sites')} · {period_label} · "
           "synthetic sample data")

# ---------------------------------------------------------------------------
# AI assistant (turnover_agent: Claude parses the question, the scenario
# engine reruns the calculator, Claude narrates the computed result). First
# on the page, above the headline metrics.
# ---------------------------------------------------------------------------


def anthropic_api_key() -> Optional[str]:
    """Streamlit secrets first (Community Cloud, or a local .streamlit/secrets.toml), then the environment."""
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        key = None  # no secrets.toml at all (e.g. first local run): fall through to the env var
    return key or os.environ.get("ANTHROPIC_API_KEY")


def ai_call(fn, *args):
    """Runs an AI call and turns its failures into page messages. Returns None on failure."""
    try:
        return fn(agent_ai.get_client(api_key=api_key), *args)
    except ValueError as e:
        st.warning(f"Couldn't answer that: {e}")
    except anthropic.AuthenticationError:
        st.error("The AI service rejected the API key. Check ANTHROPIC_API_KEY in your secrets.")
    except anthropic.RateLimitError:
        st.error("The AI service is rate-limiting requests. Wait a moment and try again.")
    except (anthropic.APIError, RuntimeError) as e:
        st.error(f"The AI request failed: {e}")
    return None


def flag_figures(figures):
    if figures:
        st.warning(
            "Check these figures before using this text; they don't match any number the "
            f"calculator produced: {', '.join(figures)}"
        )


@st.cache_resource
def get_org():
    return agent_ai.Org(data, sites, default_costs)


def settings_key(s: sc.Settings):
    return (s.ramp_on, s.ramp_start, s.ramp_months, s.basis, tuple(sorted(s.caseload.items())),
            s.assumptions[list(EDITABLE_COSTS)].to_json())


AI_DELTAS = {  # metric: label; lower is better for all of them
    "turnover_rate": "Turnover rate",
    "vacancy_rate": "Vacancy rate",
    "continuity_score": "Care-continuity score",
    "avg_days_per_affected_family": "Days without a worker, per affected family",
    "annual_turnover_cost": "Cost of turnover / yr",
}


def delta_text(metric: str, after: float, before: float) -> str:
    diff = after - before
    if metric in ("turnover_rate", "vacancy_rate"):
        return f"{diff * 100:+.1f} pts"
    if metric == "continuity_score":
        return f"{diff:+.2f}"
    if metric == "avg_days_per_affected_family":
        return f"{diff:+,.1f} days"
    return f"{diff:+,.0f}"

api_key = anthropic_api_key()
org = get_org()
st.subheader("Ask about turnover and continuity")
if not api_key:
    st.info(
        "**The AI assistant needs an Anthropic API key.** The dashboard below works without one.\n\n"
        "- **Locally:** copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in the key.\n"
        "- **Streamlit Community Cloud:** add `ANTHROPIC_API_KEY` under the app's Settings → Secrets."
    )
else:
    with st.form("ask"):
        question = st.text_area(
            "Question", label_visibility="collapsed", height=80,
            placeholder="e.g. What happens to the care-continuity score if caseworker turnover drops 10%? "
                        "· What if we raised the clinician caseload cap to 25? "
                        "· Which site has the worst break-even tenure for caseworkers?",
        )
        asked = st.form_submit_button("Ask", type="primary")
    if asked and question.strip():
        with st.spinner("Reading the question, rerunning the calculator, and writing the answer…"):
            result = ai_call(agent_ai.ask, question.strip(), org, view, settings)
        if result is not None:
            st.session_state["ai_answer"] = {"question": question.strip(), "result": result}

    answer = st.session_state.get("ai_answer")
    if answer:
        result = answer["result"]
        parsed = result["parsed"]
        st.markdown(f"**Q:** {answer['question']}")
        st.info(result["explanation"])
        flag_figures(result["unverified_figures"])
        answered = agent_ai.describe_view(result["view"], org)
        st.caption("Answered for: " + " · ".join(f"{k.replace('_', ' ')} {v}" for k, v in answered.items()))
        if result["changes"]:
            st.caption("What-if: " + "; ".join(result["changes"]) + ".")
        if parsed["assumption_notes"]:
            st.caption(f"Assumption: {parsed['assumption_notes']}")
        if parsed["unsupported"]:
            st.caption(f"Not modeled: {parsed['unsupported']}")
        if result["after"]:
            st.caption("What-if result, after vs. before:")
            cols = st.columns(len(AI_DELTAS))
            for col, (metric, label) in zip(cols, AI_DELTAS.items()):
                b, a = result["before"].get(metric), result["after"].get(metric)
                if a is None or b is None:
                    col.metric(label, "—")
                    continue
                col.metric(label, agent_ai.fmt(metric, a), delta_text(metric, a, b), delta_color="inverse",
                           help=f"Before: {agent_ai.fmt(metric, b)}")
        if result["breakdown"] and result["breakdown"]["rows"]:
            st.dataframe(pd.DataFrame(result["breakdown"]["rows"]), hide_index=True, width="stretch")

    with st.expander(f"📝 Funder-report summary · {site_names_short.get(site, 'All sites')} · {period_label}"):
        st.caption("A short staff-stability and continuity paragraph for a grant report, written from this "
                   "view's computed numbers (site, staff and period from the sidebar).")
        report_key = (view, settings_key(settings))
        reports = st.session_state.setdefault("funder_reports", {})
        if st.button("Draft funder summary", key="funder_report"):
            with st.spinner("Pulling the numbers and drafting the summary…"):
                drafted = ai_call(agent_ai.funder_report, org, view, settings)
            if drafted is not None:
                reports[report_key] = drafted
        drafted = reports.get(report_key)
        if drafted:
            st.text_area("Funder summary — edit before using", drafted["text"], height=220,
                         key=f"funder_text_{hash(report_key)}")
            flag_figures(drafted["unverified_figures"])
            st.caption("Written from synthetic sample data for a fictional organization.")

st.divider()

if positions.empty:
    st.info("No seats match these filters. Widen the site, program area, staff type or role selection.")
    st.stop()

# ---------------------------------------------------------------------------
# Headline metrics
# ---------------------------------------------------------------------------


def fmt_pct(x: float) -> str:
    return "—" if pd.isna(x) else f"{x:.1%}"


overall = turnover_rates(staff, start, end, by=None).loc["All"]
vacancy = vacancy_report(staff, positions, start, end, by=None).loc["All"]
by_role = turnover_rates(staff, start, end, by="role").reindex(view_roles)
vacancy_by_role = vacancy_report(staff, positions, start, end, by="role").reindex(view_roles)
costs_view = assumptions.loc[view_roles]
costs_applied = cost_per_departure(costs_view, vacancy_by_role["avg_days_to_fill"], applied_schedule)
annual_cost = annual_turnover_cost(by_role.dropna(subset=["turnover_rate"]), costs_applied)["annual_cost"].sum()
continuity = care_continuity_score(assignments, start, end, weights) if weights and not assignments.empty else None
gap_total = family_days_without_assigned_worker(staff, positions, start, end, caseload, by=None)
gap_total = gap_total.loc["All"] if not gap_total.empty else None

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Turnover rate", fmt_pct(overall["turnover_rate"]),
          help="Separations / average headcount, annualized.")
c1.caption(f"{fmt_pct(overall['voluntary_rate'])} voluntary · {fmt_pct(overall['involuntary_rate'])} involuntary "
           f"· {int(overall['separations'])} departures")
c2.metric("Vacancy rate", fmt_pct(vacancy["vacancy_rate"]), help="Vacant seat-days / total seat-days.")
c2.caption(f"{int(vacancy['open_at_end'])} of {int(vacancy['positions'])} seats open at period end")
c3.metric("Care-continuity score", "—" if continuity is None or pd.isna(continuity) else f"{continuity:.2f}",
          help="Distinct caregivers per child, weighted by role criticality. 1.00 = one person in every role.")
c3.caption("1.00 = no caregiver changes" if continuity is not None
           else "No child-facing roles in this view" if not weights
           else "No children served in this view")
c4.metric("Cost of turnover / yr", f"${annual_cost:,.0f}",
          help="Departures per year x fully-loaded cost per departure, for the roles in view.")
c4.caption("with ramp-up discount" if ramp_on else "ramp-up discount off")
GAP_LABEL = "Days without a worker, per affected family"
GAP_HELP = ("For families whose caseworker or clinician seat went vacant: the average number of days they "
            "went without that worker. Affected families = the typical caseload of each seat that had a "
            "gap, counted once per seat.")
if gap_total is None:
    c5.metric(GAP_LABEL, "—", help=GAP_HELP)
    c5.caption("No caseworker or clinician vacancies in this view")
else:
    c5.metric(GAP_LABEL, f"{gap_total['avg_days_per_affected_family']:.1f} days", help=GAP_HELP)
    c5.caption(f"{gap_total['affected_families']:,.0f} families affected · "
               f"{gap_total['family_days']:,.0f} family-days in all")

# ---------------------------------------------------------------------------
# Cost of turnover and break-even tenure
# ---------------------------------------------------------------------------

st.subheader("Cost of turnover and break-even tenure by role")
st.caption(f"Ramp-up: {ramp_label} productivity in months 1–{len(schedule)}. "
           f"Break-even basis: {BASIS_LABELS[basis].lower()}. "
           f"Days-to-fill is the observed average where seats were backfilled in the period.")

costs_ramp = cost_per_departure(costs_view, vacancy_by_role["avg_days_to_fill"], schedule)
costs_flat = cost_per_departure(costs_view, vacancy_by_role["avg_days_to_fill"], None)
be_ramp = breakeven_tenure(costs_view, schedule, basis=basis)
be_applied = breakeven_tenure(costs_view, applied_schedule, staff=staff, start=start, end=end, basis=basis)
cost_table = pd.DataFrame({
    "Departures": be_applied["departures"],
    "Cost / departure (no ramp)": costs_flat["cost_per_departure"],
    "Cost / departure (with ramp)": costs_ramp["cost_per_departure"],
    "Break-even months (no ramp)": be_ramp["breakeven_months_no_ramp"],
    "Break-even months (with ramp)": be_ramp["breakeven_months"],
    "Left before break-even": be_applied["share_left_before_breakeven"] * 100,
}).rename_axis("Role").reset_index()

left, right = st.columns([3, 2])
dollars = {"format": "dollar", "step": 1}
left.dataframe(cost_table, hide_index=True, width="stretch", column_config={
    "Cost / departure (no ramp)": st.column_config.NumberColumn(**dollars),
    "Cost / departure (with ramp)": st.column_config.NumberColumn(**dollars),
    "Break-even months (no ramp)": st.column_config.NumberColumn(format="%.1f"),
    "Break-even months (with ramp)": st.column_config.NumberColumn(format="%.1f"),
    "Left before break-even": st.column_config.NumberColumn(
        format="%.0f%%", help="Share of the period's departures who left before break-even"
                              + (" (with ramp)." if ramp_on else " (no ramp).")),
})

be_long = cost_table.melt(id_vars="Role", value_vars=["Break-even months (no ramp)", "Break-even months (with ramp)"],
                          var_name="Series", value_name="Months")
be_long["Series"] = be_long["Series"].str.extract(r"\((.*)\)")[0].str.capitalize()
right.altair_chart(
    alt.Chart(be_long).mark_bar(cornerRadiusEnd=4, height=10).encode(
        y=alt.Y("Role:N", sort=view_roles, title=None),
        yOffset=alt.YOffset("Series:N", sort=["No ramp", "With ramp"]),
        x=alt.X("Months:Q", title="Break-even tenure (months)"),
        color=alt.Color("Series:N", sort=["No ramp", "With ramp"],
                        scale=alt.Scale(range=SERIES[:2]), legend=alt.Legend(orient="top", title=None)),
        tooltip=["Role", "Series", alt.Tooltip("Months:Q", format=".1f")],
    ).properties(height=max(160, 44 * len(view_roles))),
    use_container_width=True,
)

st.subheader("Vacancy cost crossover by role")
st.caption("Months of overtime / temp coverage that cost as much as recruiting and onboarding a replacement. "
           "A seat covered for longer than this has cost more than hiring.")
crossover = vacancy_cost_crossover(costs_view)
crossover_table = pd.DataFrame({
    "Recruiting + onboarding": crossover["recruiting_and_onboarding_cost"],
    "Coverage premium / month": crossover["monthly_coverage_premium"],
    "Crossover months": crossover["crossover_months"],
    "Avg days to fill": vacancy_by_role["avg_days_to_fill"],
}).rename_axis("Role").reset_index()
st.dataframe(crossover_table, hide_index=True, width="stretch", column_config={
    "Recruiting + onboarding": st.column_config.NumberColumn(**dollars),
    "Coverage premium / month": st.column_config.NumberColumn(**dollars),
    "Crossover months": st.column_config.NumberColumn(format="%.1f"),
    "Avg days to fill": st.column_config.NumberColumn(format="%.0f",
                                                      help="Observed, in this period. Blank: no backfills."),
})

# ---------------------------------------------------------------------------
# Continuity gap: family-days without an assigned worker
# ---------------------------------------------------------------------------

st.subheader("Continuity gap: days without an assigned caseworker or clinician")
st.caption("When a caseworker or clinician seat sits empty, the families on its caseload go without their "
           "worker of record. This counts how long, not how often caregivers change (that's the continuity "
           "score). Affected families = each vacant seat's typical caseload, counted once per seat.")
if gap_total is None:
    st.info("No caseload-carrying seats (Caseworker, Clinician) were vacant in this view."
            if set(caseload) & set(roles) else
            "No caseload-carrying roles in this view. Direct care is shift-based and central admin "
            "has no family contact, so neither produces family-days.")
else:
    g1, g2 = st.columns([1, 3])
    g1.metric(GAP_LABEL, f"{gap_total['avg_days_per_affected_family']:.1f} days", help=GAP_HELP)
    g1.metric("Families affected", f"{gap_total['affected_families']:,.0f}")
    g1.caption(f"{gap_total['affected_families']:,.0f} families experienced a gap, averaging "
               f"{gap_total['avg_days_per_affected_family']:.1f} days each: "
               f"{gap_total['family_days']:,.0f} family-days in all, across {int(gap_total['vacancies'])} "
               f"vacancies ({int(gap_total['still_open'])} still open at period end).")
    by_site = family_days_without_assigned_worker(staff, positions, start, end, caseload, by="site_id")
    by_site = by_site.assign(Site=by_site.index.map(site_names_short)).reset_index(drop=True)
    g2.altair_chart(
        alt.Chart(by_site).mark_bar(cornerRadiusEnd=4, height=14, color=SERIES[0]).encode(
            y=alt.Y("Site:N", sort="-x", title=None),
            x=alt.X("avg_days_per_affected_family:Q", title="Days without a worker, per affected family"),
            tooltip=["Site",
                     alt.Tooltip("avg_days_per_affected_family:Q", title="Days per affected family", format=".1f"),
                     alt.Tooltip("affected_families:Q", title="Families affected", format=",.0f"),
                     alt.Tooltip("family_days:Q", title="Family-days in all", format=",.0f")],
        ).properties(height=max(120, 34 * len(by_site))),
        use_container_width=True,
    )
    with st.expander("By site and role"):
        detail = family_days_without_assigned_worker(staff, positions, start, end, caseload,
                                                     by=["site_id", "role"]).reset_index()
        st.dataframe(pd.DataFrame({
            "Site": detail["site_id"].map(site_names_short),
            "Role": detail["role"],
            "Families affected": detail["affected_families"],
            "Days per affected family": detail["avg_days_per_affected_family"],
            "Vacancies": detail["vacancies"],
            "Vacant seat-days": detail["vacant_days"],
            "Family-days": detail["family_days"],
        }), hide_index=True, width="stretch", column_config={
            "Days per affected family": st.column_config.NumberColumn(format="%.1f"),
            "Family-days": st.column_config.NumberColumn(format="localized"),
        })

# ---------------------------------------------------------------------------
# Retention cohorts
# ---------------------------------------------------------------------------

st.subheader("Retention by hire cohort")
curves = retention_curves(staff, end, RETENTION_MONTHS, by="hire_year", hired_from=start).reset_index()
curves = curves.dropna(subset=["retention"])
if curves.empty:
    st.info("No hires in this view and period are old enough to measure retention yet.")
else:
    curves["Cohort"] = "Hired " + curves["hire_year"].astype(str)
    cohorts = [f"Hired {y}" for y in YEARS]   # fixed slot per cohort, whatever the filter
    st.caption("Share of each hire cohort still employed N months after hire. A cohort's line stops where "
               "its hires aren't yet old enough to measure.")
    base = alt.Chart(curves).encode(
        x=alt.X("months:Q", title="Months since hire", scale=alt.Scale(domain=[0, max(RETENTION_MONTHS)]),
                axis=alt.Axis(values=list(RETENTION_MONTHS))),
        y=alt.Y("retention:Q", title="Still employed", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        color=alt.Color("Cohort:N", sort=cohorts, scale=alt.Scale(domain=cohorts, range=SERIES[:len(cohorts)]),
                        legend=alt.Legend(orient="top", title=None,
                                          values=[c for c in cohorts if c in set(curves["Cohort"])])),
        tooltip=["Cohort", alt.Tooltip("months:Q", title="Months"),
                 alt.Tooltip("retention:Q", title="Still employed", format=".0%"),
                 alt.Tooltip("eligible:Q", title="Hires measured")],
    )
    st.altair_chart(base.mark_line(strokeWidth=2) + base.mark_point(size=64, filled=True),
                    use_container_width=True)
    with st.expander("Table"):
        st.dataframe(curves.pivot(index="Cohort", columns="months", values="retention")
                     .rename(columns=lambda m: f"{m} mo").style.format("{:.0%}", na_rep="—"),
                     width="stretch")

# ---------------------------------------------------------------------------
# Open seats
# ---------------------------------------------------------------------------

as_of_label = (end - pd.Timedelta(days=1)).strftime("%b %d, %Y").replace(" 0", " ")
st.subheader(f"Open seats as of {as_of_label}")
seats = open_seats(staff, positions, end)
if seats.empty:
    st.info("Every seat in this view was filled at the end of the period.")
else:
    seats_table = pd.DataFrame({
        "Site": seats["site_id"].map(site_names_short),
        "Role": seats["role"],
        "Staff type": seats["staff_type"].map({FIELD: "Field", CENTRAL_ADMIN: "Central admin"}),
        "Seat": seats["position_id"],
        "Vacant since": seats["vacant_since"].dt.date,
        "Days open": seats["days_open"],
    }).sort_values("Days open", ascending=False)
    st.dataframe(seats_table, hide_index=True, width="stretch", column_config={
        "Days open": st.column_config.NumberColumn(format="%.0f", help="Blank: empty since before the data starts."),
    })
