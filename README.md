# Turnover & Care Continuity Impact Tool — Build Log

Workforce tool for a child welfare nonprofit. Measures staff turnover,
vacancies and retention, prices what each departure costs, works out how
long a new hire has to stay before replacing them pays off, and scores how
often children have to start over with a new caseworker, clinician or
youth care worker because staff left.

**Market: New York City child welfare / youth services.**

> **The organization is fictional.** Kinbridge Family Services and all of
> its sites are invented. The org is modeled on the general structure of
> real NYC child welfare agencies (family foster care and prevention
> offices, youth residences, a clinic and a central office, in Manhattan,
> Brooklyn and the Bronx) for realism. It is not based on any specific
> agency. Sites sit in real neighborhoods so the borough mix and map look
> right, but the street names and addresses are made up and every site
> name ends in "(SYN)". Every staff member, child and assignment is
> synthetic.

## Status

- [x] **MVP 1 — Data & calculation engine (no UI, no AI).** Synthetic org
      generator, turnover / vacancy / retention math, cost of turnover,
      break-even tenure with a ramp-up discount and a choice of
      vacancy-cost basis, vacancy cost crossover, the care-continuity
      score, and family-days without an assigned caseworker or clinician.
      Every record is tagged field or central admin. 48 passing unit
      tests. *Done.*
- [x] **MVP 2 — Streamlit dashboard.** Sidebar filters (site, program
      area, field / central admin, role, period) and every engine setting
      (per-role costs, ramp-up on/off and its two settings, vacancy-cost
      basis, caseloads), with headline metrics, cost and break-even with and
      without the ramp, vacancy cost crossover, family-days by site,
      retention curves and open seats. Built for Streamlit Community Cloud.
      88 passing tests in all, including 38 that run the app headlessly.
- [x] **MVP 3 — AI layer.** Plain-English scenario questions (lookups,
      site / role comparisons and what-ifs) at the top of the page, and a
      funder-report summary for any site and period, in the app and as a
      CLI. Claude parses and writes; the calculator does every
      calculation, and every number Claude writes is checked against the
      calculator's output. 156 passing tests, with every Claude call
      mocked (no API key or spend needed). *Built; not yet run against
      the live API.*

## MVP 3 — What's here

- `scenarios.py` — the deterministic scenario engine (no AI). Views
  (site, program area, staff type, roles, period) and settings (costs,
  ramp-up, vacancy-cost basis, caseloads) as objects, the dashboard's
  headline / per-role / per-site numbers, and the what-if changes:
  - **Turnover reduction** ("caseworker turnover drops 10%"): that share
    of the period's departures, taken evenly by date, don't happen. Each
    of those staff stays in the seat instead of being replaced, so the
    seat never goes vacant and the children keep their caregiver. Turnover,
    vacancy, continuity and family-days are all recomputed from the
    adjusted data. "10 points" is converted to a share using the actual
    rate.
  - Caseloads, per-role costs (set or ±%), ramp-up on/off and its two
    settings, and the vacancy-cost basis.
  - The dashboard uses the same `filter_view`, so an answer's "before"
    numbers are the numbers on the page.
- `turnover_agent.py` — the AI layer, same discipline as the EDO, RevOps
  and recruiting agents: the model never does the math.
  - **`ask()`**: the question goes to Claude, which returns a structured
    view (site, staff type, roles, program areas, years), what-if changes
    from the list above, an optional site or role breakdown and the
    metric the question is about. Structured outputs guarantee the shape;
    `validate_question` checks the meaning before anything reruns. For
    example, it rejects "10%" parsed as 10 and caseloads for shift-based
    roles. The engine runs before and after, ranks any breakdown itself
    (highest / lowest, which direction is worse, all-equal), and Claude
    writes 2–4 sentences from the computed results.
  - **`funder_report()`**: a view's children served, turnover,
    vacancies, 12-month retention, days-to-fill, continuity score,
    families affected by a vacancy gap and their average days without a
    worker (plus year-by-year figures for multi-year periods) go to
    Claude, which writes a 120–180 word grant-report paragraph. Costs are
    left out.
  - **Number check**: every number the model writes (dollars,
    percentages, months, scores, counts) is compared with the numbers it
    was given (`unknown_figures`). Anything else, such as a rounded
    "$1.2M" or a made-up "90%", is flagged in the app and CLI.
  - Model `claude-opus-5-5` at low effort, with server-side refusal
    fallback (`fallbacks: "default"`) and structured outputs for the parse.
- `app.py` — "Ask about turnover and continuity" is the first thing on the
  page, above the headline metrics. It shows the answer, what was modeled,
  a what-if after-vs-before row and any breakdown table. Below it, a
  "Funder-report summary" expander drafts the paragraph for the sidebar's
  site and period into an editable box. Answers and drafts are kept in the
  session, so reruns don't call the AI again. Without a key, the
  assistant explains how to add one and the dashboard works unchanged.
- `test_scenarios.py`, `test_turnover_agent.py`, AI tests in
  `test_app.py`. In the AI tests, Claude is replaced by a fake that returns
  canned replies and records what it was sent.

## MVP 2 — What's here

- `app.py` — the dashboard. Every number comes from `turnover_calculator.py`;
  the app filters the data, passes the sidebar settings through, and shows
  the results.
  - **Sidebar:** site, program area, staff type (all / field / central
    admin, read from `staff_type`), role, and period (any run of years in
    2023–2025). Assumptions: an editable per-role cost table (recruiting,
    training, OT / temp premium, salary) seeded from
    `role_cost_assumptions.csv`; ramp-up on/off plus
    `RAMP_START_PRODUCTIVITY` and `RAMP_MONTHS_TO_FULL`; vacancy-cost basis;
    caseloads per caseload-carrying role; an editable program budget per
    site.
  - **Main panel:** headline cards in two rows. The first row has turnover
    rate, vacancy rate and the care-continuity score. The second row has
    the annual cost of turnover, with its supporting figures underneath
    (cost per departure, monthly cost, share of the program budget), and
    days without a worker per affected family; cost per departure and break-even tenure by
    role, with and without the ramp (the toggle decides which one feeds the
    annual cost and the "left before break-even" share); vacancy cost
    crossover; days without an assigned caseworker or clinician per
    affected family (the headline), families affected, and the family-days
    total as detail, overall and by site; retention by hire cohort; open seats with days open
    so far.
  - **Opens on a full view:** all sites, all staff, 2023–2025. A sweep of
    every site × staff type × period showed it is the broadest view with
    no empty section. Narrower views (for example central admin, which has
    no caseload or children) say what's missing instead of showing zeros.
  - With one site selected, the continuity score covers the children the
    site serves: those placed there, plus those assigned a worker from
    there. That way the Fordham clinic, where no children are placed,
    still gets a score.
- `test_app.py` — runs the app headlessly with Streamlit's AppTest: the
  default view is non-empty, headline numbers match the engine, every
  setting flows through, and every site × staff type renders.

## MVP 1 — What's here

- `generate_sample_data.py` — writes the sample data (seed 42, byte-identical
  on every run):
  - `sites.csv` — 8 fictional Kinbridge sites: a central office, four
    family services centers, two youth residences and a clinic.
  - `synthetic_positions.csv` — 199 budgeted seats by site and role.
  - `synthetic_staff.csv` — everyone who held a seat during 2023–2025:
    hire date, termination date, role, program area, `staff_type`, site,
    voluntary / involuntary, tenure. IDs start `SYN-S-`.
  - `synthetic_children.csv`, `synthetic_assignments.csv` — children (IDs
    `SYN-C-`, no names or case details) and which staff member covered
    each of their child-facing roles, day by day.
- `program_budget_assumptions.csv` — **synthetic** annual program budget
  per site ($22.0M in all). Kinbridge's data had no budget figure, so this
  was added as a configurable input. Each site is sized from its loaded
  payroll (seats × salary × fringe), assuming personnel is about 68% of a
  program budget. It is used only for turnover cost as a share of budget;
  edit the file, or the sidebar table, to change it.
- `role_cost_assumptions.csv` — per-role cost assumptions (salary, fringe,
  recruiting, training, planning days-to-fill, coverage premium). Edit
  this file to change the cost model.
- `turnover_calculator.py` — the engine. `python turnover_calculator.py`
  prints the full report for the sample data.
- `test_turnover_calculator.py` — every calculation checked against
  hand-worked cases. `test_generate_sample_data.py` — checks the sample
  data is fictional, labeled, deterministic and shaped realistically.

### How the sample data is shaped

Turnover is not flat. Each role has its own rate, and direct-care and
casework roles churn several times faster than admin. The rates below are
the generator's targets; seed 42 measures 47% direct care, 31% case
management, 22% clinical, 13% admin.

| Program area | Roles | Target annual turnover |
|---|---|---|
| Direct Care | Youth Care Worker, Residential Counselor | 55%, 45% |
| Case Management | Caseworker, Case Supervisor | 42%, 20% |
| Clinical | Clinician, Clinical Supervisor | 25%, 15% |
| Admin | Administrative Assistant, Finance Associate | 14%, 10% |

- Tenure is Weibull-distributed with shape 0.8, so most departures come
  early: nearly half of a hire cohort is gone within a year.
- Every departure leaves the seat empty for a role-specific number of days
  (clinical seats take longest to fill).
- Children are assigned to seats. When staff leave, the child gets a
  coworker who covers the gap, then the new hire, so caregiver churn comes
  out of the same staff history.

### Field vs. central admin

Every staff and seat record has a `staff_type` column. Scope by this
column, not by role name; MVP 2's field / central-admin filter reads
it directly.

| staff_type | Meaning | Roles |
|---|---|---|
| `field` | Direct family / child contact | Direct Care (Youth Care Worker, Residential Counselor), Case Management (Caseworker, Case Supervisor), Clinical (Clinician, Clinical Supervisor) |
| `central_admin` | Internal only, no family contact | Admin (Administrative Assistant, Finance Associate) and any future back-office role |

Field roles that carry an individual family caseload: **Caseworker** and
**Clinician**. Supervisors are field staff without a caseload. Direct care
is field but shift-based.

### What the engine computes

Periods are half-open, `[start, end)`. A person is employed from
`hire_date` up to (not including) `termination_date`.

| Output | Function | Definition |
|---|---|---|
| Turnover rate | `turnover_rates(staff, start, end, by=...)` | Separations / average headcount, annualized; split voluntary vs. involuntary. By role, program area, staff type, site, or overall. |
| Vacancy & days-to-fill | `vacancy_report(staff, positions, start, end, by=...)` | Vacant seat-days / total seat-days; average days from a departure to the backfill hire; seats open at the end. |
| Retention curves | `retention_curves(staff, as_of, months=...)` | Share of each hire cohort still employed 3, 6, 12, 18 and 24 months after hire. Only hires old enough to be measured count. |
| Cost per departure | `cost_per_departure(assumptions, days_to_fill=...)` | Recruiting + training + vacancy coverage (days-to-fill × loaded monthly cost × coverage premium) + ramp-up productivity loss. The report uses observed days-to-fill. |
| Break-even tenure | `breakeven_tenure(assumptions, schedule=..., basis=...)` | (Recruiting + training) ÷ what a filled seat saves each month, with that saving discounted by the ramp-up. Shows the months with and without the ramp, and the share of departures who left before break-even. The monthly saving follows `VACANCY_COST_BASIS` (below). |
| Cost of turnover summary | `turnover_cost_summary(staff, start, end, costs, program_budget)` | The annual cost plus its supporting figures. **Per departure** = the whole period's departures × their cost ÷ the number of departures (the annual total ÷ a three-year departure count would understate it about 3×). **Monthly** = annual ÷ 12. **Share of budget** = annual ÷ the program budget of the sites in view. |
| Vacancy cost crossover | `vacancy_cost_crossover(assumptions)` | Recruiting and onboarding cost ÷ monthly coverage premium: how many months of overtime/temp coverage cost as much as hiring a replacement. Uses the coverage premium regardless of `VACANCY_COST_BASIS`, and no ramp, so under the default basis it equals break-even tenure without the ramp. |
| Care-continuity score | `care_continuity_score(assignments, start, end)` | Per child: distinct people in each child-facing role, weighted by criticality (caseworker 1.0, youth care worker 0.9, clinician 0.8). 1.0 means one person in every role. Averaged across children, overall or by site. Counts **caregiver changes**. |
| Continuity gap: family-days without an assigned caseworker or clinician | `family_days_without_assigned_worker(staff, positions, start, end, by=...)` | For every vacancy in a caseload-carrying field seat: days the seat was empty × that seat's typical caseload, summed. Caseloads come from `TYPICAL_CASELOAD` (Caseworker 10, Clinician 20), with optional per-seat overrides. Only field seats in those roles count: a vacant Direct Care seat (shift-based, no individual cases) or central admin seat produces no family-days. Measures **how long families go without a worker of record**, not how often their caregiver changes. It is seat-based, so it counts the gap even when a coworker covers the cases. Seats still open at the end count up to the period end. Also reports **affected families** (the typical caseload of every seat that had a gap, counted once per seat even if it was vacant twice) and **average days per affected family** (family-days ÷ affected families). The dashboard leads with these two ("680 families experienced a gap, averaging 93.8 days each") and keeps the family-days total as detail. |

**Ramp-up discount.** By default a new hire works at 50% in month 1, 75%
in month 2 and 100% from month 3, a linear ramp. It is set by
`RAMP_START_PRODUCTIVITY` and `RAMP_MONTHS_TO_FULL` in
`turnover_calculator.py`. Any function that uses it also takes a
`schedule` argument, and `schedule=None` turns the ramp off. The ramp
feeds two numbers: it adds a ramp-up cost to each departure, and it pushes
break-even tenure later.

**Vacancy-cost basis.** `VACANCY_COST_BASIS` in `turnover_calculator.py`
sets what a filled seat is counted as saving each month in break-even
tenure:

- `"coverage_premium"` (default): the overtime/temp premium above normal
  pay (loaded monthly cost × `coverage_premium`). The new hire's salary
  replaces the coverage spend, so only the premium is saved. Caseworker
  break-even comes out at 6.5 months with the ramp.
- `"full_salary"`: the seat's full loaded monthly cost. Caseworker
  break-even comes out at 3.6 months.

`breakeven_tenure` also takes a `basis` argument for one-off comparisons.

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python generate_sample_data.py      # rewrite the sample CSVs
.venv/bin/python turnover_calculator.py       # print the report
.venv/bin/streamlit run app.py                # the dashboard
.venv/bin/python -m pytest -q                 # 156 tests
.venv/bin/python turnover_agent.py --ask "Which site has the worst break-even tenure for caseworkers?"
```

## Deploy (Streamlit Community Cloud)

1. Push `main` to GitHub.
2. At share.streamlit.io, choose **Create app**, pick
   `dcduboisiv/turnover-continuity-tool`, branch `main`, main file `app.py`.
3. For the AI features, add `ANTHROPIC_API_KEY = "..."` under the app's
   **Settings → Secrets** (see `.streamlit/secrets.toml.example`). The
   dashboard works without it.
4. After a later push that changes an imported module
   (`turnover_calculator.py`, `scenarios.py`, `turnover_agent.py`,
   `generate_sample_data.py`), reboot the app
   (**Manage app → Reboot**), because Cloud keeps imported modules in
   memory.
