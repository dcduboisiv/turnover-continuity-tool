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
      break-even tenure with a ramp-up discount, and the care-continuity
      score. 36 passing unit tests. *Done.*
- [ ] **MVP 2 — Streamlit dashboard.**
- [ ] **MVP 3 — AI layer.**

## MVP 1 — What's here

- `generate_sample_data.py` — writes the sample data (seed 42, byte-identical
  on every run):
  - `sites.csv` — 8 fictional Kinbridge sites: a central office, four
    family services centers, two youth residences and a clinic.
  - `synthetic_positions.csv` — 199 budgeted seats by site and role.
  - `synthetic_staff.csv` — everyone who held a seat during 2023–2025:
    hire date, termination date, role, program area, site, voluntary /
    involuntary, tenure. IDs start `SYN-S-`.
  - `synthetic_children.csv`, `synthetic_assignments.csv` — children (IDs
    `SYN-C-`, no names or case details) and which staff member covered
    each of their child-facing roles, day by day.
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

### What the engine computes

Periods are half-open, `[start, end)`. A person is employed from
`hire_date` up to (not including) `termination_date`.

| Output | Function | Definition |
|---|---|---|
| Turnover rate | `turnover_rates(staff, start, end, by=...)` | Separations / average headcount, annualized; split voluntary vs. involuntary. By role, program area, site, or overall. |
| Vacancy & days-to-fill | `vacancy_report(staff, positions, start, end, by=...)` | Vacant seat-days / total seat-days; average days from a departure to the backfill hire; seats open at the end. |
| Retention curves | `retention_curves(staff, as_of, months=...)` | Share of each hire cohort still employed 3, 6, 12, 18 and 24 months after hire. Only hires old enough to be measured count. |
| Cost per departure | `cost_per_departure(assumptions, days_to_fill=...)` | Recruiting + training + vacancy coverage (days-to-fill × loaded monthly cost × coverage premium) + ramp-up productivity loss. The report uses observed days-to-fill. |
| Break-even tenure | `breakeven_tenure(assumptions, schedule=...)` | (Recruiting + training) ÷ the monthly coverage premium a filled seat avoids, with that saving discounted by the ramp-up. Shows the months with and without the ramp, and the share of departures who left before break-even. |
| Care-continuity score | `care_continuity_score(assignments, start, end)` | Per child: distinct people in each child-facing role, weighted by criticality (caseworker 1.0, youth care worker 0.9, clinician 0.8). 1.0 means one person in every role. Averaged across children, overall or by site. |

**Ramp-up discount.** By default a new hire works at 50% in month 1, 75%
in month 2 and 100% from month 3, a linear ramp. It is set by
`RAMP_START_PRODUCTIVITY` and `RAMP_MONTHS_TO_FULL` in
`turnover_calculator.py`. Any function that uses it also takes a
`schedule` argument, and `schedule=None` turns the ramp off. The ramp
feeds two numbers: it adds a ramp-up cost to each departure, and it pushes
break-even tenure later.

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python generate_sample_data.py      # rewrite the sample CSVs
.venv/bin/python turnover_calculator.py       # print the report
.venv/bin/python -m pytest -q                 # 36 tests
```
