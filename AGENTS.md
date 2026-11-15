# Kessler Sentinel — Autonomous Agent & Intelligence System Guide

Welcome to **Kessler Sentinel** (`kessler-sentinel`), the machine learning intelligence platform for Low Earth Orbit (LEO) orbital debris risk prediction, satellite conjunction assessment, and collision mitigation within Systems Studio (Tier 2).

This file is the single authoritative source of truth for all autonomous coding agents, LLMs, and engineers operating in this repository.

---

## 1. Front-Page Context Firewall

To maintain context hygiene and prevent hallucination, agents MUST observe the following firewall boundaries:

| Action | Permitted Boundaries | Forbidden Boundaries |
| :--- | :--- | :--- |
| **Where to Read on Startup** (Max 4 files) | 1. `AGENTS.md` (this file)<br>2. `README.md`<br>3. `predict.py`<br>4. `tests/test_model.py` | Do **NOT** scan arbitrary files or historical branches without explicit task scope. |
| **Where to Write** | `src/`, `predict.py`, `train.py`, `tests/`, `docs/`, `Makefile`, `Dockerfile` | Do **NOT** write operational logs or cross-portfolio items locally; route portfolio-wide tasks to `portfolio-ops/MASTER_BACKLOG.md`. |
| **Where NOT to Read** | Active source code and configuration files | Strictly NEVER inspect or parse files in `archive/`, `past_version/`, `.venv/`, or `.pytest_cache/`. |

---

## 2. Proactive Strategic Co-Pilot Protocol & SVED 6-Validator Engine

Every agent turn operating on non-trivial architecture or operational decisions MUST execute the Proactive Co-Pilot workflow:
1. **ITR Scoring (Impact / Toxicity / Reversibility):** Calculate risk metrics before generating irreversible code changes.
2. **SVED 6-Validator Engine:**
   - *Schema Compatibility:* Verify feature schemas and orbital parameters match production definitions.
   - *Security / Secrets:* Zero API keys, credentials, or private tokens committed.
   - *Type Safety & Linting:* Adhere to Python type hints and clean execution.
   - *Regression Guard:* Ensure unit and feature tests pass (`pytest`).
   - *Dimensional Invariance:* Physical orbital metrics (semi-major axis, eccentricity, inclination, miss distance) must remain dimensionally consistent.
   - *Provenance Integrity:* Zero AI foot-printing or unverified attribution comments.
3. Present execution proposals using the canonical header:
   `[PROACTIVE EXECUTION QUEUE — AWAITING YOUR "YES"]`

---

## 3. Four-Layer Agentic Architecture (Grigorev + gstack Standard)

1. **Layer 1 — Entry Point & Routing:** Guided by `AGENTS.md` and context firewalls.
2. **Layer 2 — Empirical Domain Skills:** Astro-dynamics formulas, conjunction data ingestion, feature engineering, and model inference calibration.
3. **Layer 3 — 3-Role Fresh-Session Cycle:**
   - *Product Manager (PM):* Defines conjunction risk thresholds, SLA, and feature scope.
   - *Software Craftsman (Engineer):* Implements vectorized feature pipelines and inference endpoints.
   - *Fresh-Session QA (Auditor):* Tests edge cases (extreme eccentricity, missing telemetry, sensor noise) in clean sessions.
4. **Layer 4 — Scoped Tool Boundaries:** Scoped read-only access where applicable; safe local sandboxing.

---

## 4. Systems Studio Universal Rules (#10–#13)

- **Rule #10 (Backlog Auto-Registration):** Any new cross-portfolio feature, risk, or bug discovered must be formally logged in `portfolio-ops/MASTER_BACKLOG.md`.
- **Rule #11 (CI Workflow Parity):** Local test invocations (`pytest`) must match continuous integration pipelines exactly.
- **Rule #12 (Dimensional Invariance, Anti-Tautology & Statistical Rigor):** Physical orbital dynamics must maintain units (km, degrees, seconds). Never use tautological surrogate targets; use calibrated survival probabilities or exact physical thresholds.
- **Rule #13 (Automated Code-Rule Checker & Root Pytest Isolation):** Root test runs MUST isolate tests directory via `pytest.ini` (`testpaths = tests`, `norecursedirs = archive past_version .* .venv`).

---

## 5. C-Suite Communication & Spanish-Friendly Pronunciation Standard

When providing executive debriefs or technical term coaching:
- Chat responses to the principal investigator are conducted in professional Spanish.
- Repository documentation and code artifacts remain in clean American English.
- English term pronunciation pills MUST use **intuitive Spanish-friendly phonetic respelling** (e.g., *KEHS-ler SEN-tih-nel*). Strictly avoid International Phonetic Alphabet (IPA) symbols like `/ˈkɛslər/`.
