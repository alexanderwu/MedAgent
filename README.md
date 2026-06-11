# MedAgent 🩺🤖

An AI agent for clinical data intelligence, built over an 8-week summer internship.
MedAgent assists with **Real-World Evidence (RWE) data analytics** and **clinical text
mining / NLP**, with stretch goals in **market analysis & forecasting** and
**clinical trial design & analysis**.

The end product is an **interactive demo app** (Streamlit) where a user can ask
natural-language questions over public clinical datasets — e.g.
*"What are the most common comorbidities among ICU patients with sepsis?"* or
*"Summarize the adverse-event signal for metformin"* — and the agent plans,
queries the data, runs the analysis, and explains the result.

---

## Team

| Role | Person | Ownership |
|---|---|---|
| Mentor / Lead | Alexander Wu | Architecture, code review, weekly planning, unblocking |
| Mentee — CS undergrad | TBD | **Agent engineering**: orchestration, tool interfaces, app/UI, infrastructure, testing |
| Mentee — DS undergrad | TBD | **Analytics & NLP**: cohort analysis, statistical methods, NLP models, evaluation |

**Cadence:** 2× weekly sync (kickoff Monday, demo/review Friday), async check-ins
via chat, PR review for all merged code. Each week ends with something runnable.

---

## Tech Stack (zero-cost)

| Layer | Choice | Notes |
|---|---|---|
| Language | Python 3.11+ | |
| LLM (agent brain) | Llama 3.x / Mistral via [Ollama](https://ollama.com) | Runs locally; Google Colab free tier as fallback for heavier jobs |
| Clinical NLP | Bio_ClinicalBERT, BioBERT, scispaCy | Hugging Face / spaCy — free, purpose-built for clinical text |
| Agent framework | LangGraph (or a minimal hand-rolled tool loop) | Decide in Week 2 spike |
| Data layer | DuckDB + pandas | MIMIC-IV ships as CSVs; DuckDB gives fast local SQL |
| App / UI | Streamlit | Fast to build, free to host (Streamlit Community Cloud — demo data only) |
| Forecasting | statsmodels, Prophet | Stretch-goal phase |
| Versioning | GitHub, PRs + reviews | |

---

## Datasets (all free / public)

| Dataset | Use | Access |
|---|---|---|
| [MIMIC-IV Demo](https://physionet.org/content/mimic-iv-demo/) (100 patients) | Development from day 1 | Open, no credentialing |
| [MIMIC-IV](https://physionet.org/content/mimiciv/) (~300k patients) | Full RWE analytics | **Requires CITI training + PhysioNet credentialing — started Week 1** |
| [MIMIC-IV-Note](https://physionet.org/content/mimic-iv-note/) | Clinical notes for NLP | Same credentialing |
| [openFDA / FAERS](https://open.fda.gov/) | Adverse-event mining, drug safety signals | Open API |
| [ClinicalTrials.gov](https://clinicaltrials.gov/data-api/api) | Trial landscape, trial-design analysis | Open API |
| [PubMed / PMC](https://www.ncbi.nlm.nih.gov/home/develop/api/) | Literature mining | Open API |
| [CMS public datasets](https://data.cms.gov/) | Utilization & market analysis | Open |

> ⚠️ **Data use:** MIMIC data must never be committed to this repo, uploaded to
> any hosted service, or sent to any external API. All MIMIC processing stays
> local. The hosted demo uses only the open demo subset and synthetic examples.

---

## 8-Week Plan

### Phase 0 — Foundations (Week 1)
**Goal: environments ready, data access in motion, shared understanding of the domain.**

- All: complete **CITI training and submit PhysioNet credentialing** for full MIMIC-IV (Day 1–2 — it takes 1–2 weeks to clear, so it cannot wait).
- All: download MIMIC-IV **demo** subset; walkthrough of its schema (patients, admissions, diagnoses, labs, prescriptions, notes).
- CS mentee: repo scaffolding — project layout, `pyproject.toml`, pre-commit, CI lint/test workflow; get Ollama + a small Llama model running locally.
- DS mentee: exploratory notebook on the demo subset — patient demographics, top diagnoses, lab distributions; reading list on RWE study design (cohorts, confounding).
- Mentor: finalize 2–3 "north-star" demo questions the agent must answer by Week 8.

**Deliverable:** running dev environments, EDA notebook, credentialing submitted.

### Phase 1 — RWE Analytics Core (Weeks 2–3)
**Goal: a library of analytics tools the agent will later call.**

- DS: cohort-building utilities (define cohort by ICD code / drug / demographics), descriptive statistics, comorbidity analysis, length-of-stay and readmission analyses on the demo subset — swapping in full MIMIC-IV when credentialing clears.
- CS: wrap each analysis as a clean, documented **tool function** with typed inputs/outputs (the agent's future API); DuckDB ingestion pipeline; unit tests.
- Joint spike (Week 2): LangGraph vs. minimal hand-rolled agent loop — pick one and commit.
- First agent milestone (end of Week 3): LLM picks and calls **one** analytics tool from a natural-language question.

**Deliverable:** `medagent.analytics` tool library + a single-tool agent demo in a notebook.

### Phase 2 — Text Mining & NLP (Weeks 4–5)
**Goal: the agent can read and extract from clinical text and literature.**

- DS: clinical NER (scispaCy / Bio_ClinicalBERT) on MIMIC notes — extract conditions, medications, procedures; note summarization with the local LLM; evaluation set with precision/recall on a hand-labeled sample.
- CS: PubMed and openFDA/FAERS API clients as agent tools (search, fetch, adverse-event counts); retrieval layer (chunking + embedding with a free sentence-transformer) so the agent can do RAG over notes and abstracts.
- Joint: multi-tool agent — it now routes between structured analytics (Phase 1) and text tools (Phase 2), and can chain them (e.g., extract a drug from notes → query FAERS for its adverse events).

**Deliverable:** NLP tool suite + multi-tool agent answering cross-source questions in a notebook.

### Phase 3 — Demo App & Hardening (Weeks 6–7)
**Goal: the interactive demo app, working end-to-end on the north-star questions.**

- CS: Streamlit app — chat interface, streaming agent responses, tables/charts rendered from tool outputs, "show your work" panel exposing the agent's plan and queries.
- DS: evaluation harness — a question bank (~30 questions) with expected answers; measure correctness, tool-selection accuracy, and hallucination rate; tune prompts and tool descriptions from failures.
- Joint: guardrails (refuse out-of-scope medical advice; always cite which dataset/query produced a number), error handling, demo-mode dataset so the app runs without credentials.
- **Stretch goals slot in here if Phases 1–2 finished on schedule** (see below).

**Deliverable:** working Streamlit app passing the north-star questions; eval report.

### Phase 4 — Stretch Goals, Polish & Handoff (Week 8)
**Goal: ship it and tell the story.**

- Stretch (if time permits):
  - **Market analysis & forecasting:** drug-utilization trends from CMS/openFDA data, time-series forecasts (statsmodels/Prophet) exposed as an agent tool.
  - **Clinical trial design & analysis:** ClinicalTrials.gov landscape queries (enrollment, endpoints, phases for a condition); power/sample-size calculator tool.
- All: documentation (architecture diagram, tool catalog, setup guide), recorded demo video, final presentation.
- Mentees: short write-up each on their track — portfolio-ready.

**Deliverable:** tagged v1.0 release, demo video, final presentations.

---

## Weekly Milestones at a Glance

| Week | Milestone |
|---|---|
| 1 | Environments up, credentialing submitted, EDA on MIMIC demo |
| 2 | Analytics tools v1, agent framework chosen |
| 3 | Single-tool agent answers an RWE question |
| 4 | Clinical NER + summarization working on notes |
| 5 | Multi-tool agent chains analytics + text tools |
| 6 | Streamlit app MVP |
| 7 | Eval harness, guardrails, north-star questions passing |
| 8 | Stretch tools, docs, demo video, final presentation |

---

## Repository Layout (planned)

```
MedAgent/
├── medagent/
│   ├── agent/          # orchestration, prompts, tool routing
│   ├── analytics/      # RWE cohort & statistical tools
│   ├── nlp/            # NER, summarization, retrieval
│   ├── data/           # ingestion, DuckDB layer, API clients
│   └── app/            # Streamlit UI
├── notebooks/          # EDA, experiments, weekly demos
├── eval/               # question bank + evaluation harness
├── tests/
└── docs/
```

## Getting Started (placeholder — filled in during Week 1)

```bash
git clone https://github.com/alexanderwu/MedAgent.git
cd MedAgent
pip install -e ".[dev]"
ollama pull llama3.1:8b
streamlit run medagent/app/main.py   # demo mode, no credentials needed
```

---

## Learning Outcomes for Mentees

- **CS mentee:** agent architectures and tool use, API design, data engineering with DuckDB, app deployment, software craftsmanship (tests, CI, PR reviews).
- **DS mentee:** RWE study design on real EHR data, clinical NLP with transformer models, LLM evaluation methodology, communicating analytical results.

## Disclaimer

MedAgent is an educational research project. It is **not a medical device** and
must not be used for clinical decision-making.
