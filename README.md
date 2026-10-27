# Conversation2FHIR
**An evidence-linked, safety-checked clinical documentation agent.**
A doctor–patient conversation goes in. A verified, FHIR-formatted record and automated patient follow-up come out. The doctor approves every record before it is saved.

> **Disclaimer:** This is a learning and engineering project built on **synthetic data only**. It is a documentation aid, not a medical device. It does not diagnose, prescribe or replace clinical judgment. Every record requires clinician approval.

---

## 1. Problem

In busy clinics, doctors have minutes per patient. Notes end up incomplete, inconsistent or missing, which hurts follow-up care, medication safety and reporting. Existing AI scribes turn conversations into text notes, but three problems remain:

1. **Trust:** the doctor cannot easily tell which parts of an AI note were actually said and which were invented.
2. **Safety:** documentation tools rarely check the prescription itself (allergy conflicts, unusual doses).
3. **Last mile:** the note is not the end of the work. Patient instructions, reminders and referrals still have to be produced by hand.

## 2. What this project does differently

| Feature | Description |
|---|---|
| **Evidence-linked fields** | Every extracted fact (symptom, drug, dose, allergy) carries the exact transcript sentence that supports it. Click a field, see the proof. |
| **Grounding verifier** | A deterministic check rejects any field whose quote is not in the transcript or does not support the value. |
| **Safety agent** | Checks allergy conflicts, dose ranges and missing information before the record can be saved. |
| **Mandatory human approval** | The pipeline pauses until the doctor approves or edits. Nothing is written without it. |
| **Real FHIR output** | Approved records are written as FHIR resources to a FHIR server. |
| **Post-visit automation** | Approved visits trigger patient instructions, reminders and referral drafts. |
| **Local-first** | Runs on a local LLM so conversation data stays on the machine. Cloud model is only a comparison baseline. |
| **Measured** | Evaluated against synthetic ground truth with planted errors. |

## 3. Architecture

```
Audio ──► Speech-to-text ──► transcript (+ speaker labels)
                                  │
      ┌───────────────────────────▼────────────────────────────┐
      │  LangGraph pipeline  (LangChain + Ollama inside nodes) │
      │  1. Extractor agent       → structured note + quotes   │
      │  2. Grounding verifier    → deterministic quote check  │
      │  3. Safety agent          → allergy / dose / gaps      │
      │         └─ MCP drug server ─► LlamaIndex ─► vector DB  │
      │  4. ⏸ Doctor review       → approve / edit / reject    │
      │  5. FHIR writer           → MCP FHIR server ─► HAPI    │
      └───────────────────────────┬────────────────────────────┘
                                  ▼
                    CrewAI "care crew" (after approval)
          Educator · Translator · Scheduler · Referral writer
                                  ▼
                       n8n workflow (webhook)
       send instructions · create reminder · referral email · log

   Cross-cutting: Pydantic schemas · Langfuse tracing · FastAPI · Streamlit · Docker
```

### Why two orchestration frameworks?
- **LangGraph** runs the safety-critical path. It needs explicit state, retries, checkpointing and a hard approval pause.
- **CrewAI** runs the post-approval content generation, where a team of role-based writers fits naturally and mistakes are lower stakes.

A single framework could do both. The split is a deliberate design choice made to separate high-risk and low-risk work, and to learn both styles.

## 4. Pipeline stages in detail

### 4.1 Transcription
Audio is converted to text with speaker separation (doctor vs patient). Development starts with **text transcripts** so LLM work is not blocked by audio quality. Audio is added afterwards.

### 4.2 Extractor agent
Produces a `ClinicalNote` (schema below). Every item includes a `quote` copied from the transcript. Output is validated with Pydantic; invalid output is retried with the validation error fed back to the model.

### 4.3 Grounding verifier (deterministic, not an LLM)
For each field:
1. Does the `quote` appear in the transcript (after whitespace and case normalization)?
2. Does the `value` appear in or follow from the quote? (rule checks for drug names and numbers; LLM judge only for fuzzy cases)

Failing fields are flagged or removed. Using plain code for the first check removes a whole class of hallucination.

### 4.4 Safety agent
- **Allergy conflicts:** curated cross-reaction table (e.g. penicillin-class allergy vs amoxicillin) as the reliable base layer.
- **Dose and frequency sanity checks:** curated ranges for the drugs in the synthetic set, plus label text retrieved through RAG.
- **Missing information:** e.g. a drug prescribed with no dose, an allergy status never asked about.
- Output: a list of flags with severity and the evidence used.

### 4.5 Doctor review (human-in-the-loop)
LangGraph pauses at an interrupt. The review screen highlights flagged fields, shows the evidence for each field, and lets the doctor edit. **Edits are logged** and used as the "doctor edit rate" metric.

### 4.6 FHIR writer
Approved data is mapped to FHIR resources (`Patient`, `Encounter`, `Condition`, `MedicationRequest`, `AllergyIntolerance`) and posted to a HAPI FHIR server through an MCP tool.

### 4.7 Care crew (CrewAI) and workflow (n8n)
- **Educator:** plain-language instructions.
- **Translator:** English ↔ Tamil (quality varies by model; flagged as best-effort).
- **Scheduler:** follow-up date and medicine schedule.
- **Referral writer:** draft referral letter for clinician review.
- **n8n** receives a webhook with the approved output and handles delivery steps (email/notification stubs, reminder creation, logging).

## 5. Core data schema

```python
from pydantic import BaseModel

class Evidenced(BaseModel):
    value: str
    quote: str                  # exact sentence from the transcript

class Medication(BaseModel):
    name: str
    dose: str | None = None
    frequency: str | None = None
    duration: str | None = None
    quote: str

class ClinicalNote(BaseModel):
    complaints: list[Evidenced]
    allergies: list[Evidenced]
    medications: list[Medication]
    diagnosis: list[Evidenced]
    follow_up: list[Evidenced]
```

Minimal grounding check:

```python
import re

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()

def is_grounded(quote: str, transcript: str) -> bool:
    return _norm(quote) in _norm(transcript)
```

## 6. Tech stack: what, why, where from

> Versions and APIs change quickly. Always follow each project's **current official docs** and pin versions in `requirements.txt` / `docker-compose.yml`.

| Layer | Tool | Role in this project | Source |
|---|---|---|---|
| Speech-to-text | **faster-whisper** (Whisper models) | Converts consultation audio to text; CPU-friendly with int8 quantization | `pip install faster-whisper` · github.com/SYSTRAN/faster-whisper |
| Local LLM runtime | **Ollama** | Serves the local chat and embedding models | ollama.com · model library at ollama.com/library |
| LLM components | **LangChain** | Chat model wrappers, prompt templates, structured-output parsing, text splitters used inside nodes | `pip install langchain langchain-ollama` · python.langchain.com |
| Orchestration (core) | **LangGraph** | Stateful pipeline, retries, checkpointing, human-approval interrupt | `pip install langgraph` · langchain-ai.github.io/langgraph |
| Tool protocol | **MCP (Model Context Protocol)** | Standard interface to tools: one server for FHIR, one for drug knowledge | modelcontextprotocol.io · official Python SDK |
| Retrieval | **LlamaIndex** | Indexes and queries drug label / guidance text for the safety agent | `pip install llama-index` · docs.llamaindex.ai |
| Vector store | **Qdrant** (or Chroma for a lighter setup) | Stores embeddings of drug guidance | Docker image `qdrant/qdrant` · qdrant.tech |
| Multi-agent crew | **CrewAI** | Role-based post-visit content generation | `pip install crewai` · docs.crewai.com |
| Workflow automation | **n8n** | Post-approval delivery workflow via webhook | Docker image `n8nio/n8n` · n8n.io |
| Schemas and validation | **Pydantic** | Typed agent outputs, validation, retry feedback | `pip install pydantic` |
| Interoperability | **FHIR + HAPI FHIR server** | Standard clinical data format and a local test server | hl7.org/fhir · hapifhir.io · Docker image `hapiproject/hapi` |
| Observability | **Langfuse** | Traces, latency and cost per run, debugging | langfuse.com (cloud free tier or self-host per its docs) |
| Evaluation | **Custom harness** + DeepEval or RAGAS-style metrics | Field accuracy, grounding rate, safety recall | `pip install deepeval` / `ragas` (optional) |
| API / UI | **FastAPI**, **Streamlit** | Backend service and doctor review screen | `pip install fastapi uvicorn streamlit` |
| Packaging | **Docker / docker-compose** | Reproducible multi-service setup | docker.com |
| CI | **GitHub Actions** | Runs the evaluation set on every change | docs.github.com/actions |

### Data sources
| Data | Source | Notes |
|---|---|---|
| Consultations | **Synthetic**, generated by this repo from a structured ground-truth record | No real patient data is used anywhere |
| Drug label text | **openFDA** drug label API (open.fda.gov) | Public data; check current terms and rate limits |
| Drug name normalization | **RxNorm** (NLM) | Optional; confirm which RxNorm/RxNav services are currently available before depending on them |
| Interaction and allergy rules | **Small curated table in this repo** | The reliable base layer for the planted-error tests. Not a clinical-grade knowledge base |

## 7. Computational design and reliable options

### 7.1 Hardware tiers

Approximate guidance. Test on your machine early.

| Tier | Hardware | Speech-to-text | LLM | What to run |
|---|---|---|---|---|
| **Minimum** | 8 GB RAM, no GPU | `base`/`small` model, int8 | ~3–4B quantized model | Text-transcript mode, core pipeline only; use cloud tiers for tracing; skip n8n/CrewAI locally |
| **Recommended** | 16 GB RAM, optional 6–8 GB GPU | `small` | ~7–8B instruct model, 4-bit quantized | Full docker-compose stack |
| **Comfortable** | 32 GB RAM / 12 GB+ GPU | `medium` or larger | ~14B-class model | Full stack plus parallel evaluation runs |

Model choice: pick an instruct model from the Ollama library that supports **structured (JSON-schema) output** well. Test two candidates on Day 1 against a few synthetic consultations and keep the better one. Model availability moves fast, so check the library for current options.

### 7.2 Techniques used for reliable output

| Risk | Mitigation |
|---|---|
| Model breaks JSON format | Schema-constrained output where supported, plus Pydantic validation and retry with the error message |
| Hallucinated facts | Evidence quote required for every field; **deterministic** string-level grounding check |
| Run-to-run variation | Temperature 0 (or near), fixed seeds where supported, saved prompts and model versions |
| Unsafe prescriptions | Rule-based safety layer first, LLM/RAG second; any uncertainty becomes a flag, not a guess |
| Weak local-model performance | Measured local vs cloud comparison; cloud model only as optional fallback baseline |
| Tool or service failure | Timeouts, bounded retries, and a clear failed state in the UI. No silent fallbacks |
| Crashes mid-run | LangGraph checkpointing so a run resumes where it stopped |
| Cost and latency creep | Per-step latency tracked in Langfuse; cache embeddings and repeated lookups |
| Resource exhaustion | Container memory limits in compose; start heavy services only when needed |
| Non-reproducible results | Pinned dependency versions, fixed evaluation set, evaluation run in CI |

### 7.3 Design principle
**Use an LLM only where judgment or language is needed. Use code wherever a rule can do the job.** Quote matching, schema validation and curated safety rules are plain code. The LLM extracts and drafts; it does not get the last word on safety.

## 8. Evaluation

### Synthetic dataset
~15–20 consultations generated from a structured ground-truth record, so the correct answer is known. Each includes some of these planted traps:
- Prescribed drug conflicts with a stated allergy
- Dose outside the normal range
- Medication mentioned vaguely with no dose
- Details corrected mid-conversation ("no, not Tuesday, Thursday")
- Code-mixed (Tamil–English) phrases (stretch goal)

### Metrics
| Metric | Definition |
|---|---|
| Field precision / recall | Extracted fields vs ground truth |
| Grounding rate | Share of fields whose quote supports the value |
| Unsupported-field catch rate | Planted fabrications caught by the verifier |
| Safety recall | Planted unsafe prescriptions flagged |
| Safety false-alarm rate | Safe prescriptions wrongly flagged |
| Doctor edit rate | Share of fields changed during review |
| Local vs cloud | Accuracy and latency difference between models |
| Baseline | Single plain prompt with no verifier, on the same cases |
| Latency | Seconds per consultation, per pipeline stage |

### Results
> Fill in after running the evaluation. Do not publish numbers that were not measured.

| System | Field F1 | Grounding rate | Safety recall | False alarms | Latency |
|---|---|---|---|---|---|
| Baseline (single prompt) | TBD | TBD | n/a | n/a | TBD |
| Full pipeline, local model | TBD | TBD | TBD | TBD | TBD |
| Full pipeline, cloud model | TBD | TBD | TBD | TBD | TBD |

Include a **Failure cases** section with real examples of where the system went wrong.

## 9. Safety, privacy and limitations

- Synthetic data only. Never commit real patient information.
- Human approval is mandatory before any record is saved.
- The system flags and documents. It does not diagnose, prescribe or give treatment advice.
- The curated safety rules cover only the drugs and cases in the test set and are **not** clinical-grade.
- Speech recognition and local-model quality are weaker for Tamil and code-mixed speech.
- Not validated clinically. Not for use with real patients.

## 10. Repository structure

```
consult-to-care/
├── README.md
├── docker-compose.yml
├── pyproject.toml / requirements.txt
├── data/
│   ├── ground_truth/            # structured records
│   ├── transcripts/             # generated consultations
│   └── drug_rules/              # curated allergy/dose tables
├── src/
│   ├── schemas.py               # Pydantic models
│   ├── transcribe.py
│   ├── agents/
│   │   ├── extractor.py
│   │   ├── verifier.py          # deterministic grounding
│   │   └── safety.py
│   ├── graph.py                 # LangGraph pipeline + approval interrupt
│   ├── mcp_servers/
│   │   ├── fhir_server.py
│   │   └── drug_server.py       # LlamaIndex retrieval behind MCP
│   ├── crew/                    # CrewAI care crew
│   ├── api.py                   # FastAPI
│   └── ui.py                    # Streamlit review screen
├── workflows/n8n/               # exported n8n workflow JSON
├── eval/
│   ├── generate_dataset.py
│   ├── run_eval.py
│   └── results/
└── .github/workflows/eval.yml
```

## 11. Quickstart (adjust to your final setup)

```bash
# 1. Local model
ollama pull <chosen-model>

# 2. Services
docker compose up -d      # HAPI FHIR, Qdrant, n8n (and Langfuse if self-hosted)

# 3. App
pip install -r requirements.txt
python eval/generate_dataset.py
uvicorn src.api:app --reload
streamlit run src/ui.py

# 4. Evaluation
python eval/run_eval.py
```

## 12. Roadmap / status

Mark items done only when built and tested.

- [ ] Synthetic dataset with planted traps
- [ ] Extractor with evidence quotes and Pydantic retries
- [ ] Deterministic grounding verifier
- [ ] LangGraph pipeline with approval interrupt
- [ ] Safety agent (rules + retrieval)
- [ ] MCP FHIR server and HAPI FHIR integration
- [ ] MCP drug server with LlamaIndex retrieval
- [ ] Streamlit review screen with evidence view
- [ ] Langfuse tracing
- [ ] Evaluation harness and results table
- [ ] CrewAI care crew
- [ ] n8n post-visit workflow
- [ ] Audio input (Whisper)
- [ ] Tamil–English support (stretch)
- [ ] GitHub Actions evaluation job
- [ ] Cloud deployment (stretch)

## 13. References

- FHIR specification: hl7.org/fhir
- HAPI FHIR: hapifhir.io
- Model Context Protocol: modelcontextprotocol.io
- LangGraph: langchain-ai.github.io/langgraph
- LlamaIndex: docs.llamaindex.ai
- CrewAI: docs.crewai.com
- n8n: n8n.io
- Ollama: ollama.com
- Langfuse: langfuse.com
- openFDA: open.fda.gov
- faster-whisper: github.com/SYSTRAN/faster-whisper

## 14. License

Choose a license (e.g. MIT) and check the licenses and terms of every model, dataset and API you use.
