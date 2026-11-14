# Praktika runtime

**Code and neutral synthetic examples only — 0.3-public, 14 November 2026.**

A local development application for versioned agent configurations, JSON-contract evaluation and bounded file-validation workflows. This repository contains no educational program, level matrix, company documents, lecture notes, pedagogical corpus, course project identifiers, source references, user records or prebuilt knowledge index.

## Run

Python 3.12+, macOS or Linux (`fcntl` is used for process locking):

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python education/scripts/server.py --port 8001
```

Open http://127.0.0.1:8001. No API credentials are needed for synthetic fixture runs. The server is loopback-only; this repository is not a deployment of a public web service.

## Runtime features

- Agent connections, immutable configuration versions, JSON contract evaluation, result imports and explicit review records.
- Responses, OpenAI-compatible, Ollama and custom HTTP adapters. Endpoints and credential environment-variable names are supplied by the operator.
- Fixed eight-role sequential workflow with file checks, bounded revisions, stored snapshots, cancellation and two review gates.
- Static GLB/PNG inspectors, eight procedural geometry fixtures and six rendered previews.
- A new, neutral numeric JSON benchmark. The fixture mode is deterministic and does not call a model.
- Minimal UI for fixture workflows and an explicit JSON API console for configuration and inspection.

Text roles produce specifications; no image/mesh generator is connected. The critic receives text and measurements, not images. Human review is separate from technical checks. This is a single-user local prototype, not a credential vault, arbitrary-code sandbox or authenticated production service.

Runtime uploads and SQLite files are stored in `education/data/` and ignored by Git. Keep them private. Do not expose this server directly to the Internet.

## Test

```sh
python -m unittest discover -s education/scripts -p 'test_*.py'
```

Tests use temporary files and mock loopback HTTP servers, not paid services. The model builder is `education/scripts/build_pipeline_assets.py`; PNG previews were rendered from the included synthetic models.

## Included files

Only backend Python code/tests, a minimal new HTML/CSS/JavaScript client, procedural media fixtures, a neutral fixture registry and numeric benchmark, dependency declarations and this README are included. No old content archives or screenshots are distributed.

No explicit license grant is included. Contact the repository owner before redistribution or commercial reuse.
