# Customer Churn Prediction Platform

<!-- project-guide:start -->
## Project guide

[Project architecture](PROJECT_ARCHITECTURE.md) · [Interview questions and answers](INTERVIEW_QA.md)

Use the architecture document for the component diagram, implementation boundaries, and verification entry points. The interview guide includes source-backed answers and project walkthroughs.

### Implementation map

| Component | Responsibility |
| --- | --- |
| [`src/churnml/main.py`](src/churnml/main.py) | HTTP handlers: `GET /healthz`, `GET /model`, `POST /score` |
| [`src/churnml/model.py`](src/churnml/model.py) | Functions: `rows`, `trained`, `score` |
| [`requirements.txt`](requirements.txt) | Implementation or supporting configuration |
| [`tests/test_churn.py`](tests/test_churn.py) | Executable checks and regression examples |
| [`.github/workflows/ci.yml`](.github/workflows/ci.yml) | GitHub Actions job definitions |
| [`README.md`](README.md) | Project explanations or operating notes |

### Local setup and verification

From the repository root (the commands follow the checked-in manifests):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

To serve the FastAPI application locally, install the server separately if it is not already available:

```bash
python -m pip install uvicorn
PYTHONPATH=src python -m uvicorn churnml.main:app --reload
```

<!-- project-guide:end -->

Phase 1

Skills: pandas-shaped rows, scikit-learn, evaluation, feature contributions

Raw rows → split → train GradientBoosting → metrics → explain. Not a notebook.

```bash
pip install -r requirements.txt
pytest -q
```

Laptop proof. No hosted model. Cluster apply stays false until a human approves.

## Ops plane

Workspaces, tenant isolation, job approval, and audit live under `/v1`. Production apply is refused. See `docs/ARCHITECTURE.md`.

## Documentation checks

Project architecture, interview guides, and local source links are checked automatically on pushes and pull requests. Run the same check locally:

```bash
python3 .github/scripts/validate_project_docs.py
```

See [service improvements and local run instructions](docs/UPGRADES.md).
