# churn-prediction-platform — interview questions and answers

[README](README.md) · [Project architecture](PROJECT_ARCHITECTURE.md)

Answers below use this repository’s files and implementation. They distinguish existing behavior from suggested extensions; source links let you verify each walkthrough.

## 1. What problem does churn-prediction-platform address, and what can you demonstrate?

Raw rows → split → train GradientBoosting → metrics → explain. Not a notebook.

I would demonstrate the linked implementation or examples and distinguish that evidence from any planned production features. Start with [`README.md`](README.md).

## 2. How is this repository organized?

- [`src/churnml/main.py`](src/churnml/main.py): Implementation or supporting configuration.
- [`src/churnml/model.py`](src/churnml/model.py): Implementation or supporting configuration.
- [`requirements.txt`](requirements.txt): Implementation or supporting configuration.
- [`tests/test_churn.py`](tests/test_churn.py): Executable checks and regression examples.
- [`.github/workflows/ci.yml`](.github/workflows/ci.yml): GitHub Actions job definitions.
- [`README.md`](README.md): Project explanations or operating notes.

[PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md) contains the component diagram and the implementation walkthrough.

## 3. Can you walk through `trained` and explain the decision it makes?

The main walkthrough here is `trained()` in [`src/churnml/model.py`](src/churnml/model.py#L23).

```python
def trained():
    data = rows()
    X = [[d[f] for f in FEATURES] for d in data]
    y = [d["churned"] for d in data]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=7)
    clf = GradientBoostingClassifier(random_state=7)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    return {
        "clf": clf,
        "metrics": {
            "accuracy": round(float(accuracy_score(yte, pred)), 4),
            "precision": round(float(precision_score(yte, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(yte, pred, zero_division=0)), 4),
        },
        "importances": {name: round(float(v), 4) for name, v in zip(FEATURES, clf.feature_importances_)},
    }
```

The implementation calls `GradientBoostingClassifier`, `accuracy_score`, `clf.fit`, `clf.predict`, `float`, `lru_cache`, `precision_score`, `recall_score`, `round`. In an interview, trace those calls in execution order using a fixture input.

## 4. What responsibility does `rows` have?

`rows()` is defined in [`src/churnml/model.py`](src/churnml/model.py#L8).

Its return expressions include:

- `data`

It uses `data.append`, `enumerate`. This is the code path I would compare against the caller to explain responsibility boundaries.

## 5. What input validation and failure behavior are implemented?

Explicit failure paths include:

- `HTTPException(422, str(exc))` in [`src/churnml/main.py`](src/churnml/main.py#L19).
- `InputError(f'{f} must be a number')` in [`src/churnml/model.py`](src/churnml/model.py#L45).

I would test both the condition that reaches each exception and the caller that translates it. An explicit raise does not mean every malformed input or dependency failure is handled.

## 6. Which test would you use to demonstrate correctness?

[`tests/test_churn.py`](tests/test_churn.py#L5) contains `test_risky_vs_loyal_and_metrics`:

```python
def test_risky_vs_loyal_and_metrics():
    high = client.post("/score", json={"tenure_months":1,"monthly_charges":95,"support_tickets":6}).json()
    low = client.post("/score", json={"tenure_months":36,"monthly_charges":40,"support_tickets":0}).json()
    assert high["probability"] > low["probability"]
    assert client.get("/model").json()["metrics"]["accuracy"] >= 0.5
    assert client.post("/score", json={"tenure_months":1}).status_code == 422
```

This is a concrete regression example from the repository. Its assertions establish that case; they do not establish behavior for every input or under production load.

## 7. What HTTP interface does the code expose?

- `GET /healthz` → `healthz` in [`src/churnml/main.py`](src/churnml/main.py#L6).
- `GET /model` → `model` in [`src/churnml/main.py`](src/churnml/main.py#L10).
- `POST /score` → `post_score` in [`src/churnml/main.py`](src/churnml/main.py#L15).

These are literal decorators. Application/router prefixes, authentication, and middleware must be checked in the corresponding setup code.

## 8. Where does state live, and what happens with multiple workers?

Module-level containers include `FEATURES` in [`src/churnml/model.py`](src/churnml/model.py).

These containers belong to a Python process. Inspect which are constant fixtures and which are mutated. Mutable process state needs an explicit shared-storage or synchronization strategy before multiple workers can provide consistent behavior.

## 9. How would another engineer reproduce your walkthrough?

Start from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

These commands follow repository manifests; environment setup and command results still need to be checked on the target machine.

## 10. What does automation verify, and what does it not prove?

Inspect [`.github/workflows/ci.yml`](.github/workflows/ci.yml) for triggers, permissions, and job commands. I would name the checks that those definitions run and show the latest run separately. A workflow definition alone does not establish a successful deployment, security review, or production SLO.

## 11. How would you present this project in a Forward Deployed Engineer interview?

Start with the user and operational problem described in [`README.md`](README.md). Explain one constraint that changes the implementation, show the linked code or example, and walk through a success case and a failure case. Agree on a measurable acceptance criterion before expanding the solution, and leave a handoff with data boundaries and rollback ownership. Any proposed production or business metric should be identified as a target until measured.

## 12. What is the input-to-output contract of `trained`?

In [`src/churnml/model.py`](src/churnml/model.py#L23), `trained()` receives the inputs. The function computes these intermediate values:

- `data = rows()`
- `X = [[d[f] for f in FEATURES] for d in data]`
- `y = [d['churned'] for d in data]`
- `Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=7)`
- `clf = GradientBoostingClassifier(random_state=7)`
- `pred = clf.predict(Xte)`

Its result is defined by:

- `{'clf': clf, 'metrics': {'accuracy': round(float(accuracy_score(yte, pred)), 4), 'precision': round(float(precision_score(yte, pred, zero_division=0)), 4), 'recall': round(float(recall_score(yte, pred, zero_division=0)), 4)}, 'importances': {name: round(float(v), 4) for name, v in zip(FEATURES, clf.feature_importances_)}}`
