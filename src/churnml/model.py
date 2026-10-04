from functools import lru_cache
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

FEATURES = ["tenure_months", "monthly_charges", "support_tickets"]

def rows():
    data = []
    for i, (t, c, s, y) in enumerate([
        (1,95,5,1),(2,85,4,1),(3,90,6,1),(4,70,3,1),(5,99,2,1),(6,80,5,1),
        (12,40,0,0),(24,55,0,0),(36,40,0,0),(48,70,1,0),(30,50,2,0),(60,90,0,0),
        (8,88,4,1),(10,75,6,1),(18,45,1,0),(40,85,1,0),(27,35,0,0),(20,95,4,1),
        (50,60,0,0),(7,30,1,0),(15,80,5,1),(22,42,0,0),(9,91,5,1),(33,48,1,0),
    ]):
        data.append({"tenure_months": t, "monthly_charges": c, "support_tickets": s, "churned": y})
    return data

class InputError(ValueError):
    pass

@lru_cache(maxsize=1)
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

def score(body):
    for f in FEATURES:
        v = body.get(f)
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise InputError(f"{f} must be a number")
    model = trained()
    proba = float(model["clf"].predict_proba([[body[f] for f in FEATURES]])[0][1])
    parts = sorted(({"feature": n, "importance": i} for n, i in model["importances"].items()), key=lambda x: -x["importance"])
    return {"probability": round(proba, 4), "label": "high" if proba >= 0.5 else "low", "factors": parts, "metrics": model["metrics"]}
