from fastapi import FastAPI, HTTPException
from churnml.model import InputError, score, trained
app = FastAPI(title="Customer Churn Prediction Platform")

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/model")
def model():
    body = trained()
    return {"metrics": body["metrics"], "importances": body["importances"]}

@app.post("/score")
def post_score(body: dict):
    try:
        return score(body)
    except InputError as exc:
        raise HTTPException(422, str(exc)) from exc
