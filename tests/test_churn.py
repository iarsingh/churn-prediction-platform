from fastapi.testclient import TestClient
from churnml.main import app
client = TestClient(app)

def test_risky_vs_loyal_and_metrics():
    high = client.post("/score", json={"tenure_months":1,"monthly_charges":95,"support_tickets":6}).json()
    low = client.post("/score", json={"tenure_months":36,"monthly_charges":40,"support_tickets":0}).json()
    assert high["probability"] > low["probability"]
    assert client.get("/model").json()["metrics"]["accuracy"] >= 0.5
    assert client.post("/score", json={"tenure_months":1}).status_code == 422
