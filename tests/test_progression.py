from churnml.progression import ds_workflow

def test_ds_workflow_stages():
    rows = [{"tenure_months": 1, "monthly_charges": 90, "support_tickets": 4, "churned": 1}]
    out = ds_workflow(rows)
    assert out["notebook_only"] is False
    assert "explain" in out["stages"]

