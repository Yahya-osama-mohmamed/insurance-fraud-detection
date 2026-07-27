"""API tests: response shape + sanity of the risk ordering."""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.schemas import ClaimInput

client = TestClient(app)

# A deliberately suspicious claim: at-fault policy holder, sports car priced
# far above its category, accident days after policy start, address changed
# right before the claim, no police report, no witness.
FRAUD_LIKE = {
    "Month": "Nov", "WeekOfMonth": 4, "DayOfWeek": "Sunday", "Make": "Honda",
    "AccidentArea": "Rural", "DayOfWeekClaimed": "Monday", "MonthClaimed": "Dec",
    "WeekOfMonthClaimed": 1, "Sex": "Male", "MaritalStatus": "Single",
    "Age": 21, "Fault": "Policy Holder", "PolicyType": "Sport - All Perils",
    "VehicleCategory": "Sport", "VehiclePrice": "more than 69000",
    "RepNumber": 3, "Deductible": 300, "DriverRating": 4,
    "Days_Policy_Accident": "1 to 7", "Days_Policy_Claim": "8 to 15",
    "PastNumberOfClaims": "more than 4", "AgeOfVehicle": "new",
    "AgeOfPolicyHolder": "21 to 25", "PoliceReportFiled": "No",
    "WitnessPresent": "No", "AgentType": "Internal",
    "NumberOfSuppliments": "more than 5", "AddressChange_Claim": "under 6 months",
    "NumberOfCars": "1 vehicle", "Year": 1996, "BasePolicy": "All Perils",
}

# A boring, verifiable claim: long-standing policy, third-party fault,
# police report and witness, modest sedan, no history.
CLEAN_LIKE = {
    "Month": "Mar", "WeekOfMonth": 2, "DayOfWeek": "Tuesday", "Make": "Toyota",
    "AccidentArea": "Urban", "DayOfWeekClaimed": "Wednesday", "MonthClaimed": "Mar",
    "WeekOfMonthClaimed": 2, "Sex": "Female", "MaritalStatus": "Married",
    "Age": 45, "Fault": "Third Party", "PolicyType": "Sedan - Liability",
    "VehicleCategory": "Sedan", "VehiclePrice": "20000 to 29000",
    "RepNumber": 7, "Deductible": 400, "DriverRating": 1,
    "Days_Policy_Accident": "more than 30", "Days_Policy_Claim": "more than 30",
    "PastNumberOfClaims": "none", "AgeOfVehicle": "5 years",
    "AgeOfPolicyHolder": "41 to 50", "PoliceReportFiled": "Yes",
    "WitnessPresent": "Yes", "AgentType": "External",
    "NumberOfSuppliments": "none", "AddressChange_Claim": "no change",
    "NumberOfCars": "2 vehicles", "Year": 1995, "BasePolicy": "Liability",
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert "status" in r.json()


def test_predict_response_shape():
    r = client.post("/predict", json=FRAUD_LIKE)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"fraud_probability", "flagged", "threshold", "top_reasons"}
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert isinstance(body["flagged"], bool)
    assert isinstance(body["top_reasons"], list)


def test_validation_rejects_bad_categorical():
    bad = dict(FRAUD_LIKE, VehicleCategory="Spaceship")
    assert client.post("/predict", json=bad).status_code == 422


def test_risk_ordering():
    """An obviously suspicious claim must outscore an obviously clean one."""
    p_fraud = client.post("/predict", json=FRAUD_LIKE).json()["fraud_probability"]
    p_clean = client.post("/predict", json=CLEAN_LIKE).json()["fraud_probability"]
    assert p_fraud > p_clean


def test_queue_lists_flagged_claims():
    client.post("/predict", json=FRAUD_LIKE)
    r = client.get("/queue")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_derived_fields_ignore_contradictory_client_values():
    """Age bracket and PolicyType are recomputed server-side, so a client
    sending contradictory values gets the SAME score as a consistent one —
    the model can never see inputs outside the training distribution."""
    consistent = dict(CLEAN_LIKE, Age=22, AgeOfPolicyHolder="21 to 25",
                      PolicyType="Sedan - Liability")
    contradictory = dict(CLEAN_LIKE, Age=22, AgeOfPolicyHolder="over 65",
                         PolicyType="Sport - All Perils")
    p1 = client.post("/predict", json=consistent).json()["fraud_probability"]
    p2 = client.post("/predict", json=contradictory).json()["fraud_probability"]
    assert p1 == p2


def test_predict_works_without_derived_fields():
    """Clients (like the built-in UI) may omit the derived fields entirely."""
    payload = {k: v for k, v in CLEAN_LIKE.items()
               if k not in ("AgeOfPolicyHolder", "PolicyType")}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200


def test_age_omitted_requires_bracket():
    """If Age is unknown, the bracket must be supplied explicitly."""
    payload = dict(CLEAN_LIKE)
    payload.pop("AgeOfPolicyHolder")
    payload["Age"] = None
    assert client.post("/predict", json=payload).status_code == 422
