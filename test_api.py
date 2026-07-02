import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


# â”€â”€ Root endpoint â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API is running"}


# â”€â”€ Valid SMILES â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_valid_smiles_ethanol():
    response = client.post("/predict", json={"smiles": "CCO"})
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is True
    assert data["smiles"] == "CCO"
    assert data["molecular_weight"] > 0
    assert data["logp"] is not None
    assert data["tpsa"] is not None
    assert data["hbd"] is not None
    assert data["hba"] is not None
    assert data["rotatable_bonds"] is not None


def test_valid_smiles_aspirin():
    response = client.post("/predict", json={"smiles": "CC(=O)Oc1ccccc1C(=O)O"})
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is True
    assert data["molecular_weight"] == pytest.approx(180.159, abs=0.01)


def test_valid_smiles_caffeine():
    response = client.post("/predict", json={"smiles": "Cn1cnc2c1c(=O)n(c(=O)n2C)C"})
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is True
    assert data["hbd"] == 0          # caffeine has no H-bond donors


# â”€â”€ Invalid SMILES â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_invalid_smiles_returns_valid_false():
    response = client.post("/predict", json={"smiles": "NOT_A_SMILES"})
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is False
    assert data["molecular_weight"] is None
    assert data["logp"] is None
    assert data["tpsa"] is None
    assert data["hbd"] is None
    assert data["hba"] is None
    assert data["rotatable_bonds"] is None


def test_invalid_smiles_empty_string():
    response = client.post("/predict", json={"smiles": ""})
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is False


# â”€â”€ Missing field â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_missing_smiles_field_returns_422():
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_wrong_field_name_returns_422():
    response = client.post("/predict", json={"molecule": "CCO"})
    assert response.status_code == 422


# â”€â”€ Response shape â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_response_contains_all_fields():
    response = client.post("/predict", json={"smiles": "CCO"})
    data = response.json()
    expected_keys = {
        "smiles", "valid", "molecular_weight",
        "logp", "tpsa", "hbd", "hba", "rotatable_bonds"
    }
    assert expected_keys.issubset(data.keys())