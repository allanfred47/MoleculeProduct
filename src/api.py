from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
import traceback

# --- RDKit ---
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_OK = True
except Exception as e:
    RDKIT_OK = False
    print(f"RDKit not available: {e}")

# --- ML Model ---
MODEL = None
try:
    import joblib
    import numpy as np
    model_path = os.path.join(os.path.dirname(__file__), "solubility_model.pkl")
    if os.path.exists(model_path):
        MODEL = joblib.load(model_path)
        print(f"Solubility model loaded from {model_path}")
    else:
        print(f"Model file not found at {model_path}")
except Exception as e:
    print(f"Could not load solubility model: {e}")
    traceback.print_exc()

app = FastAPI(
    title="MoleculeProduct API",
    description="Predict molecular properties and drug-likeness from SMILES strings.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    smiles: str

    model_config = {
        "json_schema_extra": {
            "examples": [{"smiles": "CCO"}]
        }
    }


def predict_solubility(features: list):
    """Run ML model to predict logS and solubility class."""
    if MODEL is None:
        return None, "Model not loaded"
    try:
        import numpy as np
        X = np.array([features])
        log_s = float(MODEL.predict(X)[0])
        if log_s > 0:
            sol_class = "Highly Soluble"
        elif log_s > -1:
            sol_class = "Soluble"
        elif log_s > -2:
            sol_class = "Moderately Soluble"
        elif log_s > -4:
            sol_class = "Poorly Soluble"
        else:
            sol_class = "Insoluble"
        return round(log_s, 3), sol_class
    except Exception as e:
        print(f"Solubility prediction error: {e}")
        traceback.print_exc()
        return None, "Prediction error"


@app.get("/")
def root():
    return {
        "message": "MoleculeProduct API is running",
        "docs": "/docs",
        "model_loaded": MODEL is not None,
        "rdkit_ok": RDKIT_OK
    }


@app.get("/lookup")
def lookup(name: str = Query(..., description="Molecule name to look up on PubChem")):
    """Look up a molecule by name and return its SMILES string from PubChem."""
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{requests.utils.quote(name)}/property/IsomericSMILES,CanonicalSMILES/JSON"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            raise HTTPException(status_code=404, detail=f"Molecule '{name}' not found on PubChem")
        data = resp.json()
        props = data["PropertyTable"]["Properties"][0]
        smiles = props.get("IsomericSMILES") or props.get("CanonicalSMILES") or props.get("SMILES")
        if not smiles:
            raise HTTPException(status_code=404, detail=f"No SMILES found for '{name}'")
        return {"name": name, "smiles": smiles}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
def predict(req: PredictRequest):
    """Predict molecular properties from a SMILES string."""
    if not RDKIT_OK:
        raise HTTPException(status_code=500, detail="RDKit is not available on this server")

    smiles = req.smiles.strip()
    if not smiles:
        raise HTTPException(status_code=422, detail="SMILES string cannot be empty")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise HTTPException(status_code=422, detail=f"Invalid SMILES string: '{smiles}'")

    # Basic descriptors
    mw = round(Descriptors.MolWt(mol), 3)
    logp = round(Descriptors.MolLogP(mol), 4)
    tpsa = round(Descriptors.TPSA(mol), 2)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    rot = rdMolDescriptors.CalcNumRotatableBonds(mol)

    # Extended descriptors
    ring_count = rdMolDescriptors.CalcNumRings(mol)
    aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    heavy_atom_count = mol.GetNumHeavyAtoms()
    fraction_csp3 = round(rdMolDescriptors.CalcFractionCSP3(mol), 4)
    molar_refractivity = round(Descriptors.MolMR(mol), 3)

    # Lipinski violations
    violations = 0
    if mw > 500: violations += 1
    if logp > 5: violations += 1
    if hbd > 5: violations += 1
    if hba > 10: violations += 1
    drug_likeness_score = round(1 - (violations / 4), 2)
    drug_like = violations == 0

    # ML solubility prediction
    features = [mw, logp, tpsa, hbd, hba, rot, ring_count, aromatic_rings,
                heavy_atom_count, fraction_csp3, molar_refractivity]
    predicted_logs, solubility_class = predict_solubility(features)

    return {
        "smiles": smiles,
        "molecular_weight": mw,
        "logp": logp,
        "tpsa": tpsa,
        "hbd": hbd,
        "hba": hba,
        "rotatable_bonds": rot,
        "ring_count": ring_count,
        "aromatic_rings": aromatic_rings,
        "heavy_atom_count": heavy_atom_count,
        "fraction_csp3": fraction_csp3,
        "molar_refractivity": molar_refractivity,
        "lipinski_violations": violations,
        "drug_likeness_score": drug_likeness_score,
        "drug_like": drug_like,
        "predicted_logs": predicted_logs,
        "solubility_class": solubility_class
    }