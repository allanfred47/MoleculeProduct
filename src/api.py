from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_OK = True
except Exception as e:
    RDKIT_OK = False
    print(f"RDKit not available: {e}")

# Load models
import joblib
import numpy as np

def load_model(filename):
    try:
        path = os.path.join(os.path.dirname(__file__), filename)
        if os.path.exists(path):
            m = joblib.load(path)
            print(f"Loaded {filename}")
            return m
        else:
            print(f"Not found: {path}")
            return None
    except Exception as e:
        print(f"Could not load {filename}: {e}")
        return None

TOXICITY_MODEL = load_model("toxicity_model.pkl")
VISCOSITY_MODEL = load_model("viscosity_model.pkl")
CO2_MODEL = load_model("co2_model.pkl")

app = FastAPI(
    title="MoleculeProduct API",
    description="Predict molecular properties, drug-likeness, toxicity, solubility, and viscosity from SMILES strings.",
    version="4.0.0"
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


def get_features(mw, logp, tpsa, hbd, hba, rot, ring_count, aromatic_rings, heavy_atom_count, fraction_csp3, molar_refractivity):
    return [mw, logp, tpsa, hbd, hba, rot, ring_count, aromatic_rings, heavy_atom_count, fraction_csp3, molar_refractivity]


def estimate_solubility(mw, logp, hbd, tpsa):
    log_s = 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * hbd - 0.74 * (tpsa / 100)
    log_s = round(log_s, 3)
    if log_s > 0: sol_class = "Highly Soluble"
    elif log_s > -1: sol_class = "Soluble"
    elif log_s > -2: sol_class = "Moderately Soluble"
    elif log_s > -4: sol_class = "Poorly Soluble"
    else: sol_class = "Insoluble"
    return log_s, sol_class


def estimate_boiling_point(mol, mw, hbd, hba):
    ring_count = rdMolDescriptors.CalcNumRings(mol)
    aromatic = rdMolDescriptors.CalcNumAromaticRings(mol)
    bp = 198.2 + 0.32 * mw + 12.5 * hbd + 5.5 * hba + 8.0 * ring_count + 10.0 * aromatic
    return round(bp, 1)


def estimate_melting_point(mw, ring_count, aromatic_rings, hbd):
    mp = -50.0 + 0.18 * mw + 20.0 * ring_count + 15.0 * aromatic_rings + 10.0 * hbd
    return round(mp, 1)


def estimate_density(mol, mw):
    heavy = mol.GetNumHeavyAtoms()
    if heavy == 0: return None
    molar_volume = 10.0 + 10.5 * heavy
    return round(mw / molar_volume, 3)


def get_pka_class(mol):
    for s in ["[OH][CX3](=O)", "[OH][SX4](=O)(=O)", "[OH][PX4](=O)"]:
        patt = Chem.MolFromSmarts(s)
        if patt and mol.HasSubstructMatch(patt): return "Acidic"
    for s in ["[NX3;H2,H1;!$(NC=O)]", "[NX3;H0;!$(NC=O)]", "n"]:
        patt = Chem.MolFromSmarts(s)
        if patt and mol.HasSubstructMatch(patt): return "Basic"
    return "Neutral"


def predict_toxicity(features):
    if TOXICITY_MODEL is None:
        return None, "Model not loaded"
    try:
        X = np.array([features])
        pred = TOXICITY_MODEL.predict(X)[0]
        proba = TOXICITY_MODEL.predict_proba(X)[0]
        tox_prob = round(float(proba[1]) * 100, 1)
        if pred == 1: tox_class = "Potentially Toxic"
        elif tox_prob > 30: tox_class = "Low Toxicity Risk"
        else: tox_class = "Non-toxic"
        return tox_prob, tox_class
    except Exception as e:
        print(f"Toxicity error: {e}")
        return None, "Prediction error"


def predict_viscosity(features):
    if VISCOSITY_MODEL is None:
        return None, "Model not loaded"
    try:
        X = np.array([features])
        log_visc = float(VISCOSITY_MODEL.predict(X)[0])
        viscosity = round(10 ** log_visc, 3)
        if viscosity < 0.5: visc_class = "Very Low"
        elif viscosity < 2.0: visc_class = "Low"
        elif viscosity < 10.0: visc_class = "Moderate"
        elif viscosity < 100.0: visc_class = "High"
        else: visc_class = "Very High"
        return viscosity, visc_class
    except Exception as e:
        print(f"Viscosity error: {e}")
        return None, "Prediction error"
    
def predict_co2_solubility(features):
    if CO2_MODEL is None:
        return None, "Model not loaded"
    try:
        X = np.array([features])
        co2 = round(float(CO2_MODEL.predict(X)[0]), 4)
        if co2 > 0.35: co2_class = "Very High"
        elif co2 > 0.28: co2_class = "High"
        elif co2 > 0.22: co2_class = "Moderate"
        elif co2 > 0.15: co2_class = "Low"
        else: co2_class = "Very Low"
        return co2, co2_class
    except Exception as e:
        print(f"CO2 error: {e}")
        return None, "Prediction error"


@app.get("/")
def root():
    return {
        "message": "MoleculeProduct API is running",
        "docs": "/docs",
        "rdkit_ok": RDKIT_OK,
        "toxicity_model": TOXICITY_MODEL is not None,
        "viscosity_model": VISCOSITY_MODEL is not None,
        "version": "4.0.0"
    }


@app.get("/lookup")
def lookup(name: str = Query(..., description="Molecule name to look up on PubChem")):
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
    if not RDKIT_OK:
        raise HTTPException(status_code=500, detail="RDKit is not available on this server")

    smiles = req.smiles.strip()
    if not smiles:
        raise HTTPException(status_code=422, detail="SMILES string cannot be empty")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise HTTPException(status_code=422, detail=f"Invalid SMILES string: '{smiles}'")

    mw = round(Descriptors.MolWt(mol), 3)
    logp = round(Descriptors.MolLogP(mol), 4)
    tpsa = round(Descriptors.TPSA(mol), 2)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    ring_count = rdMolDescriptors.CalcNumRings(mol)
    aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    heavy_atom_count = mol.GetNumHeavyAtoms()
    fraction_csp3 = round(rdMolDescriptors.CalcFractionCSP3(mol), 4)
    molar_refractivity = round(Descriptors.MolMR(mol), 3)

    violations = 0
    if mw > 500: violations += 1
    if logp > 5: violations += 1
    if hbd > 5: violations += 1
    if hba > 10: violations += 1
    drug_likeness_score = round(1 - (violations / 4), 2)
    drug_like = violations == 0

    predicted_logs, solubility_class = estimate_solubility(mw, logp, hbd, tpsa)
    boiling_point = estimate_boiling_point(mol, mw, hbd, hba)
    melting_point = estimate_melting_point(mw, ring_count, aromatic_rings, hbd)
    density = estimate_density(mol, mw)
    pka_class = get_pka_class(mol)
    stereocenters = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    formal_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())

    features = get_features(mw, logp, tpsa, hbd, hba, rot, ring_count, aromatic_rings, heavy_atom_count, fraction_csp3, molar_refractivity)
    toxicity_probability, toxicity_class = predict_toxicity(features)
    viscosity, viscosity_class = predict_viscosity(features)
    co2_solubility, co2_class = predict_co2_solubility(features)

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
        "solubility_class": solubility_class,
        "boiling_point": boiling_point,
        "melting_point": melting_point,
        "density": density,
        "pka_class": pka_class,
        "stereocenters": stereocenters,
        "formal_charge": formal_charge,
        "toxicity_probability": toxicity_probability,
        "toxicity_class": toxicity_class,
        "viscosity": viscosity,
        "viscosity_class": viscosity_class,
        "co2_solubility": co2_solubility,
        "co2_class": co2_class
    }