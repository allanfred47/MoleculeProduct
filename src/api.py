import joblib
import numpy as np
import requests as req
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors

# Load the trained solubility model
import os
MODEL_PATH = os.path.join(os.path.dirname(__file__), "solubility_model.pkl")
solubility_model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="MoleculeProduct API",
    description="""
A molecular property prediction API powered by RDKit, PubChem, and Machine Learning.

## Features
- Predict molecular properties from a SMILES string
- Look up any molecule by name using PubChem (100M+ compounds)
- Predict aqueous solubility using a trained Random Forest model
- Returns Lipinski Rule of 5 properties for drug-likeness assessment

## Properties Returned
- **Molecular Weight** — mass of the molecule in Daltons
- **LogP** — measure of lipophilicity (fat solubility)
- **TPSA** — topological polar surface area in Å²
- **HBD** — number of hydrogen bond donors
- **HBA** — number of hydrogen bond acceptors
- **Rotatable Bonds** — measure of molecular flexibility
- **Ring Count** — total number of rings in the molecule
- **Aromatic Rings** — number of aromatic rings
- **Heavy Atom Count** — number of non-hydrogen atoms
- **Fraction CSP3** — fraction of sp3 carbons
- **Predicted LogS** — machine learning predicted aqueous solubility
- **Drug-likeness Score** — estimated score based on Lipinski Rule of 5
    """,
    version="2.0.0",
    contact={
        "name": "MoleculeProduct",
        "url": "https://moleculeproduct.onrender.com",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SmilesInput(BaseModel):
    smiles: str = Field(..., json_schema_extra={"example": "CCO"})


class MoleculeResponse(BaseModel):
    smiles: str
    valid: bool
    molecular_weight: float | None = None
    logp: float | None = None
    tpsa: float | None = None
    hbd: int | None = None
    hba: int | None = None
    rotatable_bonds: int | None = None
    ring_count: int | None = None
    aromatic_rings: int | None = None
    heavy_atom_count: int | None = None
    fraction_csp3: float | None = None
    molar_refractivity: float | None = None
    predicted_logs: float | None = None
    solubility_class: str | None = None
    lipinski_violations: int | None = None
    drug_likeness_score: float | None = None
    drug_like: bool | None = None


def get_features(mol):
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=256)
    mw   = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd  = rdMolDescriptors.CalcNumHBD(mol)
    hba  = rdMolDescriptors.CalcNumHBA(mol)
    rb   = rdMolDescriptors.CalcNumRotatableBonds(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)
    return np.array(list(fp) + [mw, logp, tpsa, hbd, hba, rb, rings]).reshape(1, -1)


def solubility_label(logs):
    if logs > 0:    return "Highly Soluble"
    if logs > -1:   return "Soluble"
    if logs > -2:   return "Moderately Soluble"
    if logs > -4:   return "Poorly Soluble"
    return "Insoluble"


def calculate_drug_likeness(mw, logp, hbd, hba):
    violations = 0
    if mw > 500:  violations += 1
    if logp > 5:  violations += 1
    if hbd > 5:   violations += 1
    if hba > 10:  violations += 1
    score = round(1 - (violations / 4), 2)
    return violations, score


@app.get("/")
def root():
    return {"message": "MoleculeProduct API v2.0 is running"}


@app.get("/lookup")
def lookup(name: str):
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/IsomericSMILES/JSON"
        r = req.get(url, timeout=10)
        if r.status_code != 200:
            return {"smiles": None}
        data = r.json()
        smiles = data["PropertyTable"]["Properties"][0].get("IsomericSMILES") or data["PropertyTable"]["Properties"][0].get("SMILES")
        return {"smiles": smiles}
    except:
        return {"smiles": None}


@app.post("/predict", response_model=MoleculeResponse)
def predict(data: SmilesInput):
    if not data.smiles.strip():
        return MoleculeResponse(smiles=data.smiles, valid=False)

    mol = Chem.MolFromSmiles(data.smiles)
    if mol is None:
        return MoleculeResponse(smiles=data.smiles, valid=False)

    mw   = round(Descriptors.MolWt(mol), 3)
    logp = round(Crippen.MolLogP(mol), 4)
    tpsa = round(rdMolDescriptors.CalcTPSA(mol), 2)
    hbd  = Lipinski.NumHDonors(mol)
    hba  = Lipinski.NumHAcceptors(mol)
    rb   = Lipinski.NumRotatableBonds(mol)
    ring_count     = rdMolDescriptors.CalcNumRings(mol)
    aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    heavy_atoms    = mol.GetNumHeavyAtoms()
    frac_csp3      = round(rdMolDescriptors.CalcFractionCSP3(mol), 4)
    mr             = round(Crippen.MolMR(mol), 3)

    # ML solubility prediction
    features = get_features(mol)
    predicted_logs = round(float(solubility_model.predict(features)[0]), 3)
    sol_class = solubility_label(predicted_logs)

    violations, score = calculate_drug_likeness(mw, logp, hbd, hba)

    return MoleculeResponse(
        smiles=data.smiles,
        valid=True,
        molecular_weight=mw,
        logp=logp,
        tpsa=tpsa,
        hbd=hbd,
        hba=hba,
        rotatable_bonds=rb,
        ring_count=ring_count,
        aromatic_rings=aromatic_rings,
        heavy_atom_count=heavy_atoms,
        fraction_csp3=frac_csp3,
        molar_refractivity=mr,
        predicted_logs=predicted_logs,
        solubility_class=sol_class,
        lipinski_violations=violations,
        drug_likeness_score=score,
        drug_like=violations == 0,
    )