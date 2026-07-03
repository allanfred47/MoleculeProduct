import requests as req
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors

app = FastAPI(
    title="MoleculeProduct API",
    description="""
A molecular property prediction API powered by RDKit and PubChem.

## Features
- Predict molecular properties from a SMILES string
- Look up any molecule by name using PubChem (100M+ compounds)
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
- **Fraction CSP3** — fraction of sp3 carbons (molecular complexity)
- **Drug-likeness Score** — estimated score based on Lipinski Rule of 5

## Lipinski Rule of 5
A molecule is considered drug-like if:
- Molecular Weight ≤ 500 Da
- LogP ≤ 5
- HBD ≤ 5
- HBA ≤ 10
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
    # Basic properties
    molecular_weight: float | None = None
    logp: float | None = None
    tpsa: float | None = None
    hbd: int | None = None
    hba: int | None = None
    rotatable_bonds: int | None = None
    # New descriptors
    ring_count: int | None = None
    aromatic_rings: int | None = None
    heavy_atom_count: int | None = None
    fraction_csp3: float | None = None
    molar_refractivity: float | None = None
    # Drug-likeness
    lipinski_violations: int | None = None
    drug_likeness_score: float | None = None
    drug_like: bool | None = None


def calculate_drug_likeness(mw, logp, hbd, hba):
    """Calculate a simple drug-likeness score from 0 to 1."""
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
        lipinski_violations=violations,
        drug_likeness_score=score,
        drug_like=violations == 0,
    )