import requests as req
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors

app = FastAPI(title="Molecule Product API")

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


@app.get("/")
def root():
    return {"message": "API is running"}


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

    return MoleculeResponse(
        smiles=data.smiles,
        valid=True,
        molecular_weight=round(Descriptors.MolWt(mol), 3),
        logp=round(Crippen.MolLogP(mol), 4),
        tpsa=round(rdMolDescriptors.CalcTPSA(mol), 2),
        hbd=Lipinski.NumHDonors(mol),
        hba=Lipinski.NumHAcceptors(mol),
        rotatable_bonds=Lipinski.NumRotatableBonds(mol),
    )