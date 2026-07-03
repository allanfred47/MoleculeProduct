from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import requests
import os
import traceback

# --- RDKit ---
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_OK = True
except Exception as e:
    RDKIT_OK = False
    print(f"RDKit not available: {e}")

# --- ML Models ---
TOXICITY_MODEL   = None
TOXICITY2_MODELS = None
VISCOSITY_MODEL  = None
CO2_MODEL        = None

try:
    import joblib
    import numpy as np

    base = os.path.dirname(__file__)

    tox_path = os.path.join(base, "toxicity_model.pkl")
    if os.path.exists(tox_path):
        TOXICITY_MODEL = joblib.load(tox_path)
        print("Toxicity model loaded")

    tox2_path = os.path.join(base, "toxicity2_models.pkl")
    if os.path.exists(tox2_path):
        TOXICITY2_MODELS = joblib.load(tox2_path)
        print("Toxicity2 (12-endpoint) model loaded")

    visc_path = os.path.join(base, "viscosity_model.pkl")
    if os.path.exists(visc_path):
        VISCOSITY_MODEL = joblib.load(visc_path)
        print("Viscosity model loaded")

    co2_path = os.path.join(base, "co2_model.pkl")
    if os.path.exists(co2_path):
        CO2_MODEL = joblib.load(co2_path)
        print("CO2 model loaded")

except Exception as e:
    print(f"Model loading error: {e}")
    traceback.print_exc()


app = FastAPI(
    title="MoleculeProduct API",
    description="Predict molecular properties and drug-likeness from SMILES strings.",
    version="9.0.0"
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
    model_config = {"json_schema_extra": {"examples": [{"smiles": "CCO"}]}}


# ─── Serve the UI HTML ────────────────────────────────────────────────────────
UI_PATH = os.path.join(os.path.dirname(__file__), "..", "molecule_ui.html")

@app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
@app.get("/app", response_class=HTMLResponse, include_in_schema=False)
def serve_ui():
    """Serve the MoleculeProduct web interface."""
    try:
        with open(UI_PATH, "r", encoding="utf-8") as f:
            html = f.read()
        return HTMLResponse(content=html)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>UI not found</h1><p>molecule_ui.html missing.</p>", status_code=404)

# ─── Serve the Designer HTML ──────────────────────────────────────────────────
DESIGNER_PATH = os.path.join(os.path.dirname(__file__), "..", "molecule_designer.html")

@app.get("/designer", response_class=HTMLResponse, include_in_schema=False)
@app.get("/design", response_class=HTMLResponse, include_in_schema=False)
def serve_designer():
    """Serve the Generative Molecular Designer interface."""
    try:
        with open(DESIGNER_PATH, "r", encoding="utf-8") as f:
            html = f.read()
        return HTMLResponse(content=html)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Designer not found</h1><p>molecule_designer.html missing.</p>", status_code=404)

# ─── Serve the Process Optimizer HTML ─────────────────────────────────────────
PROCESS_PATH = os.path.join(os.path.dirname(__file__), "..", "process_optimizer.html")

@app.get("/process", response_class=HTMLResponse, include_in_schema=False)
@app.get("/optimizer", response_class=HTMLResponse, include_in_schema=False)
def serve_process():
    """Serve the Industrial Process Optimizer interface."""
    try:
        with open(PROCESS_PATH, "r", encoding="utf-8") as f:
            html = f.read()
        return HTMLResponse(content=html)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Process Optimizer not found</h1><p>process_optimizer.html missing.</p>", status_code=404)


# ─── ESOL Solubility ──────────────────────────────────────────────────────────
def esol_solubility(mol):
    try:
        mw   = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        rb   = rdMolDescriptors.CalcNumRotatableBonds(mol)
        ap   = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
        log_s = 0.16 - 0.63*logp - 0.0062*mw + 0.066*rb - 0.74*ap
        if log_s > 0:    cls = "Highly Soluble"
        elif log_s > -1: cls = "Soluble"
        elif log_s > -2: cls = "Moderately Soluble"
        elif log_s > -4: cls = "Poorly Soluble"
        else:            cls = "Insoluble"
        return round(log_s, 3), cls
    except:
        return None, "Error"


def get_toxicity_features(mol):
    try:
        return [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            rdMolDescriptors.CalcTPSA(mol),
            rdMolDescriptors.CalcNumHBD(mol),
            rdMolDescriptors.CalcNumHBA(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol),
            rdMolDescriptors.CalcNumRings(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            mol.GetNumHeavyAtoms(),
            Descriptors.FractionCSP3(mol),
            Descriptors.MolMR(mol),
        ]
    except:
        return None


def get_extended_features(mol):
    return get_toxicity_features(mol)


def estimate_bp_mp(mol):
    try:
        mw    = Descriptors.MolWt(mol)
        logp  = Descriptors.MolLogP(mol)
        tpsa  = rdMolDescriptors.CalcTPSA(mol)
        hbd   = rdMolDescriptors.CalcNumHBD(mol)
        rings = rdMolDescriptors.CalcNumRings(mol)
        bp = 80  + 0.5*mw + 10*logp + 0.3*tpsa + 15*hbd + 20*rings
        mp = -50 + 0.35*mw + 8*logp + 0.2*tpsa + 10*hbd + 15*rings
        return round(bp, 1), round(mp, 1)
    except:
        return None, None


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service":           "MoleculeProduct API",
        "version":           "9.0.0",
        "ui":                "/ui",
        "designer":          "/designer",
        "process":           "/process",
        "rdkit":             RDKIT_OK,
        "toxicity2_loaded":  TOXICITY2_MODELS is not None,
        "viscosity_loaded":  VISCOSITY_MODEL is not None,
        "co2_loaded":        CO2_MODEL is not None,
        "endpoints":         ["/ui", "/designer", "/process", "/predict", "/structure", "/lookup", "/docs"]
    }


@app.get("/lookup")
def lookup(name: str = Query(..., description="Common molecule name")):
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{requests.utils.quote(name)}/property/IsomericSMILES,CanonicalSMILES/JSON"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            raise HTTPException(status_code=404, detail=f"Molecule '{name}' not found on PubChem")
        props = r.json()["PropertyTable"]["Properties"][0]
        smiles = props.get("IsomericSMILES") or props.get("CanonicalSMILES")
        if not smiles:
            raise HTTPException(status_code=404, detail="No SMILES found")
        return {"name": name, "smiles": smiles}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/structure")
def get_structure(smiles: str = Query(...), size: int = 400):
    if not RDKIT_OK:
        raise HTTPException(status_code=503, detail="RDKit not available")
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise HTTPException(status_code=400, detail="Invalid SMILES")
        from rdkit.Chem import AllChem
        AllChem.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DSVG(size, size)
        drawer.drawOptions().addStereoAnnotation = True
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return Response(content=drawer.GetDrawingText(), media_type="image/svg+xml")
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
def predict(req: PredictRequest):
    if not RDKIT_OK:
        raise HTTPException(status_code=503, detail="RDKit not available")

    mol = Chem.MolFromSmiles(req.smiles)
    if mol is None:
        return {"valid": False, "smiles": req.smiles}

    mw    = round(Descriptors.MolWt(mol), 3)
    logp  = round(Descriptors.MolLogP(mol), 4)
    tpsa  = round(rdMolDescriptors.CalcTPSA(mol), 2)
    hbd   = rdMolDescriptors.CalcNumHBD(mol)
    hba   = rdMolDescriptors.CalcNumHBA(mol)
    rb    = rdMolDescriptors.CalcNumRotatableBonds(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)
    arom  = rdMolDescriptors.CalcNumAromaticRings(mol)
    hac   = mol.GetNumHeavyAtoms()
    fcsp3 = round(Descriptors.FractionCSP3(mol), 4)
    mr    = round(Descriptors.MolMR(mol), 3)

    viol     = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    dl_score = max(0.0, round(1.0 - viol*0.25, 2))
    drug_like = viol == 0

    log_s, sol_class = esol_solubility(mol)
    bp, mp = estimate_bp_mp(mol)
    density = round(mw / (mr*0.6 + 10), 3) if mr > 0 else None

    if hbd >= 2 and tpsa > 60:   pka_class = "Acidic"
    elif hba >= 2 and logp < 2:  pka_class = "Basic"
    else:                         pka_class = "Neutral"

    stereocenters = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    formal_charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())

    tox_prob, tox_class = None, None
    if TOXICITY_MODEL is not None:
        feats = get_toxicity_features(mol)
        if feats:
            try:
                X = np.array([feats])
                tox_prob  = round(float(TOXICITY_MODEL.predict_proba(X)[0][1]), 4)
                tox_class = "Toxic" if tox_prob > 0.5 else "Non-Toxic"
            except Exception as e:
                print(f"Toxicity error: {e}")

    toxicity_endpoints = []
    if TOXICITY2_MODELS is not None:
        feats = get_toxicity_features(mol)
        if feats:
            try:
                X = np.array([feats])
                for ep in TOXICITY2_MODELS["columns"]:
                    m = TOXICITY2_MODELS["models"].get(ep)
                    if m:
                        prob = round(float(m.predict_proba(X)[0][1]), 4)
                        toxicity_endpoints.append({"endpoint": ep, "probability": prob})
            except Exception as e:
                print(f"Toxicity2 error: {e}")

    viscosity, viscosity_class = None, None
    if VISCOSITY_MODEL is not None:
        feats = get_extended_features(mol)
        if feats:
            try:
                log_v = float(VISCOSITY_MODEL.predict(np.array([feats]))[0])
                viscosity = round(10**log_v, 3)
                if viscosity < 1:     viscosity_class = "Very Low"
                elif viscosity < 5:   viscosity_class = "Low"
                elif viscosity < 50:  viscosity_class = "Medium"
                elif viscosity < 500: viscosity_class = "High"
                else:                 viscosity_class = "Very High"
            except Exception as e:
                print(f"Viscosity error: {e}")

    co2_sol, co2_class = None, None
    if CO2_MODEL is not None:
        feats = get_extended_features(mol)
        if feats:
            try:
                co2_sol = round(float(CO2_MODEL.predict(np.array([feats]))[0]), 4)
                if co2_sol < 0.001:  co2_class = "Negligible"
                elif co2_sol < 0.01: co2_class = "Very Low"
                elif co2_sol < 0.05: co2_class = "Low"
                elif co2_sol < 0.2:  co2_class = "Moderate"
                else:                co2_class = "High"
            except Exception as e:
                print(f"CO2 error: {e}")

    return {
        "valid": True, "smiles": req.smiles,
        "molecular_weight": mw, "logp": logp, "tpsa": tpsa,
        "hbd": hbd, "hba": hba, "rotatable_bonds": rb,
        "ring_count": rings, "aromatic_rings": arom,
        "heavy_atom_count": hac, "fraction_csp3": fcsp3,
        "molar_refractivity": mr,
        "lipinski_violations": viol, "drug_likeness_score": dl_score, "drug_like": drug_like,
        "predicted_logs": log_s, "solubility_class": sol_class,
        "boiling_point": bp, "melting_point": mp, "density": density,
        "pka_class": pka_class, "stereocenters": stereocenters, "formal_charge": formal_charge,
        "toxicity_probability": tox_prob, "toxicity_class": tox_class,
        "toxicity_endpoints": toxicity_endpoints,
        "viscosity": viscosity, "viscosity_class": viscosity_class,
        "co2_solubility": co2_sol, "co2_class": co2_class,
    }