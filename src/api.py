from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import requests
import os
import re
import traceback
import urllib.parse

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
    description="Neagenics — Molecular property prediction, process optimization, formulation generation, and literature extraction.",
    version="11.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Pydantic Models ──────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    smiles: str
    model_config = {"json_schema_extra": {"examples": [{"smiles": "CCO"}]}}

class ProcessRequest(BaseModel):
    reaction_type: str = "general"
    temperature: float = 25.0
    pressure: float = 1.0
    catalyst: str = "none"
    residence_time: float = 60.0
    reactor_type: str = "batch"

class FormulateRequest(BaseModel):
    product_type: str = "anti_aging_cream"
    skin_type: str = "normal"
    philosophy: List[str] = []
    claims: List[str] = []

class ExtractRequest(BaseModel):
    text: str
    options: Optional[List[str]] = None
    filename: Optional[str] = None


# ─── HTML File Paths ──────────────────────────────────────────────────────────

ROOT = os.path.join(os.path.dirname(__file__), "..")

UI_PATH          = os.path.join(ROOT, "molecule_ui.html")
DESIGNER_PATH    = os.path.join(ROOT, "molecule_designer.html")
PROCESS_PATH     = os.path.join(ROOT, "process_optimizer.html")
FORMULATION_PATH = os.path.join(ROOT, "formulation_generator.html")
LITERATURE_PATH  = os.path.join(ROOT, "literature_extractor.html")


def serve_html(path: str, label: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content=f"<h1>{label} not found</h1>", status_code=404)


# ─── HTML Routes ──────────────────────────────────────────────────────────────

@app.get("/ui",          response_class=HTMLResponse, include_in_schema=False)
@app.get("/app",         response_class=HTMLResponse, include_in_schema=False)
def serve_ui():
    return serve_html(UI_PATH, "Molecular Property Predictor")

@app.get("/designer",    response_class=HTMLResponse, include_in_schema=False)
@app.get("/design",      response_class=HTMLResponse, include_in_schema=False)
def serve_designer():
    return serve_html(DESIGNER_PATH, "Generative Molecular Designer")

@app.get("/process",     response_class=HTMLResponse, include_in_schema=False)
@app.get("/optimizer",   response_class=HTMLResponse, include_in_schema=False)
def serve_process():
    return serve_html(PROCESS_PATH, "Process Optimizer")

@app.get("/formulation", response_class=HTMLResponse, include_in_schema=False)
@app.get("/formulator",  response_class=HTMLResponse, include_in_schema=False)
def serve_formulation():
    return serve_html(FORMULATION_PATH, "Formulation Generator")

@app.get("/literature",  response_class=HTMLResponse, include_in_schema=False)
@app.get("/extractor",   response_class=HTMLResponse, include_in_schema=False)
def serve_literature():
    return serve_html(LITERATURE_PATH, "Literature Knowledge Extractor")


# ─── Root ─────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "MoleculeProduct API",
        "version": "11.0.0",
        "brand":   "Neagenics",
        "modules": {
            "1_predictor":  "/ui",
            "2_designer":   "/designer",
            "3_optimizer":  "/process",
            "4_formulator": "/formulation",
            "5_extractor":  "/literature",
        },
        "rdkit":            RDKIT_OK,
        "toxicity2_loaded": TOXICITY2_MODELS is not None,
        "viscosity_loaded": VISCOSITY_MODEL is not None,
        "co2_loaded":       CO2_MODEL is not None,
        "endpoints": [
            "/ui", "/designer", "/process", "/formulation", "/literature",
            "/predict", "/structure", "/lookup",
            "/process-calc", "/formulate", "/extract", "/extract-pdf",
            "/docs"
        ]
    }


# ─── Chemistry Helpers ────────────────────────────────────────────────────────

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


# ─── /lookup ──────────────────────────────────────────────────────────────────

@app.get("/lookup")
def lookup(name: str = Query(..., description="Common molecule name e.g. benzene, aspirin")):
    """Look up a molecule by common name and return its SMILES from PubChem."""
    try:
        encoded = urllib.parse.quote(name)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/property/IsomericSMILES,CanonicalSMILES/JSON"
        r = requests.get(url, timeout=12)
    
        if not r.ok:
            return {
                "status": r.status_code,
                "url": url,
                "response": r.text[:500]
    }
        props = r.json()["PropertyTable"]["Properties"][0]
        smiles = props.get("IsomericSMILES") or props.get("CanonicalSMILES")
        if not smiles:
            raise HTTPException(status_code=404, detail="No SMILES found")
        return {"name": name, "smiles": smiles}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── /structure ───────────────────────────────────────────────────────────────

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


# ─── /predict ─────────────────────────────────────────────────────────────────

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

    viol      = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    dl_score  = max(0.0, round(1.0 - viol*0.25, 2))
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


# ─── /process-calc ────────────────────────────────────────────────────────────

REACTION_EA = {
    "hydrogenation": 45000, "esterification": 60000, "oxidation": 55000,
    "polymerization": 70000, "nitration": 80000, "halogenation": 65000,
    "fermentation": 35000, "condensation": 50000, "reforming": 90000,
    "general": 55000,
}
CATALYST_BOOST = {
    "none": 1.0, "acid": 1.25, "base": 1.20, "metal": 1.35,
    "enzyme": 1.45, "zeolite": 1.30, "platinum": 1.50, "palladium": 1.48,
}
REACTOR_EFFICIENCY = {
    "batch": 0.80, "cstr": 0.85, "pfr": 0.92, "fluidized_bed": 0.88,
    "packed_bed": 0.90, "microreactor": 0.95, "membrane": 0.87, "autoclave": 0.82,
}

@app.post("/process-calc")
def process_calc(req: ProcessRequest):
    """Arrhenius-based yield and efficiency calculation for Module 3."""
    import math
    R = 8.314
    T = req.temperature + 273.15
    Ea = REACTION_EA.get(req.reaction_type.lower(), 55000)
    k = 1e10 * math.exp(-Ea / (R * T))
    base_conv = 1 - math.exp(-k * req.residence_time / 3600)
    base_conv = max(0.0, min(base_conv, 1.0))

    cat_boost       = CATALYST_BOOST.get(req.catalyst.lower(), 1.0)
    reactor_eff     = REACTOR_EFFICIENCY.get(req.reactor_type.lower(), 0.85)
    pressure_factor = 1 + 0.02 * (req.pressure - 1)

    yield_pct = round(base_conv * cat_boost * reactor_eff * pressure_factor * 100, 1)
    yield_pct = max(0.0, min(yield_pct, 99.9))

    return {
        "yield_percent":          yield_pct,
        "selectivity":            round(min(95, 70 + yield_pct * 0.25), 1),
        "e_factor":               round(max(1, 50 - yield_pct * 0.4), 2),
        "toc_ppm":                round(max(5, 200 - yield_pct * 1.8), 1),
        "conversion":             round(base_conv * 100, 1),
        "catalyst_boost":         cat_boost,
        "reactor_efficiency":     reactor_eff,
        "temperature_K":          round(T, 1),
        "activation_energy_J_mol": Ea,
    }


# ─── /formulate ───────────────────────────────────────────────────────────────

@app.post("/formulate")
def formulate(req: FormulateRequest):
    """
    Acknowledges the request — full formulation logic lives client-side
    in formulation_generator.html. Returning 200 tells the HTML the
    server is alive so it proceeds with its built-in database.
    """
    return {
        "status":       "ok",
        "message":      "Formulation engine active. Using client-side database.",
        "product_type": req.product_type,
        "skin_type":    req.skin_type,
        "philosophy":   req.philosophy,
        "claims":       req.claims,
        "source":       "client_db",
    }


# ─── /extract ─────────────────────────────────────────────────────────────────

@app.post("/extract")
def extract(req: ExtractRequest):
    """Extract catalysts, conditions, synthesis steps, molecules, yields from scientific text."""
    text    = req.text
    options = req.options or ["catalysts","reaction_conditions","synthesis_steps","molecules","yields","insights"]
    result  = {}

    if "catalysts" in options:
        pats = [
           r'\b([A-Z][a-z]?\d*(?:/[A-Z][a-z]?\d*)*)\s+catalyst\b'
           r'\bcatalyzed by\s+([A-Za-z0-9\-/,\s]+)'
            r'\b(Pd|Pt|Rh|Ru|Ni|Cu|Fe|Au|Ag|Ir|Os|Co|Mn|Zn|Al|Ti|Zr|Ce|Mo|W|V)\b[^.]{0,40}',
            r'\b(zeolite|alumina|silica|MOF|enzyme|lipase|protease|acid|base|BINAP|chiral)\b',
        ]
        found = set()
        for pat in pats:
            for m in re.finditer(pat, text, re.IGNORECASE):
                v = m.group(0).strip()
                if 3 < len(v) < 60:
                    found.add(v[:50])
        result["catalysts"] = [
            {"name": c, "type": "Heterogeneous" if "/" in c else "Homogeneous",
             "activity": round(70 + hash(c) % 30, 1)}
            for c in list(found)[:8]
        ]

    if "reaction_conditions" in options:
        cond = {}
        m = re.search(r'(\d+)\s*[°]?C\b', text)
        if m: cond["temperature"] = m.group(0)
        m = re.search(r'(\d+(?:\.\d+)?)\s*(MPa|bar|atm|kPa|psi)', text, re.IGNORECASE)
        if m: cond["pressure"] = m.group(0)
        m = re.search(r'(\d+(?:\.\d+)?)\s*(h|hr|hours?|min|minutes?|s\b)', text, re.IGNORECASE)
        if m: cond["time"] = m.group(0)
        m = re.search(r'\b(methanol|ethanol|water|DMF|DMSO|THF|acetone|toluene|hexane|acetonitrile|DCM|chloroform)\b', text, re.IGNORECASE)
        if m: cond["solvent"] = m.group(0)
        m = re.search(r'\bpHs*(\d+(?:\.\d+)?)\b', text, re.IGNORECASE)
        if m: cond["pH"] = m.group(0)
        result["reaction_conditions"] = cond

    if "synthesis_steps" in options:
        kws   = ["dissolve","heat","stir","add","filter","wash","dry","evaporate","reflux",
                 "cool","mix","centrifuge","purify","recrystallize","neutralize","extract",
                 "separate","concentrate","precipitate","calcine","reduce","oxidize"]
        steps = []
        for sent in re.split(r'(?<=[.!?])s+', text):
            for kw in kws:
                if kw in sent.lower() and len(sent) > 20:
                    steps.append({"step": len(steps)+1, "action": kw.capitalize(), "description": sent.strip()[:200]})
                    break
            if len(steps) >= 8:
                break
        result["synthesis_steps"] = steps

    if "molecules" in options:
        names = re.findall(r'\b([A-Z][a-z]{2,}s*(?:acid|oxide|hydroxide|chloride|bromide|sulfate|nitrate|phosphate|amine|ether|ester|aldehyde|ketone|alcohol)?)\b', text)
        smis  = re.findall(r'\b([CNOSPFClBrI][a-zA-Z0-9@+-[]()=#%]{4,})\b', text)
        mols  = [{"name": n.strip(), "smiles": None, "source": "text"} for n in list(set(names))[:5]]
        mols += [{"name": s[:20],    "smiles": s,    "source": "pattern"} for s in list(set(smis))[:5]]
        result["molecules"] = mols[:8]

    if "yields" in options:
        result["yields"] = {
            "yield": (m := re.search(r'(\d+(?:\.\d+)?)\s*%\s*(?:yield|conversion|selectivity)', text, re.IGNORECASE)) and m.group(0),
            "ee":    (m := re.search(r'(\d+(?:\.\d+)?)\s*%\s*ee\b', text, re.IGNORECASE))    and m.group(0),
            "TON":   (m := re.search(r'TON\s*(?:of|=|:)?\s*(\d+(?:,\d+)?)', text, re.IGNORECASE)) and m.group(1),
            "TOF":   (m := re.search(r'TOF\s*(?:of|=|:)?\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)) and m.group(1),
        }

    if "insights" in options:
        kws      = ["novel","significant","first","demonstrate","show","report","achieve",
                    "improve","enhance","increase","decrease","exceed","outperform","selectivity"]
        insights = []
        for sent in re.split(r'(?<=[.!?])\s+', text):
            for kw in kws:
                if kw in sent.lower() and len(sent) > 30:
                    insights.append(sent.strip()[:250])
                    break
            if len(insights) >= 5:
                break
        result["insights"] = insights

    return {"status": "ok", "filename": req.filename or "uploaded_text", "char_count": len(text), "extraction": result}


# ─── /extract-pdf ─────────────────────────────────────────────────────────────

@app.post("/extract-pdf")
async def extract_pdf(file: UploadFile = File(...)):
    """Accept PDF or TXT upload and return extracted text."""
    content  = await file.read()
    filename = file.filename or "upload"

    if filename.lower().endswith(".txt"):
        text = content.decode("utf-8", errors="replace")
        return {"status": "ok", "filename": filename, "text": text[:50000], "method": "txt"}

    text = "  "
    try:
        import io, pdfplumber
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            text = "".join(p.extract_text() or "" for p in pdf.pages[:40])
        method = "pdfplumber"
    except Exception:
        try:
            import io
            from PyPDF2 import PdfReader
            reader = PdfReader(io.BytesIO(content))
            text   = "  ".join(p.extract_text() or "" for p in reader.pages[:40])
            method = "pypdf2"
        except Exception as e:
            return {"status": "error", "filename": filename, "text": "", "error": str(e), "method": "none"}

    if not text.strip():
        return {"status": "error", "filename": filename, "text": "", "error": "No text extracted from PDF.", "method": method}

    return {"status": "ok", "filename": filename, "text": text[:50000], "method": method}