import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors, Crippen
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Dataset of molecules with known solubility (logS values)
# logS = log of aqueous solubility. Higher = more soluble.
data = [
    ("CCO", 0.31),           # Ethanol
    ("CC(=O)O", -0.72),      # Acetic acid
    ("c1ccccc1", -1.90),     # Benzene
    ("CC(=O)Oc1ccccc1C(=O)O", -1.19),  # Aspirin
    ("Cn1cnc2c1c(=O)n(c(=O)n2C)C", -1.22),  # Caffeine
    ("CC(=O)Nc1ccc(O)cc1", -1.07),  # Paracetamol
    ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", -3.19),  # Ibuprofen
    ("OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", 0.54),  # Glucose
    ("C", 0.54),             # Methane
    ("CC", 0.11),            # Ethane
    ("CCC", -0.23),          # Propane
    ("CCCC", -0.69),         # Butane
    ("CCCCC", -1.12),        # Pentane
    ("CCCCCC", -1.55),       # Hexane
    ("Oc1ccccc1", -1.20),    # Phenol
    ("Nc1ccccc1", -1.10),    # Aniline
    ("OC(=O)c1ccccc1", -1.77),  # Benzoic acid
    ("O", 0.00),             # Water
    ("CO", -0.20),           # Methanol
    ("CCCO", -0.11),         # Propanol
    ("CCCCO", -0.50),        # Butanol
    ("CC(=O)C", -0.24),      # Acetone
    ("NC(=O)N", 0.20),       # Urea
    ("NCC(=O)O", 0.14),      # Glycine
    ("OCC(O)CO", 0.10),      # Glycerol
    ("CC(O)C(=O)O", -0.50),  # Lactic acid
    ("Cc1ccccc1", -2.20),    # Toluene
    ("ClC(Cl)Cl", -1.80),    # Chloroform
    ("CS(=O)C", -1.00),      # DMSO
    ("OO", 0.20),            # Hydrogen peroxide
    ("NCCc1ccc(O)c(O)c1", -1.00),  # Dopamine
    ("NCCc1c[nH]c2ccc(O)cc12", -1.50),  # Serotonin
    ("CN1CCC[C@H]1c1cccnc1", -1.80),  # Nicotine
    ("Nc1ncnc2[nH]cnc12", -1.40),  # Adenine
    ("Nc1nc2[nH]cnc2c(=O)[nH]1", -1.60),  # Guanine
    ("Nc1ccnc(=O)[nH]1", -1.10),  # Cytosine
    ("Cc1cnc(=O)[nH]c1=O", -1.30),  # Thymine
    ("OC(=O)c1cccnc1", -1.20),  # Niacin
    ("CC(N)C(=O)O", -0.10),  # Alanine
    ("NC(Cc1ccccc1)C(=O)O", -1.80),  # Phenylalanine
    ("CSCCC(N)C(=O)O", -0.80),  # Methionine
    ("NC(CS)C(=O)O", -0.30),  # Cysteine
    ("CC(C)CC(N)C(=O)O", -1.20),  # Leucine
    ("OC(CC(O)(C(=O)O)CC(=O)O)(C(=O)O)=O", -0.20),  # Citric acid
    ("C=C", 0.20),           # Ethylene
    ("C#C", 0.10),           # Acetylene
    ("CCOCC", -1.30),        # Diethyl ether
    ("CC(C)=O", -0.24),      # Acetone alt
    ("OC[C@H](O)[C@H]1OC(=O)C(O)=C1O", -0.20),  # Vitamin C
]

smiles_list = [d[0] for d in data]
logS_list   = [d[1] for d in data]

def get_fingerprint_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=256)
    mw   = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd  = rdMolDescriptors.CalcNumHBD(mol)
    hba  = rdMolDescriptors.CalcNumHBA(mol)
    rb   = rdMolDescriptors.CalcNumRotatableBonds(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)
    return list(fp) + [mw, logp, tpsa, hbd, hba, rb, rings]

X, y = [], []
for s, val in zip(smiles_list, logS_list):
    feat = get_fingerprint_features(s)
    if feat:
        X.append(feat)
        y.append(val)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2  = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R² Score : {r2:.3f}")
print(f"MSE      : {mse:.3f}")
print("Model trained successfully.")

joblib.dump(model, "src/solubility_model.pkl")
print("Model saved to src/solubility_model.pkl")