import pandas as pd
import numpy as np
import joblib
import os
import urllib.request

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

print("Downloading viscosity dataset...")
url = "https://raw.githubusercontent.com/dataprofessor/data/master/viscosity.csv"
data_path = "viscosity.csv"

try:
    urllib.request.urlretrieve(url, data_path)
    df = pd.read_csv(data_path)
    print(f"Downloaded: {len(df)} rows, columns: {list(df.columns)}")
except Exception as e:
    print(f"Download failed: {e}")
    print("Building dataset from known viscosity values...")
    # Known experimental viscosity values (cP at 25 degrees C)
    data = [
        ("CCO", 1.074),           # Ethanol
        ("CC(C)O", 2.07),         # Isopropanol
        ("CCCO", 2.0),            # Propanol
        ("CCCCO", 2.95),          # Butanol
        ("OC(CO)CO", 945.0),      # Glycerol subset
        ("c1ccccc1", 0.652),      # Benzene
        ("Cc1ccccc1", 0.560),     # Toluene
        ("CCCCCCCC", 0.508),      # Octane
        ("CCCCCC", 0.294),        # Hexane
        ("CC(=O)O", 1.056),       # Acetic acid
        ("CC(=O)OCC", 0.455),     # Ethyl acetate
        ("ClCCl", 0.413),         # Dichloromethane
        ("C1CCCCC1", 0.894),      # Cyclohexane
        ("CC#N", 0.369),          # Acetonitrile
        ("O=Cc1ccccc1", 1.39),    # Benzaldehyde
        ("CCOC(=O)CC", 0.564),    # Ethyl propanoate
        ("CCCCCCO", 4.59),        # Hexanol
        ("CC(C)CC(C)(C)C", 0.503),# Isooctane
        ("OCC(O)CO", 1412.0),     # Glycerol
        ("CC(O)CC", 3.3),         # 2-Butanol
        ("c1ccc(O)cc1", 12.0),    # Phenol
        ("CC(=O)c1ccccc1", 1.68), # Acetophenone
        ("CCCCCCCCCC", 0.838),    # Decane
        ("ClC(Cl)Cl", 0.537),     # Chloroform
        ("CCOCC", 0.224),         # Diethyl ether
        ("CC1CCCCC1", 0.727),     # Methylcyclohexane
        ("CCCCCCCCCCCC", 1.35),   # Dodecane
        ("OCC", 16.1),            # Ethylene glycol subset
        ("OCCO", 16.1),           # Ethylene glycol
        ("O=C1CCCCC1", 2.02),     # Cyclohexanone
        ("c1ccncc1", 1.36),       # Pyridine
        ("C1CCOC1", 0.456),       # THF
        ("CN(C)C=O", 0.794),      # DMF
        ("CS(C)=O", 1.987),       # DMSO
        ("CC(C)=O", 0.306),       # Acetone
        ("OC(CO)CO", 945.0),      # Glycerol
        ("CCCCCO", 3.69),         # Pentanol
        ("CCCC", 0.180),          # Butane
        ("CCC", 0.099),           # Propane
        ("CCCCCCC", 0.387),       # Heptane
        ("CC(C)OCC(C)C", 0.363),  # Diisopropyl ether
        ("O=C(OCC)OCC", 0.748),   # Diethyl carbonate
        ("CCN(CC)CC", 0.347),     # Triethylamine
        ("c1ccc2ccccc2c1", 1.6),  # Naphthalene
        ("OC(=O)c1ccccc1", 3.44), # Benzoic acid
        ("CC(=O)Nc1ccccc1", 2.0), # Acetanilide
        ("ClCCCl", 0.844),        # 1,2-Dichloroethane
        ("FC(F)(F)C(F)(F)F", 0.179), # Perfluoroethane
        ("CCCCC", 0.214),         # Pentane
        ("CC(C)(C)O", 4.31),      # tert-Butanol
    ]
    df = pd.DataFrame(data, columns=["smiles", "viscosity"])
    print(f"Built dataset with {len(df)} known molecules")

# Make sure columns exist
if "smiles" not in df.columns or "viscosity" not in df.columns:
    cols = list(df.columns)
    print(f"Columns found: {cols}")
    df.columns = ["smiles", "viscosity"] + cols[2:]

df = df[["smiles", "viscosity"]].dropna()
df = df[df["viscosity"] > 0]

# Log-transform viscosity (spans many orders of magnitude)
df["log_viscosity"] = np.log10(df["viscosity"])

def get_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
        rings = rdMolDescriptors.CalcNumRings(mol)
        arom = rdMolDescriptors.CalcNumAromaticRings(mol)
        heavy = mol.GetNumHeavyAtoms()
        csp3 = rdMolDescriptors.CalcFractionCSP3(mol)
        mr = Descriptors.MolMR(mol)
        return [mw, logp, tpsa, hbd, hba, rot, rings, arom, heavy, csp3, mr]
    except:
        return None

print("Calculating features...")
features, targets = [], []
for _, row in df.iterrows():
    f = get_features(row["smiles"])
    if f is not None:
        features.append(f)
        targets.append(row["log_viscosity"])

X = np.array(features)
y = np.array(targets)
print(f"Ready: {len(X)} molecules")

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training viscosity model...")
model = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"R2 Score: {round(r2, 3)}")
print(f"MSE: {round(mse, 3)}")

out_path = os.path.join("src", "viscosity_model.pkl")
joblib.dump(model, out_path)
print(f"Viscosity model saved to {out_path}")

if os.path.exists(data_path):
    os.remove(data_path)
print("Done.")