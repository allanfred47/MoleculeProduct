import pandas as pd
import numpy as np
import joblib
import os
import urllib.request

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

print("Downloading Tox21 dataset...")
url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
data_path = "tox21.csv.gz"
urllib.request.urlretrieve(url, data_path)
print("Download complete.")

df = pd.read_csv(data_path, compression="gzip")
print(f"Dataset loaded: {len(df)} molecules")

TOX_COLS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
]

df = df[["smiles"] + TOX_COLS].dropna(subset=["smiles"])

def get_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            rdMolDescriptors.CalcNumHBD(mol),
            rdMolDescriptors.CalcNumHBA(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol),
            rdMolDescriptors.CalcNumRings(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            mol.GetNumHeavyAtoms(),
            rdMolDescriptors.CalcFractionCSP3(mol),
            Descriptors.MolMR(mol),
        ]
    except:
        return None

print("Calculating features...")
features = []
valid_idx = []
for i, row in df.iterrows():
    f = get_features(row["smiles"])
    if f is not None:
        features.append(f)
        valid_idx.append(i)

df = df.loc[valid_idx].reset_index(drop=True)
X = np.array(features)
print(f"Features ready: {len(X)} molecules")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

models = {}
results = {}

for col in TOX_COLS:
    y_raw = df[col].fillna(0).astype(int).values
    n_pos = y_raw.sum()
    if n_pos < 10:
        print(f"Skipping {col} — too few positive samples ({n_pos})")
        continue
    X_train, X_test, y_train, y_test = train_test_split(X, y_raw, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced")
    model.fit(X_train, y_train)
    try:
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    except:
        auc = 0.0
    models[col] = model
    results[col] = round(auc, 3)
    print(f"{col}: AUC = {round(auc, 3)}")

out_path = os.path.join("src", "toxicity2_models.pkl")
joblib.dump({"models": models, "columns": TOX_COLS}, out_path)
print(f"All models saved to {out_path}")
print(f"Average AUC: {round(np.mean(list(results.values())), 3)}")

if os.path.exists(data_path):
    os.remove(data_path)
print("Done.")