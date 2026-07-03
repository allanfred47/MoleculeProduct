import pandas as pd
import numpy as np
import joblib
import os
import urllib.request

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

print("Downloading Tox21 dataset...")
url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
data_path = "tox21.csv.gz"
urllib.request.urlretrieve(url, data_path)
print("Download complete.")

df = pd.read_csv(data_path, compression="gzip")
print(f"Dataset loaded: {len(df)} molecules")

# Use SR-ARE column as general toxicity indicator (1=toxic, 0=non-toxic)
# Fill missing with 0
target_col = "SR-ARE"
df[target_col] = df[target_col].fillna(0)
df = df[df["smiles"].notna()]

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

print("Calculating molecular descriptors...")
features = []
labels = []
for _, row in df.iterrows():
    f = get_features(row["smiles"])
    if f is not None:
        features.append(f)
        labels.append(int(row[target_col]))

X = np.array(features)
y = np.array(labels)
print(f"Features ready: {len(X)} molecules")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training toxicity model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {round(acc * 100, 1)}%")
print(classification_report(y_test, y_pred, target_names=["Non-toxic", "Toxic"]))

out_path = os.path.join("src", "toxicity_model.pkl")
joblib.dump(model, out_path)
print(f"Toxicity model saved to {out_path}")

# Clean up
os.remove(data_path)
print("Done.")