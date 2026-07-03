import numpy as np
import joblib
import os
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# Known CO2 solubility data (mol CO2 per kg solvent at 25C, 1 atm)
# Source: experimental literature values
data = [
    ("CCO", 0.280),
    ("CC(C)O", 0.310),
    ("CCCCO", 0.290),
    ("c1ccccc1", 0.208),
    ("Cc1ccccc1", 0.224),
    ("CCCCCCCC", 0.185),
    ("CCCCCC", 0.190),
    ("CC(=O)O", 0.260),
    ("CC(=O)OCC", 0.331),
    ("ClCCl", 0.349),
    ("C1CCCCC1", 0.210),
    ("CC#N", 0.272),
    ("CC(C)=O", 0.352),
    ("CCOCC", 0.340),
    ("CS(C)=O", 0.180),
    ("CN(C)C=O", 0.175),
    ("C1CCOC1", 0.330),
    ("OCCO", 0.140),
    ("OCC(O)CO", 0.090),
    ("OC(CO)CO", 0.080),
    ("CCOC(=O)CC", 0.310),
    ("ClCCCl", 0.320),
    ("CC(C)(C)O", 0.295),
    ("CCCO", 0.275),
    ("CCCCCCC", 0.188),
    ("CCCCCCCCCC", 0.182),
    ("c1ccncc1", 0.240),
    ("CC1CCCCC1", 0.205),
    ("O=C1CCCCC1", 0.255),
    ("CCN(CC)CC", 0.265),
    ("CCCCC", 0.192),
    ("CC(O)CC", 0.270),
    ("CCCCCO", 0.285),
    ("CC(=O)c1ccccc1", 0.230),
    ("c1ccc(O)cc1", 0.195),
    ("ClC(Cl)Cl", 0.360),
    ("FC(F)(F)C(F)(F)F", 0.450),
    ("CCCCCCCCCCCC", 0.178),
    ("CC(C)CC(C)(C)C", 0.183),
    ("CC(=O)Nc1ccccc1", 0.190),
    ("O=Cc1ccccc1", 0.220),
    ("OC(=O)c1ccccc1", 0.170),
    ("c1ccc2ccccc2c1", 0.195),
    ("CCCCCCO", 0.282),
    ("CC(C)OCC(C)C", 0.325),
    ("C(F)(F)(F)C(=O)O", 0.380),
    ("CC(F)(F)F", 0.420),
    ("CCOP(=O)(OCC)OCC", 0.210),
    ("c1ccc(Cl)cc1", 0.215),
    ("c1ccc(F)cc1", 0.218),
]

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

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

features, targets = [], []
for smiles, co2 in data:
    f = get_features(smiles)
    if f:
        features.append(f)
        targets.append(co2)

X = np.array(features)
y = np.array(targets)
print(f"Ready: {len(X)} molecules")

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"R2 Score: {round(r2_score(y_test, y_pred), 3)}")
print(f"MSE: {round(mean_squared_error(y_test, y_pred), 4)}")

out_path = os.path.join("src", "co2_model.pkl")
joblib.dump(model, out_path)
print(f"CO2 solubility model saved to {out_path}")
print("Done.")