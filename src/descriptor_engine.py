import pandas as pd
from rdkit.Chem import Descriptors

from src.smiles_utils import parse_smiles


def compute_basic_properties(smiles: str):
    mol = parse_smiles(smiles)
    if mol is None:
        return {
            "smiles": smiles,
            "valid": False,
            "molecular_weight": None,
            "logp": None,
            "tpsa": None,
            "hbd": None,
            "hba": None,
            "rotatable_bonds": None,
        }

    return {
        "smiles": smiles,
        "valid": True,
        "molecular_weight": round(Descriptors.MolWt(mol), 4),
        "logp": round(Descriptors.MolLogP(mol), 4),
        "tpsa": round(Descriptors.TPSA(mol), 4),
        "hbd": int(Descriptors.NumHDonors(mol)),
        "hba": int(Descriptors.NumHAcceptors(mol)),
        "rotatable_bonds": int(Descriptors.NumRotatableBonds(mol)),
    }


def compute_properties_table(smiles_list):
    rows = [compute_basic_properties(smiles) for smiles in smiles_list]
    return pd.DataFrame(rows)