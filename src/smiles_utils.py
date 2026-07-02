from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def parse_smiles(smiles: str):
    if smiles is None:
        return None

    smiles = str(smiles).strip()
    if not smiles:
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except Exception:
        return None


def canonicalize_smiles(smiles: str):
    mol = parse_smiles(smiles)
    if mol is None:
        return None

    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def is_valid_smiles(smiles: str) -> bool:
    return parse_smiles(smiles) is not None