from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
TESTS_DIR = BASE_DIR / "tests"

APP_NAME = "MoleculeProduct"
APP_VERSION = "0.1.0"

RANDOM_STATE = 42
FP_RADIUS = 2
FP_BITS = 2048

SUPPORTED_INPUT_TYPES = ["smiles"]
SUPPORTED_TASKS = [
    "molecular_weight",
    "logp",
    "tpsa",
    "hbd",
    "hba",
    "rotatable_bonds",
]

for folder in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, TESTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)