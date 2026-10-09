from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "raw" / "bcw_data.csv"
DESCRIPTION_PATH = ROOT_DIR / "data" / "bcw_description.md"
ARTIFACTS_DIR = ROOT_DIR / "data" / "processed"
RANDOM_STATE = 42
TARGET = "diagnosis"
ID_COLUMN = "id"

TARGET_LABELS = {"B": "Bénigne", "M": "Maligne"}
