from pathlib import Path
from typing import Final

# config.py -> src/building_perceptron/config.py
# parent = src/building_perceptron/
# parent.parent = src/
# parent.parent.parent = project root

# /home/.../.../.../building-perceptron/src/building_perceptron
# .parents[0] -> .../src/building_perceptron
# .parents[1] -> .../src
# .parents[2] -> .../building-perceptron (racine)
PACKAGE_DIR: Final[Path] = Path(__file__).resolve().parent
ROOT_DIR: Final[Path] = PACKAGE_DIR.parents[1]

DATA_DIR: Final[Path] = ROOT_DIR / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw_data"
CLEAN_DATA_DIR: Final[Path] = DATA_DIR / "clean_data"

DATA: Final[Path] = RAW_DATA_DIR / "bcw_data.csv"
TARGET: Final[str] = "diagnosis"



# ================== Legacy code for reference ==================
# # current directory
# ROOT_DIR = Path(".")
# # absolute path
# print(ROOT_DIR.resolve())

# DATA_DIR = ROOT_DIR / "data"
# RAW_DATA_DIR = DATA_DIR / "raw_data"
# DATA = RAW_DATA_DIR / "bcw_data.csv"
# CLEAN_DATA_DIR = DATA_DIR / "clean_data"
# TARGET = 'diagnosis'
