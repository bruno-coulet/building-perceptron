import sys
from pathlib import Path
import numpy as np
import pandas as pd

from pyprojroot import here

# Ajout de la racine au sys.path pour rendre les dossiers importables
root_dir = here()
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from scripts.utils import fetch_and_save_dataset


# Load Data
output_dir = Path(root_dir) / "data"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "breast_cancer_wisconsin.csv"

if not output_path.is_file():
    df, X, y = fetch_and_save_dataset(output_path)
else:
    df = pd.read_csv(output_path)
    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]
