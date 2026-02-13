"""
Boîte à outils pour l'analyse exploratoire des données (EDA) et la préparation de données.

Auteur     : Bruno Coulet d'après Eric Demangel "Maitrisez la data science avec pyhton"
Date       : Février 2026
Description: Module utilitaire pour nettoyer, explorer et préparer les données pour le ML
Version    : 1.0.0
"""

import pandas as pd
import numpy as np
import unicodedata
import re
import math
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional, Any, List, Iterable
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===============================
# DÉTECTION DES COLONNES PROBLÉMATIQUES
# ===============================

# Identifie toutes les colonnes complètement vides (100% NaN)
def empty_columns(df: pd.DataFrame) -> list:
    """Retourne les colonnes dont toutes les valeurs sont NaN"""
    return df.columns[df.isna().all()].tolist()

# Détecte les colonnes avec une seule modalité unique
def unique_value_columns(df: pd.DataFrame) -> list:
    """Retourne les colonnes avec une seule valeur unique"""
    return df.columns[df.nunique(dropna=False) <= 1].tolist()

# Récupère toutes les colonnes de type texte (object ou string)
def string_columns(df: pd.DataFrame) -> list:
    """Retourne les colonnes de type string ou object"""
    return df.select_dtypes(include=['object', 'string']).columns.tolist()

# Identifie les colonnes contenant uniquement des valeurs booléennes
def boolean_columns(df: pd.DataFrame) -> list:
    """Retourne les colonnes booléennes ou contenant seulement True/False/NaN"""
    bool_cols = []
    for col in df.columns:
        s = df[col].dropna().unique()
        if set(s).issubset({True, False}):
            bool_cols.append(col)
    return bool_cols

# Récupère toutes les colonnes numériques (int ou float)
def numeric_columns(df: pd.DataFrame) -> list:
    """Retourne les colonnes numériques (int, float)"""
    return df.select_dtypes(include=['number']).columns.tolist()

# Repère les colonnes avec un taux de données manquantes excessif
def high_na_columns(df: pd.DataFrame, threshold: float = 0.5) -> list:
    """Retourne les colonnes avec plus de `threshold` proportion de NaN"""
    return df.columns[df.isna().mean() > threshold].tolist()

# Détecte les colonnes catégoriques avec trop de modalités
def high_cardinality_columns(df: pd.DataFrame, max_modalities: int = 20) -> list:
    """
    Retourne les colonnes object/string qui ont plus de `max_modalities` valeurs uniques
    """
    cat_cols = string_columns(df)
    high_card_cols = [col for col in cat_cols if df[col].nunique(dropna=True) > max_modalities]
    return high_card_cols

# Détecte les colonnes contenant des valeurs manquantes implicites (na, null, etc)
def missing_like_columns(df: pd.DataFrame) -> list:
    """
    Colonnes contenant des valeurs manquantes explicites : NaN, None, '', 'na', 'null'
    """
    missing_vals = {np.nan, None, '', 'na', 'NA', 'null', 'NULL'}
    cols = []
    for col in df.columns:
        if df[col].isin(missing_vals).any():
            cols.append(col)
    return cols

# Détecte les lignes en doublon dans le DataFrame
def duplicate_rows(df: pd.DataFrame, subset: Optional[List[str]] = None, keep: str = 'first') -> pd.DataFrame:
    """
    Retourne les lignes en doublon dans le DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame à analyser
    subset : list | None
        Liste des colonnes à considérer pour détecter les doublons.
        Si None, toutes les colonnes sont utilisées.
    keep : str
        'first' : marque les doublons sauf la première occurrence
        'last' : marque les doublons sauf la dernière occurrence
        False : marque toutes les occurrences comme doublons
    
    Returns
    -------
    pd.DataFrame
        DataFrame contenant uniquement les lignes en doublon
    """
    return df[df.duplicated(subset=subset, keep=keep)]

# Compte le nombre de doublons par groupe de colonnes
def count_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> int:
    """
    Retourne le nombre total de lignes en doublon (hors première occurrence).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame à analyser
    subset : list | None
        Liste des colonnes à considérer pour détecter les doublons.
        Si None, toutes les colonnes sont utilisées.
    
    Returns
    -------
    int
        Nombre de lignes en doublon
    """
    return df.duplicated(subset=subset, keep='first').sum()




# ===============================
# OPÉRATIONS DE NETTOYAGE
# ===============================

# Supprime des colonnes et enregistre les modifications
def drop_columns(df: pd.DataFrame, cols: list, dropped: list = None) -> pd.DataFrame:
    """
    Supprime les colonnes et optionnellement ajoute à la liste `dropped`
    """
    if dropped is not None:
        dropped.extend(cols)
    return df.drop(columns=cols)

# --- Suppression d'une colonne et enregistre les modifications

def drop_one_column(df: pd.DataFrame, col: str, dropped: list = None) -> pd.DataFrame:
    """
    Supprime une colonne d'un DataFrame et l'ajoute à la liste `dropped` si fournie.
    Attention : la fonction retourne un nouveau DataFrame, il faut le réaffecter.
    """
    if dropped is not None:
        dropped.append(col)
    return df.drop(columns=[col])

# Convertit les colonnes booléennes en entiers (0, 1, NaN)
def convert_bool_to_uint8(df: pd.DataFrame, cols: list, keep_na: bool = True) -> pd.DataFrame:
    """
    Convertit True/False/NaN en UInt8
    keep_na=True : True->1, False->0, NaN->NaN
    keep_na=False: True->1, False->0, NaN->0
    """
    for col in cols:
        if keep_na:
            df[col] = df[col].astype('boolean').astype('UInt8')
        else:
            df[col] = df[col].astype('boolean').fillna(False).astype('UInt8')
    return df

# Convertit le texte en minuscules pour les colonnes spécifiées
def lower_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Convertit en minuscules uniquement les colonnes existantes et typées string/object
    """
    # Colonnes existantes dans le DataFrame
    existing_cols = [c for c in cols if c in df.columns]
    
    # Filtrer pour ne garder que les colonnes string/object
    string_cols = [c for c in existing_cols if pd.api.types.is_string_dtype(df[c])]
    
    for col in string_cols:
        df[col] = df[col].str.lower()
    
    return df


# Crée une colonne catégorielle basée sur un mapping de mots-clés
def add_type_column(df: pd.DataFrame, col_source: str, mapping: dict, col_dest: str = 'type') -> pd.DataFrame:
    """
    Crée une nouvelle colonne selon un mapping de mots-clés dans la colonne source
    mapping = {'piso': 'piso', 'casa': 'casa o chalet', ...}
    """
    df[col_dest] = None
    for key, value in mapping.items():
        mask = df[col_source].str.contains(key, na=False)
        df.loc[mask & df[col_dest].isna(), col_dest] = value
    return df

# Impute les valeurs manquantes numériques (médiane, moyenne ou zéro)
def impute_numeric(df: pd.DataFrame, cols: list = None, strategy: str = 'median') -> pd.DataFrame:
    """
    Impute les valeurs manquantes des colonnes numériques
    strategy: 'mean', 'median', 'zero'
    """
    if cols is None:
        cols = numeric_columns(df)
    for col in cols:
        if strategy == 'median':
            df[col] = df[col].fillna(df[col].median())
        elif strategy == 'mean':
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == 'zero':
            df[col] = df[col].fillna(0)
    return df

# Impute les colonnes catégoriques avec une valeur par défaut
def impute_categorical(df: pd.DataFrame, cols: list = None, fill_value: str = 'missing') -> pd.DataFrame:
    """Remplit les valeurs manquantes des colonnes object/string par fill_value"""
    if cols is None:
        cols = string_columns(df)
    for col in cols:
        df[col] = df[col].fillna(fill_value)
    return df

# Calcule le pourcentage de remplissage par colonne
def fill_rate(df):
    """Calcule le taux de remplissage (%) par colonne : 100 * (nb de valeurs non nulles / nb total de lignes).
    Retourne une Series indexée par nom de colonne.
    """
    return df.count() / len(df) * 100




# ===============================
# NORMALISATION DU TEXTE
# ===============================

# Normalise une chaîne : minuscules, supprime accents (protège True/False)
def normalize_string(text):
    """
    Convertit une chaîne en minuscules et supprime les accents/caractères diacritiques,
    tout en protégeant les booléens (True/False) et les NaN.
    """
    
    # 1. Protection contre les NaN et None
    if pd.isna(text) or text is None:
        return text
    
    text_str = str(text)

    # 2. **PROTECTION BOOLÉENNE (NOUVEAU)**
    # Nous vérifions si la chaîne (en ignorant la casse) est 'true' ou 'false'
    if text_str.lower() in ['true', 'false']:
        # On peut soit laisser la chaîne telle quelle, soit la convertir en booléen Python natif.
        # Nous la laissons en chaîne pour le moment, mais non modifiée.
        return text
    
    # 3. Traitement standard du texte (minuscules et accents)
    
    # Convertir en minuscules (UNIQUEMENT les chaînes qui ne sont pas True/False)
    text_str_lower = text_str.lower() 

    # Décomposer les caractères (NFD)
    normalized = unicodedata.normalize('NFD', text_str_lower)
    
    # Retirer les marques d'accent
    text_no_accents = re.sub(r'[\u0300-\u036f]', '', normalized)
    
    return text_no_accents


# Normalise toutes les colonnes texte d'un DataFrame
def normalize_all_text_columns(df):
    string_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    
    print(f"Normalisation des colonnes de texte : {string_cols}")

    for col in string_cols:
        # Utiliser la fonction sécurisée
        df[col] = df[col].apply(normalize_string)
        
    return df





# ===============================
# SÉLECTION ET EXPLORATION DES FEATURES
# ===============================

# Filtre une liste de features pour ne garder que celles présentes
def select_existing_features(features: Iterable[str], columns: Iterable[str]) -> List[str]:
    """
    Filtre une liste de features pour ne conserver que celles présentes.

    Parameters
    ----------
    features : iterable
        Liste des colonnes souhaitées.
    columns : iterable
        Colonnes effectivement présentes (ex: X.columns).

    Returns
    -------
    list
        Liste filtrée dans l'ordre d'origine.
    """
    col_set = set(columns)
    return [c for c in features if c in col_set]

# Recherche de colinéarité (entre features)
def feature_collinearity(df: pd.DataFrame, target_col: str, threshold: float = 0.8):
    """
    Identifie les paires de features fortement corrélées entre elles (colinéarité),
    en excluant la colonne cible.
    """
    # On ne garde que les colonnes numériques et on exclut la cible
    df_features = df.select_dtypes(include=[np.number]).drop(columns=[target_col], errors='ignore')
    
    corr_matrix = df_features.corr().abs()
    
    # On ne récupère que la partie supérieure de la matrice pour éviter les doublons (A/B et B/A)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # On filtre par le seuil
    collinear_features = [
        (column, row, upper.loc[row, column]) 
        for column in upper.columns 
        for row in upper.index 
        if upper.loc[row, column] > threshold
    ]
    
    return sorted(collinear_features, key=lambda x: x[2], reverse=True)

# Affiche une heatmap des corrélations entre features (excluant la cible)
def plot_feature_collinearity(df: pd.DataFrame, target_col: str, figsize: tuple = (12, 10)):
    """Affiche la heatmap des corrélations entre features (exclut la cible)."""
    # Sélection des numériques et exclusion de la cible
    df_features = df.select_dtypes(include=[np.number]).drop(columns=[target_col], errors='ignore')
    
    corr = df_features.corr()
    
    plt.figure(figsize=figsize)
    # On utilise un masque pour ne voir que le triangle inférieur (plus lisible)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', center=0)
    plt.title(f"Colinéarité entre Features (excluant {target_col})")
    plt.show()

# Recherche de corrélations avec la Cible
def target_correlations(df: pd.DataFrame, target: pd.Series, n_top: int = 10):
    """
    Calcule la corrélation entre toutes les features numériques et la cible.
    Retourne les n premières features les plus corrélées.
    """
    # S'assurer que la cible est numérique pour le calcul (ex: diagnosis M/B -> 1/0)
    temp_df = df.copy()
    if target.dtype == 'object' or target.dtype == 'category':
        # Encodage rapide si c'est une catégorie (B=0, M=1 par exemple)
        from sklearn.preprocessing import LabelEncoder
        temp_df[target.name] = LabelEncoder().fit_transform(target)
    
    correlations = temp_df.corr()[target.name].abs().sort_values(ascending=False)
    
    # On retire la cible elle-même (corrélation de 1.0)
    return correlations.drop(labels=[target_col], errors='ignore').head(n_top)

# Affiche un barplot des n variables les plus corrélées à la cible
def plot_target_correlations(df: pd.DataFrame, target_col: str, n_top: int = 15):
    """Affiche un barplot des n variables les plus corrélées à la cible."""
    # Réutilisation de ta fonction existante pour obtenir les données
    corrs = target_correlations(df, target_col, n_top=n_top)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x=corrs.values, y=corrs.index, hue=corrs.index, palette='viridis', legend=False)
    plt.title(f"Top {n_top} des corrélations avec la cible : {target_col}")
    plt.xlabel("Coefficient de corrélation (valeur absolue)")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()

# ===============================
# VISUALISATIONS
# ===============================

# Affiche les histogrammes de toutes les colonnes numériques
def plot_numeric_histograms(
    X: pd.DataFrame,
    bins: int = 40,
    n_cols: int = 3,
    figsize_per_col: Tuple[int, int] = (5, 3),
) -> None:
    """
    Affiche les histogrammes de toutes les colonnes numériques.

    Parameters
    ----------
    X : pd.DataFrame
        Données d'entrée.
    bins : int
        Nombre de bins pour les histogrammes.
    n_cols : int
        Nombre de graphes par ligne.
    figsize_per_col : tuple
        Taille d'un subplot (largeur, hauteur).
    """
    num_cols = X.select_dtypes(include=["number"]).columns
    n_plots = len(num_cols)
    if n_plots == 0:
        return
    n_rows = math.ceil(n_plots / n_cols)
    plt.figure(figsize=(n_cols * figsize_per_col[0], n_rows * figsize_per_col[1]))
    for i, col in enumerate(num_cols, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(X[col].dropna(), bins=bins)
        plt.xlabel("")   # masque axe X
        plt.ylabel("")   # masque axe Y
        plt.grid(True, alpha=0.3)
        plt.title(col)
    plt.tight_layout()
    plt.show()

# Affiche les diagrammes en barres des colonnes qualitatives
def plot_qualitative(
    X: pd.DataFrame,
    top_n: int = 20,
    n_cols: int = 2,
    figsize_per_col: Tuple[int, int] = (6, 4),
    figsize: Optional[Tuple[int, int]] = None,
    height_per_row: int = 4,
) -> None:
    """
    Affiche des barplots pour les colonnes qualitatives.

    Parameters
    ----------
    X : pd.DataFrame
        Données d'entrée.
    top_n : int
        Nombre de modalités affichées par colonne.
    n_cols : int
        Nombre de graphes par ligne.
    figsize_per_col : tuple
        Taille d'un subplot (largeur, hauteur).
    """
    cat_cols = X.select_dtypes(include=["object", "category", "string", "bool"]).columns
    n_plots = len(cat_cols)
    if n_plots == 0:
        return

    n_rows = math.ceil(n_plots / n_cols)
    if figsize is None:
        figsize = (n_cols * figsize_per_col[0], n_rows * height_per_row)

    plt.figure(figsize=figsize)
    

    for i, col in enumerate(cat_cols, 1):
        plt.subplot(n_rows, n_cols, i)
        vc = X[col].astype("string").value_counts(dropna=False).head(top_n)
        sns.barplot(x=vc.values, y=vc.index, color="#439cc8")
        plt.xlabel("")
        plt.ylabel("")
        plt.title(col)
        plt.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.show()


# Affiche le taux de valeurs manquantes par colonne
def plot_missing_bar(
    X: pd.DataFrame,
    top_n: Optional[int] = None,
    figsize: Tuple[int, int] = (8, 4),
) -> None:
    """
    Affiche un bar plot du % de valeurs manquantes par colonne.

    Parameters
    ----------
    X : pd.DataFrame
        Données d'entrée.
    top_n : int | None
        Affiche uniquement les top_n colonnes les plus manquantes.
    figsize : tuple
        Taille de la figure.
    """
    missing_pct = (X.isna().mean() * 100).sort_values(ascending=False)
    if top_n is not None:
        missing_pct = missing_pct.head(top_n)

    plt.figure(figsize=figsize)
    sns.barplot(x=missing_pct.values, y=missing_pct.index, color="#439cc8")
    plt.xlabel("% de valeurs manquantes")
    plt.ylabel("Colonnes")
    plt.title("Taux de valeurs manquantes par colonne")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# Trace des scatter plots de variables vs la target
def plot_scatter_vs_target(
    X: pd.DataFrame,
    y: pd.Series,
    cols: Iterable[str],
    transform_y: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10),
    alpha: float = 0.2,
    s: int = 10,
) -> None:
    """
    Trace des scatter plots de variables vs la target.

    Parameters
    ----------
    X : pd.DataFrame
        Features d'entrée.
    y : pd.Series
        Target associée à X.
    cols : iterable
        Colonnes à visualiser.
    transform_y : {"log1p", None}
        Applique \\log(1+y) à la target si "log1p".
    figsize : tuple
        Taille de la figure.
    alpha : float
        Transparence des points.
    s : int
        Taille des points.
    """
    if transform_y == "log1p":
        y_vals = np.log1p(y.values)
    else:
        y_vals = y.values

    cols = list(cols)
    if not cols:
        return

    n_rows = math.ceil(len(cols) / 3)
    plt.figure(figsize=figsize)
    for i, col in enumerate(cols, 1):
        plt.subplot(n_rows, 3, i)
        mask = X[col].notna()
        sns.scatterplot(x=X.loc[mask, col], y=y_vals[mask], s=s, alpha=alpha)
        plt.title(f"{transform_y + ' ' if transform_y else ''}target vs {col}")
    plt.tight_layout()
    plt.show()


# Affiche une heatmap des corrélations entre variables numériques
def plot_corr_heatmap(
    df: pd.DataFrame,
    method: str = "pearson",
    title: str = "Heatmap des corrélations",
    figsize: Tuple[int, int] = (12, 10),
    annot: bool = True,
    fmt: str = ".2f",
    vmin: float = -1,
    vmax: float = 1,
    cmap: str = "coolwarm",
) -> None:
    """
    Affiche une heatmap de corrélation pour les colonnes numériques.

    Parameters
    ----------
    df : pd.DataFrame
        Données d'entrée.
    title : str
        Titre du graphique.
    figsize : tuple
        Taille de la figure.
    annot : bool
        Afficher les valeurs numériques dans la heatmap.
    fmt : str
        Format des annotations.
    vmin, vmax : float
        Bornes de l'échelle de couleur.
    cmap : str
        Palette de couleurs.
    """
    corr = df.select_dtypes(include=[np.number]).corr(method=method)
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=annot, fmt=fmt, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ===============================
# EXPORT ET PREPROCESSING
# ===============================

# Supprime les colonnes dépassant le seuil de données manquantes
def _drop_high_na(
    X: pd.DataFrame,
    threshold: float,
    stats: Dict[str, Any],
) -> pd.DataFrame:
    """Supprime les colonnes dont le taux de NA dépasse un seuil."""
    if "cols_to_drop" not in stats:
        cols_to_drop = X.columns[X.isna().mean() > threshold]
        stats["cols_to_drop"] = list(cols_to_drop)
    return X.drop(columns=stats.get("cols_to_drop", []), errors="ignore")


# Impute les colonnes binaires avec le mode appris sur l'entraînement
def _fill_bin_with_mode(
    X: pd.DataFrame,
    cols: List[str],
    stats: Dict[str, Any],
) -> pd.DataFrame:
    """Impute les colonnes binaires avec le mode (appris sur train)."""
    cols = [c for c in cols if c in X.columns]
    if "bin_modes" not in stats:
        stats["bin_modes"] = {}
        for c in cols:
            if len(X[c].mode()) > 0:
                stats["bin_modes"][c] = X[c].mode()[0]
    for c in cols:
        if c in stats.get("bin_modes", {}):
            X[c] = X[c].fillna(stats["bin_modes"][c])
    return X


# Impute les colonnes numériques avec leur médiane
def _fill_numeric_with_median(
    X: pd.DataFrame,
    cols: List[str],
    stats: Dict[str, Any],
    key: str,
) -> pd.DataFrame:
    """Impute les colonnes numériques avec leur médiane (apprise sur train)."""
    if key not in stats:
        stats[key] = {}
        for c in cols:
            if c in X.columns:
                X[c] = pd.to_numeric(X[c], errors="coerce")
                stats[key][c] = X[c].median()
    for c in cols:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
            if c in stats.get(key, {}):
                X[c] = X[c].fillna(stats[key][c])
    return X


# Nettoie une colonne (remplacement/validation) puis impute par médiane
def _replace_and_median(
    X: pd.DataFrame,
    col: str,
    stats: Dict[str, Any],
    replace_map: Optional[Dict[Any, Any]] = None,
    invalid_below: Optional[float] = None,
    key: Optional[str] = None,
) -> pd.DataFrame:
    """Nettoie une colonne (replace/invalid) puis impute par médiane."""
    if col not in X.columns:
        return X
    if replace_map:
        X[col] = X[col].replace(replace_map)
    X[col] = pd.to_numeric(X[col], errors="coerce")
    if invalid_below is not None:
        X.loc[X[col] < invalid_below, col] = np.nan
    median_key = key or f"{col}_median"
    if median_key not in stats:
        stats[median_key] = X[col].median()
    X[col] = X[col].fillna(stats[median_key])
    return X


# Nettoyage configurable et reproductible des données
def clean_data(
    X: pd.DataFrame,
    config: Dict[str, Any],
    stats: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleaning configurable et réutilisable.

    Parameters
    ----------
    X : pd.DataFrame
        Données d'entrée.
    config : dict
        Paramètres de nettoyage (exemple ci-dessous).
    stats : dict | None
        Statistiques apprises sur le train pour répliquer sur le test.

    Config attendu (exemple) :
    {
        "drop_na_threshold": 0.40,
        "binary_cols": [...],
        "floor_col": "floor",
        "floor_replace": {"bajo": 0},
        "numeric_median_cols": ["sq_mt_built", "n_bathrooms"],
        "rent_col": "rent_price",
        "rent_invalid_below": 0,
    }

    Returns
    -------
    X_clean : pd.DataFrame
        Données nettoyées.
    stats : dict
        Statistiques apprises pour réutilisation sur d'autres jeux.
    """
    X = X.copy()
    stats = {} if stats is None else dict(stats)

    threshold = config.get("drop_na_threshold")
    if threshold is not None:
        X = _drop_high_na(X, threshold=threshold, stats=stats)

    bin_cols = config.get("binary_cols", [])
    if bin_cols:
        X = _fill_bin_with_mode(X, cols=bin_cols, stats=stats)

    floor_col = config.get("floor_col")
    if floor_col:
        X = _replace_and_median(
            X,
            col=floor_col,
            stats=stats,
            replace_map=config.get("floor_replace"),
            key=config.get("floor_median_key", "floor_median"),
        )

    num_cols = config.get("numeric_median_cols", [])
    if num_cols:
        X = _fill_numeric_with_median(
            X,
            cols=num_cols,
            stats=stats,
            key=config.get("numeric_median_key", "num_medians"),
        )

    rent_col = config.get("rent_col")
    if rent_col:
        X = _replace_and_median(
            X,
            col=rent_col,
            stats=stats,
            invalid_below=config.get("rent_invalid_below"),
            key=config.get("rent_median_key", "rent_median"),
        )

    return X, stats


# Exporte les données train/test au format Feather
def export_train_test_feather(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: str = "data_model",
    target_name: str = "log_buy_price",
    transform_y: Optional[str] = None,
    drop_cols: Optional[List[str]] = None,
) -> None:
    """
    Exporte X/y train/test au format Feather.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Jeux de features.
    y_train, y_test : pd.Series
        Targets associées.
    output_dir : str
        Dossier de sortie.
    target_name : str
        Nom de la target exportée.
    transform_y : {"log1p", None}
        Applique \\log(1+y) si "log1p".
    drop_cols : list | None
        Colonnes à retirer de X_train avant export.
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    X_train_final = X_train.drop(columns=drop_cols or [], errors="ignore").reset_index(drop=True)
    X_test_final = X_test.reset_index(drop=True)

    if transform_y == "log1p":
        y_train_final = pd.Series(np.log1p(y_train.values), name=target_name).reset_index(drop=True)
        y_test_final = pd.Series(np.log1p(y_test.values), name=target_name).reset_index(drop=True)
    else:
        y_train_final = pd.Series(y_train.values, name=target_name).reset_index(drop=True)
        y_test_final = pd.Series(y_test.values, name=target_name).reset_index(drop=True)

    X_train_final.to_feather(f"{output_dir}/X_train.feather")
    X_test_final.to_feather(f"{output_dir}/X_test.feather")
    y_train_final.to_frame().to_feather(f"{output_dir}/y_train.feather")
    y_test_final.to_frame().to_feather(f"{output_dir}/y_test.feather")




# ===============================
# ENTRAÎNEMENT ET ÉVALUATION
# ===============================

# Entraîne un modèle avec GridSearchCV ou RandomizedSearchCV
def evaluate_model(
        algo,
        param_grid,
        X_train,
        y_train,
        X_test,
        y_test,
        search_type='grid',
        scoring='r2',
        cv=5):
    """
    Entraine un modele avec GridSearchCV ou RandomizedSearchCV et affiche les résultats
    prédit les valeurs de test et calcule les métriques
    
    :parma algo: instance de l'algorithme à utiliser
    :param param_grid: dictionnaire des paramètres à testerNone si vide)
    :param X_train: features d'entrainement
    :param y_train: target d'entrainement
    :param X_test: features de test
    :param y_test: target de test
    :param search_type: type de recherche, 'grid' pour GridSearchCV, 'random' pour RandomizedSearchCV
    :param scoring: métrique d'évaluation
    :param cv: nombre de folds pour la validation croisée
    :return: dict des meilleurs paramètres, R2, RMSE, MAE, les résultats et le modele
    """
    # Si la param grid est vide, entrainement sans optimisation
    if param_grid is None:
        algo.fit(X_train, y_train)
        best_model = algo
        best_params = algo.get_params()
        cv_results = cross_val_score(best_model,
                                      X_train, y_train,
                                      cv=cv,
                                      scoring=scoring)
    else:
        # Choisir le type de recherche d'hyperparamètres
        if search_type == 'grid':
            search = GridSearchCV(algo, 
                                  param_grid, 
                                  cv=cv, 
                                  scoring=scoring)
        elif search_type == 'random':
            search = RandomizedSearchCV(algo, 
                                        param_grid, 
                                        cv=cv, 
                                        scoring=scoring)
        else:
            raise ValueError("search_type doit être 'grid' ou 'random'")
        
        # Entrainement et optimisation
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_params = search.best_params_
        cv_results = search.cv_results_

        # Prédiction avec le meilleur modele
        y_pred = best_model.predict(X_test)

        # Calcul des métriques
        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        mae = mean_absolute_error(y_test, y_pred)

        # Affichage des résultats
        print(f"Modèle : {algo.__class__.__name__}")
        print(f"Meilleurs paramètres : {best_params}")
        print(f"R2 (sur le test): {r2:.4f}")
        print(f"RMSE : {rmse:.4f}")
        print(f"MAE : {mae:.4f}")

        return {
            "best_params": best_params,
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "cv_results": cv_results,
            "best_model": best_model
        }



# ===============================
# EXEMPLES D'UTILISATION
# ===============================
if __name__ == "__main__":
    # --------- Chargement des données ---------
    df = pd.read_csv("raw_data/houses_Madrid.csv", index_col=1)
    
    # --------- Détection des problèmes ---------
    print("=" * 60)
    print("DIAGNOSTIC DES DONNÉES")
    print("=" * 60)
    print(f"Colonnes vides: {empty_columns(df)}")
    print(f"Colonnes avec une seule valeur: {unique_value_columns(df)}")
    print(f"Colonnes booléennes: {boolean_columns(df)}")
    print(f"Colonnes texte: {string_columns(df)}")
    print(f"Colonnes numériques: {numeric_columns(df)}")
    print(f"Colonnes avec >50% de NaN: {high_na_columns(df, threshold=0.5)}")
    print(f"Colonnes catégoriques haute cardinalité (>2): {high_cardinality_columns(df, max_modalities=2)}")
    print(f"Colonnes avec valeurs 'nulles' implicites: {missing_like_columns(df)}")
    
    # --------- Analyse du remplissage ---------
    print("\n" + "=" * 60)
    print("TAUX DE REMPLISSAGE PAR COLONNE")
    print("=" * 60)
    print(fill_rate(df))
    
    # --------- Normalisation du texte (optionnel) ---------
    # df_normalized = normalize_all_text_columns(df.copy())
    
    # --------- Visualisations ---------
    # plot_numeric_histograms(df, bins=40, n_cols=3)
    # plot_qualitative(df, top_n=20, n_cols=2)
    # plot_missing_bar(df, top_n=10)
    # plot_corr_heatmap(df, title="Corrélations des variables numériques")