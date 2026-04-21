"""Entraînement des modèles de prédiction du prix au m²"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def prepare_features(df):
    """Prépare X et y pour la modélisation."""
    
    y = df["prix_m2"]

    features_num = [
        "surface_reelle_bati",
        "nombre_pieces_principales",
        "mediane_niveau_vie",
        "population",
        "taux_chomage",
        "taux_logements_sociaux",
        "densite"
    ]
 
    X = pd.get_dummies(df[features_num + ["type_local"]], columns=["type_local"], drop_first=True)
    X = X.apply(pd.to_numeric, errors="coerce")
    
    # Supprimer les NaN restants
    mask = X.notna().all(axis=1)
    X = X[mask]
    y = y.loc[X.index]

    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """Split train/test."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_linear_regression(X_train, y_train):
    """Entraîne une régression linéaire."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """Entraîne un Random Forest."""
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train, n_estimators=100, random_state=42):
    """Entraîne un Gradient Boosting."""
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model