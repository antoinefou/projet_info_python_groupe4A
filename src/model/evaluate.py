import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data.collect import (load__data_url_zip_txt, 
                        load_insee_dossier_complet)

# Prix au m² (variable cible)
df["prix_m2"] = df["valeur_fonciere"] / df["surface_reelle_bati"]

# Année de mutation
df["annee_mutation"] = pd.to_datetime(df["date_mutation"]).dt.year

# Taux de chômage
df["taux_chomage"] = (df["P22_CHOM1564"] / df["P22_ACT1564"]) * 100

# Variable cible
y = df["prix_m2"]

# Variables explicatives
# Les numériques
features_num = [
    "surface_reelle_bati",
    "surface_terrain",
    "nombre_pieces_principales",
    "annee_mutation",       # extraite de date_mutation
    "MED21",                # revenu médian
    "TP6021",               # taux de pauvreté
    "P22_POP",              # population
    "taux_chomage",         # calculé : P22_CHOM1564 / P22_ACT1564 * 100
    "taux_logements_sociaux"
]

# La catégorielle : type_local → one-hot encoding
X = pd.get_dummies(df[features_num + ["type_local"]], columns=["type_local"], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train : {X_train.shape[0]} lignes")
print(f"Test  : {X_test.shape[0]} lignes")

#Régression_linéaire

# Entraîner
reg = LinearRegression()
reg.fit(X_train, y_train)

# Prédire sur le test
y_pred_reg = reg.predict(X_test)

# Évaluer
mae_reg = mean_absolute_error(y_test, y_pred_reg)
rmse_reg = np.sqrt(mean_squared_error(y_test, y_pred_reg))
r2_reg = r2_score(y_test, y_pred_reg)

print("=== Régression Linéaire ===")
print(f"MAE  : {mae_reg:.2f} €/m²")
print(f"RMSE : {rmse_reg:.2f} €/m²")
print(f"R²   : {r2_reg:.4f}")

#Random_Forest 
# Entraîner
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Prédire sur le test
y_pred_rf = rf.predict(X_test)

# Évaluer
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print("=== Random Forest ===")
print(f"MAE  : {mae_rf:.2f} €/m²")
print(f"RMSE : {rmse_rf:.2f} €/m²")
print(f"R²   : {r2_rf:.4f}")


#Comparaison_des modèles 
resultats = pd.DataFrame({
    "Modèle": ["Régression Linéaire", "Random Forest"],
    "MAE (€/m²)": [mae_reg, mae_rf],
    "RMSE (€/m²)": [rmse_reg, rmse_rf],
    "R²": [r2_reg, r2_rf]
})
print(resultats.to_string(index=False))



# Importance des variables (Random Forest)
importances = pd.DataFrame({
    "variable": X.columns,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(importances["variable"], importances["importance"], color="steelblue")
plt.title("Importance des variables - Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()