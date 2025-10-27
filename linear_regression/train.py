import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Chargement des données
data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modèles
model_lr = LinearRegression()
model_ridge = Ridge(alpha=1.0)
model_lasso = Lasso(alpha=0.1)

# Entraînement
model_lr.fit(X_train_scaled, y_train)
model_ridge.fit(X_train_scaled, y_train)
model_lasso.fit(X_train_scaled, y_train)

# Prédictions
y_pred_lr = model_lr.predict(X_test_scaled)
y_pred_ridge = model_ridge.predict(X_test_scaled)
y_pred_lasso = model_lasso.predict(X_test_scaled)

# Évaluations
print("Régression Linéaire classique:")
print(f"  MSE: {mean_squared_error(y_test, y_pred_lr):.2f}")
print(f"  R2: {r2_score(y_test, y_pred_lr):.2f}")

print("\nRidge Regression:")
print(f"  MSE: {mean_squared_error(y_test, y_pred_ridge):.2f}")
print(f"  R2: {r2_score(y_test, y_pred_ridge):.2f}")

print("\nLasso Regression:")
print(f"  MSE: {mean_squared_error(y_test, y_pred_lasso):.2f}")
print(f"  R2: {r2_score(y_test, y_pred_lasso):.2f}")

# Visualisation des coefficients
plt.figure(figsize=(10,6))
plt.plot(model_lr.coef_, 'o-', label='Linéaire')
plt.plot(model_ridge.coef_, 's-', label='Ridge')
plt.plot(model_lasso.coef_, 'x-', label='Lasso')
plt.xlabel('Index des caractéristiques')
plt.ylabel('Valeur du coefficient')
plt.title('Comparaison des coefficients des modèles')
plt.legend()
plt.show()