import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix
# Chargement des données:
data = load_breast_cancer()
x = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# separation x et y:
x_train, x_test ,y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

# model 
model = LogisticRegression(max_iter=10000)

# Entraînement
model.fit(x_train,y_train)

print("Modèle entraîné avec succès.")

# Prédictions
y_pred = model.predict(x_test)
print("Prédictions réalisées sur le jeu de test.")
print("Extrait des prédictions :", y_pred[:10])

# Évaluations
ac = accuracy_score(y_test,y_pred)
print(f"the scoor is :{ac:.4f}")

cm = confusion_matrix(y_test,y_pred)
print("Matrice de confusion :")
print(cm)

cr= classification_report(y_test,y_pred,target_names=data.target_names)
print ("Rapport de classification :")
print(cr)
# 🔹 Visualisation de la matrice de confusion
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
plt.title("Matrice de confusion — Régression Logistique")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.tight_layout()
plt.show()