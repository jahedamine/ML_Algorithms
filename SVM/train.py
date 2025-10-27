import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Charger les données
iris = load_iris()
x = iris.data[:, :2]  # Utiliser seulement 2 features pour la visualisation
y = iris.target

# Séparer en train/test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Entraîner le modèle SVM linéaire
model = SVC(kernel="linear")
model.fit(x_train, y_train)

# Créer la figure
plt.figure(figsize=(8, 6))
plt.scatter(x[:, 0], x[:, 1], c=y, s=30, cmap="coolwarm", edgecolors="k")

# Limites des axes
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Création de la grille
xx = np.linspace(xlim[0], xlim[1], 200)
yy = np.linspace(ylim[0], ylim[1], 200)
XX, YY = np.meshgrid(xx, yy)
xy = np.c_[XX.ravel(), YY.ravel()]
z = model.predict(xy).reshape(XX.shape)

# Affichage des régions de décision
plt.contourf(XX, YY, z, cmap="coolwarm", alpha=0.3)

# Vecteurs de support
ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
           s=100, linewidth=1, facecolors='none', edgecolors='k')

# Titres et labels
plt.title("Séparation des classes avec SVM")
plt.xlabel("Caractéristique 1")
plt.ylabel("Caractéristique 2")
plt.show()