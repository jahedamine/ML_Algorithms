# K-Nearest Neighbors (KNN) — Classification multiclasse

Ce projet implémente un modèle KNN pour classer les fleurs du dataset Iris en trois espèces distinctes.

## Objectif
Utiliser l’algorithme des k plus proches voisins (KNN) pour prédire la classe d’une fleur en fonction de ses caractéristiques.

## Ce que j’ai appris
- Le fonctionnement de KNN comme algorithme basé sur la distance
- L’impact du choix de `k` sur la performance
- Comment visualiser les erreurs de classification avec une matrice de confusion

## Structure du code
- Chargement du dataset Iris
- Séparation des données en train/test
- Entraînement du modèle `KNeighborsClassifier` avec `k=3`
- Évaluation avec `accuracy_score` et `confusion_matrix`
- Visualisation de la matrice de confusion avec Seaborn

## Exécution
```bash
pip install -r requirements.txt
python train.py
