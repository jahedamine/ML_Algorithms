# Régression Logistique — Classification binaire

Ce projet implémente une régression logistique pour prédire la présence ou non d’un cancer du sein à partir de données médicales.

## Objectif
Utiliser le dataset `load_breast_cancer` de Scikit-learn pour entraîner un modèle de classification binaire et évaluer ses performances.

## Ce que j’ai appris
- L’importance de la stratification dans le split des données
- Le rôle de la régression logistique dans les tâches de classification
- Comment interpréter une matrice de confusion et un rapport de classification

## Structure du code
- Chargement et préparation des données
- Entraînement du modèle `LogisticRegression`
- Prédictions sur le jeu de test
- Évaluation avec `accuracy_score`, `confusion_matrix`, `classification_report`
- Visualisation de la matrice de confusion avec Seaborn

## Exécution
```bash
pip install -r requirements.txt
python train.py
