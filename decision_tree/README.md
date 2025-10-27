# Arbre de Décision — Classification binaire

Ce projet implémente un arbre de décision pour prédire la présence ou non d’un cancer du sein à partir de deux caractéristiques médicales.

## Objectif
Utiliser un modèle `DecisionTreeClassifier` pour classer les patients en fonction de leurs données médicales, et visualiser l’arbre de décision généré.

## Ce que j’ai appris
- Comment fonctionne un arbre de décision et le rôle de la profondeur (`max_depth`)
- L’interprétation de la matrice de confusion
- La visualisation de l’arbre pour comprendre les décisions prises par le modèle

## Structure du code
- Chargement du dataset `load_breast_cancer`
- Sélection de deux caractéristiques pour simplifier la visualisation
- Entraînement du modèle `DecisionTreeClassifier`
- Évaluation avec `accuracy_score` et `confusion_matrix`
- Visualisation de la matrice de confusion et de l’arbre de décision

## Exécution
```bash
pip install -r requirements.txt
python train.py
