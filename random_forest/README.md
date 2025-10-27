# Random Forest — Classification binaire synthétique

Ce projet implémente un modèle Random Forest pour classer des données générées artificiellement à l’aide de `make_classification`.

## Objectif
Utiliser un ensemble d’arbres de décision (Random Forest) pour effectuer une classification binaire sur un dataset synthétique à deux dimensions.

## Ce que j’ai appris
- Le fonctionnement d’un modèle Random Forest et l’effet du paramètre `n_estimators`
- L’importance de la variance entre les arbres pour améliorer la robustesse
- Comment visualiser la séparation des classes dans un espace 2D

## Structure du code
- Génération d’un dataset synthétique avec `make_classification`
- Séparation des données en train/test
- Entraînement du modèle `RandomForestClassifier`
- Évaluation avec `accuracy_score`, `confusion_matrix`, `classification_report`
- Visualisation de la distribution des classes dans l’espace des features

## Exécution
```bash
pip install -r requirements.txt
python train.py