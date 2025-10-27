# Régression Linéaire — Comparaison des variantes

Ce projet implémente et compare trois modèles de régression :
- Régression Linéaire classique
- Régression Ridge (régularisation L2)
- Régression Lasso (régularisation L1)

##  Objectif
Prédire la progression de la maladie chez des patients diabétiques à partir de données médicales.

##  Ce que j’ai appris
- L’impact de la régularisation sur les coefficients
- La différence entre L1 et L2
- L’importance de la normalisation avant la régression

##  Structure du code
- Chargement et préparation des données
- Entraînement des trois modèles
- Évaluation avec MSE et R²
- Visualisation des coefficients

##  Exécution
```bash
pip install -r requirements.txt
python train.py