# Support Vector Machine (SVM) — Classification multiclasse

Ce projet implémente un modèle SVM linéaire pour classer les fleurs du dataset Iris en trois espèces distinctes, avec visualisation des régions de décision.

##  Objectif
Utiliser un modèle `SVC` avec noyau linéaire pour séparer les classes du dataset Iris en fonction de deux caractéristiques.

##  Ce que j’ai appris
- Le fonctionnement d’un SVM linéaire et le rôle des vecteurs de support
- Comment tracer les frontières de décision dans un espace 2D
- L’impact du choix des features sur la visualisation

##  Structure du code
- Chargement du dataset Iris
- Sélection de deux caractéristiques pour simplifier la visualisation
- Entraînement du modèle `SVC(kernel="linear")`
- Visualisation des régions de décision et des vecteurs de support

##  Exécution
```bash
pip install -r requirements.txt
python train.py
