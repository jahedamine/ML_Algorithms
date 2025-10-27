# Algorithmes de Machine Learning — Implémentations chirurgicales

Ce dossier regroupe mes implémentations fondamentales des algorithmes classiques de machine learning. Chaque projet est structuré, documenté, et accompagné d’une visualisation ou d’une évaluation chirurgicale. Mon objectif est de transmettre une compréhension profonde et modulaire de chaque méthode.

---

## Vision

Je ne cherche pas à utiliser des frameworks tout faits — je construis chaque brique moi-même. Ce dossier est une preuve vivante de ma maîtrise des algorithmes classiques, avant d’entrer dans le deep learning, le reinforcement learning, et l’alignement GenRL++.

---

## Projets inclus

| Algorithme          | Description                                               | Visualisation        |
|---------------------|----------------------------------------------------------------------------------|
| Linear Regression   | Régression linéaire + Ridge + Lasso sur données médicales | Coefficients         |
| Logistic Regression | Classification binaire sur cancer du sein                 | Matrice de confusion |
| KNN                 | Classification multiclasse sur Iris avec `k=3`            | Matrice de confusion |
| Decision Tree       | Arbre de décision sur 2 features du cancer du sein        | Arbre + Matrice      |
| Random Forest       | Classification sur données synthétiques 2D                | Scatter plot         |
| SVM                 | SVM linéaire sur Iris avec visualisation des vecteurs de support | Frontières de décision |

---

## Exécution

Chaque projet contient :
- Un fichier `train.py` exécutable
- Un `README.md` local avec explication chirurgicale
- Un `requirements.txt` pour les dépendances

```bash
cd ML_Algorithms/linear_regression/
pip install -r requirements.txt
python train.py
