import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score , confusion_matrix

# Chargement des données Iris
iris = load_iris()
x = iris.data
y = iris.target

# Séparation train/test
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.25 , random_state=42)

# Création du modèle KNN avec K=3
knn = KNeighborsClassifier(n_neighbors=3)

# Entraînement
knn.fit(x_train,y_train)

# Prédictions
y_pred = knn.predict(x_test)

# Évaluation
ac = accuracy_score(y_test,y_pred)
print("accuracy score")
print(f"{ac:.2f}")
cm = confusion_matrix(y_test,y_pred) 
print("confusion")
print(cm)

# Visualisation de la matrice de confusion:
plt.figure(figsize=(6,4))
sns.heatmap(cm , annot=True, fmt="d",
            xticklabels=iris.target_names,
            yticklabels=iris.target_names, cmap="Blues")
plt.xlabel('Prédiction')
plt.ylabel('Vérité terrain')
plt.title('Matrice de confusion KNN')
plt.show()