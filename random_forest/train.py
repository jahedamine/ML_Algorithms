import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report , confusion_matrix, accuracy_score

# Generate a synthetic classification dataset
X, y = make_classification(n_samples=200, n_features=2, n_informative=2,
                           n_redundant=0, n_classes=2, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test,y_train,y_test = train_test_split(X, y, random_state=42,test_size=0.3)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
acs = accuracy_score(y_pred, y_test)
print(f"accuracy score : {acs:.2f}")

# Confusion matrix
cm = confusion_matrix(y_pred,y_test)
print("confusion matrix :")
print (cm)

# classification report
clr = classification_report(y_pred, y_test)
print("classification rapport  :")
print (clr)

# Visualize the results
plt.figure(figsize=(12, 8))
plt.scatter(X[:,0], X[:,1],c=y , edgecolors='k' , cmap= "coolwarm" , s= 30 , alpha=0.6)
plt.title("Classification avec Random Forest")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()