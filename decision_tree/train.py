import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier , plot_tree
from sklearn.metrics import accuracy_score , confusion_matrix

data = load_breast_cancer()
x = data.data[:,:2]
y = data.target

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.3, random_state=42)


model = DecisionTreeClassifier(max_depth= 3 , random_state=42)
model.fit(x_train,y_train)

y_pred = model.predict(x_test)


acs = accuracy_score(y_test,y_pred)
print (f"accuracy score : {acs:.2f}")

cm = confusion_matrix(y_test,y_pred) 
print("confusion matrix : ")
print(cm)

plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plot_tree(model , feature_names=data.feature_names, class_names=data.target_names, filled=True)
plt.title("classification tree")
plt.show()