from sklearn import *

iris = datasets.load_iris()
x, y = iris.data, iris.target

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.8, random_state=50)

model = tree.DecisionTreeClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
print("Confusion Matrix: \n", metrics.confusion_matrix(y_test, y_pred))