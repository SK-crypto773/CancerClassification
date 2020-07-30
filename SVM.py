# Comparing two algorithms on detecting if a tumor is malignant or benign

import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

#print(cancer.feature_names)
#print(cancer.target_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

classes = ['malignant', 'benign']

knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(x_train, y_train)

clf = svm.SVC(kernel="linear", C=2)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
knn_y_pred = knn.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)
knn_acc = metrics.accuracy_score(y_test, knn_y_pred)

print("SVM: ", acc * 100, "%")
print("KNN: ", knn_acc * 100, "%")