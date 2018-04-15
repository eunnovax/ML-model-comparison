from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

#[height, wight, shoe size]
X = [[181,80,44],[177,70,43],[160,60,38],[154,54,37]
,[166,65,40],[190,95,43],[175,64,39],[177,70,40],[159,55,37]
,[171,75,42],[181,85,43]]

y = ['male','female','female','female','male','male','male'
,'female','male','female','male']

#Classifiers
clf_tree = tree.DecisionTreeClassifier()
clf_svm = SVC()
clf_regression = LogisticRegression()
clf_knn = KNeighborsClassifier()

# Train/test split
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4,random_state=4)

#Training the models
clf_tree.fit(X_train,y_train)
clf_svm.fit(X_train,y_train)
clf_regression.fit(X_train,y_train)
clf_knn.fit(X_train,y_train)

#Testing using train-test split
pred_tree = clf_tree.predict(X_test)
acc_tree = accuracy_score(y_test,pred_tree)
print ('Accuracy for DecisionTree: {}'.format(acc_tree))

pred_svm = clf_svm.predict(X_test)
acc_svm = accuracy_score(y_test, pred_svm)
print ('Accuracy for support vector machine:{}'.format(acc_svm))

pred_regression = clf_regression.predict(X_test)
acc_regression = accuracy_score(y_test, pred_regression)
print ('Accuracy for Logistic Regression:{}'.format(acc_regression))

pred_knn = clf_knn.predict(X_test)
acc_knn = accuracy_score(y_test,pred_knn)
print('Accuracy for KNN:{}'.format(acc_knn))

# The best classifier from sv, tree, reg, KNN
index = np.argmax([acc_tree,acc_svm,acc_regression,acc_knn])
classifiers = {0:'tree', 1:'svm', 2:'regression', 3:'knn'}
print ('Best shoe gender classifier is {}'.format(classifiers[index]))

