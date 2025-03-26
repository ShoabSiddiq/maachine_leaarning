import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as mtp

data_set=pd.read_csv("Brain_Tumor_Dataset.csv")
x=data_set.iloc[:,1:17]
y=data_set.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=0)

y_test = np.nan_to_num(y_test)
x_test = np.nan_to_num(x_test)
x_train = np.nan_to_num(x_train)
y_train = np.nan_to_num(y_train)

st_x=StandardScaler()
x_train=st_x.fit_transform(x_train)
x_test=st_x.fit_transform(x_test)

lg=LogisticRegression(random_state=1)
lg.fit(x_train,y_train)
y_pred=lg.predict(x_test)

print("Report for Logistic Regration:\n")
print("Confution Metrix : \n",metrics.confusion_matrix(y_test,y_pred))
print("Accuracy: \n", metrics.accuracy_score(y_test, y_pred)*100)
print(metrics.classification_report(y_test, y_pred))

sv=SVC(probability=(True))
sv.fit(x_train, y_train)
y_pred_svm=sv.predict(x_test)

print("Report for Support Vector Matchine: \n")
print("Confution Matrix: \n",metrics.confusion_matrix(y_test, y_pred_svm))
print("Accuracy: \n", metrics.accuracy_score(y_test,y_pred_svm)*100)
print(metrics.classification_report(y_test,y_pred_svm))



ADB=AdaBoostClassifier(n_estimators=4)
ADB.fit(x_train,y_train)
y_pred_ADB=ADB.predict(x_test)

print("Report for ADB: \n")
print("Confusion matrix: \n", metrics.confusion_matrix(y_test, y_pred_ADB))
print("Accuracy: ",metrics.accuracy_score(y_test,y_pred_ADB)*100)
print(metrics.classification_report(y_test, y_pred_ADB))


RF=RandomForestClassifier(n_estimators=2000,random_state=4)
RF.fit(x_train, y_train)
y_pred_rf=RF.predict(x_test)

print("Report for Random Forest: \n")
print("Confution Matrix: \n",metrics.confusion_matrix(y_test, y_pred_rf))
print("Accuracy: \n", metrics.accuracy_score(y_test,y_pred_rf)*100)
print(metrics.classification_report(y_test,y_pred_rf))

knn=KNeighborsClassifier(leaf_size=100, n_neighbors=4)
knn.fit(x_train,y_train)
y_pred_knn=knn.predict(x_test)

print("Report for KNN: \n")
print("Confusion matrix: \n", metrics.confusion_matrix(y_test, y_pred_knn))
print("Accuracy: ",metrics.accuracy_score(y_test,y_pred_knn)*100)
print(metrics.classification_report(y_test, y_pred_knn))












