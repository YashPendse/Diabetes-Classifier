# Diabetes-Classifier
ML project to classify individual as diabetic and non diabetic using knn algorithm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('/content/diabetes.csv')

data.shape

data

x=data.drop(['Outcome'],axis=1)
x.head()

y=data['Outcome']
y.head()

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()
x=scaler.fit_transform(x)
x

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)

knn.fit(x_train,y_train)

y_pred=knn.predict(x_test)
y_pred

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

error_rate=[]
for i in range(1,40):
  knn=KNeighborsClassifier(n_neighbors=i)
  knn.fit(x_train,y_train)
  pred_i=knn.predict(x_test)
  error_rate.append(np.mean(pred_i!=y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue',linestyle='--',markersize=10,markerfacecolor='red',marker='o')
plt.title('k vs error rate')
plt.xlabel('k')
plt.ylabel('errorrate')

knn=KNeighborsClassifier(n_neighbors=13)
knn.fit(x_train,y_train)
predictions=knn.predict(x_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test,predictions))




