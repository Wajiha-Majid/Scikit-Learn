from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
#Accuracies using all models and Dataset is Breast cancer
X,y= load_breast_cancer(return_X_y=True)
Xtrain,Xtest,ytrain,ytest= train_test_split(X,y,train_size=0.7, random_state=0)
rf=RandomForestClassifier()
rf.fit(Xtrain,ytrain)
acc=accuracy_score(rf.predict(Xtest),ytest)
print(acc)
lr=LogisticRegression()
lr.fit(Xtrain,ytrain)
acc1=accuracy_score(lr.predict(Xtest),ytest)
print(acc1)
cv1= cross_val_score(lr, X, y, cv=5)
print(cv1)
knn=KNeighborsClassifier()
knn.fit(Xtrain,ytrain)
acc2=accuracy_score(knn.predict(Xtest),ytest)
print(acc2)
dt=DecisionTreeClassifier()
dt.fit(Xtrain,ytrain)
acc3=accuracy_score(dt.predict(Xtest),ytest)
print(acc3)
nb=GaussianNB()
nb.fit(Xtrain,ytrain)
acc4=accuracy_score(nb.predict(Xtest),ytest)
print(acc4)
#Accuracies using all models and Dataset is load iris
X,y= load_iris(return_X_y=True)
X1train,X1test,y1train,y1test= train_test_split(X,y,train_size=0.7, random_state=0)
rf=RandomForestClassifier()
rf.fit(X1train,y1train)
acc5=accuracy_score(rf.predict(X1test),y1test)
print(acc5)
lr=LogisticRegression()
lr.fit(X1train,y1train)
acc6=accuracy_score(lr.predict(X1test),y1test)
print(acc6)
knn=KNeighborsClassifier()
knn.fit(X1train,y1train)
acc7=accuracy_score(knn.predict(X1test),y1test)
print(acc7)
dt=DecisionTreeClassifier()
dt.fit(X1train,y1train)
acc8=accuracy_score(dt.predict(X1test),y1test)
print(acc8)
nb=GaussianNB()
nb.fit(X1train,y1train)
acc9=accuracy_score(nb.predict(X1test),y1test)
print(acc9)
cv= cross_val_score(nb, X, y, cv=5)
print(cv)