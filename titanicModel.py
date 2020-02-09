import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import svm
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
test=pd.read_csv(r"test.csv")
train=pd.read_csv(r"rain.csv")
gender_submission=pd.read_csv(r"gender_submission.csv")
colss=[ 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
ss=test[colss]
tt=train[colss]
ss['Sex'].replace('female',0, inplace=True)
ss['Sex'].replace('male',1, inplace=True)
ss['Age'].fillna(ss['Age'].mean(), inplace=True)
ss['Embarked'].replace('Q',0, inplace=True)
ss['Embarked'].replace('S',1, inplace=True)
ss['Embarked'].replace('C',2, inplace=True)
ss['Fare'].fillna(ss['Fare'].mean(), inplace=True)
tt['Sex'].replace('female',0, inplace=True)
tt['Sex'].replace('male',1, inplace=True)
tt['Age'].fillna(tt['Age'].mean(), inplace=True)
tt['Embarked'].replace('Q',0, inplace=True)
tt['Embarked'].replace('S',1, inplace=True)
tt['Embarked'].replace('C',2, inplace=True)
tt['Embarked'].fillna(0, inplace=True)
X=tt
Y=train['Survived']
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
prediction=clf.predict(ss)
accuracy_score(prediction,gender_submission['Survived'])