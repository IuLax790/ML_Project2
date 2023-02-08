import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
import plotly.express as px

Poverty = pd.read_csv("poverty_brazil.csv")

Poverty['non_white'].fillna(Poverty['non_white'].value_counts().idxmax(), inplace=True)

#Education_type:
# 1 - No education and less than 1 year of study
# 2 - Incomplete elementary or equivalent
# 3 - Complete fundamental or equivalent
# 4 - Incomplete audio or equivalent
# 5 - Complete audio or equivalent
# 6 - Incomplete higher or equivalent
# 7 - Superior complete
colored = Poverty.groupby(['non_white','education'])['poverty'].value_counts().reset_index(name='Count')
fig = px.treemap(colored, path=['education','non_white','poverty','Count'],color='non_white')
fig.update_layout(
    title='Non White or Not with education type, poverty and count')
fig.show()

#Working type:
# 1 - Agriculture, livestock, forestry, fisheries and aquaculture
# 2 - General industry
# 3 - Construction
# 4 - Trade, repair of motor vehicles and motorcycles
# 5 - Transport, storage and mail
# 6 - Accommodation and food
# 7 - Information, communication and financial, real estate, professional and administrative
# 8 - Public administration, defense and social security
# 9 - Education, human health and social services
# 10 - Other Services
# 11 - Home Services
# 12 - Ill-defined activities

metropole = Poverty.groupby(['metropolitan_area','work'])['poverty'].value_counts().reset_index(name='Count')
fig = px.treemap(metropole, path=['metropolitan_area','work','poverty','Count'],color='metropolitan_area')
fig.update_layout(
    title='Metropole or not with working type,poverty and count')
fig.show()

print(Poverty.isnull().sum())


X = Poverty.iloc[:,:-1]
y = Poverty.iloc[:,-1]



plt.figure(figsize=(10,10))
print(sns.heatmap(Poverty.corr(), annot=True, fmt='.2f'))
print(plt.show())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
sc = StandardScaler()
X_train= sc.fit_transform(X_train)
X_test = sc.transform(X_test)

def logistic_regression(X_train,X_test,y_train,y_test):
  from sklearn.metrics import accuracy_score,classification_report
  LR = LogisticRegression(random_state = 0)
  LR.fit(X_train,y_train)
  print("Logistic Regression\n",classification_report(y_test, LR.predict(X_test)),"\n")
  print(accuracy_score(y_test, LR.predict(X_test)),"\n")


def SVM(X_train,X_test,y_train,y_test):
  from sklearn.svm import SVC
  from sklearn.metrics import accuracy_score,classification_report
  svc = SVC(kernel = 'linear', random_state = 0)
  svc.fit(X_train,y_train)
  print("SVM\n",classification_report(y_test, svc.predict(X_test)),"\n")
  print(accuracy_score(y_test, svc.predict(X_test)),"\n")

def kernSVM(X_train,X_test,y_train,y_test):
  from sklearn.svm import SVC
  from sklearn.ensemble import AdaBoostClassifier
  from sklearn.metrics import accuracy_score,classification_report
  kernelsvm = SVC(kernel = 'rbf', random_state = 42)
  kernelsvm.fit(X_train,y_train)
  print("KernSVM\n",classification_report(y_test, kernelsvm.predict(X_test)),"\n")
  print(accuracy_score(y_test, kernelsvm.predict(X_test)),"\n")

def Naive_Bayes(X_train,X_test,y_train,y_test):
  from sklearn.naive_bayes import GaussianNB
  from sklearn.metrics import accuracy_score,classification_report
  NB = GaussianNB()
  NB.fit(X_train,y_train)
  print("Naive_Bayes\n",classification_report(y_test, NB.predict(X_test)),"\n")
  print(accuracy_score(y_test, NB.predict(X_test)),"\n")

logistic_regression(X_train,X_test,y_train,y_test)
SVM(X_train,X_test,y_train,y_test)
kernSVM(X_train,X_test,y_train,y_test)
Naive_Bayes(X_train,X_test,y_train,y_test)

