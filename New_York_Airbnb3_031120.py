# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 14:28:38 2020
Prep for Insight interview showcase 3-5 min
should include:
    1) data wrangling
    2) exploratory data analysis
    3) statistics
    4) visualization
    5) machine learning
@author: lw1365
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Import data
#df = pd.read_csv('AB_NYC_2019.csv')
ori_df = pd.read_csv(r'C:\Users\lw1365\Desktop\jobs\data science\Kaggle datasets\NYC_AirBB\AB_NYC_2019.csv')
df = ori_df.copy()
df.head()

df.columns

df['neighbourhood_group'].unique()

df = df[df['neighbourhood_group'] == 'Manhattan']
df['neighbourhood_group'].unique()

df['neighbourhood'].unique()

#plt.figure(figsize=(30,30))
g = sns.catplot(x = 'price', y = 'neighbourhood', data = df, kind = 'point')
g.set(xlim = (np.min(df['price']),np.percentile(df['price'], 95))) 

g = sns.catplot(x = 'price', y = 'room_type', data = df, kind = 'point')
g.set(xlim = (np.min(df['price']),np.percentile(df['price'], 95))) 

# Convert the categorical data to numerical values
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
le = LabelEncoder()
df1a = pd.DataFrame([df['neighbourhood'], df['room_type']]).T 
df1a['neighbourhood'] = le.fit_transform(df1a['neighbourhood']) #convert neighhourhood groups to distinct numbers
df1a['room_type'] = le.fit_transform(df1a['room_type']) #convert room types to distinct numbers
ohe = OneHotEncoder()
df1a = ohe.fit_transform(df1a).toarray() # use one hot encoder to convert distinct numbers to columns of 0s and 1s

co_name = sorted(df['neighbourhood'].unique()) + sorted(df['room_type'].unique())
df1b = pd.DataFrame(df1a, columns = co_name)

X = df1b
#y = df['price']
y = [1 if df['price'].iloc[i] >= 200 else 0 for i in range(len(df))] # covert to a classifier problem

''' adding latitude and longitude barely changed the predictor. 
Only incresed the logistic regression: #0.75 with neighbourhood and room type; 
0.80 with neighbourhood, room type, and latitude and longitude
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df1c= pd.DataFrame([df['latitude'], df['longitude']]).T
df1c =  scaler.fit_transform(df1c)
X['latitude'] = df1c[:,0]
X['longitude'] = df1c[:,1]
'''
''' adding number of review on top of neighbourhood and roomtypes also did not improve the algorithm by much. 
LogisticRegression increased to 0.80 from 0.75 
X['review'] = df['number_of_reviews']
X['review'] = X['review'].fillna(0)
sns.distplot(X['review']) #quite skewed'''

'''adding number of review per month on top of neighbourhood and roomtypes also did not improve the algorithm by much. 
LogisticRegression increased to 0.80 from 0.75 
X['reviews_per_month'] = df['reviews_per_month']
X['reviews_per_month'] = X['reviews_per_month'].fillna(0)'''

''' adding all the extra information only improved lr to 0.8 from 0.75. did not improve adaboost
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df1c= pd.DataFrame([df['latitude'], df['longitude']]).T
df1c =  scaler.fit_transform(df1c)
X['latitude'] = df1c[:,0]
X['longitude'] = df1c[:,1]
X['review'] = df['number_of_reviews']
X['review'] = X['review'].fillna(0)
X['reviews_per_month'] = df['reviews_per_month']
X['reviews_per_month'] = X['reviews_per_month'].fillna(0)'''

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.linear_model import LinearRegression
linreg = LinearRegression().fit(X_train, y_train)
print(linreg.score(X_train, y_train)) 
print(linreg.score(X_test, y_test)) 

# Algorithm 4 -- Lasso Regression
# Not sensitive to scaling, slightly better result pre-scaling
from sklearn.linear_model import Lasso
grid_values = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'max_iter': [10, 100, 1000, 10000]}
lr2 = Lasso()
gg = GridSearchCV(lr2, param_grid = grid_values, scoring = 'r2') 
gg.fit(X_train, y_train)
gg.best_score_

linlasso = Lasso(alpha = 1, max_iter = 10000).fit(X_train_scaled, y_train) 

print(linlasso.score(X_train_scaled, y_train)) 
print(linlasso.score(X_test_scaled, y_test)) 


from sklearn.linear_model import LogisticRegression # classification problem
# GridSearch for better parameters
from sklearn.model_selection import GridSearchCV
grid_values = {'C': [1e-6, 1e-5, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}
lr2 = LogisticRegression()
gg = GridSearchCV(lr2, param_grid = grid_values, scoring = 'roc_auc') 
gg.fit(X_train, y_train)
gg.best_score_ #0.75 with neighbourhood and room type; 0.80 with neighbourhood, room type, and latitude and longitude;

clf = gg.best_estimator_
print(clf.score(X_train, y_train)) #0.75
print(clf.score(X_test, y_test)) #0.74

# Adaptive Boosting
from sklearn.ensemble import AdaBoostClassifier
grid_values = {'n_estimators': [5, 10, 25, 50, 100], 'learning_rate': [0.001, 0.01, 0.1, 1, 10]}
ab = AdaBoostClassifier()
gg = GridSearchCV(ab, param_grid = grid_values, scoring = 'roc_auc') # based on 
gg.fit(X_train, y_train)
gg.best_score_ #0.80 with neighbourhood and room type; 0.80 with neighbourhood, room type, and latitude and longitude;

# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB().fit(X_train, y_train)
print(gnb.score(X_train, y_train)) #0.75, adding latitude and longitude made it worse
print(gnb.score(X_test, y_test)) #0.74

from sklearn.metrics import roc_curve, auc
lr = LogisticRegression()
y_score_lr = lr.fit(X_train, y_train).decision_function(X_test)
fpr, tpr, _ = roc_curve(y_test, y_score_lr)
roc_auc_lr = auc(fpr, tpr) #0.80

plt.figure()
plt.plot(fpr, tpr, label = 'ROC')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

# Use decision tree to determine the most prominent features
from sklearn.tree import DecisionTreeClassifier
feature_names = X.columns
class_names = ['price']
clf = DecisionTreeClassifier(max_depth = 5).fit(X_train, y_train)
print(clf.score(X_train, y_train)) 
print(clf.score(X_test, y_test)) 

# Visualize the decision tree
!pip install graphviz
import graphviz
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.externals.six import StringIO
!pip install pydotplus
import pydotplus
from IPython.display import Image, display
dot_data = StringIO()
export_graphviz(clf, dot_data, filled = True, rounded = True, special_characters = True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
display(Image(graph.create_png()))


