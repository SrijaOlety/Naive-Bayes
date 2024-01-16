# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 00:52:27 2023

@author: dell
"""

#import the data

import numpy as np
import pandas as pd
df1 = pd.read_csv("D:\\data science python\\NEW DS ASSESSMENTS\\SalaryData_Train(1).csv")
df1
df1.info()
pd.set_option('display.max_columns', None)
df1
df1.isnull().sum()

df2 = pd.read_csv("D:\\data science python\\NEW DS ASSESSMENTS\\SalaryData_Test(1).csv")
df2
df2.info()
pd.set_option('display.max_columns', None)
df2
df2.isnull().sum()


# EDA
#EDA----->EXPLORATORY DATA ANALYSIS

# removing outliers from the training data sample
import seaborn as sns
import matplotlib.pyplot as plt
data = ['age','educationno','capitalgain','capitalloss','hoursperweek']
for column in data:
    plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
    sns.boxplot(x=df1[column])
    plt.title(f" Horizontal Box Plot of {column}")
    plt.show()
#so basically we have seen the ouliers at once without doing everytime for each variable using seaborn#

"""removing the ouliers"""
# List of column names with continuous variables
continuous_columns = ['age','educationno','capitalgain','capitalloss','hoursperweek' ]  
# Create a new DataFrame without outliers for each continuous column
data_without_outliers = df1.copy()
for columns in continuous_columns:
    Q1 = data_without_outliers[column].quantile(0.25)
    Q3 = data_without_outliers[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_whisker_Length = Q1 - 1.5 * IQR
    upper_whisker_Length = Q3 + 1.5 * IQR
    data_without_outliers = data_without_outliers[(data_without_outliers[column] >= lower_whisker_Length) & (data_without_outliers[column]<= upper_whisker_Length)]
# Print the cleaned data without outliers

print(data_without_outliers)
df_train = data_without_outliers
print(df_train.shape) 
print(df_train.info())
df_train

# removing outliers from the testing data sample

continuous_columns = ['age','educationno','capitalgain','capitalloss','hoursperweek']

# Create a new DataFrame without outliers for each continuous column
data_without_outliers = df2.copy()
for column in continuous_columns:
    Q1 = data_without_outliers[column].quantile(0.25)
    Q3 = data_without_outliers[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_whisker = Q1 - 1.5 * IQR
    upper_whisker = Q3 + 1.5 * IQR
    data_without_outliers = data_without_outliers[(data_without_outliers[column] >= lower_whisker) & (data_without_outliers[column] <= upper_whisker)]

# Print the cleaned data without outliers
print(data_without_outliers)
df_test = data_without_outliers
df_test
# Check the shape and info of the cleaned DataFrame
print(df_test.shape)
print(df_test.info())




#HISTOGRAM BUILDING, SKEWNESS AND KURTOSIS CALCULATION #
df_train.hist()
df_train.skew()
df_train.kurt()
df_train.describe() 

df_test.hist()
df_test.skew()
df_test.kurt()
df_test.describe() 




#data partition

# Salary is our target variable

X_train = df_train.drop(columns=['Salary']) 
Y_train = df_train['Salary']               

X_test = df_test.drop(columns=['Salary'])    
Y_test = df_test['Salary']                  

# Standardize the data
# for categorical variables

from sklearn.preprocessing import LabelEncoder
categorical_columns = ['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 'race', 'sex', 'native']
LE = LabelEncoder()
for column in categorical_columns:
    X_train[column] = LE.fit_transform(X_train[column])
    X_test[column] = LE.transform(X_test[column])  

#model fitting using Naive Bayes

from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train,Y_train)
Y_pred_train = mnb.predict(X_train)
Y_pred_train 
Y_pred_test = mnb.predict(X_test)
Y_pred_test

#metrics
from sklearn.metrics import accuracy_score
AC1 = accuracy_score(Y_train,Y_pred_train)
print("Training Accuracy score : ",AC1.round(3))
AC2 = accuracy_score(Y_test,Y_pred_test)
print("Testing Accuracy score : ",AC2.round(3))


""" Training Accuracy score :  0.791
Testing Accuracy score :  0.791"""



# Preparing a Categorical naive bayes model on training data set
from sklearn.naive_bayes import CategoricalNB

CNB = CategoricalNB()

CNB.fit(X_train,Y_train) # Bo + b1x1 + B2x2
Y_pred_train = CNB.predict(X_train)
Y_pred_test = CNB.predict(X_test)

from sklearn.metrics import accuracy_score
# Model Accuracy on train set
training_accuracy = accuracy_score(Y_train,Y_pred_train).round(3)
# Model Accuracy on test set
test_accuracy = accuracy_score(Y_test,Y_pred_test).round(3)

print(training_accuracy)
print(test_accuracy)



# Preparing a Gaussian naive bayes model on training data set
from sklearn.naive_bayes import GaussianNB
GB=GaussianNB()
GB.fit(X_train,Y_train) # Bo + b1x1 + B2x2
Y_pred_train = GB.predict(X_train)
Y_pred_test = GB.predict(X_test)

from sklearn.metrics import accuracy_score
# Model Accuracy on train set
training_accuracy = accuracy_score(Y_train,Y_pred_train).round(3)
# Model Accuracy on test set
test_accuracy = accuracy_score(Y_test,Y_pred_test).round(3)

print(training_accuracy)
print(test_accuracy)