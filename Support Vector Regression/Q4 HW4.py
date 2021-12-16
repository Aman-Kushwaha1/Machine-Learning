import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

# You have have to install the libraries below.
# sklearn, csv
import csv

# The csv file air-quality-train.csv contains the training data.
# After loaded, each row of X_train will correspond to CO, NO2, O3, SO2.
# The vector y_train will contain the PM2.5 concentrations.
# Each row of X_train corresponds to the same timestamp.
X_train = []
y_train = []

with open('air-quality-train.csv', 'r') as air_quality_train:
    air_quality_train_reader = csv.reader(air_quality_train)
    next(air_quality_train_reader)
    for row in air_quality_train_reader:
        row = [float(string) for string in row]
        row[0] = int(row[0])
        
        X_train.append([row[1], row[2], row[3], row[4]])
        y_train.append(row[5])
        
# The csv file air-quality-test.csv contains the testing data.
# After loaded, each row of X_test will correspond to CO, NO2, O3, SO2.
# The vector y_test will contain the PM2.5 concentrations.
# Each row of X_train corresponds to the same timestamp.
X_test = []
y_test = []

with open('air-quality-test.csv', 'r') as air_quality_test:
    air_quality_test_reader = csv.reader(air_quality_test)
    next(air_quality_test_reader)
    for row in air_quality_test_reader:
        row = [float(string) for string in row]
        row[0] = int(row[0])
        
        X_test.append([row[1], row[2], row[3], row[4]])
        y_test.append(row[5])

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
d, n = X_train.shape
print(y_train[4])

# TODOs for part (a)
#    1. Use SVR loaded to train a SVR model with rbf kernel, regularizer (C) set to 1 and rbg kernel parameter (gamma) 0.1
#    2. Print the RMSE on the test dataset

regressor = SVR(kernel= 'rbf', C= 1, gamma= 0.1)
regressor.fit(X_train, y_train)
pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, pred)
print("The value of RMSE for Q1 is: {}".format(np.sqrt(mse)))

# TODOs for part (b)
#    1. Use KernelRidge to train a Kernel Ridge  model with rbf kernel, regularizer (C) set to 1 and rbg kernel parameter (gamma) 0.1
#    2. Print the RMSE on the test dataset 

regressor2 = KernelRidge(kernel= 'rbf', alpha = 0.5, gamma= 0.1)
regressor2.fit(X_train, y_train)
pred2 = regressor2.predict(X_test)
mse2 = mean_squared_error(y_test, pred2)
print("The value of RMSE for Q2 is: {}".format(np.sqrt(mse2)))

# Use this seed.
seed = 0
np.random.seed(seed) 

K = 5 #The number of folds we will create 

# TODOs for part (c)
#   1. Create a partition of training data into K=5 folds 
#   Hint: it suffice to create 5 subarrays of indices   


# Specify the grid search space 
reg_range = np.logspace(-1,1,3)       # Regularization paramters
kpara_range = np.logspace(-2, 0, 3)   # Kernel parameters

#Finding best parameter values for SVR
clf = GridSearchCV(SVR(), {
    'C': reg_range , 'gamma': kpara_range ,
    'kernel': ['rbf']
}, cv=5, return_train_score=False)
clf.fit(X_train, y_train)
print()
print("The best parsmeters for SVR is: {} ".format(clf.best_params_))

#Finding best parameter values for KernalRidge
clf = GridSearchCV(KernelRidge(), {
    'alpha': 1/(2*reg_range) , 'gamma': kpara_range ,
    'kernel': ['rbf']
}, cv=5, return_train_score=False)
clf.fit(X_train, y_train)
print()
print("The best parsmeters for Kernal Ridge is: {} ".format(clf.best_params_))

# TODOs for part (d)
#   1.  Select the best parameters for both SVR and KernelRidge based on k-fold cross-validation error estimate (use RMSE as the performance metric)
#   2.  Print the best paramters for both SVR and KernelRidge selected
#   3.  Train both SVR and KernelRidge on the full training data with selected best parameters
#   4.  Print both the RMSE on the test dataset of SVR and KernelRidge

regressor = SVR(kernel= 'rbf', C= 10, gamma= 0.01)
regressor.fit(X_train, y_train)
pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, pred)
print()
print("The value of RMSE for best values of C and gamma in SVR is: {}".format(np.sqrt(mse)))

regressor2 = KernelRidge(kernel= 'rbf', alpha = 0.05, gamma= 0.01)
regressor2.fit(X_train, y_train)
pred2 = regressor2.predict(X_test)
mse2 = mean_squared_error(y_test, pred2)
print("\nThe value of RMSE for best values of alpha and gamma in KernalRidge is: {}".format(np.sqrt(mse2)))