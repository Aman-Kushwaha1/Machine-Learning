import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

Xtrain = np.load("housing_train_features.npy")
Xtest = np.load("housing_test_features.npy")
ytrain = np.load("housing_train_labels.npy")
ytrain = np.reshape(ytrain,(2000,1))
ytest = np.load("housing_test_labels.npy")

"""
feature_names = np.load("housing_feature_names.npy", allow_pickle=True)
print("First feature name: ", feature_names[0])
print("Lot frontage for first train sample:", Xtrain[0,0])
"""

  #Defining a 58x2000 zero matrix
d,n = Xtrain.shape #Calculating shape of the given dataset of Iowa Housing
#Now let us calculate the mean, variance and then the new "sphered" matrix in one for loop
for i in range(0,d):
    mean = np.mean(Xtrain[i,:])
    var  = np.var(Xtrain[i,:])
    Xtrain[i,:] = (Xtrain[i,:] - mean)/np.sqrt(var)




#Xtrain_new = np.transpose(Xtrain_new)
d,n = Xtrain.shape

W = np.ones(58) #Defining a 58x1 column vector for weights w1,w2,....,w58 as W
w_0 = np.mean(ytrain) #Defining the offset w_0

#editing
def calc(d, Xtrain_new, W, w_0, ytrain, lamda, j):

    Xtrain_new = np.transpose(Xtrain_new)
    n1 = Xtrain_new*W
    y_hat =(n1.sum(axis = 1) + w_0 - W[j]*Xtrain_new[:,j])
    m = ytrain - y_hat
    m = m * Xtrain_new[:,j]
    ans = np.sum(m)
    C_i = ans
    C_i = C_i*2
    A_i = np.sum(Xtrain_new[:, j]**2)
    wi = C_i/(A_i + 2*lamda)
    return wi

#comment
lamda = 100
for j in range(50):
    for i in range(58):
        W[i] = calc(58, Xtrain, W, w_0, ytrain, lamda, i)

print(W)