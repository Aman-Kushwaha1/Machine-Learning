import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
x = np.load("pulsar_features.npy")
y = np.load("pulsar_labels.npy")
print(y[0,5])
print(np.random.permutation(8))

#c_i += (((ytrain[i]) - w0 - (np.sum(theta*Xtrain[:,i]) + Xtrain[j,i]*theta[j]))*Xtrain[j,i]*2)
#a_i = 2*np.sum(Xtrain[j,:])
#w_i = (c_i/(a_i + 2*lamda))
#print("wi is {}".format(w_i))

n = np.ones(6)
n = n+ 2
print(n)
a = np.array([[1, 2, 3, 4, 5, 6], [ 2, 2, 2, 2, 2, 2]])
print(a)
print()
print(a*n)
print()

print(a.sum(axis=1))