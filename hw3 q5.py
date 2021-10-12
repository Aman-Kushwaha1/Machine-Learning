import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

Xtrain = np.load("housing_train_features.npy")
Xtest = np.load("housing_test_features.npy")
ytrain = np.load("housing_train_labels.npy")
ytest = np.load("housing_test_labels.npy")

d, n = np.shape(Xtrain)
print(d,n)

#Sphering the training data
for i in range(d):
    mean = np.mean(Xtrain[i, :])
    var = np.var(Xtrain[i, :])
    Xtrain[i, :] = ((Xtrain[i, :] - mean)/np.sqrt(var))



# Function for finding out Minimizer

def minimizer(j1, n1, ytrain1, w01, theta1, Xtrain1, lamda1):
    Xtrain1 = np.transpose(Xtrain1)
    m = Xtrain1*theta1
    n = m.sum(axis = 1)
    y = ((ytrain1 - w01) -n)
    y +=  theta[j1]*Xtrain1[:, j1]
    y1 = 2*y*Xtrain1[:, j1]
    c_i = np.sum(y1)
    a_i = 2 * ((np.sum(Xtrain1[:, j1] ** 2)))
    if c_i > lamda1:
        wi = (c_i - lamda1)/a_i
    elif c_i < (-1*lamda1):
        wi = (c_i + lamda1)/a_i
    else:
        wi = 0
    return wi


# Function for finding out mean squared prediction error

def mse(n, ytrain, w0, theta, Xtrain):
    m = 0
    for i in range(n):
        m += ((-ytrain[i] + np.sum(theta * Xtrain[:, i]) + w0) ** 2)
    m = (m / n)
    return m


theta = np.ones(d)
lamda = 100
w0 = np.mean(ytrain)
no_of_iteration = 50
theta_itr = np.zeros([50, 58])
MSE = np.zeros(no_of_iteration)
for m in range(no_of_iteration):
    for i in range(d):
        theta[i] = minimizer(i, n, ytrain, w0, theta, Xtrain, lamda)
    theta_itr[m] = theta
    MSE[m] = mse(n, ytrain, w0, theta, Xtrain)

x_val = np.linspace(1, 50, 50)
for i in range(no_of_iteration):
    plt.plot(x_val, theta_itr[:, i])

plt.xlabel('No of Iterations')
plt.ylabel('Value of Weights')
plt.show()                               #Plotting graph for no of iterations vs value of Weight

plt.plot(x_val, MSE)
plt.xlabel('No of Iterations')
plt.ylabel('Mean Squared Error')
plt.show()                             #Plotting graph for no of iterations vs Mean Sqaured Error

i1 = 0
for i in range(d):
    if theta[i] == 0:
        i1 += 1

print("No of times the weight vector=0 is: {}".format(i1))