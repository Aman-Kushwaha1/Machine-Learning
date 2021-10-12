import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
x = np.load("pulsar_features.npy")
y = np.load("pulsar_labels.npy")

negInd = y == -1
posInd = y == 1
plt.scatter(x[0, negInd[0, :]], x[1, negInd[0, :]], color='b', s=0.3)
plt.scatter(x[0, posInd[0, :]], x[1, posInd[0, :]], color='r', s=0.3)
plt.figure(1)

d,n = x.shape
w = np.zeros(2)
b = 1
lamda = 0.001
j_i = np.zeros(n)

for i in range(n):
    m1 = (1 - ((np.sum(w*x[:, i])+b)*y[0, i]))
    m2 = max(0, m1)
    j_i[i] = ((m1 + (np.sum(w**2)*0.5*lamda))/n)

print(np.sum(j_i))
# Finding out Gradient of w1, w2 and b
def grad(x, y, w, lamda, b, n):
    sub_w1 = np.zeros(n)
    sub_w2 = np.zeros(n)
    sub_b = np.zeros(n)
    for i in range(n):
        if (x[0, i]*y[0,i]*w[0]) < 1:
            sub_w1[i] = (((x[0, i]*y[0,i]*-1) + lamda*w[0])/n)
        if (x[1, i]*y[0,i]*w[1]) < 1:
            sub_w2[i] = (((x[1, i]*y[0,i]*-1) + lamda*w[1])/n)
        if (y[0,i]*b) < 1:
            sub_b[i] = ((-1*y[0,i])/n)
    sub_w1 = np.sum(sub_w1)
    sub_w2 = np.sum(sub_w2)
    sub_b = np.sum(sub_b)
    return sub_w1, sub_w2, sub_b

# Finding out value of j
def j(x, y, w, lamda, b, n):
    for i in range(n):
        m = 1 - (np.sum(w*x[:,i]*y[0,i]) + b*y[0,i])
        m = max(0, m)
        m = m + (lamda*np.sum(w**2))
    return m

alpha = 100
i = 0
j_plot = np.zeros(10)
while i<10:
    i += 1
    grad_w1, grad_w2, grad_b = grad(x, y, w, lamda, b, n)
    w[0] -= (alpha/i)*grad_w1
    w[1] -= (alpha/i)*grad_w2
    b -= (alpha/i)*grad_b
    j_plot[i-1] = j(x, y, w, lamda, b, n)


print(w, b)
x_graph = np.linspace(0, 1, 500)
y_graph = (((-1*b)/w[1]) - ((w[0]*x_graph)/w[1]))
plt.plot(x_graph, y_graph)                                 #plotting The data and the learned line
plt.show()
x_graph = np.linspace(1, 10, 10)
plt.plot(x_graph, j_plot)                                  #plotting J vs no of iterations
plt.xlabel('Iteration number')
plt.ylabel("J")
plt.show()

# Q1 c
i1 = np.random.permutation(10)
alpha = 100
j_plot = np.zeros(10)
for i in range(10):
    i += 1
    grad_w1, grad_w2, grad_b = grad(x, y, w, lamda, b, n)
    w[0] -= ((alpha/(i*n))*grad_w1)
    w[1] -= (alpha/(i*n))*grad_w2
    b -= (alpha/(i*n))*grad_b
    j_plot[i-1] = j(x, y, w, lamda, b, n)

plt.scatter(x[0, negInd[0, :]], x[1, negInd[0, :]], color='b', s=0.3)
plt.scatter(x[0, posInd[0, :]], x[1, posInd[0, :]], color='r', s=0.3)
plt.figure(1)

print(w, b)
x_graph = np.linspace(0, 1, 500)
y_graph = (((-1*b)/w[1]) - ((w[0]*x_graph)/w[1]))
plt.plot(x_graph, y_graph)
plt.title("Part c")
plt.show()                                            #plotting The data and the learned line
x_graph = np.linspace(1, 10, 10)
plt.plot(x_graph, j_plot)                             #plotting J vs no of iterations
plt.xlabel('Lamda (λ)')
plt.ylabel("J")
plt.title("Part c")
plt.show()



