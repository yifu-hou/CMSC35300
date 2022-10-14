# CMSC 35300 - HW1
# Author: Yifu Hou
# Program: MSCAPP

import numpy as np
from random import randint

# Q1 - e: script for question 1

# a) build the matrix

X = np.array([
    [2500, 350, 200], 
    [2000, 405, 250], 
    [2000, 325, 400], 
    [2000, 210, 450]]
    )
print(X)

# b) get total expense for each month:

w1 = np.array([1, 1, 1])
by_month = np.dot(X, w1)
print(by_month)

# c) calculate the total expenses in each category across all months

w2 = np.array([1, 1, 1, 1])
by_category = np.dot(X.T, w2)
print(by_category)

# d) calculate total expenses across all categories and months

total = np.dot(np.dot(X, w1).T, w2)
print(total)


# Q2 - g) varify the calculation above

X = np.array([
    [8, 0, 1, 1],
    [9, 2, 9, 4],
    [1, 5, 9, 9], 
    [9, 9, 4, 7], 
    [6, 9, 8, 9]
])

# a) get the fourth row of X with 5 * 1 vector y:

y = np.array([0, 0, 0, 1, 0])
rv = np.dot(y.T, X)
print(rv)

# b) what vector should y be?

k = randint(1, 5)

y = np.zeros(5)
y[k - 1] = 1

print("Get the {} row of X:".format(k))
print(np.dot(y.T, X))

# c) use y vector to get a * k-th row and b * j-th row 

# let's choose a, b randomly in range [1, 10]
a = randint(1, 10)
b = randint(1, 10)
k = randint(1, 5)
j = randint(1, 5)
assert(k != j)

y = np.zeros(5)
y[k - 1] = a
y[j - 1] = b

rv = np.dot(y, X)

print("Expect:")
print(X[k - 1] * a + X[j - 1] * b)
print("Result by matrix multiplication:")
print(rv)

# d) find vector to get the third column of X

w = np.array([0, 0, 1, 0])
rv = np.dot(X, w)

print(rv)

# e) construct a vector w to make Xw the k-th column of X

k = randint(1,4)

w = np.zeros(4)
w[k - 1] = 1

print("Get the {} column of X:".format(k))
print(np.dot(X, w))

# f) construct a vector w to make Xw a times the k-th column of X 
#    plus b times the j-th column of X

# let's choose a, b randomly in range [1, 10]
a = randint(1, 10)
b = randint(1, 10)
k = randint(1, 4)
j = randint(1, 4)
assert(k != j)

w = np.zeros(4)

w[k - 1] = a
w[j - 1] = b
rv = np.dot(X, w)

print("Choose vector w so that Xw = col{} * {} + col{} * {}".format(
    k, a, j, b
))
print("Result by matrix multiplication:")
print(rv)


# Q5 - c Create the polinomial model

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# n = number of points
# z = points where polynomial is evaluated
# p = array to store the values of the interpolated polynomials
n = 100
z = np.linspace(-1, 1, n)

d = 3 # degree
w = np.random.rand(d) 
X = np.zeros((n,d))

# generate X-matrix



# evaluate polynomial at all points z, and store the result in p # do NOT use a loop for this
# plot the datapoints and the best-fit polynomials
plt.plot(z, p, linewidth=2)
plt.xlabel('z')
plt.ylabel('y') 
plt.title('polynomial␣with␣coefficients␣w␣=␣%s' %w ) 
plt.show()