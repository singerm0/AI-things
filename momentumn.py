# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 12:41:28 2018

@author: Matthew
"""

# with MOMENTUM

# This is SGD without momentum. Add momentum to this code and compare the performance
import matplotlib.pyplot as plt
import numpy as np
import math

#this is where we define the function f. this functio nreturns back f(x)
def non_convex_f(z):
  return (z**4-3*z**2+2*z-2)

#this function returns back the derivative of the defined the function f at different values x, i.e. it returns back df/d(x) evaluated at x.
def d_non_convex_f(z):
  return(4*z**3-6*z+2)

alpha = 0.01  #step size (learning rate in context of machine learning)
n_iterations = 20

colors = plt.cm.jet(np.linspace(0,1,n_iterations))

np.random.seed(seed=1)
x = 2 #initial condition
trajectory=np.array([x])


plt.plot(x,non_convex_f(x),'k.')

precision=0.001
step_size=1
iteration=0
momentum  = [0]*10
print(step_size)

while (step_size > precision) & (iteration < n_iterations):
    descent = alpha*d_non_convex_f(x+float(np.mean(momentum))*4)
    new_x=x-descent - float(np.mean(momentum))*4
    momentum[iteration%10] = descent
    step_size = abs(new_x)
    x=new_x
    plt.plot(x,d_non_convex_f(x))
    iteration=iteration+1
    trajectory=np.append(trajectory,x)
    

plt.plot(trajectory,non_convex_f(trajectory),'r*')
plt.quiver(trajectory[:-1], non_convex_f(trajectory)[:-1], trajectory[1:]-trajectory[:-1], non_convex_f(trajectory)[1:]-non_convex_f(trajectory)[:-1], scale_units='xy', angles='xy', scale=1)
plt.axis([-5, 5, -8, 8])
xx=np.linspace(-5,5,100)
yy=non_convex_f(xx)
plt.plot(xx,yy,'b')
plt.show()
print()