import numpy as np
from scipy import optimize
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib
import random
import math
from copy import deepcopy
import time
#Here we define function f. f is the function that we like to find its global minimum. Here f is the
#famous Rastrigin Function that has many local minima.

time_start = time.time()

def f(z):
    x1 = z[0]
    y1 = z[1]
    h = (x1**2 - 10 * math.cos(2 * 3.14 * x1)) +(y1**2 - 10 * math.cos(2 * 3.14 * y1)) + 20
    return (h)
  
x_start = [0.8, -0.5]
# Let's take a look at how this function looks like. 

i1 = np.arange(-5, 5, 0.002)
i2 = np.arange(-5, 5, 0.002)
x1m, x2m = np.meshgrid(i1, i2)
fm = np.zeros(x1m.shape)
for i in range(x1m.shape[0]):
  for j in range(x1m.shape[1]):
    fm[i][j] = (x1m[i][j]**2 - 10 * np.cos(2 * 3.14 * x1m[i][j])) \
    + (x2m[i][j]**2 - 10 * np.cos(2 * 3.14 * x2m[i][j])) + 20

    
plt.figure()

CS = plt.contour(x1m,x2m,fm)
plt.clabel(CS,inline=1,fontsize=2)
plt.title('Rastrigin Function')
plt.xlabel('x1')
plt.ylabel('y1')

#SA
n = 50; m = 50; na = 0.0; p1 = .7; p50 = .0001
t1 = -1.0/math.log(p1)
t50 = -1.0/math.log(p50)
frac = (t50/t1)**(1.0/n-1.0)

x = np.zeros((n+1,2))
x[0] = deepcopy(x_start)
xi = np.zeros(2)
xi = deepcopy(x_start)
na = na + 1.0

xc = np.zeros(2)
xc = deepcopy(x[0])
fc = f(xi)
fs = np.zeros(n+1)
fs[0] = deepcopy(fc)
    
t = t1

DeltaE_avg = 0.0
#%%
for i in range(n):
    for j in range(m):
        # Generate new trial points
        xi[0] = xc[0] + (random.random() - 0.5)*5
        xi[1] = xc[1] + (random.random() - 0.5)*5
        # Clip to upper and lower bounds
        xi[0] = max(min(xi[0],5.0),-5.0)
        xi[1] = max(min(xi[1],5.0),-5.0)
        DeltaE = abs(f(xi)-fc)
        
        if (f(xi)>fc):
            # Initialize DeltaE_avg if a worse solution was found
            #   on the first iteration
            if (i==0 and j==0): DeltaE_avg = deepcopy(DeltaE)
            # objective function is worse
            # generate probability of acceptance
            p = math.exp(-DeltaE/(DeltaE_avg * t))
            # determine whether to accept worse point
            if (random.random()>p):
                # accept the worse solution
                accept = True
            else:
                # don't accept the worse solution
                accept = False
        else:
            # objective function is lower, automatically accept
            accept = True
            
        if (accept==True):
            # update currently accepted solution
            xc[0] = xi[0]
            xc[1] = xi[1]
            fc = f(xc)
            # increment number of accepted solutions
            na = na + 1.0
            # update DeltaE_avg
            DeltaE_avg = (DeltaE_avg * (na-1.0) +  DeltaE) / na
    # Record the best x values at the end of every cycle
    
    x[i + 1][0] = deepcopy(xc[0])
    x[i + 1][1] = deepcopy(xc[1])
    fs[i + 1] = deepcopy(fc)

    # Lower the temperature for next cycle
    t = frac * t

# print solution
# find best sol:

bestSol = min(fs)
bestIndex = np.where(fs == bestSol)
bestIndex = int(bestIndex[0][0])
bestXY = [x[bestIndex][0],x[bestIndex][1]]

print('My Best sol: ' + str(bestXY))
print('my Best obj: ' + str(bestSol))

print('Best solution: ' + str(xc))
print('Best objective: ' + str(fc))

plt.plot(x[:,0],x[:,1],'y-o')
plt.savefig('contour.png')

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(fs,'r.-')
ax1.legend(['Objective'])
ax2 = fig.add_subplot(212)
ax2.plot(x[:,0],'b.-')
ax2.plot(x[:,1],'g--')
ax2.legend(['x1','x2'])

# Save the figure as a PNG
plt.savefig('iterations.png')

plt.show()

print('This took: ' + str(time.time()-time_start) + ' seconds')