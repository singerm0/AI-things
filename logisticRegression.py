from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn import preprocessing

#This was solved using basic Stochastic Gradient Descent, then mini batch, then regular gradient descent

Data_set_size = 400
X, Y = make_blobs(n_samples=Data_set_size, centers=2, n_features=2,cluster_std=1.0, center_box=(-4.0, 4.0),random_state=1)
df = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=Y))
colors = {0:'red', 1:'blue'}

test_fraction = .0
training_size = Data_set_size*(1-test_fraction)
test_size = Data_set_size*test_fraction

x_train, x_test, y_train, y_test=train_test_split (X, Y, test_size=test_fraction, random_state=6)

def LogFunc(X,W):
    y_hat = 1/(1+np.exp(-1*(np.dot(X,W))))
    return y_hat
def y_hatFunc(X,W):
    y_hat = np.apply_along_axis(LogFunc,1,X,W)
    return(y_hat)
    
def ErrorForLogReg(X,W,Y):
    y_hat = np.apply_along_axis(LogFunc,1,X,W)
    error = [-1*el[0]*np.log(el[1]) -1*(1-el[0])*np.log(el[1]) for el in zip(Y,y_hat)]
    avgError = np.sum(error)/len(X)
    return(y_hat,avgError)

scaler = preprocessing.StandardScaler().fit(x_train)
x_train_scaled=scaler.transform(x_train)
x_train_scaledp=np.c_[np.ones((int(training_size))),x_train_scaled]  #padding x_train with ones

m = int(training_size)
n_epochs = 50
t0,t1 = 7,50

#colors = plt.cm.jet(np.linspace(0,1,n_epochs))

def learning_schedule(t):
    return t0/(t+t1)


#%% stochasstic gradient descent
W = np.random.randn(3,1)  
Wtrajectory=np.array(W)
inner_loop_num=2

for i in range(n_epochs):
  random_index = np.random.randint(m)
  xi = x_train_scaledp[random_index:random_index+1]
  yi = y_train[random_index:random_index+1]
  gradients = 2 * xi.T.dot(y_hatFunc(xi,W) - yi)
  alpha = learning_schedule(i)
  W = W - alpha * gradients
  Wtrajectory=np.hstack((Wtrajectory,W))
  y_hat = y_hatFunc(x_train_scaledp,W)


fig,ax = plt.subplots(1)
plt.title('Iteration SGD %i'%(i+1))
colors = ['r','b']
for group in range(2):
    points = np.array([x_train[j] for j in range(len(x_train)) if int(np.round(y_hat[j])) == group])
    ax.scatter(points[:,0], points[:,1] , c= colors[group],s=8)
#%% mini batch gradient descent
W2 = np.random.randn(3,1)
Wtrajectory2=np.array(W2)
minibatch_size = 20
t = 0
for epoch in range(n_epochs):
    shuffled_indices = np.random.permutation(m)
    x_train_scaledp_shuffled = x_train_scaledp[shuffled_indices]
    y_train_scaled_shuffled = y_train[shuffled_indices]
    for i in range(0, m, minibatch_size):
        t += 1
        xi = x_train_scaledp_shuffled[i:i+minibatch_size]
        yi = y_train_scaled_shuffled[i:i+minibatch_size]
        gradients = 1/minibatch_size * xi.T.dot(y_hatFunc(xi,W2) - yi)
        alpha = learning_schedule(t)
        W2 = W2 - alpha * gradients
    Wtrajectory2=np.hstack((Wtrajectory2,W2))
    y_hat=y_hatFunc(x_train_scaledp,W2[:,0]) 
    
fig,ax = plt.subplots(1)
plt.title('Iteration Mini-Batch GD %i'%(i+1))
colors = ['r','b']
for group in range(2):
    points = np.array([x_train[j] for j in range(len(x_train)) if int(np.round(y_hat[j])) == group])
    ax.scatter(points[:,0], points[:,1] , c= colors[group],s=8)
#%% regular gradient descent
n_iterations = 50
W3 = np.random.randn(3,1)
Wtrajectory3 = np.array(W3)

for iteration in range(n_iterations):
    gradients = 2/m * x_train_scaledp.T.dot(x_train_scaledp.dot(W) - y_train)
    W3 = W3 - alpha * gradients
    y_hat= y_hatFunc(x_train_scaledp,W3[:,-1])
    Wtrajectory3=np.hstack((Wtrajectory3,W))
    
fig,ax = plt.subplots(1)
plt.title('Iteration regular GD %i'%(i+1))
colors = ['r','b']
for group in range(2):
    points = np.array([x_train[j] for j in range(len(x_train)) if int(np.round(y_hat[j])) == group])
    ax.scatter(points[:,0], points[:,1] , c= colors[group],s=8)

#%% plotting original      
colors = {0:'red', 1:'blue'}      
fig, ax = pyplot.subplots()
plt.title('Initial Data')
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key,s=8, color=colors[key])
pyplot.show()
