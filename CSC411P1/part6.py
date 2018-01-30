from __future__ import division
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib
import pandas as pd
from numpy import *
from numpy.linalg import norm
from glob import glob
from random import shuffle
from PIL import Image

def f_part6 (x, y, theta):
    #implement using trace: notice that the cost function is equivalent to:
    #tr[(theta.T*X - Y).T(theta.T*X-Y)]
    x = vstack((ones((1, x.shape[1])), x))
    return trace(dot((dot(theta.T,x)-y),(dot(theta.T,x)-y).T))

def df_part6 (x,y, theta): 
    x = vstack((ones((1,x.shape[1])),x))
    return 2*dot(x,(dot(theta.T,x)-y).T)
    
def correctness (x,y,theta,H,h):
    diff = f_part6(x,y,theta+H)-f_part6(x,y,theta)
    return diff/h


#verifying correctness of the gradient: computing along 5 coordinates:
numPeople = 6
numFeatures = 1024
numTraining = 70

np.random.seed(5)
x = np.random.rand(numFeatures,numTraining)
y = np.random.rand(numPeople, numTraining)
theta = np.random.rand(numFeatures+1,numPeople)
for i in range (numFeatures+1):
    for j in range (numPeople):
        if theta[i][j] == 0.:
            theta[i][j] = np.random.random()    #making sure that there is no zero
H = zeros((numFeatures+1,numPeople))

h = 0.1
counter = 5

output = []
hs = []

while counter >= 0:
    hs.append(h)
    h = h/10
    np.random.seed(5)
    row = int(np.random.random()*(numFeatures+1))
    column = int(np.random.random()*numPeople)
    H[row, column] += h

    #dimensions:
    #theta: (# of features, # of people)
    #x: (# of features, # of training images)
    #y: (# of people, # of training images) 
    #
    #dot(theta.T,x): # of people x # of test images    ....ok
    
    a = df_part6(x,y,theta)[row,column]
    b = correctness(x,y,theta,H,h)
    
    print ('When h = '+ str(H[row,column])+', df_part6 gives ' +str(a))
    print ('When h = '+ str(H[row,column])+', finte difference gives '+ str(b))
    percDiff = abs(a-b)/(abs(a)+abs(b))
    output.append(percDiff)
    print ('The percent difference is: '+str(percDiff))
    print ('\n')
    counter = counter - 1
    H = zeros((numFeatures+1,numPeople))
    

plt.plot(hs,output,label = 'Percent Difference')
plt.xlabel('log(h)')
plt.ylabel('log(Error)')
plt.xscale ('log')
plt.yscale ('log')
plt.title('Error of Finite Difference Approximation vs. h')
plt.legend()
plt.show()

#print ('It is clear to see that as we decrease h, the derivative we approximate using finite difference approaches that we compute using vectorization.')


