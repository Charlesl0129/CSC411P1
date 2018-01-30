##goal: distinguish Alec Baldwin from Steve Carell
#cost function: Quadratic 
#transform baldwin => 1
#transform carell => -1
#if prediction > 0 => baldwin, otherwise carell

from __future__ import division
import random
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
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

os.chdir(os.path.dirname(__file__))


def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray/255.
    
def f(x, y, theta):
    x = vstack( (ones((1, x.shape[1])), x))
    return sum( (y - dot(theta.T,x)) ** 2)
    
def df(x, y, theta):
    x = vstack( (ones((1, x.shape[1])), x))
    return -2*sum((y-dot(theta.T, x))*x, 1)
    
def grad_descent(f, df, x, y, init_t, alpha):
    EPS = 1e-5   #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = 30000
    iter  = 0
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha*df(x, y, t)
        if iter % 500 == 0:
            print "Iter", iter
            print "x = (%.2f, %.2f, %.2f), f(x) = %.2f" % (t[0], t[1], t[2], f(x, y, t)) 
            print "Gradient: ", df(x, y, t), "\n"
        iter += 1
    return t

direct = os.getcwd()
#fetching the files names:
filelist_male = []
filelist_female = []
for filename in os.walk('final_female'):
    filelist_female.append(filename)
for filename in os.walk('final_male'):
    filelist_male.append(filename)
filelist_male = filelist_male[0][2]
filelist_female = filelist_female [0][2]

#initializing lists for actors:
baldwin_list = []
carell_list = []
hader_list = []
radcliffe_list = []
butler_list = []
vartan_list = []
bracco_list = []
gilpin_list = []
harmon_list = []
chenoweth_list = []
drescher_list = []
ferrera_list = []

for file in filelist_male:
    if 'baldwin' in file:
        baldwin_list.append(file)
    elif 'carell' in file:
        carell_list.append(file)
    elif 'hader' in file:
        hader_list.append(file)
    elif 'radcliffe' in file:
        radcliffe_list.append(file)
    elif 'butler' in file:
        butler_list.append(file)
    elif 'vartan' in file:
        vartan_list.append(file)

for file in filelist_female:
    if 'bracco' in file:
        bracco_list.append(file)
    elif 'gilpin' in file:
        gilpin_list.append(file)
    elif 'harmon' in file:
        harmon_list.append(file)
    elif 'chenoweth' in file:
        chenoweth_list.append(file)
    elif 'drescher' in file:
        drescher_list.append(file)
    elif 'ferrera' in file:
        ferrera_list.append(file)

#randomize:
np.random.seed(2)
np.random.shuffle(baldwin_list)
np.random.shuffle(carell_list)
np.random.shuffle(hader_list)
np.random.shuffle(radcliffe_list)
np.random.shuffle(butler_list)
np.random.shuffle(vartan_list)
np.random.shuffle(bracco_list)
np.random.shuffle(gilpin_list)
np.random.shuffle(harmon_list)
np.random.shuffle(chenoweth_list)
np.random.shuffle(drescher_list)
np.random.shuffle(ferrera_list)

#for 4(a)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
training_baldwin_4b = baldwin_list
training_carell_4b = carell_list
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#initializing lists for training, validation, and test sets:
training_baldwin = []
training_carell = []
training_hader = []
training_radcliffe = []
training_butler = []
training_vartan = []
training_bracco = []
training_gilpin = []
training_harmon = []
training_chenoweth = []
training_drescher = []
training_ferrera = []

validation_baldwin = []
validation_carell = []
validation_hader = []
validation_radcliffe = []
validation_butler = []
validation_vartan = []
validation_bracco = []
validation_gilpin = []
validation_harmon = []
validation_chenoweth = []
validation_drescher = []
validation_ferrera = []

test_baldwin = []
test_carell = []
test_hader = []
test_radcliffe = []
test_butler = []
test_vartan = []
test_bracco = []
test_gilpin = []
test_harmon = []
test_chenoweth = []
test_drescher = []
test_ferrera = []


#forming training, validation, and test sets:
for i in range (70):
    training_baldwin.append(baldwin_list.pop())
    training_carell.append(carell_list.pop())
    training_hader.append(hader_list.pop())
    training_radcliffe.append(radcliffe_list.pop())
    training_butler.append(butler_list.pop())
    training_vartan.append(vartan_list.pop())
    training_bracco.append(bracco_list.pop())
    training_harmon.append(harmon_list.pop())
    training_chenoweth.append(chenoweth_list.pop())
    training_drescher.append(drescher_list.pop())
    training_ferrera.append(ferrera_list.pop())
    
#since there are only 87 usable images for gilpin:
for i in range (67):
    training_gilpin.append(gilpin_list.pop())

for i in range (10):
    validation_baldwin.append(baldwin_list.pop())
    validation_carell.append(carell_list.pop())
    validation_hader.append(hader_list.pop())
    validation_radcliffe.append(radcliffe_list.pop())
    validation_butler.append(butler_list.pop())
    validation_vartan.append(vartan_list.pop())
    validation_bracco.append(bracco_list.pop())
    validation_gilpin.append(gilpin_list.pop())
    validation_harmon.append(harmon_list.pop())
    validation_chenoweth.append(chenoweth_list.pop())
    validation_drescher.append(drescher_list.pop())
    validation_ferrera.append(ferrera_list.pop())
    
for i in range (10):
    test_baldwin.append(baldwin_list.pop())
    test_carell.append(carell_list.pop())
    test_hader.append(hader_list.pop())
    test_radcliffe.append(radcliffe_list.pop())
    test_butler.append(butler_list.pop())
    test_vartan.append(vartan_list.pop())
    test_bracco.append(bracco_list.pop())
    test_gilpin.append(gilpin_list.pop())
    test_harmon.append(harmon_list.pop())
    test_chenoweth.append(chenoweth_list.pop())
    test_drescher.append(drescher_list.pop())
    test_ferrera.append(ferrera_list.pop())

x = array([1]*1024) 
y = array([1]*70)
y = append(y,[-1]*70)

#building the big X matrix, to compute theta
while len(training_baldwin)!= 0:
    toRead = training_baldwin.pop()
    im = imread ('final_male/'+toRead)
    im = im[:,:,0]/255
    im_1d = im.flatten()
    x = vstack((x,im_1d))
  
while len(training_carell)!= 0:
    toRead = training_carell.pop()
    im = imread ('final_male/'+toRead)
    im = im[:,:,0]/255
    im_1d = im.flatten()
    x = vstack((x,im_1d))

x = delete(x,0,0)
x = x.T
x_training = x
theta0 = np.random.normal(scale=0.0,size = 1025)

theta = grad_descent (f, df, x, y, theta0,0.00001)
print ('The value of the cost function on the training set is: '+str(f(x,y,theta)))

#now for VALIDATION SET:
x = array([1]*1024) 
y = array([1]*10)
y = append(y,[-1]*10)

while len(validation_baldwin)!= 0:
    toRead =validation_baldwin.pop()
    im = imread ('final_male/'+toRead)
    im = im[:,:,0]/255
    im_1d = im.flatten()
    x = vstack((x,im_1d))
  
while len(validation_carell)!= 0:
    toRead = validation_carell.pop()
    im = imread ('final_male/'+toRead)
    im = im[:,:,0]/255
    im_1d = im.flatten()
    x = vstack((x,im_1d))

x = delete(x,0,0)
x = x.T
print ('The value of the cost function on the validation set is: ' + str(f(x,y,theta)))

x_4b = array([1]*1024)
y_4b = array([1]*2)
y_4b = append(y_4b,[0]*2)
for i in range (2):
    toRead_4b = training_baldwin_4b.pop()
    im_4b = imread('final_male/'+toRead_4b)
    im_4b = im_4b[:,:,0]/225
    im_1d_4b = im_4b.flatten()
    x_4b = vstack((x_4b,im_1d_4b))
for i in range (2):
    toRead_4b = training_carell_4b.pop()
    im_4b = imread('final_male/'+toRead_4b)
    im_4b = im_4b[:,:,0]/225
    im_1d_4b = im_4b.flatten()
    x_4b = vstack((x_4b,im_1d_4b))
    
x_4b = delete(x_4b,0,0)
x_4b = x_4b.T
x_training_4b = x_4b
theta0_4b = np.random.normal(scale=0.0,size = 1025)
theta_4b = grad_descent (f, df, x_4b, y_4b, theta0_4b,0.000001)

x = vstack( (ones((1, x.shape[1])), x))
prediction = dot(theta.T,x)
numCorrect = 0
for i in range (10):
    if prediction [i] > 0.0: #output: baldwin
        numCorrect += 1
for i in range (10,20):
    if prediction [i] <= 0.0: #output: carell
        numCorrect += 1
        
print ('The success rate on the validation set is: ' + str(numCorrect/20))

x_training = vstack((ones((1,x_training.shape[1])),x_training))
prediction_training = dot(theta.T,x_training)

numCorrect = 0
for i in range (70):
    if prediction_training [i] > 0.0:
        numCorrect += 1
for i in range (70,140):
    if prediction_training [i] <= 0.0:
        numCorrect += 1
print ('The success rate on the training set is: ' + str(numCorrect/140))

## Part 4 (a)
## Using 70 images:
theta_to_imageA = reshape(theta[1:],(32,32))
plt.imshow (theta_to_imageA,cmap = "RdBu")
#plt.savefig ('Theta70.png')
plt.show()


## Using 2 images: (Umcomment this to see visualization using two images each)
#theta_to_imageB = reshape(theta_4b[1:],(32,32))
#plt.imshow (theta_to_imageB,cmap = "RdBu")
#plt.savefig ('Theta2.png')
#plt.show()