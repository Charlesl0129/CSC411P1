from __future__ import division
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imsave
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

def f_part6 (x, y, theta):
    #implement using trace: notice that the cost function is equivalent to:
    #tr[(theta.T*X - Y).T(theta.T*X-Y)]
    x = vstack((ones((1, x.shape[1])), x))
    return trace(dot((dot(theta.T,x)-y),(dot(theta.T,x)-y).T))

def df_part6 (x,y, theta): 
    x = vstack((ones((1,x.shape[1])),x))
    return 2*dot(x,(dot(theta.T,x)-y).T)
    
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
            # print "x = (%.2f, %.2f, %.2f), f(x) = %.2f" % (t[0], t[1], t[2], f(x, y, t)) 
            # print "Gradient: ", df(x, y, t), "\n"
        iter += 1
    return t

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#dimensions:
#theta: (# of features, # of people)
#x: (# of features, # of training images)
#y: (# of people, # of training images) 
#
#dot(theta.T,x): # of people x # of test images    ....ok

#theta: (1024+1,6)
#x: (1024+1,6*70)
#y: (6, 6*70)

#fetching the files names:
filelist_male = []
filelist_female = []
for filename in os.walk('final_female'):
    filelist_female.append(filename)
for filename in os.walk('final_male'):
    filelist_male.append(filename)
filelist_male = filelist_male[0][2]
filelist_female = filelist_female [0][2]

#getting all images of actors:
baldwin_list = []
carell_list = []
hader_list = []
for file in filelist_male:
    if 'baldwin' in file:
        baldwin_list.append(file)
    elif 'carell' in file:
        carell_list.append(file)
    elif 'hader' in file:
        hader_list.append(file)

#getting all images of actresses:
bracco_list = []
gilpin_list = []
harmon_list = []
for file in filelist_female:
    if 'bracco' in file:
        bracco_list.append(file)
    elif 'gilpin' in file:
        gilpin_list.append(file)
    elif 'harmon' in file:
        harmon_list.append(file)
        
#shuffle the lists for randomness:
np.random.seed(0)
np.random.shuffle(baldwin_list)
np.random.shuffle(carell_list)
np.random.shuffle(hader_list)
np.random.shuffle(bracco_list)
np.random.shuffle(gilpin_list)
np.random.shuffle(harmon_list)

#initialize training sets:
training_baldwin = []
training_carell = []
training_hader = []
training_bracco = []
training_gilpin = []
training_harmon = []

#initialize validation sets:
validation_baldwin = []
validation_carell = []
validation_hader = []
validation_bracco = []
validation_gilpin = []
validation_harmon = []

#initialize test sets:
test_baldwin = []
test_carell = []
test_hader = []
test_bracco = []
test_gilpin = []
test_harmon = []

#forming training sets:
for i in range (70):
    training_baldwin.append(baldwin_list.pop())
    training_carell.append(carell_list.pop())
    training_hader.append(hader_list.pop())
    training_bracco.append(bracco_list.pop())
    training_harmon.append(harmon_list.pop())
    training_gilpin.append(gilpin_list.pop())

#since there are only 87 usable images for gilpin, I reduce the number of training images to 67 instead of 70 for gilpin:
# for i in range (67):
#     training_gilpin.append(gilpin_list.pop())
    
#forming validation sets:
for i in range (10):
    validation_baldwin.append(baldwin_list.pop())
    validation_carell.append(carell_list.pop())
    validation_hader.append(hader_list.pop())
    validation_bracco.append(bracco_list.pop())
    validation_gilpin.append(gilpin_list.pop())
    validation_harmon.append(harmon_list.pop())

#forming test sets:
# for i in range (10):
#     test_baldwin.append(baldwin_list.pop())
#     test_carell.append(carell_list.pop())
#     test_hader.append(hader_list.pop())
#     test_bracco.append(bracco_list.pop())
#     test_gilpin.append(gilpin_list.pop())
#     test_harmon.append(harmon_list.pop())
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#dimensions:
#theta: (1024+1,6)
#x: (1024+1,6*70)
#y: (6, 6*70)

#initializing theta, x, y, theta:
theta0 = random.normal(scale=0.0,size = (1025,6))
#make x.shape be (420,1024) first then take transpose for easier loading:
x = ones ((1,1024))
y = array([0.,0.,0.,0.,0.,0.])
for i in range (70):
    y = vstack((y,[1.,0.,0.,0.,0.,0.])) #bracco
for i in range (70):
    y = vstack((y,[0.,1.,0.,0.,0.,0.]))  #gilpin  
for i in range (70):
    y = vstack((y,[0.,0.,1.,0.,0.,0.]))  #harmon
for i in range (70):
    y = vstack((y,[0.,0.,0.,1.,0.,0.]))  #baldwin
for i in range (70):
    y = vstack((y,[0.,0.,0.,0.,1.,0.]))  #hader
for i in range (70):
    y = vstack((y,[0.,0.,0.,0.,0.,1.]))  #carell
y = delete(y,0,0)
y = y.T


#initializing big X matrix: using 70 images each actor/actresses for training
#(IMPORTANT) ordering: bracco - gilpin - harmon - baldwin - hader - carell

#loading all actresses images into X:
while len(training_bracco)!= 0:
    toRead = training_bracco.pop()
    im = imread ('final_female/'+toRead)
    im = im[:,:,0]/255
    im_1d = im.flatten()
    x = vstack((x,im_1d))
while len(training_gilpin)!= 0:
    toRead = training_gilpin.pop()
    im = imread ('final_female/'+toRead)
    im = im[:,:,0]/255
    im_1d = im.flatten()
    x = vstack((x,im_1d))
while len(training_harmon)!= 0:
    toRead = training_harmon.pop()
    im = imread ('final_female/'+toRead)
    im = im[:,:,0]/255
    im_1d = im.flatten()
    x = vstack((x,im_1d))

# #loading all actors images into X:
while len(training_baldwin)!= 0:
    toRead = training_baldwin.pop()
    im = imread ('final_male/'+toRead)
    im = im[:,:,0]/255
    im_1d = im.flatten()
    x = vstack((x,im_1d))
while len(training_hader)!= 0:
    toRead = training_hader.pop()
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
x_training = vstack((ones((1,x.shape[1])),x))

# 
theta = grad_descent (f_part6, df_part6, x, y, theta0,0.000001)


#performance on the training set:
# 
prediction = dot(theta.T,x_training)
prediction = prediction.T
numCorrect = 0
for i in range (70):                    #bracco
    index = prediction[i].argmax()
    if index == 0:
        numCorrect += 1
for i in range (70,140):                #gilpin
    index = prediction[i].argmax()
    if index == 1:
        numCorrect += 1
for i in range (140,210):               #harmon
    index = prediction[i].argmax()
    if index == 2:
        numCorrect += 1
for i in range (210,280):               #baldwin
    index = prediction[i].argmax()
    if index == 3:
        numCorrect += 1
for i in range (280,350):               #hader
    index = prediction[i].argmax()
    if index == 4:
        numCorrect += 1
for i in range (350,420):               #carell
    index = prediction[i].argmax()
    if index == 5:
        numCorrect += 1
print ('The success rate on the training set is: ' + str(numCorrect/420))





# #performance on the validation set:
x = ones ((1,1024))
y = array([0.,0.,0.,0.,0.,0.])
for i in range (10):
    y = vstack((y,[1.,0.,0.,0.,0.,0.])) #bracco
for i in range (10):
    y = vstack((y,[0.,1.,0.,0.,0.,0.]))  #gilpin  
for i in range (10):
    y = vstack((y,[0.,0.,1.,0.,0.,0.]))  #harmon
for i in range (10):
    y = vstack((y,[0.,0.,0.,1.,0.,0.]))  #baldwin
for i in range (10):
    y = vstack((y,[0.,0.,0.,0.,1.,0.]))  #hader
for i in range (10):
    y = vstack((y,[0.,0.,0.,0.,0.,1.]))  #carell
y = delete(y,0,0)


# #(IMPORTANT) ordering: bracco - gilpin - harmon - baldwin - hader - carell
while len(validation_bracco)!= 0:
    toRead =validation_bracco.pop()
    im = imread ('final_female/'+toRead)
    im = im[:,:,0]/255
    im_1d = im.flatten()
    x = vstack((x,im_1d))
while len(validation_gilpin)!= 0:
    toRead = validation_gilpin.pop()
    im = imread ('final_female/'+toRead)
    im = im[:,:,0]/255
    im_1d = im.flatten()
    x = vstack((x,im_1d))
while len(validation_harmon)!= 0:
    toRead = validation_harmon.pop()
    im = imread ('final_female/'+toRead)
    im = im[:,:,0]/255
    im_1d = im.flatten()
    x = vstack((x,im_1d))
while len(validation_baldwin)!= 0:
    toRead =validation_baldwin.pop()
    im = imread ('final_male/'+toRead)
    im = im[:,:,0]/255
    im_1d = im.flatten()
    x = vstack((x,im_1d))
while len(validation_hader)!= 0:
    toRead = validation_hader.pop()
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
x = vstack( (ones((1, x.shape[1])), x))
prediction = dot(theta.T,x)
prediction = prediction.T

y = y.T

numCorrect = 0
for i in range (10):                #bracco
    index = prediction[i].argmax() 
    if index == 0:                  
        numCorrect += 1
for i in range (10,20):             #gilpin
    index = prediction[i].argmax()
    if index == 1:
        numCorrect += 1
for i in range (20,30):             #harmon
    index = prediction[i].argmax()
    if index == 2:
        numCorrect += 1
for i in range (30,40):             #baldwin
    index = prediction[i].argmax()
    if index == 3:
        numCorrect += 1
for i in range (40,50):             #hader
    index = prediction[i].argmax()
    if index == 4:
        numCorrect += 1
for i in range (50,60):             #carell
    index = prediction[i].argmax()
    if index == 5:
        numCorrect += 1
        
print ('The success rate on the validation set is: ' + str(numCorrect/60))

##Part 8
#columes of theta are for: bracco, gilpin, harmon, baldwin, hader, carell
bracco_pixels = theta[:,0]
bracco_image = reshape(bracco_pixels[1:],(32,32))

gilpin_pixels = theta[:,1]
gilpin_image = reshape(gilpin_pixels[1:],(32,32))

harmon_pixels = theta[:,2]
harmon_image = reshape(harmon_pixels[1:],(32,32))

baldwin_pixels = theta[:,3]
baldwin_image = reshape(baldwin_pixels[1:],(32,32))

hader_pixels = theta[:,4]
hader_image = reshape(hader_pixels[1:],(32,32))

carell_pixels = theta[:,5]
carell_image = reshape(carell_pixels[1:],(32,32))
#Uncomment the two lines of codes corresponding to each actor/actress to show visualization:
plt.imshow (bracco_image,cmap = "RdBu")
plt.show()
##
#plt.imshow(gilpin_image,cmap = "RdBu")
#plt.show()
##
#plt.imshow(harmon_image,cmap = "RdBu")
#plt.show()
##
#plt.imshow(baldwin_image,cmap = "RdBu")
#plt.show()
##
#plt.imshow(hader_image,cmap = "RdBu")
#plt.show()
##
#plt.imshow(carell_image,cmap = "RdBu")
#plt.show()






