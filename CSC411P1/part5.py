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
    max_iter = 10000
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

def for_plotting (trainingSize):
    '''
    return: performance as a percentage
    trainingSize -- the size of training sets for EACH gender
    '''
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
    radcliffe_list = []
    butler_list = []
    vartan_list = []
    
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

    #getting all images of actresses:
    bracco_list = []
    gilpin_list = []
    harmon_list = []
    chenoweth_list = []
    drescher_list = []
    ferrera_list = []
    
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
        
    # #shuffle the lists for randomness:
    # shuffle(baldwin_list)
    # shuffle(carell_list)
    # shuffle(hader_list)
    # shuffle(bracco_list)
    # shuffle(gilpin_list)
    # shuffle(harmon_list)

    #initialize training sets:
    training_baldwin = []
    training_carell = []
    training_hader = []
    training_bracco = []
    training_gilpin = []
    training_harmon = []
    training_radcliffe = []
    training_butler = []
    training_vartan = []
    training_chenoweth = []
    training_drescher = []
    training_ferrera = []

    #initialize validation sets:
    validation_baldwin = []
    validation_carell = []
    validation_hader = []
    validation_bracco = []
    validation_gilpin = []
    validation_harmon = []
    validation_radcliffe = []
    validation_butler = []
    validation_vartan = []
    validation_chenoweth = []
    validation_drescher = []
    validation_ferrera = []

    #initialize test sets:
    test_baldwin = []
    test_carell = []
    test_hader = []
    test_bracco = []
    test_gilpin = []
    test_harmon = []
    test_radcliffe = []
    test_butler = []
    test_vartan = []
    test_chenoweth = []
    test_drescher = []
    test_ferrera = []


    #forming training sets:
    for i in range (70):
        training_baldwin.append(baldwin_list.pop())
        training_carell.append(carell_list.pop())
        training_hader.append(hader_list.pop())
        training_bracco.append(bracco_list.pop())
        training_harmon.append(harmon_list.pop())
        training_radcliffe.append(radcliffe_list.pop())
        training_butler.append(butler_list.pop())
        training_vartan.append(vartan_list.pop())
        training_chenoweth.append(chenoweth_list.pop())
        training_drescher.append(drescher_list.pop())
        training_ferrera.append(ferrera_list.pop())
    

    #since there are only 87 usable images for gilpin, I reduce the number of training images to 67 instead of 70 for gilpin:
    for i in range (67):
        training_gilpin.append(gilpin_list.pop())
    
    #forming validation sets:
    for i in range (10):
        validation_baldwin.append(baldwin_list.pop())
        validation_carell.append(carell_list.pop())
        validation_hader.append(hader_list.pop())
        validation_bracco.append(bracco_list.pop())
        validation_gilpin.append(gilpin_list.pop())
        validation_harmon.append(harmon_list.pop())
        validation_radcliffe.append(radcliffe_list.pop())
        validation_butler.append(butler_list.pop())
        validation_vartan.append(vartan_list.pop())
        validation_chenoweth.append(chenoweth_list.pop())
        validation_drescher.append(drescher_list.pop())
        validation_ferrera.append(ferrera_list.pop())

    #forming test sets:
    for i in range (10):
        test_baldwin.append(baldwin_list.pop())
        test_carell.append(carell_list.pop())
        test_hader.append(hader_list.pop())
        test_bracco.append(bracco_list.pop())
        test_gilpin.append(gilpin_list.pop())
        test_harmon.append(harmon_list.pop())
        test_radcliffe.append(radcliffe_list.pop())
        test_butler.append(butler_list.pop())
        test_vartan.append(vartan_list.pop())
        test_chenoweth.append(chenoweth_list.pop())
        test_drescher.append(drescher_list.pop())
        test_ferrera.append(ferrera_list.pop())

    #male -> 1
    #female -> 0

    #initializing and forming potential male, female lists for training:
    male_training = []
    female_training = []
    male_training.extend(training_baldwin)
    male_training.extend(training_carell)
    male_training.extend(training_hader)
    female_training.extend(training_bracco)
    female_training.extend(training_gilpin)
    female_training.extend(training_harmon)
    
    shuffle(male_training)
    shuffle(female_training)
    #initializing and forming potential male, female lists for validation:
    male_validation = []
    female_validation = []
    male_validation.extend(validation_baldwin)
    male_validation.extend(validation_carell)
    male_validation.extend(validation_hader)
    female_validation.extend(validation_bracco)
    female_validation.extend(validation_gilpin)
    female_validation.extend(validation_harmon)
    
    np.random.seed(3)
    np.random.shuffle(male_validation)
    np.random.shuffle(female_validation)
    
    #forming lists of unseen/unknown males and females:
    male_unknown = []
    female_unknown = []
    male_unknown.extend(training_radcliffe)
    male_unknown.extend(training_butler)
    male_unknown.extend(training_vartan)
    female_unknown.extend(training_chenoweth)
    female_unknown.extend(training_drescher)
    female_unknown.extend(training_ferrera)

    shuffle(male_unknown)
    shuffle(female_unknown)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    x = array([1]*1024) 
    y = array([1]*trainingSize) #fist trainingSize # of one's -> male
    y = append(y,[-1]*trainingSize)  #followed by trainingSize # of zero's -> female
    theta0 = np.random.normal(scale=0.0,size = 1025)   #initialize theta for grad. descent

    #building the big X matrix:
    for i in range (trainingSize):
        toRead = male_training.pop()
        im = imread ('final_male/'+toRead)
        im = im[:,:,0]/255
        im_1d = im.flatten()
        x = vstack((x,im_1d))
        
    for i in range (trainingSize):
        toRead = female_training.pop()
        im = imread ('final_female/'+toRead)
        im = im[:,:,0]/255
        im_1d = im.flatten()
        x = vstack((x,im_1d))
    x = delete(x,0,0)
    x = x.T
    x_training = x #for later use to calculate pridiction on training set
    theta = grad_descent (f, df, x, y, theta0,0.000005)
    
    #Now, need to get the performance as a % and return it:
    performance_training = 0
    performance_validation = 0
    performance_unknown = 0
    
    #computing performance on training set:
    x_training = vstack((ones((1,x_training.shape[1])),x_training))
    prediction_training = dot(theta.T,x_training)
   
    numCorrect = 0
    for i in range (trainingSize):
        if prediction_training [i] > 0.0:
            numCorrect += 1
    for i in range (trainingSize,2*trainingSize):
        if prediction_training [i] <= 0.0:
            numCorrect += 1
    perf_training = numCorrect/(2*trainingSize)
    
    #computing peroformance on the validation set:
    x = array([1]*1024) 
    y = array([1]*30)
    y = append(y,[-1]*30)
    
    for i in range (30):
        toRead =male_validation.pop()
        im = imread ('final_male/'+toRead)
        im = im[:,:,0]/255
        im_1d = im.flatten()
        x = vstack((x,im_1d))

    for i in range (30):
        toRead = female_validation.pop()
        im = imread ('final_female/'+toRead)
        im = im[:,:,0]/255
        im_1d = im.flatten()
        x = vstack((x,im_1d))
        
    x = delete(x,0,0)
    x = x.T
    x = vstack( (ones((1, x.shape[1])), x))
    prediction = dot(theta.T,x)
    numCorrect = 0
    
    for i in range (30):
        if prediction [i] > 0.0:
            numCorrect += 1
    for i in range (30,60):
        if prediction [i] <= 0.0:
            numCorrect += 1
            
    perf_validation = numCorrect/(2*30)
    
    #compute performance on the unknown set:
    x = array([1]*1024) 
    y = array([1]*trainingSize)
    y = append(y,[-1]*trainingSize)
    
    for i in range (trainingSize):
        toRead = male_unknown.pop()
        im = imread ('final_male/'+toRead)
        im = im[:,:,0]/255
        im_1d = im.flatten()
        x = vstack((x,im_1d))
        
    for i in range (trainingSize):
        toRead = female_unknown.pop()
        im = imread ('final_female/'+toRead)
        im = im[:,:,0]/255
        im_1d = im.flatten()
        x = vstack((x,im_1d))
        
    x = delete(x,0,0)
    x = x.T
    x = vstack( (ones((1, x.shape[1])), x))
    prediction = dot(theta.T,x)
    numCorrect = 0
    
    for i in range (trainingSize):
        if prediction [i] > 0.0:
            numCorrect += 1
    for i in range (trainingSize,2*trainingSize):
        if prediction [i] <= 0.0:
            numCorrect += 1
            
    perf_unknown = numCorrect/(2*trainingSize)
    
    
    print ('The success rate on the training set is: ' + str(perf_training))
    print ('The success rate on the validation set is: ' + str(perf_validation))
    print ('The success rate on the unknown set is: ' + str(perf_unknown))

    return (perf_training,perf_validation,perf_unknown)




perf_train_result = []
perf_valid_result = []
perf_unknown_result = []

size = []
for i in range (40,210,10):
    size.append(i)
    perf_train_result.append(for_plotting(i)[0])
    perf_valid_result.append(for_plotting(i)[1])
    perf_unknown_result.append(for_plotting(i)[2])
    

##Plotting
plt.plot(size,perf_valid_result,label = 'Validation',color ='b')
plt.plot(size,perf_train_result,label = 'Training',color ='g')
plt.plot(size,perf_unknown_result,label = 'Unknown',color ='r')
plt.xlabel('Training Size (per gender)')
plt.ylabel('Performance (%)')
plt.title('Performance on the Training, Validation, and Unknown Sets')
plt.legend()
plt.show()






