# -*- coding: utf-8 -*-
"""
This file is used for extracting features over windows of tri-axial accelerometer 
data. We recommend using helper functions like _compute_mean_features(window) to 
extract individual features.

As a side note, the underscore at the beginning of a function is a Python 
convention indicating that the function has private access (although in reality 
it is still publicly accessible).

"""

import numpy as np
import scipy.signal
import math

def _compute_mean_features(window):
    """
    Computes the mean x, y and z acceleration over the given window. 
    """
    new_window = window
    return np.mean(new_window, axis=0)

# TODO: define functions to compute more features


#we need to compute four categories of features: statistical, FFt, Other, and Peak features

#Category 1: Statistical Features
#they are pretty vague as far as what features will be useful here so I implemented the ones they listed
def _compute_max_features(window):
    #there are multiple max functions, this one propegates NaN values. nanMax ignores NaN values
    return np.amax(window, axis=0)
    
def _compute_min_features(window):
    return np.amin(window, axis=0)

def _compute_median_features(window):
    return np.median(window, axis=0)

def _compute_variance_features(window):
    return np.var(window, axis=0)

def _compute_stdv_features(window): 
    return np.std(window, axis= 0)

def _compute_mag_var(window):
    mag = []
    for bit in window:
        mag.append(math.sqrt(bit[0] ** 2 + bit[1] ** 2 + bit[2] ** 2))
    return [np.var(mag, axis= 0)]

'''
def _compute_rzc_features(window): #rate of zero- or mean-crossings?
    new_window = np.zeros(size(window))
    new_window(2:end) = window(1:end-1)
    return (1/(2*length(window))) * sum(abs(sign(window) - sign(new_window)))

'''


#Category 2: FFT Features

#rfft can create imaginary values, astype supposedly fixes that so I assume we run it through rfft then use astype after
#this is based on the category 2 description in the project
def FFT(window):
    mag = []
    for bit in window:
        mag.append(math.sqrt(bit[0] ** 2 + bit[1] ** 2 + bit[2] ** 2))
    return [max(np.fft.rfft(mag).astype(float))]

def all_FFT(window):
    x = []
    y = []
    z = []
    for bit in window:
        x.append(bit[0])
        y.append(bit[1])
        z.append(bit[2])

    return [max(np.fft.rfft(x).astype(float)), max(np.fft.rfft(y).astype(float)), max(np.fft.rfft(z).astype(float))]

#Category 3: Other Features
def entropy(window):
    hist, bin_edges = np.histogram(window) 
    #the professor said there could be issues with using np.histogram where it might not return a probability distribution. 
    # A solution is normalizing it to make it sum to 1 or implement our own histogram calculation
    hist = hist/hist.sum()
    count = 0
    for i in hist:
          if i != 0:
                count += i * math.log(i)
    return [count]

#Category 4: Peak Features

#similar to assignment 1 where we created steps
#use the peak count over each window as a feature or try something like the average duration between peaks in a window

def num_peaks(window):
    magnitude = []
    for window_list in window:
        magnitude.append(math.sqrt( window_list[0] ** 2 + window_list[1] ** 2 + window_list[2] ** 2))
    peaks, prop = scipy.signal.find_peaks(magnitude)
    return np.array([len(peaks)])

def peak_dist(window):
    magnitude = []
    for window_list in window:
        magnitude.append(math.sqrt(pow(window_list[0],2) + pow(window_list[1],2) + pow(window_list[2],2)))
    peaks, prop = scipy.signal.find_peaks(magnitude)
    
    dist = []
    for i in range(1, len(peaks) - 1):
        dist.append(peaks[i] - peaks[i - 1])
    return [np.mean(dist)]

#we need to add at least four features, meaning extract 4 functions from the data
def extract_features(window):
    """
    Here is where you will extract your features from the data over 
    the given window. We have given you an example of computing 
    the mean and appending it to the feature vector.
    
    """
    no_label = []
    for row in window:
        no_label.append(row[:3])
    
    x = []
    feature_names = []

    x.append(_compute_mean_features(no_label))
    feature_names.append("x_mean")
    feature_names.append("y_mean")
    feature_names.append("z_mean")
    
    x.append(_compute_max_features(no_label))
    feature_names.append("x_max")
    feature_names.append("y_max")
    feature_names.append("z_max")


    #ADDED
    x.append(_compute_median_features(no_label))
    feature_names.append("x_median")
    feature_names.append("y_median")
    feature_names.append("z_median")

    x.append(_compute_variance_features(no_label))
    feature_names.append("x_var")
    feature_names.append("y_var")
    feature_names.append("z_var")
    
    x.append(_compute_mag_var(no_label))
    feature_names.append("mag_var")
    
    x.append(_compute_stdv_features(no_label))
    feature_names.append("x_stdv")
    feature_names.append("y_stdv")
    feature_names.append("z_stdv")

    x.append(FFT(no_label))
    feature_names.append("mag_fft")
    
    x.append(all_FFT(no_label))
    feature_names.append("x_fft")
    feature_names.append("y_fft")
    feature_names.append("z_fft")


    x.append(entropy(no_label))
    feature_names.append("x_entropy")
    
    x.append(num_peaks(no_label))
    feature_names.append("mag_peaks")
                   
    #x.append(peak_dist(window))
    #feature_names.append("peak_distance")

    # TODO: call functions to compute other features. Append the features to x and the names of these features to feature_names

    feature_vector = np.concatenate(x, axis=0) # convert the list of features to a single 1-dimensional vector
    return feature_names, feature_vector
