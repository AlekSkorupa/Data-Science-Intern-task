#!/usr/bin/env python
###########################################################################################
#
# Class for downloading, extracting and loading cifar10 dataset. It also contains functions
#  for unpacking CNN features extracted from the pretrained Incepcion v3 model.
#
###########################################################################################
import re
import sys
import tarfile
import copy
from subprocess import call

import numpy as np
import pickle
import os
###########################################################################################
dataset_path = "/home/kyp/tooploox/MyProject/cifar-10-batches-py/"


def download_and_open():
    print("")
    print("Downloading...")
    if not os.path.exists("cifar-10-python.tar.gz"):
        call(
            "wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
            shell=True
        )
        print("Downloading done.\n")
    else:
        print("Dataset already downloaded. Did not download twice.\n")
    print("Extracting...")
    cifar_python_directory = os.path.abspath("cifar-10-batches-py")
    if not os.path.exists(cifar_python_directory):
        call(
            "tar -zxvf cifar-10-python.tar.gz",
            shell=True
        )
        print("Extracting successfully done to {}.".format(cifar_python_directory))
    else:
            print("Dataset already extracted. Did not extract twice.\n")

def unpickle(file_path):
    import pickle
    with open(file_path, mode='rb') as batch:
        dictionary = pickle.load(batch, encoding='bytes')
    return dictionary

def load_class_names():
    
    file_path ='/home/kyp/tooploox/MyProject/cifar-10-batches-py/batches.meta'
    raw = unpickle(file_path)[b'label_names']
    
    # Convert from binary strings.
    names = [x.decode('utf-8') for x in raw]

    return names

def load_training_set(dataset_path):
      
    # Pre-allocate the arrays for the images and class-numbers for efficiency.
    X_train = np.zeros(shape=[50000, 32, 32, 3], dtype=float)
    y_train = np.zeros(shape=[50000], dtype=int)
    
    # Begin-index for the current batch.
    begin = 0
    for i in range(1,6):
        
        # Load image batch
        print('Loading: data_batch_'+str(i))
        file_path = os.path.join(dataset_path,'data_batch_'+str(i))
        data = unpickle(file_path)
        images = (data[b'data'])
        labels = (data[b'labels'])
        
        # Convert the raw images from the data-files to floating-points.
        images_float = np.array(images, dtype=float) / 255.0

        # Reshape the array to 4-dimensions and reorder indices
        images = images_float.reshape([-1, 3, 32, 32]).transpose([0, 2, 3, 1])
        labels = np.asarray(labels, dtype='uint8')
        
        # End-index for the current batch.
        end = begin + len(images)
        
        # Store the images into the array.
        X_train[begin:end, :] = images

        # Store the class-numbers into the array.
        y_train[begin:end] = labels
    
        # The begin-index for the next batch is the current end-index.
        begin = end
    
    return  X_train, y_train

def load_test_set(dataset_path):
      
    print('Loading: test_batch')
    file_path = os.path.join(dataset_path,'test_batch')
    data = unpickle(file_path)
    images = (data[b'data'])
    labels = (data[b'labels'])
    
    # Convert the raw images from the data-files to floating-points.
    images_float = np.array(images, dtype=float) / 255.0

    # Reshape the array to 4-dimensions and reorder indices
    X_test = images_float.reshape([-1, 3, 32, 32]).transpose([0, 2, 3, 1])
    y_test = np.asarray(labels, dtype='uint8')
    
    return  X_test, y_test

def load_CNN(data_path):
   
    print("Loading: test_batch.npz")
    file_path = os.path.join(data_path,'test_batch.npz')
    data = np.load(file_path)
    y_test = data['y']
    X_test  = data['representations']
    
    print('Loading: data_batch_'+str(1)+'.npz' )
    file_path = os.path.join(data_path,'data_batch_1.npz')
    data = np.load(file_path)
    y_train = data['y']
    X_train = data['representations']
    for i in range(2,6):
        print('Loading: data_batch_'+str(i)+'.npz' )
        file_path = os.path.join(data_path,'data_batch_'+str(i)+'.npz')
        data = np.load(file_path)
        y_train = np.append(y_train, data['y'], axis=0)
        X_train = np.append(X_train, data['representations'], axis=0)
    
    return  X_train, y_train, X_test, y_test
