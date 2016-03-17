# -*- coding: utf-8 -*-
"""
Convolutional Autoencoder in Keras
"""

import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
import pandas as pd

from keras.layers.core import AutoEncoder, Permute, Activation
from keras.layers import containers
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Sequential



# Image generator
def image_generator(photo_l,batch_size=128,ch=3,h=128,w=128):
    rng = np.random.RandomState(290615)
    i,p_order = 0,rng.permutation(photo_l.index)
    while True:
        X_batch = np.empty((batch_size,h,w,ch))
        for j in range(batch_size):
            i_ = p_order[i+j]
            f_ = 'data/train_photos/%i.jpg' % photo_l.ix[i_]
            im = Image.open(f_).resize((w,h))
            # scale inputs [-1,+1]
            xi = np.asarray(im)/128.-1
            # flip coords horizontally (but not vertically)
            if rng.rand(1)[0] > 0.5:
                xi = np.fliplr(xi)
            # rescale slightly within a random range
            jit = (h+w)/2*0.2
            if rng.rand(1)[0] > 0.1:
                xl,xr = rng.uniform(0,jit,1),rng.uniform(w-jit,w,1)
                yu,yd = rng.uniform(0,jit,1),rng.uniform(h-jit,h,1)
                pts1 = np.float32([[xl,yu],[xr,yu],[xl,yd],[xr,yd]])
                pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
                M = cv2.getPerspectiveTransform(pts1,pts2)
                xi = cv2.warpPerspective(xi,M,(w,h))
            X_batch[j,:,:,:] = xi
        yield(X_batch,X_batch)
        i += batch_size
        if i + batch_size >= len(photo_l):
            i,p_order = 0,rng.permutation(photo_l.index)


# TODO write a convolutional autoencoder function that returns final weights
def train_CAE():

    encoder = containers.Sequential()
    encoder.add(Permute((3,1,2),input_shape=(h,w,ch))) # reorder input to ch, h, w (no sample axis)
    encoder.add(GaussianNoise(0.05)) # corrupt inputs slightly
    encoder.add(Convolution2D(16,3,3,init='glorot_uniform',border_mode='same'))
    encoder.add(MaxPooling2D((2,2)))
    encoder.add(Activation('tanh'))
    encoder.add(Convolution2D(32,3,3,init='glorot_uniform',border_mode='same'))
    encoder.add(MaxPooling2D((2,2)))
    encoder.add(Activation('tanh'))
    decoder = containers.Sequential()
    decoder.add(UpSampling2D((2,2),input_shape=(32,32,32)))
    decoder.add(Convolution2D(3,3,3,init='glorot_uniform',border_mode='same'))
    decoder.add(Activation('tanh'))
    decoder.add(UpSampling2D((2,2),input_shape=(16,64,64)))
    decoder.add(Convolution2D(3,3,3,init='glorot_uniform',border_mode='same'))
    decoder.add(Activation('tanh'))
    decoder.add(Permute((2,3,1)))
    autoencoder = AutoEncoder(encoder,decoder)

    model = Sequential()
    model.add(autoencoder)
    model.compile(optimizer='rmsprop', loss='mae')
    # if shapes don't match, check the output_shape of encoder/decoder
    genr = image_generator(biz_id_train['photo_id'], batch_size)
    model.fit_generator(genr, samples_per_epoch=len(biz_id_train), nb_epoch=10)


# TODO find clusters of +ve labels for each label and visualize samples


# viz filters from first layer of encoder
def viz_conv1(encoder):
    w,b = encoder.layers[2].get_weights()
    w = (w.transpose((2,3,1))+1)/2
    fig1 = plt.figure()
    fig1.suptitle('Filters 1st layer',fontsize=24)
    for i in range(len(w)):
        axi = fig1.add_subplot(4,4,i+1)
        axi.imshow(w[i,...])
        axi.axis('off')




def to_bool(s):
    """convert numeric labels to binary matrix"""
    return(pd.Series([1L if str(i) in str(s).split(' ') else 0L for i in range(9)]))



if __name__ == '__main__':

    # prep data to be read
    train = pd.read_csv('data/train.csv')
    biz_id_train = pd.read_csv('data/train_photo_to_biz_ids.csv')
    train[['0','1','2','3','4','5','6','7','8']] = train['labels'].apply(to_bool)
    h,w,ch,batch_size = 128,128,3,32


