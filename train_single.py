# -*- coding: utf-8 -*-
"""
A naive convnet that takes the group's label as each instance's label
"""

from time import time
import os
import h5py

import pandas as pd
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

from keras.models import Sequential, model_from_json
from keras.layers.core import Permute, Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D


# convert numeric labels to binary matrix
def to_bool(s):
    return(pd.Series([1L if str(i) in str(s).split(' ') else 0L for i in range(9)]))

# take average prediction of binary matrix and convert into a string of numeric labels
def from_bool(x):
    x = np.where(x.mean(axis=0).round())[0]
    return(' '.join(str(i) for i in x))

# sample up to 10 pictures from each business_id
def take(series,n=10):
    n = min(n,len(series))
    return(series.sample(n,random_state=290615))


# image generator + augmentor
def image_generator(i_min,i_max,batch_size,augment=True):
    rng = np.random.RandomState(290615)
    bi,b_list = 0,rng.permutation(range(i_min/batch_size,i_max/batch_size))
    while True:
        b = b_list[bi]
        b_range = range(b*batch_size,(b+1)*batch_size)
        X_batch = np.empty((batch_size,h,w,ch))
        Y_batch = np.asarray(XY.ix[b_range,['0','1','2','3','4','5','6','7','8']])
        for i_ in b_range:
            f_ = 'data/train_photos/' + str(XY.ix[i_,'photo_id']) + '.jpg'
            im = Image.open(os.path.realpath(f_))
            im_sml = im.resize((w,h))
            # scale inputs [-1,+1]
            xi = np.asarray(im_sml)/128.-1
            if augment:
                # flip coords horizontally (but not vertically)
                if rng.rand(1)[0] > 0.5:
                    xi = np.fliplr(xi)
                # rescale slightly within a random range
                jit = w*0.2
                if rng.rand(1)[0] > 0.1:
                    xl,xr = rng.uniform(0,jit,1),rng.uniform(w-jit,w,1)
                    yu,yd = rng.uniform(0,jit,1),rng.uniform(h-jit,h,1)
                    pts1 = np.float32([[xl,yu],[xr,yu],[xl,yd],[xr,yd]])
                    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
                    M = cv2.getPerspectiveTransform(pts1,pts2)
                    xi = cv2.warpPerspective(xi,M,(w,h))
            # save back to X_batch
            X_batch[i_-b*batch_size,:,:,:] = xi
        yield([X_batch],Y_batch)
        if bi < len(b_list)-1:
            bi += 1
        else:
            bi,b_list = 0,rng.permutation(range(i_min/batch_size,i_max/batch_size))


# train model
def train():
    print('Compiling Model')
    t_comp = time()
    model = Sequential()
    # reorder input to ch, h, w (no sample axis)
    model.add(Permute((3,1,2),input_shape=(h,w,ch)))
    # add conv layers
    model.add(Convolution2D(16,3,3,init='glorot_uniform',activation='relu',
                            subsample=(1,1)))
#    model.add(Convolution2D(16,3,3,init='glorot_uniform',activation='relu',
#                            subsample=(1,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.1))
    model.add(Convolution2D(32,3,3,init='glorot_uniform',activation='relu',
                            subsample=(1,1)))
#    model.add(Convolution2D(32,3,3,init='glorot_uniform',activation='relu',
#                        subsample=(1,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.1))
    model.add(Convolution2D(64,3,3,init='glorot_uniform',activation='relu',
                            subsample=(1,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(64,3,3,init='glorot_uniform',activation='relu',
                            subsample=(1,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(output_dim=500,init='glorot_uniform',activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=500,init='glorot_uniform',activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=9,init='glorot_uniform',activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', 
                  class_mode='binary')
    t_train = time()
    print('Took %.1fs' % (t_train-t_comp))


    # basic fitting with image generator
    train_p,valid_p = 0.8,0.1
    i_train = int(nrow*train_p)/batch_size*batch_size
    i_valid = int(nrow*(train_p+valid_p))/batch_size*batch_size
    train_gen = image_generator(i_min=0,i_max=i_train,batch_size=batch_size,augment=True)
    valid_gen = image_generator(i_min=i_train,i_max=i_valid,batch_size=i_valid-i_train,augment=False)
    test_gen = image_generator(i_min=i_valid,i_max=nrow,batch_size=nrow-i_valid,augment=False)
    hist_gen = model.fit_generator(train_gen, 
                                   samples_per_epoch=i_train,
                                   nb_epoch=10,
                                   verbose=2,
                                   show_accuracy=True,
                                   validation_data=valid_gen.next())
    print(hist_gen)
    # evaluate on test set
    X_test,Y_test = test_gen.next()
    loss_test = model.evaluate(X_test,Y_test)
    
    # save model
    mname = 'model_%03d' % (loss_test*100)
    model_json = model.to_json()
    open('models/%s.json' % (mname), 'w').write(model_json)
    model.save_weights('models/%s_weights.h5' % (mname))
    return(loss_test)


# store test images as a matrix for making predictions
def store_test():
    data = h5py.File('data/test_matrix.h5','w')
    X_test = data.create_dataset(
        name='X',
        shape=(len(XY_test),h,w,ch),
        dtype=np.float32)
    for i in range(len(XY_test)):
        f_ = 'data/test_photos/%s.jpg' % (XY_test['photo_id'][i])
        im_sml = Image.open(f_).resize((w,h))
        X_test[i,:,:,:] = np.asarray(im_sml)/128.-1
        if i % 1000 == 0:
            print('reading %ith image' % i)
    data.close()




if __name__ == '__main__':

    # prep data to be read
    train = pd.read_csv('data/train.csv')
    biz_id_train = pd.read_csv('data/train_photo_to_biz_ids.csv')
    train[['0','1','2','3','4','5','6','7','8']] = train['labels'].apply(to_bool)
    XY = pd.merge(biz_id_train,train,on='business_id')
    h,w,ch,batch_size = 128,128,3,32
    nrow,ncol,ny = len(XY),h*w*ch,9

    loss_test = train()
    
    # predict
    model = model_from_json(open('models/model_%03d.json' % (loss_test*100)).read())
    model.load_weights('models/model_%03d_weights.h5' % (loss_test*100))
    biz_id_test = pd.read_csv('data/test_photo_to_biz.csv')
    submit = pd.read_csv('data/sample_submission.csv')
    
    # take average of up to 10 photos for each biz_id
    test_id = biz_id_test.groupby('business_id').apply(take)
    XY_test = pd.merge(test_id,submit,on='business_id')
    # store_test()
    data = h5py.File('data/test_matrix.h5','r')
    X_test = data['X']
    Y_pred = pd.DataFrame(model.predict(X_test,batch_size=128)) # 46mins
    data.close()
    Y_pred['business_id'] = XY_test['business_id']
    labels = pd.DataFrame(Y_pred.groupby('business_id').apply(from_bool))
    labels['business_id'],labels['labels'] = labels.index,labels[0]
    submit_labels = pd.merge(submit[['business_id']],labels[['business_id','labels']])
    submit_labels.to_csv('data/sub2_singleconv.csv',index=False)
    # mostly 1 2 5 6 8. F1 score of ~0.61
    
    