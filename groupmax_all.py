# -*- coding: utf-8 -*-
"""
A convnet that takes uses the max of each predicted class across the group
to predict the group's label. Simple, still using keras
"""


from time import time
import os

import pandas as pd
import numpy as np
from PIL import Image
import cv2

from keras.models import Sequential
from keras.layers.core import Permute, Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import Callback


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
def image_generator(biz_idx,batch_size,augment=True):
    """
    Generates images from a specific business_id each batch.
    Samples without replacement (with replacement if examples < batch_size)
    
    Parameters
    ----------
    biz_idx: list
        an iterable of business_ids' indices
    batch_size: int
        standard size for each batch
    augment: bool
        if data is to be augmented (on the fly)
    """
    rng = np.random.RandomState(290615)
    bi,b_list = 0,rng.permutation(biz_idx)
    while True:
        biz_id_i = b_list[bi]
        photo_id_list = biz_id_train[biz_id_train['business_id']==biz_id_i]['photo_id']
        batch_size_i = batch_size if len(photo_id_list) >= batch_size else len(photo_id_list)
        photo_id_list = photo_id_list.sample(n=batch_size_i)
        X_batch = np.empty((batch_size_i,h,w,ch))
        for i_ in range(batch_size_i):
            f_ = 'data/train_photos/' + str(photo_id_list.iloc[i_]) + '.jpg'
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
            X_batch[i_,:,:,:] = xi
            # plt.imsave('data/aug_%i' % i_,(xi+1)/2);plt.close()
        Y_batch = np.asarray(train.ix[train['business_id']==biz_id_i,\
            ['0','1','2','3','4','5','6','7','8']])
        Y_batch = np.repeat(Y_batch, repeats=batch_size_i, axis=0)
        yield([X_batch],Y_batch)

        if bi < len(b_list)-1:
            bi += 1
        else:
            bi,b_list = 0,rng.permutation(biz_idx)


def obj_agg(Y_true,Y_pred,agg='max'):
    """
    Calculates the binary cross entropy on the aggregated outputs of each 
    batch. Y_pred is a matrix containing the predictions of each example for each label 
    """
    if agg == 'max':
        y_pred = K.max(Y_pred,axis=0)
    elif agg == 'avg':
        y_pred = K.mean(Y_pred,axis=0)
    else:
        raise NotImplementedError
    y_true = K.max(Y_true,axis=0)
    return(K.mean(K.binary_crossentropy(y_pred, y_true)))


def test(model,test_gen,n_id,agg='max'):
    """
    Evaluate on test set
    
    Parameters
    ----------
    model: keras model
        Model
    test_gen: generator
        Generates a batch of images for a business_id in each batch
    n_id: int
        number of business_ids in test set
    """
    ce_test,tp,tn,fp,fn = 0.,0.,0.,0.,0.
    for i in range(n_id):
        X_test,Y_test = test_gen.next()
        Y_pred = model.predict(X_test)
        if agg == 'max':
            y_pred = Y_pred.max(axis=0)
        elif agg == 'avg':
            y_pred = Y_pred.mean(axis=0)
        else:
            raise NotImplementedError
        y_pred = np.maximum(np.minimum(y_pred,1-1e-15,),1e-15)
        y_test = Y_test[0,:]
        ce = np.mean(- y_test * np.log(y_pred) - (1-y_test) * np.log(1-y_pred))
        ce_test += ce/n_id
        y_predr = y_pred.round()
        tp += sum((y_test == 1) & (y_predr == 1))
        tn += sum((y_test == 0) & (y_predr == 0))
        fp += sum((y_test == 0) & (y_predr == 1))
        fn += sum((y_test == 1) & (y_predr == 0))
    prec,recall,acc = tp/(tp+fp),tp/(tp+fn),(tp+tn)/n_id/len(y_test)
    F1 = 2*prec*recall/(prec+recall)
    return(ce_test,prec,recall,F1,acc)


# callback that loops over validation set generator
class Validator(Callback):
    def __init__(self, valid_gen, n_id, agg='max'):
        self.valid_gen = valid_gen
        self.n_id = n_id
        self.logs = {}
        self.agg = agg
    def on_batch_end(self, epoch, logs={}):
        print(logs)
    def on_epoch_end(self, epoch, logs={}):
        ce,prec,recall,F1,acc = test(model,self.valid_gen,self.n_id,agg=self.agg)
        print('Valid ce %.3f precision %.3f recall %.3f F1 %.3f acc %.3f' % \
            (ce,prec,recall,F1,acc))
        self.logs[epoch] = {'ce':ce,'prec':prec,'recall':recall,'F1':F1,'acc':acc}


# train model
def train_model():
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
    model.add(Dense(output_dim=100,init='glorot_uniform',activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=9,init='glorot_uniform',activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss=obj_agg, 
                  class_mode='binary')
    t_train = time()
    print('Took %.1fs' % (t_train-t_comp))


    # basic fitting with image generator
    rng = np.random.RandomState(290615)
    i_split = rng.choice([0,1,2], size=len(train), p=[0.8,0.1,0.1])
    i_train = train.ix[i_split==0,'business_id']
    i_valid = train.ix[i_split==1,'business_id']
    i_test = train.ix[i_split==2,'business_id']
    train_gen = image_generator(i_train,batch_size)
    valid_gen = image_generator(i_valid,batch_size)
    test_gen = image_generator(i_test,batch_size,augment=False)
    validator = Validator(valid_gen,len(i_valid),agg='avg')
    model.fit_generator(train_gen,
                       samples_per_epoch=0.8*len(i_train)*batch_size,
                       nb_epoch=10,
                       verbose=2,
                       show_accuracy=True,
                       callbacks=[validator])
    ce,prec,recall,F1,acc = test(model,test_gen,len(i_test),agg='avg')
    print('Test ce %.3f precision %.3f recall %.3f F1 %.3f acc %.3f' % \
        (ce,prec,recall,F1,acc))
    
    # save model
    mname = 'model_%03d' % (F1*100)
    model_json = model.to_json()
    open('models/%s.json' % (mname), 'w').write(model_json)
    model.save_weights('models/%s_weights.h5' % (mname))
    return(model,F1)


def predict_biz_id(photo_ids,n=32,m=10,augment=True):
    rng = np.random.RandomState(290615)
    t_start = time()
    n = min(n,len(photo_ids))
    all_pred = np.zeros((m,9))
    # sample n photos, augmented
    pid_rand = photo_ids['photo_id'].sample(m*n,replace=True,random_state=290615)
    for i in range(m):
        X_i = np.zeros((n,h,w,ch))
        for j in range(n):
            f_ = 'data/test_photos/%i.jpg' % pid_rand.iloc[i*n+j]
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
            X_i[j,:,:,:] = xi
        # predict and take the max across the batch
        all_pred[i,:] = model.predict(X_i).max(axis=0)
    # return mean, rounded
    avg_pred = np.where(all_pred.mean(axis=0) > 0.5)[0]
    avg_pred_s = ' '.join(str(label) for label in avg_pred)
    print('biz_id %s prediction %s' %(photo_ids.iloc[0]['business_id'],avg_pred_s))
    print('\ttook %.1fs' % (time()-t_start))
    return(avg_pred_s)



if __name__ == '__main__':

    # prep data to be read
    train = pd.read_csv('data/train.csv')
    biz_id_train = pd.read_csv('data/train_photo_to_biz_ids.csv')
    train[['0','1','2','3','4','5','6','7','8']] = train['labels'].apply(to_bool)
    h,w,ch,batch_size = 128,128,3,64

    model,F1 = train_model()
    
    # predict
    biz_id_test = pd.read_csv('data/test_photo_to_biz.csv')
    submit = pd.read_csv('data/sample_submission.csv')
    
    submit['labels'] = biz_id_test.groupby('business_id').apply(predict_biz_id,n=batch_size)
    submit_labels.to_csv('data/sub2_singleconv.csv',index=False)
    
    