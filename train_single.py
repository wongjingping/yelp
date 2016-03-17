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



def test(model,gen,n_id,threshold=0.5,verbose=True):
    """
    Evaluate on test set
    
    Parameters
    ----------
    model: keras model
        Model
    gen: generator
        Generates a batch of images for a business_id in each batch
    n_id: int
        number of business_ids in test set
    """
    t_start = time()
    ce_avg,tp,tn,fp,fn = 0.,0.,0.,0.,0.
    for i in range(n_id):
        X_test,y_test = gen.next()
        y_test = y_test.mean()
        Y_pred = model.predict(X_test)
        y_pred = Y_pred.max(axis=0)
        ce = np.mean(- y_test * np.log(y_pred) - (1-y_test) * np.log(1-y_pred))
        ce_avg += ce/n_id
        y_predr = y_pred.round()
        tp += sum((y_test == 1) & (y_predr == 1))
        tn += sum((y_test == 0) & (y_predr == 0))
        fp += sum((y_test == 0) & (y_predr == 1))
        fn += sum((y_test == 1) & (y_predr == 0))
        print(i)
    prec,recall,acc = tp/(tp+fp+1e-15),tp/(tp+fn+1e-15),(tp+tn)/n_id
    F1 = 2*tp/(2*tp+fp+fn)
    if verbose:
        print('Valid F1 %.3f tp %.3f tn %.3f fp %.3f fn %.3f' % (F1,tp,tn,fp,fn))
        print('Took %.1fs' % (time()-t_start))
    return(ce_avg,prec,recall,F1,acc,tp,tn,fp,fn)


# callback that loops over validation set generator
class Validator(Callback):
    def __init__(self, valid_gen, n_id, print_every_n=np.inf):
        self.gen = valid_gen
        self.n_id = n_id
        self.print_every_n = print_every_n
    def on_batch_end(self, epoch, logs={}):
        if epoch % self.print_every_n == 0:
            print('batch: %i acc: %.3f loss: %.3f size: %i' % \
        (logs['batch'],logs['acc'],logs['loss'],logs['size']))
    def on_epoch_end(self, epoch, logs={}):
        ce,prec,recall,F1,acc,tp,tn,fp,fn = test(model,self.gen,self.n_id)
        self.logs[epoch] = {'ce':ce,'prec':prec,'recall':recall,'F1':F1,
            'acc':acc,'tp':tp,'tn':tn,'fp':fp,'fn':fn}



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
    train_gen = image_generator(0,i_train,batch_size,augment=True)
    valid_gen = image_generator(i_train,i_valid,batch_size,augment=False)
    test_gen = image_generator(i_valid,nrow,batch_size,augment=False)
    validator = Validator(valid_gen,i_valid-i_train,100)
    hist_gen = model.fit_generator(train_gen, 
                                   samples_per_epoch=i_train,
                                   nb_epoch=100,
                                   verbose=2,
                                   show_accuracy=True,
                                   callbacks=[validator])
    # evaluate on test set
    ce,prec,recall,F1,acc,tp,tn,fp,fn = test(model,test_gen,nrow-i_valid)

    
    # save model
    mname = 'model_%03d' % (loss_test*100)
    model_json = model.to_json()
    open('models/%s.json' % (mname), 'w').write(model_json)
    model.save_weights('models/%s_weights.h5' % (mname))
    return(loss_test)



# visualize selected/flagged examples on random data
# TODO adapt to multi-label at once
def show_classified_pics(gen,model):
    X_batch,y_batch = gen.next()
    y_pred = model.predict(X_batch[0])
    y_predr = np.max(y_pred.round())
    plt.rcParams['figure.figsize'] = (14,14)
    xis = np.where(y_pred > 0.5)
    fig1 = plt.figure()
    fig1.suptitle('Label 1',fontsize=24)
    for i in range(len(xis[0])):
        axi = fig1.add_subplot(6,6,i+1)
        i_ = xis[0][i]
        axi.imshow((X_batch[0][i_,...]+1)/2)
        axi.axis('off')
        axi.set_title('%.3f' % y_pred[i_])
    xis = np.where(y_pred < 0.5)
    fig2 = plt.figure()
    fig2.suptitle('Label 0',fontsize=24)
    for i in range(len(xis[0])):
        axi = fig2.add_subplot(6,6,i+1)
        i_ = xis[0][i]
        axi.imshow((X_batch[0][i_,...]+1)/2)
        axi.axis('off')
        axi.set_title('%.3f' % y_pred[i_])
    print('Truth %i \nPredicted %i' % (y_batch[0],y_predr))
    return(X_batch,y_batch)



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
    # mostly 1 2 5 6 8. F1 score of ~0.65
    
    