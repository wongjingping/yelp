# -*- coding: utf-8 -*-
"""
Train a convnet that trains for single labels only
"""


from time import time
import os

import pandas as pd
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Permute, Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K
from keras.callbacks import Callback


# convert numeric labels to binary matrix
def to_bool(s):
    return(pd.Series([1L if str(i) in str(s).split(' ') else 0L for i in range(9)]))

# returns list of biz_id given full dataframe of photo_id, business_id, labels
def get_biz_id(df_i,train=True,max_batch_size=32):
    if train:
        if df_i.iloc[:,-1].all() and len(df_i) <= max_batch_size:
            return(df_i.iloc[0]['business_id'])
        else:
            return(0)
    else:
        return(df_i.iloc[0]['business_id'])


# image generator + augmentor
def image_generator(df,batch_size,plab,augment=True):
    """
    Generates images from a specific business_id each batch.
    Samples without replacement (with replacement if examples < batch_size)
    
    Parameters
    ----------
    df: data frame
        joined data frame of photo_id, business_id, labels
    batch_size: int
        standard size for each batch
    plab: double
        probability of sampling by business_id. Set to 1 for validation/testing.
    augment: bool
        if data is to be augmented (on the fly)
    """
    rng = np.random.RandomState(290615)
    if_train = 1 if plab < 1. else 0
    bi,b_list = 0,df.groupby('business_id').apply(get_biz_id,if_train,batch_size)
    b_list = b_list[b_list!=0]
    b_order = rng.permutation(b_list.index)
    pi,p_list = 0, df[df.iloc[:,-1]==0]['photo_id']
    p_order = rng.permutation(p_list.index)
    while True:
        if rng.rand(1)[0] < plab:
            # aggregate biz_id with outdoor-seating
            biz_id_i = b_list.ix[b_order[bi]]
            photo_train = df[df['business_id']==biz_id_i]['photo_id']
            y_batch = np.asarray(df[df['business_id']==biz_id_i].iloc[:,-1])
            # increase/loop indices for next iteration
            if bi < len(b_list)-1:
                bi += 1
            else:
                bi,b_order = 0,rng.permutation(b_list.index)
        else:
            # pic 32 random non-outdoor-seating pictures
            photo_train = p_list[p_order[pi:(pi+batch_size)]]
            y_batch = np.repeat(0, repeats=len(photo_train), axis=0)
            # increase/loop indices for next iteration
            if pi < len(p_list)-1-batch_size:
                pi += batch_size
            else:
                pi,p_order = 0,rng.permutation(p_list.index)
        batch_size_i = len(photo_train)
        # read and augment photos
        X_batch = np.empty((batch_size_i,h,w,ch))
        for i_ in range(batch_size_i):
            f_ = 'data/train_photos/' + str(photo_train.iloc[i_]) + '.jpg'
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
            # save individual image to X_batch
            X_batch[i_,:,:,:] = xi
#            plt.imsave('data/aug_%i' % i_,(xi+1)/2);plt.close()
        yield([X_batch],y_batch)



# mix of max and mean
def obj_mix(Y_true,Y_pred):
    y_true = K.mean(Y_true,axis=0)
    if y_true == 1:
        y_pred = K.max(Y_pred,axis=0)
        return(K.mean(K.binary_crossentropy(y_pred, y_true)))
    elif y_true == 0:
        return(K.mean(K.binary_crossentropy(Y_pred,Y_true)))
    else:
        print('unexpected value of y_true',y_true)
        return(K.mean(K.binary_crossentropy(Y_pred,Y_true)))

# sum of output probabilities
def obj_sum(Y_true,Y_pred):
    y_true = K.mean(Y_true,axis=0)
    y_pred = K.sum(Y_pred,axis=0)
    return(K.mean(K.binary_crossentropy(y_pred,y_true)))

# max of output probabilities
def obj_max(Y_true,Y_pred):
    y_true = K.mean(Y_true,axis=0)
    y_pred = K.sum(Y_pred,axis=0)
    return(K.mean(K.binary_crossentropy(y_pred,y_true)))


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
def train_model(lab):
    print('Compiling Model')
    t_comp = time()
    # build model
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
    model.add(Dense(output_dim=1,init='zero',activation='sigmoid'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0., nesterov=True)
    model.compile(optimizer=sgd, loss=obj_mix, 
                  class_mode='binary')
    t_train = time()
    print('Took %.1fs' % (t_train-t_comp))


    # basic fitting with image generator
    df = pd.merge(biz_id_train,train[['business_id',lab]],on='business_id')
    plab = df[lab].mean()
    rng = np.random.RandomState(290615)
    i_split = rng.choice([0,1,2], size=len(train), p=[0.8,0.10,0.10])
    i_train = train.ix[i_split==0,'business_id']
    i_valid = train.ix[i_split==1,'business_id']
    i_test = train.ix[i_split==2,'business_id']
    df_train = df[df['business_id'].isin(i_train)]
    df_valid = df[df['business_id'].isin(i_valid)]
    df_test = df[df['business_id'].isin(i_test)]
    train_gen = image_generator(df_train,batch_size,plab=plab)
    valid_gen = image_generator(df_valid,np.inf,plab=1.)
    test_gen = image_generator(df_test,np.inf,plab=1.,augment=False)
    validator = Validator(valid_gen,len(df_valid),10)
    model.fit_generator(train_gen,
                       samples_per_epoch=len(df_train),
                       nb_epoch=10,
                       verbose=2,
                       show_accuracy=True,
                       callbacks=[validator])
    ce,prec,recall,F1,acc,tp,tn,fp,fn = test(model,test_gen,len(i_test))

    
    # save model
    mname = 'groupmax_%s_%03d' % (lab,F1*100)
    model_json = model.to_json()
    open('models/%s.json' % (mname), 'w').write(model_json)
    model.save_weights('models/%s_weights.h5' % (mname))
    
    # X_batch,y_batch = show_classified_pics(train_gen,model)
    
    return(model,F1)


# visualize selected/flagged examples on random data
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




# helper function for visualizing change in label proportion by number of images
def plot_proportion_by_num_img(lab):
    df = pd.merge(biz_id_train,train[['business_id',lab]],on='business_id')
    counts = df.groupby('business_id').apply( \
        lambda x: pd.Series([np.sum(x[lab]==0),np.sum(x[lab]==1)]))
    hc1 = np.histogram(counts[1],bins=range(0,1000,20))
    hc0 = np.histogram(counts[0],bins=range(0,1000,20))
    p = 1.*hc1[0]/(hc0[0]+hc1[0])
    plt.plot(list(hc1[1][1:]),list(p))




if __name__ == '__main__':

    # prep data to be read
    train = pd.read_csv('data/train.csv')
    biz_id_train = pd.read_csv('data/train_photo_to_biz_ids.csv')
    train[['0','1','2','3','4','5','6','7','8']] = train['labels'].apply(to_bool)
    h,w,ch,batch_size = 128,128,3,32
    
    ### EDA
    # bin the number of training images for each biz_id
 # 0.49 of biz_id are classified as having outdoor seating

    # proportion of images in each bin labeled as outdoors doesn't depend 
    # linearly with number of images provided
    
    
    ### Training
    model0,F1 = train_model(lab='0')
    model1,F1 = train_model(lab='1')
    model2,F1 = train_model(lab='2')
    model3,F1 = train_model(lab='3')
    model4,F1 = train_model(lab='4')
    model5,F1 = train_model(lab='5')
    model6,F1 = train_model(lab='6')
    model7,F1 = train_model(lab='7')
    model8,F1 = train_model(lab='8')    

    
    