
import numpy as np
import pandas as pd
import h5py
from time import time
from PIL import Image
import cv2

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import backend as K
from keras.callbacks import Callback



# convert numeric labels to binary matrix
def to_bool(s):
    return(pd.Series([1L if str(i) in str(s).split(' ') else 0L for i in range(9)]))

# take average prediction of binary matrix and convert into a string of numeric labels
def from_bool(Y):
    y = np.where(Y.mean(axis=0).round())[0]
    return(' '.join(str(i) for i in y))

# returns matrix of pixel data given photo list
def generate_images(photo_list, rng=None, f_path='data/train_photos/'):

    # read and augment photos
    batch_size_i = len(photo_list)
    X_batch = np.empty((batch_size_i,ch,h,w))
    for i_ in range(batch_size_i):
        f_ = f_path + str(photo_list.iloc[i_]) + '.jpg'
        im = Image.open(f_)
        im_sml = im.resize((w,h))
        # scale inputs [-1,+1]
        xi = np.asarray(im_sml)/128.-1
        if rng:
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
        X_batch[i_,:,:,:] = xi.transpose((2,0,1))
    return X_batch


def batch_generator(df_,batch_size,gentype='train'):
    """
    Generates images from a specific business_id each batch.
    Samples batch_size examples if examples > batch_size.

    Parameters
    ----------
    df_: data frame
        joined data frame of photo_id, business_id, labels
    batch_size: int
        max size for each batch
    gentype: string
    """
    rng = np.random.RandomState(290616)
    bi,b_list = 0, df_['business_id'].unique()
    b_order = rng.permutation(b_list)
    while True:
        biz_id = b_order[bi]
        photos = df_.ix[df_['business_id']==biz_id,'photo_id']
        if len(photos) > batch_size:
            photos = photos.ix[rng.choice(photos.index,batch_size,replace=False)]
        # print('start bi %i biz_id %i n_photos' % (bi,biz_id,len(photos)))
        if gentype == 'train':
            X_batch = generate_images(photos,rng=rng,f_path='data/train_photos/')
        elif gentype == 'valid':
            X_batch = generate_images(photos,rng=rng,f_path='data/train_photos/')
        elif gentype == 'test':
            X_batch = generate_images(photos,rng=None,f_path='data/train_photos/')
        elif gentype == 'submit':
            X_batch = generate_images(photos,rng=None,f_path='data/test_photos/')
        else:
            raise NotImplementedError('%s not a valid gentype' % gentype)
        y_true = biz_lab.ix[biz_lab['business_id']==biz_id,[str(c) for c in range(9)]]
        Y_batch = np.array(y_true).repeat(len(photos),axis=0)
        # increase/loop indices for next iteration
        if bi < len(b_list)-1:
            bi += 1
        else:
            bi,b_order = 0,rng.permutation(b_list)
        # return batch data
        yield([X_batch],Y_batch)



# max of output probabilities
def obj_max(Y_true,Y_pred):
    y_true = K.mean(Y_true,axis=0)
    y_pred = K.max(Y_pred,axis=0)
    return(K.mean(K.binary_crossentropy(y_pred,y_true)))


# build VGG-16 model
def VGG_16(weights_path=None):

    vgg = Sequential()
    vgg.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    vgg.add(Convolution2D(64, 3, 3, activation='relu'))
    vgg.add(ZeroPadding2D((1,1)))
    vgg.add(Convolution2D(64, 3, 3, activation='relu'))
    vgg.add(MaxPooling2D((2,2), strides=(2,2)))

    vgg.add(ZeroPadding2D((1,1)))
    vgg.add(Convolution2D(128, 3, 3, activation='relu'))
    vgg.add(ZeroPadding2D((1,1)))
    vgg.add(Convolution2D(128, 3, 3, activation='relu'))
    vgg.add(MaxPooling2D((2,2), strides=(2,2)))

    vgg.add(ZeroPadding2D((1,1)))
    vgg.add(Convolution2D(256, 3, 3, activation='relu'))
    vgg.add(ZeroPadding2D((1,1)))
    vgg.add(Convolution2D(256, 3, 3, activation='relu'))
    vgg.add(ZeroPadding2D((1,1)))
    vgg.add(Convolution2D(256, 3, 3, activation='relu'))
    vgg.add(MaxPooling2D((2,2), strides=(2,2)))

    vgg.add(ZeroPadding2D((1,1)))
    vgg.add(Convolution2D(512, 3, 3, activation='relu'))
    vgg.add(ZeroPadding2D((1,1)))
    vgg.add(Convolution2D(512, 3, 3, activation='relu'))
    vgg.add(ZeroPadding2D((1,1)))
    vgg.add(Convolution2D(512, 3, 3, activation='relu'))
    vgg.add(MaxPooling2D((2,2), strides=(2,2)))

    vgg.add(ZeroPadding2D((1,1)))
    vgg.add(Convolution2D(512, 3, 3, activation='relu'))
    vgg.add(ZeroPadding2D((1,1)))
    vgg.add(Convolution2D(512, 3, 3, activation='relu'))
    vgg.add(ZeroPadding2D((1,1)))
    vgg.add(Convolution2D(512, 3, 3, activation='relu'))
    vgg.add(MaxPooling2D((2,2), strides=(2,2)))

    vgg.add(Flatten())
    vgg.add(Dense(4096, activation='relu'))
    vgg.add(Dropout(0.5))
    vgg.add(Dense(4096, activation='relu'))
    vgg.add(Dropout(0.5))
    vgg.add(Dense(1000, activation='softmax'))

    if weights_path:
        vgg.load_weights(weights_path)

    # replace final softmax layer with 9 output sigmoid
    vgg_9 = Sequential()
    depth = len(vgg.layers)
    for i in range(depth-1):
       vgg_9.add(vgg.layers[i])
    vgg_9.add(Dense(9, activation='sigmoid'))
    vgg_9.compile(optimizer='rmsprop',loss=obj_max,class_mode='binary') 

    return(vgg_9)


# evaluate on test set
def test(model,gen,n_biz,threshold=0.5,print_every_n=np.inf):
    t_start = time()
    ce_avg,tp,tn,fp,fn = 0.,0.,0.,0.,0.
    y = np.zeros((n_biz,2,9))
    for i in range(n_biz):
        X_test,Y_test = gen.next()
        y_test = Y_test.mean(axis=0)
        Y_pred = model.predict(X_test)
        y_pred = Y_pred.mean(axis=0)
        y[i,:,:] = np.asarray([y_test, y_pred])
        ce = np.mean(- y_test * np.log(y_pred) - (1-y_test) * np.log(1-y_pred))
        ce_avg += ce/n_biz
        y_predr = np.asarray(y_pred > threshold, dtype='uint8')
        tp += np.sum((y_test == 1) & (y_predr == 1))
        tn += np.sum((y_test == 0) & (y_predr == 0))
        fp += np.sum((y_test == 0) & (y_predr == 1))
        fn += np.sum((y_test == 1) & (y_predr == 0))
        if i % print_every_n == 0 and i > 0:
            print(i)
    prec,recall,acc = tp/(tp+fp+1e-15),tp/(tp+fn+1e-15),(tp+tn)/n_biz
    F1 = 2*tp/(2*tp+fp+fn)
    print('Valid F1 %.3f tp %.3f tn %.3f fp %.3f fn %.3f' % (F1,tp,tn,fp,fn))
    print('Took %.1fs' % (time()-t_start))
    return(ce_avg,prec,recall,F1,acc,tp,tn,fp,fn,y)


# callback that loops over validation set generator
class Validator(Callback):
    def __init__(self, model, valid_gen, n_biz, print_every_n=np.inf):
        self.model = model
        self.gen = valid_gen
        self.n_biz = n_biz
        self.print_every_n = print_every_n
        self.logs = {}
        self.t_batch = np.zeros((print_every_n if print_every_n != np.inf else 100))
    def on_batch_begin(self, epoch, logs={}):
        self.t_start = time()
    def on_batch_end(self, epoch, logs={}):
        t_batch = time()-self.t_start
        self.t_batch[epoch % len(self.t_batch)] = t_batch
        if epoch % self.print_every_n == 0 and epoch > 0:
            print('batch: %i acc: %.3f loss: %.3f size: %i took %.2fs' % \
        (logs['batch'],logs['acc'],logs['loss'],logs['size'],self.t_batch.mean()))
    def on_epoch_end(self, epoch, logs={}):
        ce,prec,recall,F1,acc,tp,tn,fp,fn,y = test(self.model,self.gen,self.n_biz)
        self.logs[epoch] = {'ce':ce,'prec':prec,'recall':recall,'F1':F1,
            'acc':acc,'tp':tp,'tn':tn,'fp':fp,'fn':fn}




if __name__ == '__main__':

    # prep data to be read
    biz_lab = pd.read_csv('data/train.csv')
    photo_biz = pd.read_csv('data/train_photo_to_biz_ids.csv')
    biz_lab[['0','1','2','3','4','5','6','7','8']] = biz_lab['labels'].apply(to_bool)
    h,w,ch,batch_size = 224,224,3,64

    # build model
    model = VGG_16('models/vgg16_weights.h5')

    # prepare generators for training separator
    train_cols = ['business_id'] + [str(i) for i in range(9)]
    df = pd.merge(photo_biz,biz_lab[train_cols], on='business_id')
    rng = np.random.RandomState(290615)
    i_split = rng.choice([0,1,2], size=len(biz_lab), p=[0.8,0.10,0.10])
    i_train = biz_lab.ix[i_split==0,'business_id']
    i_valid = biz_lab.ix[i_split==1,'business_id']
    i_test = biz_lab.ix[i_split==2,'business_id']
    df_train = df[df['business_id'].isin(i_train)]
    df_valid = df[df['business_id'].isin(i_valid)]
    df_test = df[df['business_id'].isin(i_test)]
    train_gen = batch_generator(df_train,batch_size,gentype='train')
    valid_gen = batch_generator(df_valid,batch_size,gentype='valid')
    test_gen = batch_generator(df_test,batch_size,gentype='test')
    samples_per_epoch = df_train.groupby('business_id').apply(\
        lambda x: len(x) if len(x) < batch_size else batch_size).sum()

    validator = Validator(model,valid_gen,len(i_valid))
    model.fit_generator(
        train_gen,
        samples_per_epoch=samples_per_epoch,
        nb_epoch=50,
        show_accuracy=True,
        verbose=2,
        callbacks=[validator])
    ce,prec,recall,F1,acc,tp,tn,fp,fn,y = test(model_separate,test_gen,len(i_test))

    # save model
    mname = 'vgg_mean_%03d' % (F1*100)
    model_json = model_separate.to_json()
    open('models/%s.json' % (mname), 'w').write(model_json)
    model.save_weights('models/%s_weights.h5' % (mname))

    # predict on submission data
    biz_lab_s = pd.read_csv('data/sample_submission.csv')
    photo_biz_s = pd.read_csv('data/test_photo_to_biz.csv')
    rng = np.random.RandomState(290615)
    t_submit = time()
    for i in range(len(biz_lab_s)):
        biz = biz_lab_s['business_id'][i]
        photos = photo_biz_s.ix[photo_biz_s['business_id']==biz,'photo_id']
        if len(photos) > batch_size:
            photos = photos.ix[rng.choice(photos.index,batch_size,replace=False)]
        X_batch = generate_images(photos, f_path='data/test_photos/')
        Y_batch = model.predict(X_batch, batch_size=batch_size)
        biz_lab_s['labels'][i] = from_bool(Y_batch)
        if i % 100 == 0 or i < 5:
            biz_lab_s.to_csv('data/sub6_vgg.csv',index=False)
            print('Predicted %dth biz %ds' % (i,time()-t_submit))
