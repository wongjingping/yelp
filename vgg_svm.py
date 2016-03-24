
import numpy as np
import pandas as pd
import h5py
from time import time
from PIL import Image

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D


# build VGG-16 model
def VGG_16(weights_path=None):

    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    # remove final softmax layer
    fc7 = Sequential()
    depth = len(model.layers)
    for i in range(depth-1):
       fc7.add(model.layers[i])
    # compile to get predict function. arbitrary optimizer and loss ok
    fc7.compile(optimizer='sgd',loss='mse') 

    return fc7

# build model
model = VGG_16('models/vgg16_weights.h5')

# read in images, normalize
biz_lab = pd.read_csv('data/train.csv')
photo_biz = pd.read_csv('data/train_photo_to_biz_ids.csv')
_,ch,h,w = model.input_shape
train_h5 = h5py.File('data/train.h5','w')
X_train = train_h5.create_dataset(
    name='X_train',
    shape=(len(photo_biz),ch,h,w),
    dtype=np.float32)
t = time()
for i in range(len(photo_biz)):
    f_ = 'data/train_photos/%s.jpg' % (photo_biz['photo_id'][i])
    im_sml = Image.open(f_).resize((w,h))
    # should use the specific means as used to normalize the data
    X_train[i,:,:,:] = np.asarray(im_sml).transpose((2,0,1))/128.-1
    if i % 1000 == 0:
        print('reading %ith image' % i)
train_h5.close()
print('Reading training data into hdf5 took %.1f' % (time()-t))


# calculate features on train set
t = time()
train_h5 = h5py.File('data/train.h5','r')
features_h5 = h5py.File('data/features.h5','w')
X_train = train_h5['X']
nrow,nf = len(photo_biz),model.output_shape[1]
X_feat = features_h5.create_dataset(name='features',
    shape=(nrow,nf),dtype=np.float32)
X_feat[...] = model.predict(X_train) # 114 mins
train_h5.close()
features_h5.close()
print('calculating features took %.1f' % (time()-t))

# average over (up to) batch_size sampled photos for each biz
nbiz, batch_size, nclass = len(biz_lab), 64, 9
rng = np.random.RandomState(290615)
features_h5 = h5py.File('data/features.h5','r')
features_all = features_h5['features']
X_agg = np.zeros((nbiz,nf))
y_agg = np.zeros((nbiz,nclass))
for i in biz_lab.index:
    biz_i,lab_i = biz_lab.ix[i,['business_id','labels']]
    photo_idx = photo_biz.index[photo_biz['business_id']==biz_i]
    if len(photo_idx) > batch_size:
        photo_s = rng.choice(photo_idx, size=batch_size, replace=False)
        photo_s.sort()
    else:
        photo_s = photo_idx
    X_agg[i,:] = np.mean(features_all[photo_s,:],axis=0)
    if type(lab_i) == str:
        y_agg[i,:] = np.asarray(\
            [1L if str(l) in lab_i.split(' ') else 0L for l in range(9)])
    else:
        y_agg[i,:] = np.zeros(9)
features_h5.close()

# train svm on 80% data
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score

scaler = StandardScaler()
X_sc = scaler.fit_transform(X_agg)
X_train,X_test,y_train,y_test = train_test_split(X_sc,y_agg,test_size=.2,random_state=rng)
classifier = OneVsRestClassifier(svm.SVC(kernel='rbf',C=1., random_state=rng))
t_svm = time()
classifier.fit(X_train, y_train)
print('Training SVM took %.0fs' % (time()-t_svm)) # ~70s

y_pred = classifier.predict(X_test)
F1 = f1_score(y_test, y_pred, average='micro')
print('F1 score: %.3f' % F1) # 0.775

# TODO use cross validation to select the optimal C parameter

# train on full data set
rng = np.random.RandomState(290615)
classifier = OneVsRestClassifier(svm.SVC(kernel='rbf',C=1., random_state=rng))
classifier.fit(X_sc, y_agg)


### ========== Predict for Submission ========== ###

# read in sampled test data
t_submit = time()
submit_photo_biz = pd.read_csv('data/test_photo_to_biz.csv')
submit_biz_lab = pd.read_csv('data/sample_submission.csv')
submit_h5 = h5py.File('data/submit.h5','w')

# read in sampled images for each biz_id, normalize
for i in submit_biz_lab.index:
    biz_i = submit_biz_lab.ix[i,'business_id']
    photo_s = submit_photo_biz.ix[submit_photo_biz['business_id']==biz_i,'photo_id']
    if len(photo_s) > batch_size:
        photo_s = photo_s.sample(n=batch_size,replace=False,random_state=290615)
    imgs = submit_h5.create_dataset(
        name=str(biz_i),
        shape=(len(photo_s),ch,h,w),
        dtype=np.float32)
    for j in range(len(photo_s)):
        f_ = 'data/test_photos/%s.jpg' % (photo_s.iloc[j])
        im_sml = Image.open(f_).resize((w,h))
        # should use the specific means as used to normalize the data
        imgs[j,:,:,:] = np.asarray(im_sml).transpose((2,0,1))/128.-1
    if i % 100 == 0:
        print('completed %d took %.1fs' % (i,time()-t_submit))
submit_h5.close()

# predict using mean biz feature
t_pred = time()
submit_h5 = h5py.File('data/submit.h5','r')
biz_l = submit_h5.keys()
for i in range(len(biz_l)):
    biz_i = biz_l[i]
    feat_i = np.mean(model.predict(submit_h5[biz_i]),axis=0).reshape(1,-1)
    # feat_sc = scaler.transform(feat_i)
    lab_i = np.where(classifier.predict(feat_i)[0])[0]
    submit_biz_lab.ix[submit_biz_lab['business_id']==biz_i,'labels'] = \
        ' '.join(str(l) for l in lab_i)
    if i % 10 == 0:
        print('completed %d took %.1fs' % (i,time()-t_pred))

submit_h5.close()
submit_biz_lab.to_csv('data/sub5_vgg16_svm.csv',index=False)
