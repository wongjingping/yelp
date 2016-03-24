
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

# read in photo, biz, label for train and submit sets
photo_biz = pd.read_csv('data/train_photo_to_biz_ids.csv')
biz_lab = pd.read_csv('data/train.csv')
submit_photo_biz = pd.read_csv('data/test_photo_to_biz.csv')
submit_biz_lab = pd.read_csv('data/sample_submission.csv')

# read in images, normalize, store in hdf5
t_read = time()
_,ch,h,w = model.input_shape
_,nf = model.output_shape
mean_pixels = np.array([123.68,116.779,103.939]).reshape((3,1,1))
# store train set
raw_h5 = h5py.File('data/raw_train.h5','w')
X_train = raw_h5.create_dataset(
    name='X_train',
    shape=(len(biz_lab),nf),
    dtype=np.float32)
for i in range(len(biz_lab)):
    biz_i = biz_lab['business_id'][i]
    photo_l = photo_biz.ix[photo_biz['business_id']==biz_i,'photo_id']
    raw_i = np.zeros((len(photo_l),ch,h,w))
    for j in range(len(photo_l)):
        f_ = 'data/train_photos/%s.jpg' % (photo_biz['photo_id'][j])
        im_j = Image.open(f_).resize((w,h))
        ima_j = np.asarray(im_j).transpose((2,0,1))
        raw_i[j,:,:,:] = ima_j - mean_pixels
    X_train[i,:] = np.mean(model.predict(raw_i,batch_size=32),axis=0)
    if i % 10 == 0:
        print('reading %ith biz took %ds' % (i,time()-t_read))
print('Reading training data into hdf5 took %.1f' % (time()-t_read))
raw_h5.close()
# store submit set
raw_h5 = h5py.File('data/raw_submit.h5','w')
X_submit = raw_h5.create_dataset(
    name='X_submit',
    shape=(len(submit_biz_lab),nf),
    dtype=np.float32)
for i in range(len(submit_biz_lab)):
    biz_i = submit_biz_lab['business_id'][i]
    photo_l = submit_photo_biz.ix[submit_photo_biz['business_id']==biz_i,'photo_id']
    raw_i = np.zeros((len(photo_l),ch,h,w))
    for j in range(len(photo_l)):
        f_ = 'data/test_photos/%s.jpg' % (submit_photo_biz['photo_id'][j])
        im_j = Image.open(f_).resize((w,h))
        ima_j = np.asarray(im_j).transpose((2,0,1))
        raw_i[j,:,:,:] = ima_j - mean_pixels
    X_submit[i,:] = np.mean(model.predict(raw_i),axis=0)
    if i % 100 == 0:
        print('reading %ith biz took %ds' % (i,time()-t_read))
print('Reading submiting data into hdf5 took %.1f' % (time()-t_read))
raw_h5.close()

def to_bool(x):
    return(pd.Series([1L if str(i) in str(x).split(' ') else 0L for i in range(9)]))

raw_h5 = h5py.File('data/raw_train.h5','r')
X = raw_h5['X_train'][...]
raw_h5.close()
y = biz_lab['labels'].apply(to_bool)
raw_h5 = h5py.File('data/raw_submit.h5','r')
X_submit = raw_h5['X_submit'][...]
raw_h5.close()


# train svm
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score

scaler = StandardScaler()
X_sc = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = \
    train_test_split(X_sc,y,test_size=.2,random_state=290615)
clf = OneVsRestClassifier(svm.SVC(kernel='rbf',C=1., random_state=290615))
t_svm = time()
clf.fit(X_train, y_train)
print('Training SVM took %.0fs' % (time()-t_svm)) # ~70s

y_pred = clf.predict(X_test)
F1 = f1_score(y_test, y_pred, average='micro')
print('F1 score: %.3f' % F1) # 0.775


# train on full data set
rng = np.random.RandomState(290615)
clf = OneVsRestClassifier(svm.SVC(kernel='rbf',C=1., random_state=rng))
clf.fit(X_sc, y_agg)


# predict using mean biz feature
t_pred = time()
submit_h5 = h5py.File('data/submit.h5','r')
biz_l = submit_h5.keys()
for i in range(len(biz_l)):
    biz_i = biz_l[i]
    feat_i = np.mean(model.predict(submit_h5[biz_i]),axis=0).reshape(1,-1)
    # feat_sc = scaler.transform(feat_i)
    lab_i = np.where(clf.predict(feat_i)[0])[0]
    submit_biz_lab.ix[submit_biz_lab['business_id']==biz_i,'labels'] = \
        ' '.join(str(l) for l in lab_i)
    if i % 10 == 0:
        print('completed %d took %.1fs' % (i,time()-t_pred))

submit_h5.close()
submit_biz_lab.to_csv('data/sub5_vgg16_svm.csv',index=False)
