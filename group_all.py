# -*- coding: utf-8 -*-
"""
Train a convnet for all labels simultaneously
"""


import threading
from time import time

import pandas as pd
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Permute, Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import Callback


# convert numeric labels to binary matrix
def to_bool(s):
	return(pd.Series([1L if str(i) in str(s).split(' ') else 0L for i in range(9)]))


# returns matrix of pixel data given photo list
def generate_images(photo_list, rng=None, augment=True, f_path='data/train_photos/'):
	# initiate random seed if not initialized
	if rng is None:
		rng = np.random.RandomState(290615)
	# read and augment photos
	batch_size_i = len(photo_list)
	X_batch = np.empty((batch_size_i,h,w,ch))
	for i_ in range(batch_size_i):
		f_ = f_path + str(photo_list.iloc[i_]) + '.jpg'
		im = Image.open(f_)
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
	return X_batch


# get top k images
def get_topk(X_batch,model,k):
	if len(X_batch) <= k:
		return(X_batch)
	Y_score = model.predict(X_batch)
	Y_rank = Y_score.argsort(axis=0)
	i_ = -1
	top_i = np.unique(Y_rank[i_,:])
	while len(top_i) < k:
		i_ -= 1
		top_i = np.unique(Y_rank[-1:i_:-1,:])
	top_i = top_i[:k]
	return(X_batch[top_i,...])


def batch_generator(df_,batch_size,gentype='train',model=None,k=0):
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
	model: keras model
	k: int
		top k images to select after predicting with model
	"""
	rng = np.random.RandomState(290616)
	bi,b_list = 0, df_['business_id'].unique()
	b_order = rng.permutation(b_list)
	lock = threading.Lock()
	while True:
		with lock:
			biz_id_i = b_order[bi]
			print('start bi %i biz_id %i' % (bi,biz_id_i))
			photos = df_.ix[df_['business_id']==biz_id_i,'photo_id']
			if len(photos) > batch_size:
				photos = photos.ix[rng.choice(photos.index,batch_size,replace=False)]
			if gentype == 'train':
				X_batch = generate_images(photos,rng=rng,augment=True,f_path='data/train_photos/')
			elif gentype == 'valid':
				X_batch = generate_images(photos,rng=rng,augment=True,f_path='data/train_photos/')
			elif gentype == 'test':
				X_batch = generate_images(photos,rng=rng,augment=True,f_path='data/train_photos/')
			elif gentype == 'submit':
				X_batch = generate_images(photos,rng=rng,augment=False,f_path='data/test_photos/')
			else:
				raise NotImplementedError('%s not a valid gentype' % gentype)
			y_true = train.ix[train['business_id']==biz_id_i,[str(c) for c in range(9)]]
			Y_batch = np.array(y_true).repeat(len(photos),axis=0)
			# select top k if model passed to generator
			if model is not None:
				X_batch = get_topk(X_batch,model,k)
				Y_batch = Y_batch[:k,...]
			# increase/loop indices for next iteration
			if bi < len(b_list)-1:
				bi += 1
			else:
				bi,b_order = 0,rng.permutation(b_list)
			# return batch data
			print('end bi %i biz_id %i' % (bi,biz_id_i))
			yield([X_batch],Y_batch)



# max of output probabilities
def obj_max(Y_true,Y_pred):
	y_true = K.mean(Y_true,axis=0)
	y_pred = K.max(Y_pred,axis=0)
	return(K.mean(K.binary_crossentropy(y_pred,y_true)))



def test(model,gen,n_id,threshold=0.5,print_every_n=np.inf):
	"""
	Evaluate on test set. Takes max prediction across all examples
	
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
	y = np.zeros((n_id,2,9))
	for i in range(n_id):
		X_test,Y_test = gen.next()
		y_test = Y_test.mean(axis=0)
		Y_pred = model.predict(X_test)
		y_pred = Y_pred.max(axis=0)
		y[i,:,:] = np.asarray([y_test, y_pred])
		ce = np.mean(- y_test * np.log(y_pred) - (1-y_test) * np.log(1-y_pred))
		ce_avg += ce/n_id
		y_predr = np.asarray(y_pred > threshold, dtype='uint8')
		tp += np.sum((y_test == 1) & (y_predr == 1))
		tn += np.sum((y_test == 0) & (y_predr == 0))
		fp += np.sum((y_test == 0) & (y_predr == 1))
		fn += np.sum((y_test == 1) & (y_predr == 0))
		if i % print_every_n == 0 and i > 0:
			print(i)
	prec,recall,acc = tp/(tp+fp+1e-15),tp/(tp+fn+1e-15),(tp+tn)/n_id
	F1 = 2*tp/(2*tp+fp+fn)
	print('Valid F1 %.3f tp %.3f tn %.3f fp %.3f fn %.3f' % (F1,tp,tn,fp,fn))
	print('Took %.1fs' % (time()-t_start))
	return(ce_avg,prec,recall,F1,acc,tp,tn,fp,fn,y)


# callback that loops over validation set generator
class Validator(Callback):
	def __init__(self, model, valid_gen, n_id, print_every_n=np.inf):
		self.model = model
		self.gen = valid_gen
		self.n_id = n_id
		self.print_every_n = print_every_n
		self.logs = {}
		self.t_batch = np.zeros((print_every_n))
	def on_batch_begin(self, epoch, logs={}):
		self.t_start = time()
	def on_batch_end(self, epoch, logs={}):
		t_batch = time()-self.t_start
		self.t_batch[epoch % self.print_every_n] = t_batch
		if epoch % self.print_every_n == 0 and epoch > 0:
			print('batch: %i acc: %.3f loss: %.3f size: %i took %.2fs' % \
		(logs['batch'],logs['acc'],logs['loss'],logs['size'],self.t_batch.mean()))
	def on_epoch_end(self, epoch, logs={}):
		ce,prec,recall,F1,acc,tp,tn,fp,fn,y = test(self.model,self.gen,self.n_id)
		self.logs[epoch] = {'ce':ce,'prec':prec,'recall':recall,'F1':F1,
			'acc':acc,'tp':tp,'tn':tn,'fp':fp,'fn':fn}


# compile model
def build_model():
	print('Compiling Model')
	t_comp = time()
	# build model
	model = Sequential()
	# reorder input to ch, h, w (no sample axis)
	model.add(Permute((3,1,2),input_shape=(h,w,ch)))
	# add conv layers
	model.add(Convolution2D(16,3,3,init='glorot_uniform',activation='relu',
							subsample=(1,1)))
	model.add(Convolution2D(16,3,3,init='glorot_uniform',activation='relu',
							subsample=(1,1)))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.1))
	model.add(Convolution2D(32,3,3,init='glorot_uniform',activation='relu',
							subsample=(1,1)))
	model.add(Convolution2D(32,3,3,init='glorot_uniform',activation='relu',
						subsample=(1,1)))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.1))
	model.add(Convolution2D(64,3,3,init='glorot_uniform',activation='relu',
							subsample=(1,1)))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.1))
	model.add(Convolution2D(128,3,3,init='glorot_uniform',activation='relu',
							subsample=(1,1)))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Flatten())
	model.add(Dense(output_dim=1000,init='glorot_uniform',activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(output_dim=1000,init='glorot_uniform',activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(output_dim=9,init='zero',activation='sigmoid'))

	model.compile(optimizer='rmsprop', loss=obj_max, 
				  class_mode='binary')
	t_train = time()
	print('Took %.1fs' % (t_train-t_comp))
	return(model)


# trains model for all labels simultaneously
def train(nb_epoch=10,k=5):

	# prepare generators for training separator
	train_cols = ['business_id'] + [str(i) for i in range(9)]
	df = pd.merge(biz_id_train,train[train_cols],on='business_id')
	rng = np.random.RandomState(290615)
	i_split = rng.choice([0,1,2], size=len(train), p=[0.8,0.10,0.10])
	i_train = train.ix[i_split==0,'business_id']
	i_valid = train.ix[i_split==1,'business_id']
	i_test = train.ix[i_split==2,'business_id']
	df_train = df[df['business_id'].isin(i_train)]
	df_valid = df[df['business_id'].isin(i_valid)]
	df_test = df[df['business_id'].isin(i_test)]
	train_gen = batch_generator(df_train,batch_size,gentype='train')
	valid_gen = batch_generator(df_valid,batch_size,gentype='valid')
	test_gen = batch_generator(df_test,batch_size,gentype='test')

	# train separator
	model_separate = build_model()
	validator = Validator(model_separate,valid_gen,len(i_valid),100)
	model_separate.fit_generator(
		train_gen,
		samples_per_epoch=len(df_train),
		nb_epoch=nb_epoch,
		show_accuracy=True,
		verbose=2,
		callbacks=[validator])
	ce,prec,recall,F1,acc,tp,tn,fp,fn,y = test(model_separate,test_gen,len(i_test))	

	# save separator model
	mname = 'group_sep_%03d' % (F1*100)
	model_json = model_separate.to_json()
	open('models/%s.json' % (mname), 'w').write(model_json)
	model_separate.save_weights('models/%s_weights.h5' % (mname))
	
	# prepare generators for topk separator
	train_gen = batch_generator(df_train,batch_size,gentype='train',
								model=model_separate,k=5)
	valid_gen = batch_generator(df_valid,batch_size,gentype='valid',
								model=model_separate,k=5)
	test_gen = batch_generator(df_test,batch_size,gentype='test',
							   model=model_separate,k=5)

	# train topk
	model_topk = build_model()
	validator = Validator(model_topk,valid_gen,len(i_valid),20)
	model_topk.fit_generator(
		train_gen,
		samples_per_epoch=len(df_train),
		nb_epoch=nb_epoch,
		verbose=2,
		show_accuracy=True,
		callbacks=[validator])
	ce,prec,recall,F1,acc,tp,tn,fp,fn = test(model_topk,test_gen,len(i_test))

	# save and return model
	mname = 'group_sep_%03d' % (F1*100)
	model_json = model_separate.to_json()
	open('models/%s.json' % (mname), 'w').write(model_json)
	model_separate.save_weights('models/%s_weights.h5' % (mname))

	return(model_topk,F1)




# returns max prob for all labels given a list of photos
def predict_biz_id(photo_id_l,n=128):
	n = min(n,len(photo_id_l))
	rng = np.random.RandomState(290615)
	# select up to/sample batch_size photos
	photo_l = photo_id_l.ix[rng.choice(photo_id_l.index,n,replace=False),'photo_id']
	X_batch = generate_images(photo_l, rng, augment=True, f_path='data/test_photos/')
	Y_pred = model_separate.predict(X_batch) # TODO change to topk when it's ready
	Y_maxprob = pd.Series(Y_pred.max(axis=0))
	return(Y_maxprob)


# predicts on submission set
def submit_max(batch_size):
	submit = pd.read_csv('data/sample_submission.csv')
	biz_id_test = pd.read_csv('data/test_photo_to_biz.csv')
	df_test = pd.merge(biz_id_test,submit,on='business_id')
	df_prob = biz_id_test.groupby('business_id').apply(predict_biz_id,n=batch_size)
	return(df_prob)


if __name__ == '__main__':

	# prep data to be read
	train = pd.read_csv('data/train.csv')
	biz_id_train = pd.read_csv('data/train_photo_to_biz_ids.csv')
	train[['0','1','2','3','4','5','6','7','8']] = train['labels'].apply(to_bool)
	h,w,ch,batch_size = 128,128,3,128
	
	
	# training separator
	sep,F1 = train(nb_epoch=10,k=5)
	submit = submit_max(batch_size)
	threshold = [submit[i].quantile(1-py[i]) for i in range(9)]
	submit_all = df_prob.apply(lambda x: ' '.join(str(i) for i in np.where(x>=threshold)[0]),axis=1)
	submit_all.to_csv('data/sub4_labels.csv', header=True)
