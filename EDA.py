# -*- coding: utf-8 -*-
"""
EDA on yelp dataset with naive submission
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

train = pd.read_csv('data/train.csv')
biz_id_train = pd.read_csv('data/train_photo_to_biz_ids.csv')
submit = pd.read_csv('data/sample_submission.csv')

# plot distribution of photo counts
photo_count = biz_id_train.groupby('business_id').count().sort_values('photo_id')
photo_count.hist(bins=100)

# convert numeric labels to binary matrix
def to_bool(s):
    return(pd.Series([1L if str(i) in str(s).split(' ') else 0L for i in range(9)]))
Y = train['labels'].apply(to_bool)

# get means proportion of each class
py = Y.mean()
plt.bar(Y.columns,py,color='steelblue',edgecolor='white')

# plot correlation of outputs
plt.matshow(Y.corr(),cmap=plt.cm.RdBu)
plt.colorbar()

# 3 (outdoor_seating) is rather uncorrelated with the rest
# 0 (good_for_lunch) negatively correlated with the other descriptors, except good for kids
# 1,2,4-7 are a correlated cluster

# simulate randomly based on mean proportions
np.random.seed(290615)
submit['labels'] = submit.apply(lambda x: ' '.join( \
[str(i) for i in np.where(np.random.binomial(1,py,size=(9)))[0]]),axis=1)
submit.to_csv('data/sub1_naive.csv',index=False)



################## Explore pictures by category #######################

pic_bid_lab = pd.merge(biz_id_train,train)

# investigate outdoor pics since they seem uncorrelated with the other outputs
pic_3 = pic_bid_lab.loc[pic_bid_lab['labels'].str.contains(u'3')==True,['photo_id','business_id']]

def sample_pics(df, n=25, rows=5, cols=5):
    plt.rcParams['figure.figsize'] = (10, 10)
    n = min(len(df),n)
    plt.suptitle(str(df.iloc[0]['business_id']), fontsize=24)
    for i in range(n):
        plt.subplot(rows,cols,i+1)
        im = Image.open('data/train_photos/%i.jpg' % df.iloc[i]['photo_id']).resize((128,128))
        plt.imshow(im)
        plt.axis('off')

# run the following 2 lines as many times as you'd like
bid = pic_3['business_id'].sample(1).iloc[0]
sample_pics(pic_3[pic_3['business_id']==bid])
