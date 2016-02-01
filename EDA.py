# -*- coding: utf-8 -*-
"""
EDA on yelp dataset with naive submission
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


