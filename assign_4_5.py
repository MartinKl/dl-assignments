
# coding: utf-8

# # Assignment 4 and 5
# 
# ## imports

# In[2]:

from h5py import File

from imagenet import ImagenetModel

import json

from scipy.misc import imshow

from numpy import array, where, take, vstack, hstack

from random import randint

from theano.tensor import tensor4
from theano import function

get_ipython().magic('matplotlib inline')


# ## constants

# In[3]:

IMAGE_DATA_H5PY_FILE = './data/cocotalk.h5'
IMG_DIR = './data/'
JSON_COCOTALK = './json/cocotalk.json'
JSON_COCO_RAW = './json/coco_raw.json'
MATLAB_MODEL_FILE = './model/imagenet-vgg-verydeep-16.mat'

WORD_PROBABILITY_TH = 0.01
R = .299
G = .587
B = .144


# ## load and build model

# In[4]:

imdl = ImagenetModel(MATLAB_MODEL_FILE)

x = tensor4('x', dtype='float32')
x_ = x
for layer in imdl.layers:
    x_ = layer.apply(x_)
    
the_func = function(inputs=[x], outputs=[x_], allow_input_downcast=True)


# ## sample
# ### load

# In[10]:

images_file = File(IMAGE_DATA_H5PY_FILE, 'r')
images = images_file['/images/']
label_length = images_file['/label_length/']
labels = images_file['/labels/']
labels_start = images_file['/label_start_ix/']
labels_end = images_file['/label_end_ix/']


# In[110]:

with open(JSON_COCOTALK) as f:
    js = json.loads(f.read())
    vocab = js['ix_to_word']
with open(JSON_COCO_RAW) as f:
    meta = json.loads(f.read())
    captions = {}
    pos2id_map = {}
    i = 0
    for e in js['images']:
        pos2id_map[e['id']] = i
        i+= 1
    for e in meta:
        captions[pos2id_map[e['id']]] = e['captions']
        


# ### inspect the data or network

# In[ ]:




# ### compute labels

# In[149]:

def get_label(word_probs, entry_id, sep=' '):
    p_vals = word_probs[0].flatten()
    p_vals.sort()
    ids = where(p_vals >= WORD_PROBABILITY_TH)
    label_candidates = []
    for id in ids[0]:
        label_candidates.append(vocab[str(id)])
    label_candidates.reverse()    
    return sep.join(label_candidates)


# In[133]:

entry_id = randint(0, images.shape[0]-1)
result = the_func([images[entry_id]])


# In[157]:

label = get_label(result, image_id)

print(label, '\ntrue labels:', captions[entry_id])
imshow(images[entry_id])


# # assignment 5
# ## create training data
# 
# We need to compute the 1000D-network output for each image. This output has to be set as input of the decoder k times, where k is the number of captions that exist for that particular image. E. g.:
# 
# image_0 has 5 captions.
# 
# 1. compute x = f(image_0)
# 2. align them:
# 
# $decode\bigg(\begin{bmatrix}x\\x\\x\\x\\x\end{bmatrix}\bigg) = \begin{bmatrix}repr(caption_1)\\repr(caption_2)\\repr(caption_3)\\repr(caption_4)\\repr(caption_5)\end{bmatrix}$ 

# In[ ]:

def get_repr(word_seq)

dec_x_rows = [] 
dec_y_rows = []
TRAINING_LIMIT = 2
for i in range(images.shape[0]):
    if (i == TRAINING_LIMIT): break
    a_1000_d = f([imgages[i]])
    for j in range(labels_start[i], labels_end[i])
        dec_x_rows.append(a_1000_d)
        dec_y_rows.append(labels[j])

dec_x = vstack(dec_x_rows)
dec_y = vstack(dec_y_rows)
print(dec_x.shape, dec_y.shape)


# ### close

# In[ ]:

images_file.close()


# In[ ]:



