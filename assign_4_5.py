
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


# In[11]:

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

# In[25]:

captions[0]


# ### compute labels

# In[12]:

def get_label(word_probs, entry_id, sep=' '):
    p_vals = word_probs[0].flatten()
    p_vals.sort()
    ids = where(p_vals >= WORD_PROBABILITY_TH)
    label_candidates = []
    for id in ids[0]:
        label_candidates.append(vocab[str(id)])
    label_candidates.reverse()    
    return sep.join(label_candidates)


# In[13]:

entry_id = randint(0, images.shape[0]-1)
result = the_func([images[entry_id]])


# In[15]:

label = get_label(result, entry_id)

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

# In[23]:

dec_x_rows = [] 
dec_y_rows = []
TRAINING_LIMIT = 1
for i in range(images.shape[0]):
    if (i == TRAINING_LIMIT): break
    a_1000_d = the_func([images[i]])[0]
    for j in range(labels_start[i], labels_end[i]+1):
        dec_x_rows.append(a_1000_d)
        dec_y_rows.append(labels[j])

dec_x = vstack(dec_x_rows)
dec_y = vstack(dec_y_rows)
print(dec_x.shape, dec_y.shape)


# ## Building the decoder

# In[ ]:

class MySimpleRecurrent(BaseRecurrent, Initializable):
    @lazy(allocation=['dim'])
    def __init__(self, dim, activation, **kwargs):
        self.dim = dim
        children = [activation]
        kwargs.setdefault('children', []).extend(children)
        super(MySimpleRecurrent, self).__init__(**kwargs)

    @property
    def W(self):
        return self.parameters[0]

    def get_dim(self, name):
        if name == 'mask':
            return 0
        if name in (MySimpleRecurrent.apply.sequences +
                    MySimpleRecurrent.apply.states):
            return self.dim
        return super(MySimpleRecurrent, self).get_dim(name)

    def _allocate(self):
        self.parameters.append(shared_floatx_nans((self.dim, self.dim), name="W"))
        add_role(self.parameters[0], WEIGHT)

        # NB no parameters for initial state

    def _initialize(self):
        self.weights_init.initialize(self.W, self.rng)

    @recurrent(sequences=['inputs', 'mask'], states=['states'],
               outputs=['states'], contexts=['context'])
    def apply(self, inputs, states, mask=None, **kwargs):
        next_states = inputs + tensor.dot(states, self.W)
        next_states = self.children[0].apply(next_states)
        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
        return next_states

    @application(contexts=["context"])
    def initial_states(self, batch_size, *args, **kwargs):
        init = kwargs["context"]
        return init.T

    @initial_states.property('outputs')
    def initial_states_outputs(self):
        return self.apply.states

    


# ### close

# In[ ]:

images_file.close()


# In[ ]:



