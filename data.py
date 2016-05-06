from numpy import array, unique, eye, vstack, where

import re, sys, h5py
from fuel.datasets.hdf5 import H5PYDataset

# params

MIN_SEQ_LENGTH = 100


# read data
f = open('data.txt', 'r')
raw = f.read().strip().replace('\n', ' ').replace('„', '').replace('“', '')
f.close()

vocab = unique(list(raw))
codes = eye(len(vocab), dtype='int')
seqs = re.split('\?|\.|!',raw)

# start = '^'
# end = '$'
seqs = [seq.strip() for seq in seqs]
clean_seqs = filter(lambda e: len(e) >= MIN_SEQ_LENGTH, seqs)

l = sys.maxsize
cut_seqs = []
for sq in clean_seqs:
    l = min(l, len(sq))  # use fixed length to get started
    cut_seqs.append(sq)

cut_seqs = [seq[:l] for seq in cut_seqs]

data = []
targets = []
for seq in cut_seqs:
    m = None
    t = []
    for c in seq:
        code_line = list(codes[where(vocab == c)][0])
        t.append(array(code_line, dtype='float32'))
        m = vstack(t)
        t = [m]
    tar_m = m.copy()[1:m.size]
    tar_m = vstack([tar_m, array([0]*m.shape[1])])
    data.append(m)
    targets.append(tar_m)

x_tensor = array(data)
y_vec = array(targets)

x_tensor = x_tensor.swapaxes(0, 1)
y_vec = y_vec.swapaxes(0, 1)

print(x_tensor.ndim, x_tensor.shape, y_vec.ndim, y_vec.shape)

# write tensor and targets to file
hdf5name = 'training_data.hdf5'
f = h5py.File(hdf5name, mode='w')

fx = f.create_dataset('x', x_tensor.shape, dtype='float32')
fy = f.create_dataset('y', y_vec.shape, dtype='int64')

fx[...] = x_tensor
fy[...] = y_vec

N = x_tensor.shape[0]  # FIXME
N_train = int(.9 * N)

split_dict = {
    'train': {'x': (0, N_train), 'y': (0, N_train)},
    'test': {'x': (N_train, N), 'y': (N_train, N)}}

f.attrs['split'] = H5PYDataset.create_split_array(split_dict)

f.flush()
f.close()