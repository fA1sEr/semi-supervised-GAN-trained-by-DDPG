from __future__ import print_function
import os
import tarfile
import subprocess
import argparse
import h5py
import numpy as np

def prepare_h5py(train_image, train_label, test_image, test_label, data_dir, shape=None):

    image = np.concatenate((train_image, test_image), axis=0).astype(np.uint8)
    label = np.concatenate((train_label, test_label), axis=0).astype(np.uint8)

    print('Preprocessing data...')

    import progressbar
    bar = progressbar.ProgressBar(maxval=100,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                           progressbar.Percentage()])
    bar.start()

    f = h5py.File(os.path.join(data_dir, 'data.hdf5'), 'w')
    data_id = open(os.path.join(data_dir,'id.txt'), 'w')
    for i in range(image.shape[0]):

        if i%(image.shape[0]/100)==0:
            bar.update(i/(image.shape[0]/100))

        grp = f.create_group(str(i))
        data_id.write(str(i)+'\n')
        if shape:
            grp['image'] = np.reshape(image[i], shape, order='F')
        else:
            grp['image'] = image[i]
        label_vec = np.zeros(10)
        label_vec[label[i]%10] = 1
        grp['label'] = label_vec.astype(np.bool)
    bar.finish()
    f.close()
    data_id.close()
    return

def check_file(data_dir):
    if os.path.exists(data_dir):
        if os.path.isfile(os.path.join('data.hdf5')) and \
            os.path.isfile(os.path.join('id.txt')):
            return True
    else:
        os.mkdir(data_dir)
    return False

def download_mnist(download_path):
    data_dir = os.path.join(download_path, 'mnist')

    if check_file(data_dir):
        print('MNIST was downloaded.')
        return

    data_url = 'http://yann.lecun.com/exdb/mnist/'
    keys = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
             't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']

    for k in keys:
        url = (data_url+k).format(**locals())
        target_path = os.path.join(data_dir, k)
        cmd = ['curl', url, '-o', target_path]
        print('Downloading ', k)
        subprocess.call(cmd)
        cmd = ['gzip', '-d', target_path]
        print('Unzip ', k)
        subprocess.call(cmd)

    num_mnist_train = 60000
    num_mnist_test = 10000

    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    train_image = loaded[16:].reshape((num_mnist_train,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    train_label = np.asarray(loaded[8:].reshape((num_mnist_train)).astype(np.float))

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    test_image = loaded[16:].reshape((num_mnist_test,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    test_label = np.asarray(loaded[8:].reshape((num_mnist_test)).astype(np.float))

    prepare_h5py(train_image, train_label, test_image, test_label, data_dir)

    for k in keys:
        cmd = ['rm', '-f', os.path.join(data_dir, k[:-3])]
        subprocess.call(cmd)

if __name__ == '__main__':
    path = './datasets'
    if not os.path.exists(path): os.mkdir(path)

    download_mnist('./datasets')