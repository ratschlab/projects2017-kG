# Copyright 2017 Max Planck Society - ETH Zurich
# Distributed under the BSD-3 Software license,
# (See accompanying file ./LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)
"""This class helps to handle the data.

"""

import os
import logging
import numpy as np
from six.moves import cPickle
import utils
from PIL import Image
import sys


def load_cifar_batch(fpath, label_key='labels'):
    """Internal utility for parsing CIFAR data.

    # Arguments
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.

    # Returns
        A tuple `(data, labels)`.
    """
    f = utils.o_gfile(fpath, 'rb')
    if sys.version_info < (3,):
        d = cPickle.load(f)
    else:
        d = cPickle.load(f, encoding='bytes')
        # decode utf8
        d_decoded = {}
        for k, v in d.items():
            d_decoded[k.decode('utf8')] = v
        d = d_decoded
    f.close()
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels


class DataHandler(object):
    """A class storing and manipulating the dataset.

    In this code we asume a data point is a 3-dimensional array, for
    instance a 28*28 grayscale picture would correspond to (28,28,1),
    a 16*16 picture of 3 channels corresponds to (16,16,3) and a 2d point
    corresponds to (2,1,1). The shape is contained in self.data_shape
    """
    def __init__(self, opts):
        self.data_shape = None
        self.num_points = None
        self.data = None
        self.test_data = None
        self.labels = None
        self.test_labels = None
        self._load_data(opts)

    def _load_data(self, opts):
        """Load a dataset and fill all the necessary variables.

        """
        if opts['dataset'] == 'mnist':
            self._load_mnist(opts)
        elif opts['dataset'] == 'mnist3':
            self._load_mnist3(opts)
        elif opts['dataset'] == 'gmm':
            self._load_gmm(opts)
        elif opts['dataset'] == 'circle_gmm':
            self._load_mog(opts)
        elif opts['dataset'] == 'guitars':
            self._load_guitars(opts)
        elif opts['dataset'] == 'cifar10':
            self._load_cifar(opts)
        else:
            raise ValueError('Unknown %s' % opts['dataset'])

        sym_applicable = ['mnist',
                          'mnist3',
                          'guitars',
                          'cifar10']

        if opts['input_normalize_sym'] and opts['dataset'] in sym_applicable:
            # Normalize data to [-1, 1]
            self.data = (self.data - 0.5) * 2.

    def _data_dir(self, opts):
        if opts['data_dir'].startswith("/"):
            return opts['data_dir']
        else:
            return os.path.join('./', opts['data_dir'])

    def _load_mog(self, opts):
        """Sample data from the mixture of Gaussians on circle.

        """

        # Only use this setting in dimension 2
        assert opts['toy_dataset_dim'] == 2

        # First we choose parameters of gmm and thus seed
        radius = opts['gmm_max_val']
        modes_num = opts["gmm_modes_num"]
        np.random.seed(opts["random_seed"])

        thetas = np.linspace(0, 2 * np.pi, modes_num)
        mixture_means = np.stack((radius * np.sin(thetas), radius * np.cos(thetas)), axis=1)
        mixture_variance = 0.01

        # Now we sample points, for that we unseed
        np.random.seed()
        num = opts['toy_dataset_size']
        X = np.zeros((num, opts['toy_dataset_dim'], 1, 1))
        for idx in xrange(num):
            comp_id = np.random.randint(modes_num)
            mean = mixture_means[comp_id]
            cov = mixture_variance * np.identity(opts["toy_dataset_dim"])
            X[idx, :, 0, 0] = np.random.multivariate_normal(mean, cov, 1)

        self.data_shape = (opts['toy_dataset_dim'], 1, 1)
        self.data = X
        self.num_points = len(X)

    def _load_gmm(self, opts):
        """Sample data from the mixture of Gaussians.

        """

        logging.debug('Loading GMM dataset...')
        # First we choose parameters of gmm and thus seed
        modes_num = opts["gmm_modes_num"]
        np.random.seed(opts["random_seed"])
        max_val = opts['gmm_max_val']
        
        modes_per_row = int(np.sqrt(modes_num))
        step = np.ceil(2.*opts['gmm_max_val']/modes_per_row)
        init_i = - max_val - step/2.

        mixture_means = np.zeros([modes_num, 2])
        idx = 0
        for i in range(0,modes_num/modes_per_row):
            init_i += step 
            init_j = - max_val - step/2.
            for j in range(0, modes_num/modes_per_row):
                init_j += step
                mixture_means[idx, :] = np.array([init_i,init_j])
                idx += 1
        
        #mixture_means = np.random.uniform(
        #    low=-max_val-1, high=max_val+1,
        #    size=(modes_num, opts['toy_dataset_dim']))*1.5
        
        #mixture_means = np.array([[5,5],[-5,5],[5,-5],[-5,-5]])*2.

        def variance_factor(num, dim):
            if num == 1: return 3 ** (2. / dim)
            if num == 2: return 3 ** (2. / dim)
            if num == 3: return 8 ** (2. / dim)
            if num == 4: return 20 ** (2. / dim)
            if num == 5: return 10 ** (2. / dim)
            return num ** 2.0 * 3

        mixture_variance = \
                max_val / variance_factor(modes_num, opts['toy_dataset_dim'])

        def banana(X, b=0.04):
            """Twist the second column of X into a banana."""
            X = X[0]
            X = [x for x in X.copy()]
            X[1] += b * X[0]**2 - 100 * b
            return X
        # Now we sample points, for that we unseed
        np.random.seed()
        num = opts['toy_dataset_size']
        X = np.zeros((num, opts['toy_dataset_dim'], 1, 1))
        for idx in xrange(num):
            comp_id = np.random.randint(modes_num)
            mean = mixture_means[comp_id]
            cov = mixture_variance * np.identity(opts["toy_dataset_dim"])
            if opts['banana']:
                X[idx, :, 0, 0] = np.asarray([banana(np.random.multivariate_normal(mean, cov, 1))])
            else:
                X[idx, :, 0, 0] = np.random.multivariate_normal(mean, cov, 1)
        
        self.data_shape = (opts['toy_dataset_dim'], 1, 1)
        self.data = X
        self.num_points = len(X)

        logging.debug('Loading GMM dataset done!')
    


    def _load_guitars(self, opts):
        """Load data from Thomann files.

        """
        logging.debug('Loading Guitars dataset')
        data_dir = os.path.join('./', 'thomann')
        X = None
        files = utils.listdir(data_dir)
        pics = []
        for f in sorted(files):
            if '.jpg' in f and f[0] != '.':
                im = Image.open(utils.o_gfile((data_dir, f), 'rb'))
                res = np.array(im.getdata()).reshape(128, 128, 3)
                pics.append(res)
        X = np.array(pics)

        seed = 123
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed()

        self.data_shape = (128, 128, 3)
        self.data = X/255.
        self.num_points = len(X)

        logging.debug('Loading Done.')

    def _load_mnist(self, opts):
        """Load data from MNIST files.

        """
        logging.debug('Loading MNIST')
        data_dir = self._data_dir(opts)
        # pylint: disable=invalid-name
        # Let us use all the bad variable names!
        tr_X = None
        tr_Y = None
        te_X = None
        te_Y = None

        with utils.o_gfile((data_dir, 'train-images-idx3-ubyte'), 'rb') as fd:
            loaded = np.frombuffer(fd.read(), dtype=np.uint8)
            tr_X = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

        with utils.o_gfile((data_dir, 'train-labels-idx1-ubyte'), 'rb') as fd:
            loaded = np.frombuffer(fd.read(), dtype=np.uint8)
            tr_Y = loaded[8:].reshape((60000)).astype(np.int)

        with utils.o_gfile((data_dir, 't10k-images-idx3-ubyte'), 'rb') as fd:
            loaded = np.frombuffer(fd.read(), dtype=np.uint8)
            te_X = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        with utils.o_gfile((data_dir, 't10k-labels-idx1-ubyte'), 'rb') as fd:
            loaded = np.frombuffer(fd.read(), dtype=np.uint8)
            te_Y = loaded[8:].reshape((10000)).astype(np.int)

        tr_Y = np.asarray(tr_Y)
        te_Y = np.asarray(te_Y)

        X = np.concatenate((tr_X, te_X), axis=0)
        y = np.concatenate((tr_Y, te_Y), axis=0)

        seed = 123
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)
        np.random.seed()

        self.data_shape = (28, 28, 1)
        self.data = X / 255.
        self.labels = y
        self.num_points = len(X)

        logging.debug('Loading Done.')

    def _load_mnist3(self, opts):
        """Load data from MNIST files.

        """
        logging.debug('Loading 3-digit MNIST')
        data_dir = self._data_dir(opts)
        # pylint: disable=invalid-name
        # Let us use all the bad variable names!
        tr_X = None
        tr_Y = None
        te_X = None
        te_Y = None

        with utils.o_gfile((data_dir, 'train-images-idx3-ubyte'), 'rb') as fd:
            loaded = np.frombuffer(fd.read(), dtype=np.uint8)
            tr_X = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

        with utils.o_gfile((data_dir, 'train-labels-idx1-ubyte'), 'rb') as fd:
            loaded = np.frombuffer(fd.read(), dtype=np.uint8)
            tr_Y = loaded[8:].reshape((60000)).astype(np.int)

        with utils.o_gfile((data_dir, 't10k-images-idx3-ubyte'), 'rb') as fd:
            loaded = np.frombuffer(fd.read(), dtype=np.uint8)
            te_X = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        with utils.o_gfile((data_dir, 't10k-labels-idx1-ubyte'), 'rb') as fd:
            loaded = np.frombuffer(fd.read(), dtype=np.uint8)
            te_Y = loaded[8:].reshape((10000)).astype(np.int)

        tr_Y = np.asarray(tr_Y)
        te_Y = np.asarray(te_Y)

        X = np.concatenate((tr_X, te_X), axis=0)
        y = np.concatenate((tr_Y, te_Y), axis=0)

        num = opts['mnist3_dataset_size']
        ids = np.random.choice(len(X), (num, 3), replace=True)
        if opts['mnist3_to_channels']:
            # Concatenate 3 digits ito 3 channels
            X3 = np.zeros((num, 28, 28, 3))
            y3 = np.zeros(num)
            for idx, _id in enumerate(ids):
                X3[idx, :, :, 0] = np.squeeze(X[_id[0]], axis=2)
                X3[idx, :, :, 1] = np.squeeze(X[_id[1]], axis=2)
                X3[idx, :, :, 2] = np.squeeze(X[_id[2]], axis=2)
                y3[idx] = y[_id[0]] * 100 + y[_id[1]] * 10 + y[_id[2]]
            self.data_shape = (28, 28, 3)
        else:
            # Concatenate 3 digits in width
            X3 = np.zeros((num, 28, 3 * 28, 1))
            y3 = np.zeros(num)
            for idx, _id in enumerate(ids):
                X3[idx, :, 0:28, 0] = np.squeeze(X[_id[0]], axis=2)
                X3[idx, :, 28:56, 0] = np.squeeze(X[_id[1]], axis=2)
                X3[idx, :, 56:84, 0] = np.squeeze(X[_id[2]], axis=2)
                y3[idx] = y[_id[0]] * 100 + y[_id[1]] * 10 + y[_id[2]]
            self.data_shape = (28, 28 * 3, 1)

        self.data = X3/255.
        y3 = y3.astype(int)
        self.labels = y3
        self.num_points = num

        logging.debug('Training set JS=%.4f' % utils.js_div_uniform(y3))
        logging.debug('Loading Done.')

    def _load_cifar(self, opts):
        """Load CIFAR10

        """
        logging.debug('Loading CIFAR10 dataset')

        num_train_samples = 50000
        data_dir = self._data_dir(opts)
        x_train = np.zeros((num_train_samples, 3, 32, 32), dtype='uint8')
        y_train = np.zeros((num_train_samples,), dtype='uint8')

        for i in range(1, 6):
            fpath = os.path.join(data_dir, 'data_batch_' + str(i))
            data, labels = load_cifar_batch(fpath)
            x_train[(i - 1) * 10000: i * 10000, :, :, :] = data
            y_train[(i - 1) * 10000: i * 10000] = labels

        fpath = os.path.join(data_dir, 'test_batch')
        x_test, y_test = load_cifar_batch(fpath)

        y_train = np.reshape(y_train, (len(y_train), 1))
        y_test = np.reshape(y_test, (len(y_test), 1))
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

        X = np.vstack([x_train, x_test])
        X = X/255.
        y = np.vstack([y_train, y_test])

        seed = 123
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)
        np.random.seed()

        self.data_shape = (32, 32, 3)

        self.data = X[:-1000]
        self.test_data = X[-1000:]
        self.labels = y[:-1000]
        self.test_labels = y[-1000:]
        self.num_points = len(self.data)

        logging.debug('Loading Done.')
