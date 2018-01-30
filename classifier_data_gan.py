# Copyright 2017 Max Planck Society - ETH Zurich
# Distributed under the BSD-3 Software license,
# (See accompanying file ./LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)
"""This class implements some classifier training.

"""

import logging
import tensorflow as tf
import utils
from utils import ProgressBar
from utils import TQDM
import numpy as np
import ops
from metrics import Metrics

class Classifier(object):
    """A base class for running individual GANs.

    This class announces all the necessary bits for running individual
    GAN trainers. It is assumed that a GAN trainer should receive the
    data points and the corresponding weights, which are used for
    importance sampling of minibatches during the training. All the
    methods should be implemented in the subclasses.
    """
    def __init__(self, opts, data, data_weights = None):

        # Create a new session with session.graph = default graph
        self._session = tf.Session()
        self._trained = False
        self._data = data
        # Placeholders
        self._real_points_ph = None
        self._fake_points_ph = None
        self._noise_ph = None
        self._c_loss = None # Loss of mixture discriminator
        self._c_training = None # Outputs of the mixture discriminator on data
        self._data_weights = data_weights
        self._c_optim = None

        with self._session.as_default(), self._session.graph.as_default():
            logging.debug('Building the graph...')
            self._build_model_internal(opts)
        # Make sure AdamOptimizer, if used in the Graph, is defined before
        # calling global_variables_initializer().
        init = tf.global_variables_initializer()
        self._session.run(init)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Cleaning the whole default Graph
        logging.debug('Cleaning the graph...')
        tf.reset_default_graph()
        logging.debug('Closing the session...')
        # Finishing the session
        self._session.close()

    def train_mixture_discriminator(self, opts, fake_images, other_fake):
        """Train classifier separating true data from points in fake_images.

        Return:
            prob_real: probabilities of the points from training data being the
                real points according to the trained mixture classifier.
                Numpy vector of shape (self._data.num_points,)
            prob_fake: probabilities of the points from fake_images being the
                real points according to the trained mixture classifier.
                Numpy vector of shape (len(fake_images),)

        """
        with self._session.as_default(), self._session.graph.as_default():
            return self._train_mixture_discriminator_internal(opts, fake_images, other_fake)


    def _run_batch(self, opts, operation, placeholder, feed,
                   placeholder2=None, feed2=None):
        """Wrapper around session.run to process huge data.

        It is asumed that (a) first dimension of placeholder enumerates
        separate points, and (b) that operation is independently applied
        to every point, i.e. we can split it point-wisely and then merge
        the results. The second placeholder is meant either for is_train
        flag for batch-norm or probabilities of dropout.

        TODO: write util function which will be called both from this method
        and MNIST classification evaluation as well.

        """
        assert len(feed.shape) > 0, 'Empry feed.'
        num_points = feed.shape[0]
        batch_size = opts['tf_run_batch_size']
        batches_num = int(np.ceil((num_points + 0.) / batch_size))
        result = []
        for idx in xrange(batches_num):
            if idx == batches_num - 1:
                if feed2 is None:
                    res = self._session.run(
                        operation,
                        feed_dict={placeholder: feed[idx * batch_size:]})
                else:
                    res = self._session.run(
                        operation,
                        feed_dict={placeholder: feed[idx * batch_size:],
                                   placeholder2: feed2})
            else:
                if feed2 is None:
                    res = self._session.run(
                        operation,
                        feed_dict={placeholder: feed[idx * batch_size:
                                                     (idx + 1) * batch_size]})
                else:
                    res = self._session.run(
                        operation,
                        feed_dict={placeholder: feed[idx * batch_size:
                                                     (idx + 1) * batch_size],
                                   placeholder2: feed2})

            if len(res.shape) == 1:
                # convert (n,) vector to (n,1) array
                res = np.reshape(res, [-1, 1])
            result.append(res)
        result = np.vstack(result)
        assert len(result) == num_points
        return result

    def _build_model_internal(self, opts):
        """Build a TensorFlow graph with all the necessary ops.

        """
        assert False, 'Gan base class has no build_model method defined.'

    def _train_internal(self, opts):
        assert False, 'Gan base class has no train method defined.'

    def _sample_internal(self, opts, num):
        assert False, 'Gan base class has no sample method defined.'

    def _train_mixture_discriminator_internal(self, opts, fake_images):
        assert False, 'Gan base class has no mixture discriminator method defined.'

class ToyClassifier(Classifier):
    """A simple GAN implementation, suitable for toy datasets.

    """


    def discriminator(self, opts, input_,
                      prefix='DISCRIMINATOR', reuse=False):
        """Discriminator function, suitable for simple toy experiments.

        """
        shape = input_.get_shape().as_list()
        assert len(shape) > 0, 'No inputs to discriminate.'

        with tf.variable_scope(prefix, reuse=reuse):
            h0 = ops.linear(opts, input_, 500, 'h0_lin')
            h0 = tf.nn.relu(h0)
            h1 = ops.linear(opts, h0, 500, 'h1_lin')
            h1 = tf.nn.relu(h1)
            h2 = ops.linear(opts, h1, 1, 'h2_lin')

        return h2

    def _build_model_internal(self, opts):
        """Build the Graph corresponding to GAN implementation.

        """
        data_shape = self._data.data_shape

        # Placeholders
        real_points_ph = tf.placeholder(
            tf.float32, [None] + list(data_shape), name='real_points_ph')
        fake_points_ph = tf.placeholder(
            tf.float32, [None] + list(data_shape), name='fake_points_ph')
        noise_ph = tf.placeholder(
            tf.float32, [None] + [opts['latent_space_dim']], name='noise_ph')

        # Operations

        c_logits_real = self.discriminator(
            opts, real_points_ph, prefix='CLASSIFIER')
        c_logits_fake = self.discriminator(
            opts, fake_points_ph, prefix='CLASSIFIER', reuse=True)
        c_training = tf.nn.sigmoid(
            self.discriminator(opts, real_points_ph, prefix='CLASSIFIER', reuse=True))
        
        c_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=c_logits_real, labels=tf.ones_like(c_logits_real)))
        c_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=c_logits_fake, labels=tf.zeros_like(c_logits_fake)))
        c_loss = c_loss_real + c_loss_fake

        t_vars = tf.trainable_variables()
        c_vars = [var for var in t_vars if 'CLASSIFIER/' in var.name]
        c_optim = ops.optimizer(opts).minimize(c_loss, var_list=c_vars)

        self._real_points_ph = real_points_ph
        self._fake_points_ph = fake_points_ph
        self._noise_ph = noise_ph

        self._c_loss = c_loss
        self._c_training = c_training
        self._c_optim = c_optim


    def _train_mixture_discriminator_internal(self, opts, fake_images, other_fake):
        """Train a classifier separating true data from points in fake_images.

        """

        batches_num = self._data.num_points / opts['batch_size_classifier']
        logging.debug('Training a mixture discriminator')
        for epoch in xrange(opts["mixture_c_epoch_num"]):
            for idx in xrange(batches_num):
                ids = np.random.choice(len(fake_images), opts['batch_size_classifier'],
                                       replace=False)
                batch_fake_images = fake_images[ids]
                ids = np.random.choice(len(other_fake), opts['batch_size_classifier'],
                                       replace=False)
                batch_other_fake = other_fake[ids]
                ids = np.random.choice(self._data.num_points, opts['batch_size_classifier'],
                                       replace=False, p = self._data_weights)
                batch_real_images = self._data.data[ids]
                _ = self._session.run(
                    self._c_optim,
                    feed_dict={self._real_points_ph: np.concatenate((batch_real_images,batch_other_fake), axis = 0),
                               self._fake_points_ph: batch_fake_images})

        res = self._run_batch(
            opts, self._c_training,
            self._real_points_ph, self._data.data)
        return res, None


        
class ImageClassifier(Classifier):
    """A simple GAN implementation, suitable for pictures.

    """

    def __init__(self, opts, data):

        # One more placeholder for batch norm
        self._is_training_ph = None

        Classifier.__init__(self, opts, data)

    def discriminator(self, opts, input_, is_training,
                      prefix='DISCRIMINATOR', reuse=False):
        """Discriminator function, suitable for simple toy experiments.

        """
        num_filters = opts['d_num_filters']

        with tf.variable_scope(prefix, reuse=reuse):
            h0 = ops.conv2d(opts, input_, num_filters, scope='h0_conv')
            h0 = ops.batch_norm(opts, h0, is_training, reuse, scope='bn_layer1')
            h0 = ops.lrelu(h0)
            h1 = ops.conv2d(opts, h0, num_filters * 2, scope='h1_conv')
            h1 = ops.batch_norm(opts, h1, is_training, reuse, scope='bn_layer2')
            h1 = ops.lrelu(h1)
            h2 = ops.conv2d(opts, h1, num_filters * 4, scope='h2_conv')
            h2 = ops.batch_norm(opts, h2, is_training, reuse, scope='bn_layer3')
            h2 = ops.lrelu(h2)
            h3 = ops.linear(opts, h2, 1, scope='h3_lin')

        return h3

    def _build_model_internal(self, opts):
        """Build the Graph corresponding to GAN implementation.

        """
        data_shape = self._data.data_shape

        # Placeholders
        real_points_ph = tf.placeholder(
            tf.float32, [None] + list(data_shape), name='real_points_ph')
        fake_points_ph = tf.placeholder(
            tf.float32, [None] + list(data_shape), name='fake_points_ph')
        noise_ph = tf.placeholder(
            tf.float32, [None] + [opts['latent_space_dim']], name='noise_ph')
        is_training_ph = tf.placeholder(tf.bool, name='is_train_ph')


        # Operations
        # We use conv2d_transpose in the generator, which results in the
        # output tensor of undefined shapes. However, we statically know
        # the shape of the generator output, which is [-1, dim1, dim2, dim3]
        # where (dim1, dim2, dim3) is given by self._data.data_shape

        c_logits_real = self.discriminator(
            opts, real_points_ph, is_training_ph, prefix='CLASSIFIER')
        c_logits_fake = self.discriminator(
            opts, fake_points_ph, is_training_ph, prefix='CLASSIFIER', reuse=True)
        c_training = tf.nn.sigmoid(
            self.discriminator(opts, real_points_ph, is_training_ph,
                               prefix='CLASSIFIER', reuse=True))


        c_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=c_logits_real, labels=tf.ones_like(c_logits_real)))
        c_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=c_logits_fake, labels=tf.zeros_like(c_logits_fake)))
        c_loss = c_loss_real + c_loss_fake

        t_vars = tf.trainable_variables()

        # d_optim_op = ops.optimizer(opts, 'd')
        # g_optim_op = ops.optimizer(opts, 'g')

        # def debug_grads(grad, var):
        #     _grad =  tf.Print(
        #         grad, # grads_and_vars,
        #         [tf.global_norm([grad])], 
        #         'Global grad norm of %s: ' % var.name)
        #     return _grad, var

        # d_grads_and_vars = [debug_grads(grad, var) for (grad, var) in \
        #     d_optim_op.compute_gradients(d_loss, var_list=d_vars)]
        # g_grads_and_vars = [debug_grads(grad, var) for (grad, var) in \
        #     g_optim_op.compute_gradients(g_loss, var_list=g_vars)]
        # d_optim = d_optim_op.apply_gradients(d_grads_and_vars)
        # g_optim = g_optim_op.apply_gradients(g_grads_and_vars)

        c_vars = [var for var in t_vars if 'CLASSIFIER/' in var.name]
        c_optim = ops.optimizer(opts).minimize(c_loss, var_list=c_vars)

        self._real_points_ph = real_points_ph
        self._fake_points_ph = fake_points_ph
        self._noise_ph = noise_ph
        self._is_training_ph = is_training_ph
        self._c_loss = c_loss
        self._c_training = c_training
        self._c_optim = c_optim

        logging.debug("Building Graph Done.")


    def _train_internal(self, opts):
        """Train a GAN model.

        """

        batches_num = self._data.num_points / opts['batch_size_classifier']
        train_size = self._data.num_points

        counter = 0
        logging.debug('Training GAN')
        for _epoch in xrange(opts["gan_epoch_num"]):
            for _idx in xrange(batches_num):
                # logging.debug('Step %d of %d' % (_idx, batches_num ) )
                data_ids = np.random.choice(train_size, opts['batch_size_classifier'],
                                            replace=False, p=self._data_weights)
                batch_images = self._data.data[data_ids].astype(np.float)
                batch_noise = utils.generate_noise(opts, opts['batch_size_classifier'])
                # Update discriminator parameters
                for _iter in xrange(opts['d_steps']):
                    _ = self._session.run(
                        self._d_optim,
                        feed_dict={self._real_points_ph: batch_images,
                                   self._noise_ph: batch_noise,
                                   self._is_training_ph: True})
                # Update generator parameters
                for _iter in xrange(opts['g_steps']):
                    _ = self._session.run(
                        self._g_optim,
                        feed_dict={self._noise_ph: batch_noise,
                                   self._is_training_ph: True})
                counter += 1

                if opts['verbose'] and counter % opts['plot_every'] == 0:
                    logging.debug(
                        'Epoch: %d/%d, batch:%d/%d' % \
                        (_epoch+1, opts['gan_epoch_num'], _idx+1, batches_num))
                    metrics = Metrics()
                    points_to_plot = self._run_batch(
                        opts, self._G, self._noise_ph,
                        self._noise_for_plots[0:320],
                        self._is_training_ph, False)
                    metrics.make_plots(
                        opts,
                        counter,
                        None,
                        points_to_plot,
                        prefix='sample_e%04d_mb%05d_' % (_epoch, _idx))
                if opts['early_stop'] > 0 and counter > opts['early_stop']:
                    break

    def _sample_internal(self, opts, num):
        """Sample from the trained GAN model.

        """
        noise = utils.generate_noise(opts, num)
        sample = self._run_batch(
            opts, self._G, self._noise_ph, noise,
            self._is_training_ph, False)
        # sample = self._session.run(
        #     self._G, feed_dict={self._noise_ph: noise})
        return sample

    def _train_mixture_discriminator_internal(self, opts, fake_images):
        """Train a classifier separating true data from points in fake_images.

        """

        batches_num = self._data.num_points / opts['batch_size']
        logging.debug('Training a mixture discriminator')
        logging.debug('Using %d real points and %d fake ones' %\
                      (self._data.num_points, len(fake_images)))
        for epoch in xrange(opts["mixture_c_epoch_num"]):
            for idx in xrange(batches_num):
                ids = np.random.choice(len(fake_images), opts['batch_size'],
                                       replace=False)
                batch_fake_images = fake_images[ids]
                ids = np.random.choice(self._data.num_points, opts['batch_size'],
                                       replace=False)
                batch_real_images = self._data.data[ids]
                _ = self._session.run(
                    self._c_optim,
                    feed_dict={self._real_points_ph: batch_real_images,
                               self._fake_points_ph: batch_fake_images,
                               self._is_training_ph: True})

        # Evaluating trained classifier on real points
        res = self._run_batch(
            opts, self._c_training,
            self._real_points_ph, self._data.data,
            self._is_training_ph, False)

        # Evaluating trained classifier on fake points
        res_fake = self._run_batch(
            opts, self._c_training,
            self._real_points_ph, fake_images,
            self._is_training_ph, False)
        return res, res_fake

