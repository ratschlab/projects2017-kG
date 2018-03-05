# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license,
# (See accompanying file ./LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)
"""This class implements VAE training.

"""

import os
import logging
import tensorflow as tf
import utils
from utils import ProgressBar
from utils import TQDM
import numpy as np
import ops
from metrics import Metrics
from ais import ais
#from ais2 import ais
import vae as VAE
import classifier as CLASSIFIER

class kVae(object):
    """A base class for running individual VAEs.

    """
    def __init__(self, opts, data, weights, test_data = None, test_weights = None):

        # Create a new session with session.graph = default graph
        self._session = tf.Session()
        self._trained = False
        self._data = data
        self._data_weights = np.copy(weights)
        self._test_data = test_data
        self._test_weights = np.copy(test_weights)
        # Latent noise sampled ones to apply decoder while training
        self._noise_for_plots = utils.generate_noise(opts, 500)
        # Placeholders
        self._real_points_ph = None
        self._noise_ph = None
        self._hard_assigned = np.ones(opts['number_of_kGANs'])*1.*len(data.data)
        # Main operations
        # FIX
        self._loss = None
        self._loss_reconstruct = None
        self._loss_kl = None
        self._generated = None
        self._reconstruct_x = None

        # Optimizers
        self.optim = None

        with self._session.as_default(), self._session.graph.as_default():
            logging.error('Building the graph...')
            self._build_model_internal(opts)

        # Make sure AdamOptimizer, if used in the Graph, is defined before
        # calling global_variables_initializer().
        init = tf.global_variables_initializer()
        self._session.run(init)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Cleaning the whole default Graph
        logging.error('Cleaning the graph...')
        tf.reset_default_graph()
        logging.error('Closing the session...')
        # Finishing the session
        self._session.close()

    def train_vaes(self, opts):
        """Train the kVAEs.

        """
        with self._session.as_default(), self._session.graph.as_default():
            self._train_internal_vaes(opts)
            for k in range(opts['number_of_kGANs']):  
                self.kVAEs[k]._trained = True

    def train_class(self, opts, fake_images):
        """Train the k classifiers.

        """
        with self._session.as_default(), self._session.graph.as_default():
           return  self._train_internal_classifiers(opts, fake_images)
    def reinit_classifiers(self,opts):
        logging.error("Reinitializing classifiers.")
        prefix = 'CLASSIFIER'
        for k in range(0,opts['number_of_kGANs']):
            t_vars = tf.trainable_variables()
            c_vars = [var for var in t_vars if prefix+str(k)+'/' in var.name]
            [self._session.run(var.initializer) for var in c_vars]

    def sample(self, opts, num=100):
        """Sample points from the trained VAE model.

        """
        assert self._trained, 'Can not sample from the un-trained VAE'
        with self._session.as_default(), self._session.graph.as_default():
            return self._sample_internal(opts, num)

    def train_mixture_discriminator(self, opts, fake_images):
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
            return self._train_mixture_discriminator_internal(opts, fake_images)


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
        assert False, 'VAE base class has no build_model method defined.'

    def _train_internal(self, opts):
        assert False, 'VAE base class has no train method defined.'

    def _sample_internal(self, opts, num):
        assert False, 'VAE base class has no sample method defined.'

    def _train_mixture_discriminator_internal(self, opts, fake_images):
        assert False, 'VAE base class has no mixture discriminator method defined.'


#class GeneratorAdapter(object):
#
#  def __init__(self, generator, input_dim, output_dim):
#    self._generator = generator
#    self.input_dim = input_dim
#    self.output_dim = output_dim
#
#  def __call__(self, z):
#    return self._generator(z)



class kToyVae(kVae):
    """A simple VAE implementation, suitable for toy dataset.

    """

    def __init__(self, opts, data, weights):

        # One more placeholder for batch norm
        self._is_training_ph = None

        kVae.__init__(self, opts, data, weights)



    def _build_model_internal(self, opts):
        """Build the Graph corresponding to VAE implementation.

        """
        data_shape = self._data.data_shape
        real_points_phs = []
        noise_phs = []
        is_training_phs = []
        lr_decay_phs = []
        vae_class = VAE.ToyVae
        classifier = CLASSIFIER.ToyClassifier
        kVAEs = [] 
        kCLASSs = []
        optims = []
        class_real_pts_phs = []
        class_fake_pts_phs = []
        class_optims = []
        for k in range(opts['number_of_kGANs']):
            device = k%opts['number_of_gpus']
            with tf.device('/device:GPU:%d' %device):
                new_vae = vae_class(opts, self._data, model_idx = k, sess = self._session) 
                new_class = classifier(opts, self._data, model_idx = k, sess = self._session )
                kVAEs.append(new_vae)
                kCLASSs.append(new_class)
                real_points_phs.append(new_vae._real_points_ph)
                noise_phs.append(new_vae._noise_ph)
                optims.append(new_vae._optim)
                is_training_phs.append(new_vae._is_training_ph)
                lr_decay_phs.append(new_vae._lr_decay_ph)
                class_real_pts_phs.append(new_class._real_points_ph)
                class_fake_pts_phs.append(new_class._fake_points_ph)
                class_optims.append(new_class._c_optim)

        # Operations

        #latent_x_mean, log_latent_sigmas = self.discriminator(
        #    opts, real_points_ph, is_training_ph)
        #scaled_noise = tf.multiply(
        #    tf.sqrt(1e-6 + tf.exp(log_latent_sigmas)), noise_ph)
        #loss_kl = 0.5 * tf.reduce_sum(
        #    tf.exp(log_latent_sigmas) +
        #    tf.square(latent_x_mean) -
        #    log_latent_sigmas, axis=1)
        #if opts['recon_loss'] == 'l2sq':
        #    reconstruct_x = self.generator(opts, latent_x_mean + scaled_noise,
        #                                   is_training_ph)
        #    loss_reconstruct = tf.reduce_sum(
        #        tf.square(real_points_ph - reconstruct_x), axis=[1,2,3])
        #    loss_reconstruct = loss_reconstruct / 2. / opts['vae_sigma']
        #elif opts['recon_loss'] == 'cross_entropy':
        #    if opts['input_normalize_sym']:
        #        expected = (real_points_ph + 1.0) / 2.0
        #    else:
        #        expected = real_points_ph
        #    reconstruct_x_logits = self.generator(
        #        opts, latent_x_mean + scaled_noise,
        #        is_training_ph, return_logits=True)
        #    loss_reconstruct = tf.reduce_sum(
        #        tf.nn.sigmoid_cross_entropy_with_logits(
        #            labels=expected, logits=reconstruct_x_logits),
        #        axis=[1,2,3])
        #else:
        #    raise ValueError("Unknown recon loss value %s" % opts['recon_loss'])
        #dec_enc_x = self.generator(opts, latent_x_mean,
        #                           is_training=False, reuse=True)
        #self.loss_pp = loss_reconstruct
        #loss_reconstruct = tf.reduce_mean(loss_reconstruct)
        #loss_kl = tf.reduce_mean(loss_kl)
        #loss = loss_kl + loss_reconstruct
        # loss = tf.Print(loss, [loss, loss_kl, loss_reconstruct], 'Loss, KL, reconstruct')
        #optim = ops.optimizer(opts, decay=lr_decay_ph).minimize(loss)

        #generated_images = self.generator(opts, noise_ph,
         #                                 is_training_ph, reuse=True)

        self._real_points_phs = real_points_phs
        self._noise_phs = noise_phs
        self._is_training_phs = is_training_phs
        self._optims = optims
        #self._loss = loss
        #self._loss_reconstruct = loss_reconstruct
        self._lr_decay_phs = lr_decay_phs
        self._class_real_pts_phs = class_real_pts_phs
        self._class_fake_pts_phs = class_fake_pts_phs
        self._class_optims = class_optims
        self.kVAEs = kVAEs
        self.kCLASS = kCLASSs
        #self._loss_kl = loss_kl
        #self._generated = generated_images
        #self._reconstruct_x = dec_enc_x
        #self._enc_mean = latent_x_mean
        #self._enc_log_var = log_latent_sigmas

        #saver = tf.train.Saver(max_to_keep=10)
        #tf.add_to_collection('real_points_ph', self._real_points_ph)
        #tf.add_to_collection('noise_ph', self._noise_ph)
        #tf.add_to_collection('is_training_ph', self._is_training_ph)
        #tf.add_to_collection('encoder_mean', self._enc_mean)
        #tf.add_to_collection('encoder_log_sigma', self._enc_log_var)
        #tf.add_to_collection('decoder', self._generated)
        #print "Building the AIS model..."
        #decoder = (lambda x, o=opts, i=is_training_ph:
        #           self.generator(o, x, i, reuse=True))

        #output_dim = np.prod(list(data_shape))
        #self._ais_model = ais.AIS(
        #    generator=GeneratorAdapter(decoder,
        #                               opts['latent_space_dim'],
        #                               output_dim),
        #    prior=ais.NormalPrior(),
        #    kernel=ais.ParsenDensityEstimator(),
        #    sigma=0.1,
        #    num_samples=1000,
        #    stepsize=0.1)
        #print "Building the AIS model: DONE"
        #self._saver = saver

        #logging.error("Building Graph Done.")


    def _train_internal_vaes(self, opts):
        """Train a VAE model.

        """

        batches_num = self._data.num_points / opts['batch_size']
        #if opts['one_batch'] == True:
        #    batches_num = 1
        train_size = self._data.num_points
        num_plot = 320
        sample_prev = np.zeros([num_plot] + list(self._data.data_shape))
        l2s = []

        counter = 0
        decay = 1.
        logging.error('Training VAE')
        for _epoch in xrange(opts["gan_epoch_num"]):

            if opts['decay_schedule'] == "manual":
                if _epoch == 30:
                    decay = decay / 2.
                if _epoch == 50:
                    decay = decay / 5.
                if _epoch == 100:
                    decay = decay / 10.

            if False and  _epoch > 0 and _epoch % opts['save_every_epoch'] == 0:
                os.path.join(opts['work_dir'], opts['ckpt_dir'])
                #self._saver.save(self._session,
                #                 os.path.join(opts['work_dir'],
                #                              opts['ckpt_dir'],
                #                              'trained-pot'),
                #                 global_step=counter)

            for _idx in xrange(batches_num):
                # logging.error('Step %d of %d' % (_idx, batches_num ) )
                dict_ph = {}
                optimizers = []
                for k in range(opts['number_of_kGANs']):
                    if self._hard_assigned[k] >= opts['batch_size']: 
                        data_ids = np.random.choice(train_size, opts['batch_size'],
                                replace=False, p=self._data_weights[:,k])
                        batch_images = self._data.data[data_ids].astype(np.float)
                        batch_noise = utils.generate_noise(opts, opts['batch_size'])
                        dict_ph[self._real_points_phs[k]] = batch_images
                        dict_ph[self._noise_phs[k]] = batch_noise
                        dict_ph[self._lr_decay_phs[k]] = decay
                        dict_ph[self._is_training_phs[k]] = True
                        optimizers.append(self._optims[k])
                #_, loss, loss_kl, loss_reconstruct = self._session.run(
                #    [self._optim, self._loss, self._loss_kl,
                #     self._loss_reconstruct],
                #    feed_dict={self._real_points_ph: batch_images,
                #               self._noise_ph: batch_noise,
                #               self._lr_decay_ph: decay,
                #               self._is_training_ph: True})
                self._session.run(
                    optimizers,
                    feed_dict=dict_ph)
                counter += 1

                #if False and opts['verbose'] and counter % opts['plot_every'] == 0:
                #    debug_str = 'Epoch: %d/%d, batch:%d/%d' % (
                #        _epoch+1, opts['gan_epoch_num'], _idx+1, batches_num)
                #    debug_str += '  [L=%.2g, Recon=%.2g, KLQ=%.2g]' % (
                #        loss, loss_reconstruct, loss_kl)
                #    logging.error(debug_str)

                #if False and opts['verbose'] and counter % opts['plot_every'] == 0:
                #    metrics = Metrics()
                #    points_to_plot = self._run_batch(
                #        opts, self._generated, self._noise_ph,
                #        self._noise_for_plots[0:num_plot],
                #        self._is_training_ph, False)
                #    l2s.append(np.sum((points_to_plot - sample_prev)**2))
                #    metrics.l2s = l2s[:]
                #    metrics.make_plots(
                #        opts,
                #        counter,
                #        None,
                #        points_to_plot,
                #        prefix='sample_e%04d_mb%05d_' % (_epoch, _idx))
                #    reconstructed = self._session.run(
                #        self._reconstruct_x,
                #        feed_dict={self._real_points_ph: batch_images,
                #                   self._is_training_ph: False})
                #    metrics.l2s = None
                #    metrics.make_plots(
                #        opts,
                #        counter,
                #        None,
                #        reconstructed,
                #        prefix='reconstr_e%04d_mb%05d_' % (_epoch, _idx))
                #if opts['early_stop'] > 0 and counter > opts['early_stop']:
                #    break
        #if _epoch > 0:
        #    os.path.join(opts['work_dir'], opts['ckpt_dir'])
        #    self._saver.save(self._session,
        #                     os.path.join(opts['work_dir'],
        #                                  opts['ckpt_dir'],
        #                                  'trained-pot-final'),
        #                     global_step=counter)
    
    def _train_internal_classifiers(self, opts, fake_images):
        """Train a classifier separating true data from points in fake_images.
        
        """
        batches_num = self._data.num_points / opts['batch_size_classifier']
        if opts['one_batch'] == True:
            batches_num = 1
        logging.debug('Training a mixture discriminator')
        for epoch in xrange(opts["mixture_c_epoch_num"]):
            for idx in xrange(batches_num):
                dict_ph = {}
                for k in range(0,opts['number_of_kGANs']): 
                    ids = np.random.choice(len(fake_images[0]), opts['batch_size_classifier'],replace=False)
                    batch_fake_images = fake_images[k,ids]
                    dict_ph[self._class_fake_pts_phs[k]] = batch_fake_images

                    ids = np.random.choice(self._data.num_points, opts['batch_size_classifier'],replace=False)
                    batch_real_images = self._data.data[ids]
                    dict_ph[self._class_real_pts_phs[k]] = batch_real_images
                _ = self._session.run(
                    self._class_optims,
                    feed_dict = dict_ph)
        res = np.zeros((len(self._data.data),opts['number_of_kGANs']))
        for k in range(0,opts['number_of_kGANs']):
            res[:,k] = self._run_batch(
                opts, self.kCLASS[k]._c_training,
                self.kCLASS[k]._real_points_ph, self._data.data).flatten()

        return res


    #def _sample_internal(self, opts, num):
    #    """Sample from the trained GAN model.
    #
    #    """
    #    noise = utils.generate_noise(opts, num)
    #    sample = self._run_batch(
    #        opts, self._generated, self._noise_ph, noise,
    #        self._is_training_ph, False)
    #    return sample

    #def compute_ais(self):
    #    self._ais_model.set_session(self._session)
    #    batch_size = 64
    #    train_size = self._data.num_points
    #    batch_size = min(batch_size, len(np.argwhere(self._data_weights != 0)))
    #    data_ids = np.random.choice(train_size, batch_size,
    #                                replace=False, p=self._data_weights)
    #    current_batch = self._data.data[data_ids].astype(np.float)
    #    current_batch = np.reshape(current_batch, [batch_size, -1])
    #    lld = self._ais_model.ais(current_batch,
    #                              ais.get_schedule(400, rad=4))
        #print "=== Step: {} ===".format(current_step)
        #print "loss: {}".format(loss)
        #print "log-likelihood: {}".format(lld)
        #print "mean(log-likelihood): {}".format(np.mean(lld))
     #   return np.mean(lld)

    #def compute_ais_test(self):
    #    self._ais_model.set_session(self._session)
    #    batch_size = 64
    #    train_size = len(self._test_data)
    #    batch_size = min(batch_size, train_size)
    #    data_ids = np.random.choice(train_size, batch_size,
    #                                replace=False, p=self._test_weights)
    #    current_batch = self._test_data[data_ids].astype(np.float)
    #    current_batch = np.reshape(current_batch, [batch_size, -1])
    #    lld = self._ais_model.ais(current_batch,
    #                              ais.get_schedule(400, rad=4))
        #print "=== Step: {} ===".format(current_step)
        #print "loss: {}".format(loss)
        #print "log-likelihood: {}".format(lld)
        #print "mean(log-likelihood): {}".format(np.mean(lld))
    #    return np.mean(lld)
    #def loss_pt(self, opts):
    #    num_points = self._data.num_points
    #    batch_noise = utils.generate_noise(opts, self._data.num_points)
    #    batch_size = opts['tf_run_batch_size']
    #    batches_num = int(np.ceil((num_points + 0.) / batch_size))
    #    result = []
    #    for idx in xrange(batches_num):
    #        if idx == batches_num - 1:
    #                res = self._session.run(
    #                    self.loss_pp,
    #                    feed_dict={self._real_points_ph: self._data.data[idx * batch_size:],
    #                        self._noise_ph: batch_noise[idx * batch_size:],
    #                        self._is_training_ph: False})
    #        else:
    #                res = self._session.run(
    #                    self.loss_pp,
    #                    feed_dict={self._real_points_ph: self._data.data[idx * batch_size:(idx + 1) * batch_size],
    #                        self._noise_ph: batch_noise[idx * batch_size:(idx + 1) * batch_size],
    #                        self._is_training_ph: False})
    #        if len(res.shape) == 1:
                # convert (n,) vector to (n,1) array
    #            res = np.reshape(res, [-1, 1])
    #        result.append(res)
    #    result = np.vstack(result)
    #    return result 


class kImageVae(kVae):
    """A simple VAE implementation, suitable for pictures.

    """

    def __init__(self, opts, data, weights):

        # One more placeholder for batch norm
        self._is_training_ph = None
        self._use_second_ais = True

        kVae.__init__(self, opts, data, weights)

    def generator(self, opts, noise, is_training, reuse=False, return_logits=False):
        """Generator function, suitable for simple picture experiments.

        Args:
            noise: [num_points, dim] array, where dim is dimensionality of the
                latent noise space.
            is_training: bool, defines whether to use batch_norm in the train
                or test mode.
            return_logits: bool, if true returns the "logits" instead of being
                normalized (by tanh or sigmoid depending on "input_normalize_sym".
        Returns:
            [num_points, dim1, dim2, dim3] array, where the first coordinate
            indexes the points, which all are of the shape (dim1, dim2, dim3).
        """

        output_shape = self._data.data_shape # (dim1, dim2, dim3)
        # Computing the number of noise vectors on-the-go
        dim1 = tf.shape(noise)[0]
        num_filters = opts['g_num_filters']
        num_layers = opts['g_num_layers']
        #keep_prob = opts['dropout_keep_prob']
        with tf.variable_scope("GENERATOR", reuse=reuse):

            height = output_shape[0] /  2**(num_layers - 1)
            width = output_shape[1] / 2**(num_layers - 1)
            h0 = ops.linear(opts, noise, num_filters * height * width,
                            scope='h0_lin')
            h0 = tf.reshape(h0, [-1, height, width, num_filters])
            h0 = tf.nn.relu(h0)

            layer_x = h0
            for i in xrange(num_layers-1):
                scale = 2**(i+1)
                _out_shape = [dim1, height * scale, width * scale, num_filters / scale]
                layer_x = ops.deconv2d(opts, layer_x, _out_shape, scope='h%d_deconv' % i)
                if opts['batch_norm']:
                    layer_x = ops.batch_norm(opts, layer_x, is_training, reuse, scope='bn%d' % i)
                layer_x = tf.nn.relu(layer_x)
                if opts['dropout']:
                    _keep_prob = tf.minimum(
                        1., 0.9 - (0.9 - keep_prob) * float(i + 1) / (num_layers - 1))
                    layer_x = tf.nn.dropout(layer_x, _keep_prob)

            # # h0 = ops.lrelu(h0)
            # _out_shape = [dim1, height * 2, width * 2, num_filters / 2]
            # # for 28 x 28 does 7 x 7 --> 14 x 14
            # h1 = ops.deconv2d(opts, h0, _out_shape, scope='h1_deconv')
            # h1 = ops.batch_norm(opts, h1, is_training, reuse, scope='bn_layer2')
            # h1 = tf.nn.relu(h1)
            # # h1 = ops.lrelu(h1)
            # _out_shape = [dim1, height * 4, width * 4, num_filters / 4]
            # # for 28 x 28 does 14 x 14 --> 28 x 28
            # h2 = ops.deconv2d(opts, h1, _out_shape, scope='h2_deconv')
            # h2 = ops.batch_norm(opts, h2, is_training, reuse, scope='bn_layer3')
            # h2 = tf.nn.relu(h2)
            # # h2 = ops.lrelu(h2)

            _out_shape = [dim1] + list(output_shape)
            # data_shape[0] x data_shape[1] x ? -> data_shape
            h3 = ops.deconv2d(opts, layer_x, _out_shape,
                              d_h=1, d_w=1, scope='hlast_deconv')
            # h3 = ops.batch_norm(opts, h3, is_training, reuse, scope='bn_layer4')

        if return_logits:
            return h3
        if opts['input_normalize_sym']:
            return tf.nn.tanh(h3)
        else:
            return tf.nn.sigmoid(h3)

    def discriminator(self, opts, input_, is_training,
                      prefix='DISCRIMINATOR', reuse=False):
        """Encoder function, suitable for simple toy experiments.

        """
        num_filters = opts['d_num_filters']

        with tf.variable_scope(prefix, reuse=reuse):
            h0 = ops.conv2d(opts, input_, num_filters / 8, scope='h0_conv')
            h0 = ops.batch_norm(opts, h0, is_training, reuse, scope='bn_layer1')
            h0 = tf.nn.relu(h0)
            h1 = ops.conv2d(opts, h0, num_filters / 4, scope='h1_conv')
            h1 = ops.batch_norm(opts, h1, is_training, reuse, scope='bn_layer2')
            h1 = tf.nn.relu(h1)
            h2 = ops.conv2d(opts, h1, num_filters / 2, scope='h2_conv')
            h2 = ops.batch_norm(opts, h2, is_training, reuse, scope='bn_layer3')
            h2 = tf.nn.relu(h2)
            h3 = ops.conv2d(opts, h2, num_filters, scope='h3_conv')
            h3 = ops.batch_norm(opts, h3, is_training, reuse, scope='bn_layer4')
            h3 = tf.nn.relu(h3)
            # Already has NaNs!!
            latent_mean = ops.linear(opts, h3, opts['latent_space_dim'], scope='h3_lin')
            log_latent_sigmas = ops.linear(opts, h3, opts['latent_space_dim'], scope='h3_lin_sigma')

        return latent_mean, log_latent_sigmas

    def _build_model_internal(self, opts):
        """Build the Graph corresponding to VAE implementation.

        """
        data_shape = self._data.data_shape
        real_points_phs = []
        noise_phs = []
        is_training_phs = []
        lr_decay_phs = []
        vae_class = VAE.ImageVae
        classifier = CLASSIFIER.ImageClassifier
        kVAEs = [] 
        kCLASSs = []
        optims = []
        class_real_pts_phs = []
        class_fake_pts_phs = []
        class_optims = []
        for k in range(opts['number_of_kGANs']): 
            device = k%opts['number_of_gpus']
            with tf.device('/device:GPU:%d' %device):
                new_vae = vae_class(opts, self._data, model_idx = k, sess = self._session) 
                new_class = classifier(opts, self._data, model_idx = k, sess = self._session )
                kVAEs.append(new_vae)
                kCLASSs.append(new_class)
                real_points_phs.append(new_vae._real_points_ph)
                noise_phs.append(new_vae._noise_ph)
                optims.append(new_vae._optim)
                is_training_phs.append(new_vae._is_training_ph)
                lr_decay_phs.append(new_vae._lr_decay_ph)
                class_real_pts_phs.append(new_class._real_points_ph)
                class_fake_pts_phs.append(new_class._fake_points_ph)
                class_optims.append(new_class._c_optim)


        self._real_points_phs = real_points_phs
        self._noise_phs = noise_phs
        self._is_training_phs = is_training_phs
        self._optims = optims
        #self._loss = loss
        #self._loss_reconstruct = loss_reconstruct
        self._lr_decay_phs = lr_decay_phs
        self._class_real_pts_phs = class_real_pts_phs
        self._class_fake_pts_phs = class_fake_pts_phs
        self._class_optims = class_optims
        self.kVAEs = kVAEs
        self.kCLASS = kCLASSs

        logging.error("Building Graph Done.")


    def _train_internal_vaes(self, opts):
        """Train a VAE model.

        """

        batches_num = self._data.num_points / opts['batch_size']
        #if opts['one_batch'] == True:
        #    batches_num = 1
        train_size = self._data.num_points
        num_plot = 320
        sample_prev = np.zeros([num_plot] + list(self._data.data_shape))
        l2s = []

        decay = 1.
        logging.error('Training VAE')
        for _epoch in xrange(opts["gan_epoch_num"]):

            if opts['decay_schedule'] == "manual":
                if _epoch == 30:
                    decay = decay / 2.
                if _epoch == 50:
                    decay = decay / 5.
                if _epoch == 100:
                    decay = decay / 10.

            if _epoch > 0 and _epoch % opts['save_every_epoch'] == 0:
                os.path.join(opts['work_dir'], opts['ckpt_dir'])
                #self._saver.save(self._session,
                #                 os.path.join(opts['work_dir'],
                #                              opts['ckpt_dir'],
                #                              'trained-pot'),
                #                 global_step=counter)

            for _idx in xrange(batches_num):
                # logging.error('Step %d of %d' % (_idx, batches_num ) )
                dict_ph = {}
                optimizers = []
                for k in range(opts['number_of_kGANs']):
                    if self._hard_assigned[k] != 0: 
                        data_ids = np.random.choice(train_size, opts['batch_size'],
                                replace=False, p=self._data_weights[:,k])
                        batch_images = self._data.data[data_ids].astype(np.float)
                        batch_noise = utils.generate_noise(opts, opts['batch_size'])
                        dict_ph[self._real_points_phs[k]] = batch_images
                        dict_ph[self._noise_phs[k]] = batch_noise
                        dict_ph[self._lr_decay_phs[k]] = decay
                        dict_ph[self._is_training_phs[k]] = True
                        optimizers.append(self._optims[k])
                #_, loss, loss_kl, loss_reconstruct = self._session.run(
                #    [self._optim, self._loss, self._loss_kl,
                #     self._loss_reconstruct],
                #    feed_dict={self._real_points_ph: batch_images,
                #               self._noise_ph: batch_noise,
                #               self._lr_decay_ph: decay,
                #               self._is_training_ph: True})
                self._session.run(
                    optimizers,
                    feed_dict=dict_ph)
                
                
                
    def _train_internal_classifiers(self, opts, fake_images):
        """Train a classifier separating true data from points in fake_images.
        
        """
        batches_num = self._data.num_points / opts['batch_size_classifier']
        if opts['one_batch'] == True:
            batches_num = 1
        logging.debug('Training a mixture discriminator')
        for epoch in xrange(opts["mixture_c_epoch_num"]):
            for idx in xrange(batches_num):
                dict_ph = {}
                for k in range(0,opts['number_of_kGANs']): 
                    ids = np.random.choice(len(fake_images[0]), opts['batch_size_classifier'],replace=False)
                    batch_fake_images = fake_images[k,ids]
                    dict_ph[self._class_fake_pts_phs[k]] = batch_fake_images

                    ids = np.random.choice(self._data.num_points, opts['batch_size_classifier'],replace=False)
                    batch_real_images = self._data.data[ids]
                    dict_ph[self._class_real_pts_phs[k]] = batch_real_images
                    dict_ph[self.kCLASS[k]._is_training_ph] = True
                _ = self._session.run(
                    self._class_optims,
                    feed_dict = dict_ph)
        res = np.zeros((len(self._data.data),opts['number_of_kGANs']))
        for k in range(0,opts['number_of_kGANs']):
            res[:,k] = self._run_batch(
                opts, self.kCLASS[k]._c_training,
                self.kCLASS[k]._real_points_ph, self._data.data,self.kCLASS[k]._is_training_ph, False).flatten()

        return res
