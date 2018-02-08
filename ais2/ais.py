import tensorflow as tf
import numpy as np
import scipy as sp
import time
from tqdm import tqdm


def get_pdf_gauss(loc, log_scale, sample):
  scale = tf.exp(log_scale)
  pdf = -tf.reduce_sum(0.5 * tf.square((sample - loc) / scale) + log_scale + 0.5*np.log(2*np.pi), [1])
  return pdf


def aaa_energy0(z, theta):
  z_mean = theta[0]
  log_z_std = theta[1]
  return -get_pdf_gauss(z_mean, log_z_std, z)


def aaa_get_z0(theta, batch_size=64, z_dim=10):
  z_mean = theta[0]
  z_std = tf.exp(theta[1])
  return z_mean + z_std * tf.random_normal([batch_size, z_dim])


def get_reconstr_err(decoder_out, x, config):
    cond_dist = config['cond_dist']
    if cond_dist == 'gauss':
        loc = tf.sigmoid(decoder_out[0])
        logscale = decoder_out[1]
        err = (x - loc)*tf.exp(-logscale)
        reconst_err = tf.reduce_sum(
            0.5 * err * err + logscale + 0.5 * np.log(2*np.pi),
            [1, 2, 3]
        )
    elif cond_dist == 'bernouille':
        p_logits = decoder_out[0]
        reconst_err = tf.reduce_sum(
          tf.nn.sigmoid_cross_entropy_with_logits(logits=p_logits, labels=x),
          [1, 2, 3]
        )

    return reconst_err


class AIS(object):

    def __init__(self, x_test, params_posterior, decoder, energy0, get_z0, config, eps_scale=None):
        """AIS initialization.

        Args:
          x_test: The input batch. Assumed to be TF tensor.
          params_posterior: A tuple holding [mean, log_sigma] of the posterior
            (approximated by an encoder).
          decoder: The decoder.
          energy0: aaa_energy0 as an example.
          get_z0: aaa_get_z0 as an example.
          config: dictionary holding
            "batch_size": The batch size.
            "z_dim": The latent space dimension.
            "output_size": height/width of the image.
            "c_dim": Number of color channels.
            "test_is_adaptive_eps": Typically False for VAE.
            "test_ais_nsteps": Number of steps (e.g. 100)
            "test_ais_eps": Step size for AIS (e.g. 1e-2)
        """
        self.x_in = x_test
        self.params_posterior_in = params_posterior
        self.decoder = decoder
        self.energy0 = energy0
        self.get_z0 = get_z0
        self.config = config

        if eps_scale is None:
            self.eps_scale_in = tf.ones([config['batch_size'], config['z_dim']])
        else:
            self.eps_scale_in = eps_scale

        self.build_model()

    def build_model(self):
        batch_size = self.config['batch_size']
        output_size = self.config['output_size']
        c_dim = self.config['c_dim']
        z_dim = self.config['z_dim']

        # Persist on gpu for efficiency
        self.x = tf.Variable(np.zeros([batch_size, output_size, output_size, c_dim], dtype=np.float32), trainable=False)
        self.params_posterior = [
            tf.Variable(tf.zeros(p0.get_shape()), trainable=False)
            for p0 in self.params_posterior_in
        ]
        self.eps_scale = tf.Variable(tf.zeros([batch_size, z_dim]), trainable=False)

        # Position and momentum variables
        mass = 1.#/self.var0
        mass_sqrt = 1.#/self.std0
        self.z = tf.Variable(np.zeros([batch_size, z_dim], dtype=np.float32), trainable=False)
        self.p = tf.Variable(np.zeros([batch_size, z_dim], dtype=np.float32), trainable=False)

        self.z_current = tf.Variable(np.zeros([batch_size, z_dim], dtype=np.float32), trainable=False)
        self.p_current = tf.Variable(np.zeros([batch_size, z_dim], dtype=np.float32), trainable=False)

        self.p_rnd = tf.random_normal([batch_size, z_dim]) * mass_sqrt

        self.eps = tf.placeholder(tf.float32, shape=[])
        self.beta = tf.placeholder(tf.float32, shape=[])

        # Hamiltoninan
        self.U = self.get_energy(self.z)
        self.V = 0.5 * tf.reduce_sum(tf.square(self.p)/mass, [1])
        self.H = self.U + self.V
        self.U_current = self.get_energy(self.z_current)
        self.V_current = 0.5 * tf.reduce_sum(tf.square(self.p_current)/mass, [1])
        self.H_current = self.U_current + self.V_current

        # Intialize
        self.init_batch = [
            self.x.assign(self.x_in),
            self.eps_scale.assign(self.eps_scale_in),
        ]
        self.init_batch += [
            p.assign(p_in) for (p, p_in) in zip(self.params_posterior, self.params_posterior_in)
        ]

        self.init_hmc =  self.z_current.assign(self.get_z0(self.params_posterior))

        self.init_hmc_step = [
            self.p_current.assign(self.p_rnd)
        ]

        self.init_hmc_step2 = [
            self.z.assign(self.z_current),
            self.p.assign(self.p_current),
        ]
        # Euler steps
        eps_scaled = self.eps_scale * self.eps


        self.euler_z = self.z.assign_add(eps_scaled * self.p/mass)
        gradU = tf.reshape(tf.gradients(self.U, self.z), [batch_size, z_dim])
        self.euler_p = self.p.assign_sub(eps_scaled * gradU)

        # Accept
        self.is_accept = tf.cast(tf.random_uniform([batch_size]) < tf.exp(self.H_current - self.H), tf.float32)
        self.accept_rate = tf.reduce_mean(self.is_accept)

        is_accept_rs = tf.reshape(self.is_accept, [batch_size, 1])
        self.update_z = self.z_current.assign(
            is_accept_rs * self.z + (1. - is_accept_rs) * self.z_current
        )

    def get_energy(self, z):
        E = self.beta*self.get_energy1(z) + (1 - self.beta) * self.get_energy0(z)
        return E

    def get_energy1(self, z):
        decoder_out = self.decoder(z)
        E = get_reconstr_err(decoder_out, self.x, self.config)
        # Prior
        E += tf.reduce_sum(
            0.5 * tf.square(z) + 0.5 * np.log(2*np.pi), [1]
        )

        return E

    def get_energy0(self, z):
        E = self.energy0(z, self.params_posterior)
        return E

    def read_batch(self, sess, feed_dict):
        sess.run(self.init_batch, feed_dict=feed_dict)

    def evaluate(self, sess):
        is_adaptive_eps = self.config['test_is_adaptive_eps']
        nsteps = self.config['test_ais_nsteps']
        batch_size = self.config['batch_size']
        eps = self.config['test_ais_eps']

        # logZ = sess.run(tf.reduce_sum(tf.log(self.std0), [1]))
        # logZ = self.z_dim * 0.5 * np.log(2*np.pi)
        logpx = 0.
        weights = np.zeros([100, batch_size])

        betas = np.linspace(0, 1, nsteps+1)
        accept_rate = 1.

        t = time.time()

        sess.run(self.init_hmc)

        progress = tqdm(range(nsteps), desc="HMC")
        for i in progress:
            f0 = -sess.run(self.U_current, feed_dict={self.beta: betas[i]})
            f1 = -sess.run(self.U_current, feed_dict={self.beta: betas[i+1]})
            logpx += f1 - f0

            if i < nsteps-1:
                accept_rate = self.run_hmc_step(sess, betas[i+1], eps)
                if is_adaptive_eps and accept_rate < 0.6:
                    eps = eps / 1.1
                elif is_adaptive_eps and accept_rate > 0.7:
                    eps = eps * 1.1
                progress.set_postfix(
                    accept_rate="%.2f" % accept_rate,
                    logw="%.2f+-%.2f" % (logpx.mean(), logpx.std()),
                    eps="%.2e" % eps,
                )
        samples = sess.run(self.z_current)

        return logpx, samples

    def evaluate_nchains(self, sess, ais_nchains):
      """

      Must call read_batch before !!!

      """
      batch_size = self.config["batch_size"]
      z_dim = self.config['z_dim']
      ais_samples = np.zeros([ais_nchains, batch_size, z_dim])

      ais_res = np.zeros([ais_nchains, batch_size])

      progress_ais = tqdm(range(ais_nchains), desc="AIS")
      for j in progress_ais:
        ais_res[j], ais_samples[j] = self.evaluate(sess)
        ais_lprob, ais_ess = self.average_weights(ais_res[:j+1], axis=0)
        progress_ais.set_postfix(
            lprob="%.2f+-%.2f" % (ais_lprob.mean(), ais_lprob.std()),
            ess="%.2f/%d" % (ais_ess.mean(), j+1)
        )
      return (np.mean(ais_lprob), np.std(ais_lprob),
              np.mean(ais_ess), np.std(ais_ess))

    def run_hmc_step(self, sess, beta, eps):
        L = 10 # TODO: make configuratble
        # Initialize
        sess.run(self.init_hmc_step)
        sess.run(self.init_hmc_step2)

        # Leapfrog steps
        sess.run(self.euler_p, feed_dict={self.eps: eps/2, self.beta: beta})
        for i in range(L+1):
            sess.run(self.euler_z, feed_dict={self.eps: eps, self.beta: beta})
            if i < L:
                sess.run(self.euler_p, feed_dict={self.eps: eps, self.beta: beta})
        sess.run(self.euler_p, feed_dict={self.eps: eps/2, self.beta: beta})

        # Update z
        _, accept_rate = sess.run([self.update_z, self.accept_rate], feed_dict={self.beta: beta})
        return accept_rate


    def average_weights(self, weights, axis=0):
        nchains = weights.shape[axis]
        logsumw = sp.misc.logsumexp(weights, axis=axis)
        lprob = logsumw - np.log(nchains)
        ess = np.exp(-sp.misc.logsumexp(2*(weights - logsumw), axis=axis))
        return lprob, ess
