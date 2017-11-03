# Copyright 2017 Max Planck Society - ETH Zurich
# Distributed under the BSD-3 Software license,
# (See accompanying file ./LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)
"""Training kGAN on mnist.
"""

import os
import argparse
import logging
import tensorflow as tf
import numpy as np
from datahandler import DataHandler
from kGAN import KGANS
from metrics import Metrics
import utils

flags = tf.app.flags
flags.DEFINE_float("g_learning_rate", 0.005,
                   "Learning rate for Generator optimizers [16e-4]")
flags.DEFINE_float("d_learning_rate", 0.0001,
                   "Learning rate for Discriminator optimizers [4e-4]")
flags.DEFINE_float("learning_rate", 0.003,
                   "Learning rate for other optimizers [8e-4]")
flags.DEFINE_float("adam_beta1", 0.5, "Beta1 parameter for Adam optimizer [0.5]")
flags.DEFINE_integer("zdim", 8, "Dimensionality of the latent space [100]")
flags.DEFINE_float("init_std", 0.01, "Initial variance for weights [0.02]")
flags.DEFINE_string("assignment", 'soft', "Type of update for the weights")
flags.DEFINE_string("workdir", 'results_mnist_soft_1_epoch_per_gan', "Working directory ['results']")
flags.DEFINE_bool("unrolled", False, "Use unrolled GAN training [True]")
flags.DEFINE_bool("vae", False, "Use VAE instead of GAN")
flags.DEFINE_bool("pot", False, "Use VAE instead of GAN")
flags.DEFINE_bool("is_bagging", False, "Do we want to use bagging instead of adagan? [False]")
FLAGS = flags.FLAGS

def main():
    opts = {}
    opts['random_seed'] = 66
    opts['dataset'] = 'mnist' # gmm, circle_gmm,  mnist, mnist3 ...
    opts['conditional'] = False
    opts['unrolled'] = FLAGS.unrolled # Use Unrolled GAN? (only for images)
    opts['unrolling_steps'] = 5 # Used only if unrolled = True
    opts['data_dir'] = 'mnist'
    opts['trained_model_path'] = 'models'
    opts['mnist_trained_model_file'] = 'mnist_trainSteps_19999_yhat' # 'mnist_trainSteps_20000'
    opts['gmm_max_val'] = 15.
    opts['toy_dataset_size'] = 10000
    opts['toy_dataset_dim'] = 2
    opts['mnist3_dataset_size'] = 2 * 64 # 64 * 2500
    opts['mnist3_to_channels'] = False # Hide 3 digits of MNIST to channels
    opts['input_normalize_sym'] = False # Normalize data to [-1, 1]
    opts['adagan_steps_total'] = 1
    opts['samples_per_component'] = 5000
    opts['work_dir'] = FLAGS.workdir
    opts['ckpt_dir'] = 'checkpoints'
    opts['is_bagging'] = FLAGS.is_bagging
    opts['beta_heur'] = 'uniform' # uniform, constant
    opts['weights_heur'] = 'theory_star' # theory_star, theory_dagger, topk
    opts['beta_constant'] = 0.5
    opts['topk_constant'] = 0.5
    opts["init_std"] = FLAGS.init_std
    opts["init_bias"] = 0.0
    opts['latent_space_distr'] = 'normal' # uniform, normal
    opts['optimizer'] = 'adam' # sgd, adam
    opts["batch_size"] = 64
    opts["d_steps"] = 1
    opts["g_steps"] = 2
    opts["verbose"] = True
    opts['tf_run_batch_size'] = 128

    opts['gmm_modes_num'] = 5
    opts['latent_space_dim'] = FLAGS.zdim
    opts["gan_epoch_num"] = 1
    opts["mixture_c_epoch_num"] = 5
    opts['opt_learning_rate'] = FLAGS.learning_rate
    opts['opt_d_learning_rate'] = FLAGS.d_learning_rate
    opts['opt_g_learning_rate'] = FLAGS.g_learning_rate
    opts["opt_beta1"] = FLAGS.adam_beta1
    opts['batch_norm_eps'] = 1e-05
    opts['batch_norm_decay'] = 0.9
    opts['d_num_filters'] = 32#512
    opts['g_num_filters'] = 64#1024
    opts['conv_filters_dim'] = 5
    opts["early_stop"] = -1 # set -1 to run normally
    opts["plot_every"] = 100
    opts["save_every_epoch"] = 10
    opts["eval_points_num"] = 25600
    opts['digit_classification_threshold'] = 0.999
    opts['inverse_metric'] = False # Use metric from the Unrolled GAN paper?
    opts['inverse_num'] = 100 # Number of real points to inverse.
    opts['objective'] = None
    opts['vae'] = FLAGS.vae
    opts['pot'] = FLAGS.pot
    opts['pot_pz_std'] = 2
    opts['vae_sigma'] = 0.01
    opts['pot_lambda'] = 10
    opts['convolutions'] = False

    opts['plot_kGANs'] = False
    opts['assignment'] = FLAGS.assignment
    opts['number_of_steps_made'] = 0
    opts['number_of_kGANs'] = 10
    opts['kGANs_number_rounds'] = 100
    opts['kill_threshold'] = 0.01 
    opts['annealed'] = True 
    if opts['verbose']:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    utils.create_dir(opts['work_dir'])
    utils.create_dir(os.path.join(opts['work_dir'], opts['ckpt_dir']))

    with utils.o_gfile((opts['work_dir'], 'params.txt'), 'w') as text:
        text.write('Parameters:\n')
        for key in opts:
            text.write('%s : %s\n' % (key, opts[key]))

    data = DataHandler(opts)
    assert data.num_points >= opts['batch_size'], 'Training set too small'
    
    
    kG = KGANS(opts, data)
    metrics = Metrics()

    train_size = data.num_points
    random_idx = np.random.choice(train_size, 4*320, replace=False)
    metrics.make_plots(opts, 0, data.data,
            data.data[random_idx], kG._data_weights, prefix='dataset_')
    
    
    
    for step in range(opts["kGANs_number_rounds"]):
        opts['number_of_steps_made'] = step
        logging.info('Running step {} of kGAN'.format(step + 1))
        kG.make_step(opts, data)
        num_fake = opts['eval_points_num']
        logging.debug('Sampling fake points')
        fake_points = kG.sample_mixture(opts, num_fake)
        logging.debug('Sampling more fake points')
        more_fake_points = kG.sample_mixture(opts, 500)
        num_samples_per_gan = 50
        fake_points_plot = kG.sample_mixture_separate_uniform(opts, num_samples_per_gan)
        logging.debug('Plotting results')
        metrics.make_plots(opts, step, data.data,more_fake_points, kG._data_weights, prefix = "")
        metrics._return_plots_pics(opts, step, data.data,fake_points_plot, 50,  kG._data_weights, prefix = "")
        #idx_plot_colors_end = 0
        #idx_plot_colors_start = 0
        
        #for k in range(opts['number_of_kGANs']):
        #    idx_plot_colors_start = idx_plot_colors_end
        #    idx_plot_colors_end += num_samples_per_gan
        #    metrics.make_plots(opts, step, data.data,fake_points_plot[idx_plot_colors_start:idx_plot_colors_end], kG._data_weights, prefix = str(k))

        res = metrics.evaluate(opts, step, data.data[:500],fake_points, more_fake_points, prefix='')
    logging.debug("kGANs finished working!")

if __name__ == '__main__':
    main()
