# Copyright 2017 Max Planck Society - ETH Zurich
# Distributed under the BSD-3 Software license,
# (See accompanying file ./LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)
"""Training kGAN on gmm datasets.

"""

import os
import argparse
import logging
import tensorflow as tf
import numpy as np
from datahandler import DataHandler
from metrics import Metrics
import gan as GAN
import utils
import pdb

flags = tf.app.flags
flags.DEFINE_float("g_learning_rate", 0.001,
                   "Learning rate for Generator optimizers [16e-4]")
flags.DEFINE_float("d_learning_rate", 0.0001,
                   "Learning rate for Discriminator optimizers [4e-4]")
flags.DEFINE_float("learning_rate", 0.0008,
                   "Learning rate for other optimizers [8e-4]")
flags.DEFINE_float("adam_beta1", 0.5, "Beta1 parameter for Adam optimizer [0.5]")
flags.DEFINE_integer("zdim", 5, "Dimensionality of the latent space [100]")
flags.DEFINE_float("init_std", 0.8, "Initial variance for weights [0.02]")
flags.DEFINE_string("assignment", 'soft', "Type of update for the weights") #'soft', 'hard'
flags.DEFINE_string("workdir", 'results_gmm_batch', "Working directory ['results']")
#flags.DEFINE_bool("unrolled", False, "Use unrolled GAN training [True]")
flags.DEFINE_bool("is_bagging", False, "Do we want to use bagging instead of adagan? [False]")
FLAGS = flags.FLAGS

from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def main():
    opts = {}
    opts['random_seed'] = 821
    opts['dataset'] = 'gmm' # gmm, circle_gmm,  mnist, mnist3 ...
#    opts['unrolled'] = FLAGS.unrolled # Use Unrolled GAN? (only for images)
#    opts['unrolling_steps'] = 5 # Used only if unrolled = True
#    opts['data_dir'] = 'mnist'
#    opts['trained_model_path'] = 'models'
#    opts['mnist_trained_model_file'] = 'mnist_trainSteps_19999_yhat' # 'mnist_trainSteps_20000'
    opts['gmm_max_val'] = 15.
    opts['toy_dataset_size'] = 128 * 1000 *2
    opts['toy_dataset_dim'] = 2
#    opts['mnist3_dataset_size'] = 2 * 64 # 64 * 2500
#    opts['mnist3_to_channels'] = False # Hide 3 digits of MNIST to channels
    opts['input_normalize_sym'] = False # Normalize data to [-1, 1]
#    opts['adagan_steps_total'] = 1
    opts['samples_per_component'] = 5000 # 50000
    opts['work_dir'] = FLAGS.workdir
#    opts['is_bagging'] = FLAGS.is_bagging
#    opts['beta_heur'] = 'uniform' # uniform, constant
#    opts['weights_heur'] = 'theory_star' # theory_star, theory_dagger, topk
#    opts['beta_constant'] = 0.5
#    opts['topk_constant'] = 0.5
    opts["init_std"] = FLAGS.init_std
    opts["init_bias"] = 0.0
    opts['latent_space_distr'] = 'normal' # uniform, normal
    opts['optimizer'] = 'adam' # sgd, adam
    opts["batch_size"] = 32#256
    opts["batch_size_classifier"] = 128
    opts["d_steps"] = 1
    opts["g_steps"] = 2
    opts["verbose"] = True
    opts['tf_run_batch_size'] = 100
    opts['objective'] = 'JS'

    opts['gmm_modes_num'] = 16
    opts['latent_space_dim'] = FLAGS.zdim
    opts["gan_epoch_num"] = 1
#    opts["mixture_c_epoch_num"] = 2
    opts['opt_learning_rate'] = FLAGS.learning_rate
    opts['opt_d_learning_rate'] = FLAGS.d_learning_rate
    opts['opt_g_learning_rate'] = FLAGS.g_learning_rate
    opts["opt_beta1"] = FLAGS.adam_beta1
    opts['batch_norm_eps'] = 1e-05
    opts['batch_norm_decay'] = 0.9
#    opts['d_num_filters'] = 16
#    opts['g_num_filters'] = 16
#    opts['conv_filters_dim'] = 4
    opts["early_stop"] = -1 # set -1 to run normally
    opts["plot_every"] = -1 # set -1 to run normally
    opts["eval_points_num"] = 1000 # 25600
    opts['digit_classification_threshold'] = 0.999
    opts['inverse_metric'] = False # Use metric from the Unrolled GAN paper?
    opts['inverse_num'] = 1 # Number of real points to inverse.
    
    opts['plot_kGANs'] = False #do not set to True
    opts['assignment'] = FLAGS.assignment
    opts['number_of_steps_made'] = 0
    opts['number_of_kGANs'] = 16
    opts['kGANs_number_rounds'] = 1000000
    opts['kill_threshold'] = 0.015
    opts['annealed'] = True
    opts['number_of_gpus'] = len(get_available_gpus())
    opts['reinitialize'] = True #when a gan die want to delete it (False) or re-initialize it (True)
    opts['one_batch'] = True # update weights every batch (True) or every epoch (False)
    if opts['verbose']:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    utils.create_dir(opts['work_dir'])
    with utils.o_gfile((opts['work_dir'], 'params.txt'), 'w') as text:
        text.write('Parameters:\n')
        for key in opts:
            text.write('%s : %s\n' % (key, opts[key]))
    
    print opts['work_dir']
    
    data = DataHandler(opts)
    assert data.num_points >= opts['batch_size'], 'Training set too small'
    num = data.num_points
    from kGAN import KGANS
    kG = KGANS(opts, data)
    
    metrics = Metrics()

    #train loop
    for step in range(opts['kGANs_number_rounds']):
        opts['number_of_steps_made'] = step
        logging.debug('Running step %03d' %(step))
        kG.make_step(opts, data)
        num_fake = opts['eval_points_num']
        logging.debug('Sampling fake points')
        num_fake_points = 500
        fake_points, num_samples_gans = kG.sample_mixture_separate_color(opts, num_fake_points)
        logging.debug('Sampling more fake points')
        
        opts["gan_epoch_num"] = 1
        more_fake_points = kG.sample_mixture(opts, 500)

        logging.debug('Plotting results')
        opts['plot_kGANs'] = True
        
        metrics.make_plots(opts, step, data.data[:500],
                fake_points, weights = num_samples_gans)
        
        (likelihood, C) = metrics.evaluate(
            opts, step, data.data[:500],
            fake_points, more_fake_points, prefix='')
    logging.debug("kGANs finished working!")
    

if __name__ == '__main__':
    main()
