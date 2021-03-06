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
import utils
from kVAEs import KGANS
import pdb
import sys

flags = tf.app.flags
#flags.DEFINE_float("g_learning_rate", 0.005,
#                   "Learning rate for Generator optimizers [16e-4]")
#flags.DEFINE_float("d_learning_rate", 0.0001,
#                   "Learning rate for Discriminator optimizers [4e-4]")
flags.DEFINE_float("learning_rate", 0.0008,
                   "Learning rate for other optimizers [8e-4]")
flags.DEFINE_float("adam_beta1", 0.5, "Beta1 parameter for Adam optimizer [0.5]")
flags.DEFINE_integer("zdim", 5, "Dimensionality of the latent space [100]")
flags.DEFINE_float("init_std", 0.8, "Initial variance for weights [0.02]")
flags.DEFINE_string("assignment", 'soft', "Type of update for the weights") #'soft', 'hard'
flags.DEFINE_string("workdir", 'results_gmm_cleanup', "Working directory ['results']")
flags.DEFINE_bool("vae", True, "use VAEs instead of GANs")
flags.DEFINE_integer("gmm_modes_num", 3, "Number of modes")
flags.DEFINE_integer("gan_epoch_num_first_iteration", 10, "epoch number to pretrain models")
flags.DEFINE_integer("gan_epoch_num_except_first", 10, "epoch number per iteration")
flags.DEFINE_integer("mixture_c_epoch_num", 1, "epoch number for classifier")
flags.DEFINE_integer("number_of_kGANs", 3, "Number of generative models")
flags.DEFINE_integer("kGANs_number_rounds", 3, "Number of iterations")
flags.DEFINE_bool("one_batch_class", False, "train classifier for a single batch")
flags.DEFINE_bool("one_batch", False, "train for a single batch")
flags.DEFINE_bool("reinit_class", True, "train classifiers from scratch")
flags.DEFINE_integer("AIS_every_it", 5, "run ais every x iterations")
flags.DEFINE_bool("bagging", False, "bagging instead of kVAEs")
FLAGS = flags.FLAGS

from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']



def main():
    opts = {}
    
    # dataset opts 
    opts['random_seed'] = 821
    opts['dataset'] = 'gmm' # gmm, circle_gmm,  mnist, mnist3 ...
    opts['gmm_modes_num'] = FLAGS.gmm_modes_num
    opts['gmm_max_val'] = 15.
    opts['toy_dataset_size'] = 64 * 1000 
    opts['toy_dataset_dim'] = 2
    opts['input_normalize_sym'] = False # Normalize data to [-1, 1]
    opts['samples_per_component'] = 5000 # 50000
    opts['work_dir'] = FLAGS.workdir
    opts['banana'] = True #skew the gaussians a bit
    
    # kGANs opts
    opts["gan_epoch_num_first_iteration"] = FLAGS.gan_epoch_num_first_iteration
    opts["gan_epoch_num"] = opts["gan_epoch_num_first_iteration"]
    opts["gan_epoch_num_except_first"] = FLAGS.gan_epoch_num_except_first
    opts["mixture_c_epoch_num"] = FLAGS.mixture_c_epoch_num
    opts['plot_kGANs'] = False #do not set to True
    opts['assignment'] = FLAGS.assignment
    opts['number_of_steps_made'] = 0
    opts['number_of_kGANs'] = FLAGS.number_of_kGANs
    opts['kGANs_number_rounds'] = FLAGS.kGANs_number_rounds
    opts['number_of_gpus'] = len(get_available_gpus()) # set to 1 if don't want parallel computation
    opts['one_batch'] = False# update weights every batch (True) or every epoch (False)
    opts['one_batch_class'] = FLAGS.one_batch_class# update weights every batch (True) or every epoch (False)
    opts['test'] = False #hack, don't set to true
    opts['reinit_class'] = FLAGS.reinit_class
    opts['AIS_every_it'] = FLAGS.AIS_every_it
    opts['bagging'] = FLAGS.bagging
    #VAE opts 
    opts['vae_sigma'] = 0.01
    opts['vae'] = FLAGS.vae
    opts['recon_loss'] = 'l2sq'
    opts['decay_schedule'] = 'manual'
    opts['number_units'] = 50
    # GAN opts
    opts["init_std"] = FLAGS.init_std
    opts["init_bias"] = 0.0
    opts['latent_space_distr'] = 'normal' # uniform, normal
    #opts["d_steps"] = 1
    #opts["g_steps"] = 2
    opts["verbose"] = True
    opts['tf_run_batch_size'] = 100
    #opts['objective'] = 'JS'
    opts['latent_space_dim'] = FLAGS.zdim
    opts["early_stop"] = -1 # set -1 to run normally
    opts["plot_every"] = -1 # set -1 to run normally
    opts["eval_points_num"] = 1000 # 25600
    opts['digit_classification_threshold'] = 0.999
    #opts['inverse_metric'] = False # Use metric from the Unrolled GAN paper?
    #opts['inverse_num'] = 1 # Number of real points to inverse.
    
    # Optimizer
    opts['optimizer'] = 'adam' # sgd, adam
    opts["batch_size"] = 32
    opts["batch_size_classifier"] = 32
    opts['opt_learning_rate'] = FLAGS.learning_rate
    #opts['opt_d_learning_rate'] = FLAGS.d_learning_rate
    #opts['opt_g_learning_rate'] = FLAGS.g_learning_rate
    opts["opt_beta1"] = FLAGS.adam_beta1
    opts['batch_norm_eps'] = 1e-05
    opts['batch_norm_decay'] = 0.9
    opts['one_batch'] = False
    opts['one_batch_condition'] = FLAGS.one_batch
    
    utils.create_dir(opts['work_dir'])
    #sys.stdout = sys.stderr = open(FLAGS.workdir+'/output', 'w')
    
    if opts['verbose']:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    with utils.o_gfile((opts['work_dir'], 'params.txt'), 'w') as text:
        text.write('Parameters:\n')
        for key in opts:
            text.write('%s : %s\n' % (key, opts[key]))
    
    print opts['work_dir']
    
    data = DataHandler(opts)
    data_test  = DataHandler(opts)
    data_test.data = data.data_test
    assert data.num_points >= opts['batch_size'], 'Training set too small'
    num = data.num_points
    kG = KGANS(opts, data, data_test)
    metrics = Metrics()
    logl = np.zeros((2,opts['kGANs_number_rounds']))
    #train loop
    for step in range(opts['kGANs_number_rounds']):
        opts['number_of_steps_made'] = step
        logging.debug('Running step %03d' %(step))
        
        #kG.KV.kVAEs[0].train(opts) 
        #import pdb
        #pdb.set_trace()
        kG.make_step(opts, data)
        opts["gan_epoch_num"] = opts["gan_epoch_num_except_first"]

        #opts['one_batch'] = True
        opts['one_batch'] = opts['one_batch_condition']
        num_fake = opts['eval_points_num']
        logging.debug('Sampling fake points')
        num_fake_points = 500*opts['number_of_kGANs']
        fake_points, num_samples_gans = kG.sample_mixture_separate_color(opts, num_fake_points)
        logging.debug('Sampling more fake points')
            
        more_fake_points = kG.sample_mixture(opts, 500)
        fake_pts_ll = kG.sample_mixture(opts, num_fake_points)
        logging.debug('Plotting results')
        opts['plot_kGANs'] = True #hack to print
        metrics.make_plots(opts, step, data.data[:10000],
            fake_points, weights = num_samples_gans)
        #np.savetxt(opts['work_dir']+"/ais.csv", kG.ais, delimiter=",")
        #np.savetxt(opts['work_dir']+"/ais_test.csv", kG.ais_test, delimiter=",")
        (likelihood, C) = metrics.evaluate(
                opts, step, data_test.data[:500],
                fake_pts_ll, more_fake_points, prefix='')
        print likelihood
        logl[0,step] = likelihood
        (likelihood, C) = metrics.evaluate(
                opts, step, data.data[:500],
                fake_pts_ll, more_fake_points, prefix='')
        print likelihood
        logl[1,step] = likelihood
        np.savetxt(opts['work_dir']+"/logl.csv", logl, delimiter=",")
    logging.debug("kGANs finished working!")
    

if __name__ == '__main__':
    main()
