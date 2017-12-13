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
from inception_score import get_inception_score

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
flags.DEFINE_string("workdir", 'results_mnist_pot', "Working directory ['results']")
flags.DEFINE_bool("pot", True, "Use POT instead of GAN")
flags.DEFINE_float("pot_lambda", 10., "POT regularization")
flags.DEFINE_bool("unrolled", False, "Use unrolled GAN training [True]")
flags.DEFINE_bool("vae", False, "Use VAE instead of GAN")
#flags.DEFINE_bool("pot", False, "Use VAE instead of GAN")
#flags.DEFINE_bool("is_bagging", False, "Do we want to use bagging instead of adagan? [False]")
FLAGS = flags.FLAGS


from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def main():
    opts = {}
    # Utility
    opts['random_seed'] = 66
    opts['dataset'] = 'mnist' # gmm, circle_gmm,  mnist, mnist3 ...
    opts['data_dir'] = 'mnist'
    opts['trained_model_path'] = None #'models'
    opts['mnist_trained_model_file'] = None #'mnist_trainSteps_19999_yhat' # 'mnist_trainSteps_20000'
    opts['work_dir'] = FLAGS.workdir
    opts['ckpt_dir'] = 'checkpoints'
    opts["verbose"] = 1
    opts['tf_run_batch_size'] = 128
    opts["early_stop"] = -1 # set -1 to run normally
    opts["plot_every"] = 200
    opts["save_every_epoch"] = 20
    opts['gmm_max_val'] = 15.

    # Datasets
    opts['toy_dataset_size'] = 10000
    opts['toy_dataset_dim'] = 2
    opts['mnist3_dataset_size'] = 2 * 64 # 64 * 2500
    opts['mnist3_to_channels'] = False # Hide 3 digits of MNIST to channels
    opts['input_normalize_sym'] = False # Normalize data to [-1, 1]
    opts['gmm_modes_num'] = 5
    # Generative model parameters
    opts["init_std"] = FLAGS.init_std
    opts["init_bias"] = 0.0
    opts['latent_space_distr'] = 'normal' # uniform, normal
    opts['latent_space_dim'] = FLAGS.zdim
    opts["gan_epoch_num"] = 100
    opts['convolutions'] = True # If False then encoder is MLP of 3 layers
    opts['d_num_filters'] = 1024
    opts['d_num_layers'] = 4
    opts['g_num_filters'] = 1024
    opts['g_num_layers'] = 3
    opts['e_is_random'] = False
    opts['e_pretrain'] = False
    opts['e_add_noise'] = False
    opts['e_pretrain_bsize'] = 1000
    opts['e_num_filters'] = 1024
    opts['e_num_layers'] = 4
    opts['g_arch'] = 'dcgan_mod'
    opts['g_stride1_deconv'] = False
    opts['g_3x3_conv'] = 0
    opts['e_arch'] = 'dcgan'
    opts['e_3x3_conv'] = 0
    opts['conv_filters_dim'] = 4
    # --GAN specific:
    opts['conditional'] = False
    opts['unrolled'] = FLAGS.unrolled # Use Unrolled GAN? (only for images)
    opts['unrolling_steps'] = 5 # Used only if unrolled = True
    # --VAE specific
    opts['vae'] = FLAGS.vae
    opts['vae_sigma'] = 0.01
    # --POT specific
    opts['pot'] = FLAGS.pot
    opts['pot_pz_std'] = 2.
    opts['pot_lambda'] = FLAGS.pot_lambda
    opts['adv_c_loss'] = 'none'
    opts['vgg_layer'] = 'pool2'
    opts['adv_c_patches_size'] = 5
    opts['adv_c_num_units'] = 32
    opts['adv_c_loss_w'] = 1.0
    opts['cross_p_w'] = 0.0
    opts['diag_p_w'] = 0.0
    opts['emb_c_loss_w'] = 1.0
    opts['reconstr_w'] = 1.0
    opts['z_test'] = 'gan'
    opts['gan_p_trick'] = False
    opts['pz_transform'] = False
    opts['z_test_corr_w'] = 1.0
    opts['z_test_proj_dim'] = 10

    # Optimizer parameters
    opts['optimizer'] = 'adam' # sgd, adam
    opts["batch_size"] = 100
    opts["d_steps"] = 1
    opts['d_new_minibatch'] = False
    opts["g_steps"] = 2
    opts['batch_norm'] = True
    opts['dropout'] = False
    opts['dropout_keep_prob'] = 0.5
    opts['recon_loss'] = 'l2'
    # "manual" or number (float or int) giving the number of epochs to divide
    # the learning rate by 10 (converted into an exp decay per epoch).
    opts['decay_schedule'] = 'manual'
    opts['opt_learning_rate'] = FLAGS.learning_rate
    opts['opt_d_learning_rate'] = FLAGS.d_learning_rate
    opts['opt_g_learning_rate'] = FLAGS.g_learning_rate
    opts["opt_beta1"] = FLAGS.adam_beta1
    opts['batch_norm_eps'] = 1e-05
    opts['batch_norm_decay'] = 0.9

    if opts['e_is_random']:
        assert opts['latent_space_distr'] == 'normal',\
            'Random encoders currently work only with Gaussian Pz'
    # Data augmentation
    opts['data_augm'] = False
#    opts['vae'] = FLAGS.vae
#    opts['pot'] = FLAGS.pot
#    opts['pot_pz_std'] = 2
#    opts['vae_sigma'] = 0.01
#    opts['pot_lambda'] = 10
#    opts['convolutions'] = False

    opts['plot_kGANs'] = False
    opts['assignment'] = FLAGS.assignment
    opts['number_of_steps_made'] = 0
    opts['number_of_kGANs'] = 10
    opts['kGANs_number_rounds'] = 300
    opts['kill_threshold'] = 0.001 
    opts['annealed'] = True 
    opts['number_of_gpus'] = len(get_available_gpus())
    opts['reinitialize'] = True
    opts['one_batch'] = False # update weights every batch (True) or every epoch (False)
    
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
        logging.info('Running step {} of kGAN'.format(step))
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
        #fp = []
        #for i in range(fake_points.shape[0]): fp.append(fake_points[i,:,:,:]*255)
        #inception = get_inception_score(fp)
        #logging.debug('Inception score: ' + ''.join(repr(inception)) )
        
        
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
