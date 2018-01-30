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
flags.DEFINE_float("learning_rate", 0.005,
                   "Learning rate for other optimizers [8e-4]")
flags.DEFINE_float("adam_beta1", 0.5, "Beta1 parameter for Adam optimizer [0.5]")
flags.DEFINE_integer("zdim", 8, "Dimensionality of the latent space [100]")
flags.DEFINE_float("init_std", 0.01, "Initial variance for weights [0.02]")
flags.DEFINE_string("assignment", 'soft', "Type of update for the weights")
flags.DEFINE_string("workdir", 'results_mnist_rez_batch_lastday_bw_fashion_fc2', "Working directory ['results']")
flags.DEFINE_bool("vae", True, "use VAEs instead of GANs")
flags.DEFINE_integer("gan_epoch_num_first_iteration", 10, "epoch number to pretrain models")
flags.DEFINE_integer("gan_epoch_num_except_first", 10, "epoch number per iteration")
flags.DEFINE_integer("mixture_c_epoch_num", 1, "epoch number for classifier")
flags.DEFINE_integer("number_of_kGANs", 3, "Number of generative models")
flags.DEFINE_integer("kGANs_number_rounds", 3, "Number of iterations")
flags.DEFINE_bool("one_batch_class", False, "train classifier for a single batch")
flags.DEFINE_bool("reinit_class", True, "train classifiers from scratch")
#flags.DEFINE_bool("unrolled", False, "Use unrolled GAN training [True]")
#flags.DEFINE_bool("vae", False, "Use VAE instead of GAN")
#flags.DEFINE_bool("pot", False, "Use VAE instead of GAN")
#flags.DEFINE_bool("is_bagging", False, "Do we want to use bagging instead of adagan? [False]")
FLAGS = flags.FLAGS


from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def main():
    opts = {}
    
    # dataset opts
    opts['random_seed'] = 66
    opts['dataset'] = 'mnist' # gmm, circle_gmm,  mnist, mnist3 ...
    opts['data_dir'] = 'mnist'
    opts['trained_model_path'] = 'models'
    opts['mnist_trained_model_file'] = 'mnist_trainSteps_19999_yhat' # 'mnist_trainSteps_20000'
#    opts['mnist3_dataset_size'] = 2 * 64 # 64 * 2500
#    opts['mnist3_to_channels'] = False # Hide 3 digits of MNIST to channels
    opts['input_normalize_sym'] = True # Normalize data to [-1, 1]
    opts['samples_per_component'] = 5000
    opts['work_dir'] = FLAGS.workdir
    print opts['work_dir']
    opts['ckpt_dir'] = 'checkpoints'
    
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
    opts['one_batch_class'] = False
    opts['test'] = False #hack, don't set to true
    opts['reinit_class'] = False#False
    # VAE opts
    opts['vae_sigma'] = 0.1
    opts['vae'] = FLAGS.vae
    opts['decay_schedule'] = 'manual'
    opts['convolutions'] = True # If False then encoder is MLP of 3 layers
    opts['conv_filters_dim'] = 4
    opts['d_num_filters'] = 64
    #opts['d_num_layers'] = 3
    opts['g_num_filters'] = 64
    opts['g_num_layers'] = 3
    opts['recon_loss'] = 'l2sq'
    
    # GAN opts    
    opts["init_std"] = FLAGS.init_std
    opts["init_bias"] = 0.0
    opts['latent_space_distr'] = 'normal' # uniform, normal
    #opts["d_steps"] = 1
    #opts["g_steps"] = 2
    opts["verbose"] = True
    opts['tf_run_batch_size'] = 128
    opts['latent_space_dim'] = FLAGS.zdim

    # Optimizer
    opts['optimizer'] = 'adam' # sgd, adam
    opts["batch_size"] = 32
    opts['batch_size_classifier'] = 32
    opts['opt_learning_rate'] = FLAGS.learning_rate
    #opts['opt_d_learning_rate'] = FLAGS.d_learning_rate
    #opts['opt_g_learning_rate'] = FLAGS.g_learning_rate
    opts["opt_beta1"] = FLAGS.adam_beta1
    opts['batch_norm_eps'] = 1e-05
    opts['batch_norm_decay'] = 0.9
    #opts['d_num_filters'] = 32#512
    #opts['g_num_filters'] = 64#1024
    opts["early_stop"] = -1 # set -1 to run normally
    opts["plot_every"] = 1
    opts["save_every_epoch"] = 10
    opts["eval_points_num"] = 25600
    opts['digit_classification_threshold'] = 0.999
    opts['inverse_metric'] = False # Use metric from the Unrolled GAN paper?
    opts['inverse_num'] = 100 # Number of real points to inverse.
    opts['objective'] = None
    opts['data_augm'] = False
    opts['batch_norm'] = True
    opts['dropout'] = False
    opts['dropout_keep_prob'] = 0.5

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
    opts['dataset'] = 'mnist_test'
    data_test = DataHandler(opts)
    opts['dataset'] = 'mnist'
    assert data_test.num_points >= opts['batch_size'], 'Training set too small'

    kG = KGANS(opts, data, data_test)
    metrics = Metrics()

    train_size = data.num_points
    random_idx = np.random.choice(train_size, 4*320, replace=False)
    metrics.make_plots(opts, 0, data.data,
            data.data[random_idx], kG._data_weights, prefix='dataset_')
    
    
    for step in range(opts["kGANs_number_rounds"]):
        opts['number_of_steps_made'] = step
        #if step>100:#+1%100 == 0:
        #    opts['one_batch_class'] = False
            #opts["mixture_c_epoch_num"]+=1
        logging.info('Running step {} of kGAN'.format(step))
        if step == opts["kGANs_number_rounds"]:
            opts["gan_epoch_num"] = 10
        kG.make_step(opts, data)
        #opts['one_batch'] = True
        if opts['reinit_class'] == False:
            opts['one_batch_class'] = True
        #opts["mixture_c_epoch_num"] = min(step, 1875)
        opts["gan_epoch_num"] = opts["gan_epoch_num_except_first"]
        if opts['number_of_steps_made']% opts["plot_every"] == 0:
            num_fake = opts['eval_points_num']
            logging.debug('Sampling fake points')
            fake_points = kG.sample_mixture(opts, num_fake)
            logging.debug('Sampling more fake points')
            more_fake_points = kG.sample_mixture(opts, 512)
            num_samples_per_gan = 64
            fake_points_plot = kG.sample_mixture_separate_uniform(opts, num_samples_per_gan)
            logging.debug('Plotting results')
            metrics.make_plots(opts, step, data.data,more_fake_points, kG._data_weights, prefix = "")
            metrics._return_plots_pics(opts, step, data.data,fake_points_plot, 64,  kG._data_weights, prefix = "")
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

        np.savetxt(opts['work_dir']+"/loss.csv", kG.loss, delimiter=",")
        np.savetxt(opts['work_dir']+"/train_loss.csv", kG.train_loss, delimiter=",")
    logging.debug("kGANs finished working!")

if __name__ == '__main__':
    main()
