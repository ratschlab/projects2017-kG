#!/usr/bin/env python
from time import sleep
from os import system
import numpy as np
import argparse
import tensorflow as tf
import pdb

flags = tf.app.flags
flags.DEFINE_integer("experiment", 1, "experiment index [1,2,3] same order of the paper")
FLAGS = flags.FLAGS

arg = {}

if FLAGS.experiment == 1:
    arg['gmm_modes_num'] = 3
    arg['number_of_kGANs'] = 3
    arg['kGANs_number_rounds'] = 200
    arg['gan_epoch_num_first_iteration'] =  10
    arg['gan_epoch_num_except_first'] = 10
    arg['mixture_c_epoch_num'] = 2
    arg['one_batch_class'] = False
    arg['reinit_class'] = True 
    arg['workdir'] = 'results_gmm_experiment_1'#''.join('{}:{}_'.format(key, val) for key, val in arg.items())
    arguments = ''.join(' --{} {}'.format(key, val) for key, val in arg.items())
    cmd_line = 'python gmm-vae.py'+arguments

if FLAGS.experiment == 2:
    arg['gmm_modes_num'] = 6
    arg['number_of_kGANs'] = 5
    arg['kGANs_number_rounds'] = 200
    arg['gan_epoch_num_first_iteration'] =  100
    arg['gan_epoch_num_except_first'] = 10
    arg['mixture_c_epoch_num'] = 2
    arg['one_batch_class'] = False
    arg['reinit_class'] = True 
    arg['workdir'] = 'results_gmm_experiment_2'#''.join('{}:{}_'.format(key, val) for key, val in arg.items())
    arguments = ''.join(' --{} {}'.format(key, val) for key, val in arg.items())
    cmd_line = 'python gmm-vae.py'+arguments
if FLAGS.experiment == 3:
    arg['gmm_modes_num'] = 9
    arg['number_of_kGANs'] = 9
    arg['kGANs_number_rounds'] = 200
    arg['gan_epoch_num_first_iteration'] =  1000
    arg['gan_epoch_num_except_first'] = 10
    arg['mixture_c_epoch_num'] = 5
    arg['one_batch_class'] = False
    arg['reinit_class'] = True 
    arg['workdir'] = 'results_gmm_experiment_3'#''.join('{}:{}_'.format(key, val) for key, val in arg.items())
    arguments = ''.join(' --{} {}'.format(key, val) for key, val in arg.items())
    cmd_line = 'python gmm-vae.py'+arguments
print(cmd_line)
system(cmd_line)
#sleep(0.5)
