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
    arg['number_of_kGANs'] = 15
    arg['kGANs_number_rounds'] = 100
    arg['gan_epoch_num_first_iteration'] =  100
    arg['gan_epoch_num_except_first'] = 10
    arg['mixture_c_epoch_num'] = 1
    arg['one_batch_class'] = True
    arg['reinit_class'] = False
    arg['learning_rate'] = 0.005
    arg['AIS_every_it'] = 5
    arg['workdir'] = 'results_mnist_ais_final3'#''.join('{}:{}_'.format(key, val) for key, val in arg.items())
    arguments = ''.join(' --{} {}'.format(key, val) for key, val in arg.items())
    cmd_line = 'python mnist-vae.py'+arguments
print(cmd_line)
system(cmd_line)
#sleep(0.5)
