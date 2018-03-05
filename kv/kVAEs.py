# Copyright 2017 Max Planck Society - ETH Zurich
# Distributed under the BSD-3 Software license,
# (See accompanying file ./LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)
"""The class implementing kGAN iterative training procedure.

"""

import logging
import numpy as np
from numpy.random import dirichlet
import tensorflow as tf
import vae as GAN
import kv_class as KV
from utils import ArraySaver
from metrics import Metrics
import utils
import classifier as CLASSIFIER
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

class KGANS(object):
    """This class implements the kGAN meta-algorithm.
    
    The class provides the 'make_step' method, which calls Gan.train()
    method to train the next Generator function. It also updates the
    weights of training points and takes care of mixture weights for
    newly trained mixture components.

    The same class can be used to implement the bagging, i.e. uniform
    mixture of independently trained GANs. This is controlled by
    opts['is_bagging'].
    """

    def __init__(self, opts, data, data_test = None):
        self.steps_total = opts['kGANs_number_rounds']
        self.steps_made = 0
        num = data.num_points
        self._data_num = num
        self.data_test = data_test
        self._number_of_kGANs = opts['number_of_kGANs']
        self._init_data_weights_uniform(opts, data)

        self._mixture_weights = np.ones([1, opts['number_of_kGANs']]) / (opts['number_of_kGANs'] + 0.)
        self._hard_assigned = np.ones([1, opts['number_of_kGANs']]) / (opts['number_of_kGANs'] + 0.) * self._data_num
        # Which GAN architecture should we use?
        pic_datasets = ['mnist']
        gan_class = None
        if opts['dataset'] in ('gmm'):
            kv_class = KV.kToyVae
            self._classifier = CLASSIFIER.ToyClassifier
            self.ais = np.zeros(opts["kGANs_number_rounds"])
            self.ais_test = np.zeros(opts["kGANs_number_rounds"])
            self.ais_idx = 0
        elif opts['dataset'] in pic_datasets:
            kv_class = KV.kImageVae
            self._classifier = CLASSIFIER.ImageClassifier
            self.ais = np.zeros(opts["kGANs_number_rounds"])
            self.ais_test = np.zeros(opts["kGANs_number_rounds"])
            self.ais_idx = 0
        else:
            assert False, "We don't have any other GAN implementation yet..."
        
        self._gan_class = gan_class 
        kVAEs = kv_class(opts, data, self._data_weights) 
        self.KV = kVAEs

    def make_step(self, opts, data):
        """Make step of the kGAN algorithm. First train the GANs on the reweighted training set,
        then update the weights"""
        self.KV.train_vaes(opts)
        self._update_data_weights_soft(opts, data)
        if np.abs(np.sum(self._mixture_weights)-1) > 0.001:
            assert np.abs(np.sum(self._mixture_weights)-1) <= 0.001, 'mixture weights dont sum to 1.. something went wrong'

        
    def _init_data_weights_uniform(self, opts, data):
        """initialize wegiths with uniform distribution"""
        self._data_weights = np.ones([self._data_num, opts['number_of_kGANs']])
        self._data_weights /= self._data_weights.sum(axis = 0, keepdims = True)
    
    def _init_data_weights_bag(self, opts, data):
        """initialize wegiths with uniform distribution"""
        K = opts['number_of_kGANs']
        self._data_weights = np.random.choice([0., 1.], size=(self._data_num, opts['number_of_kGANs']), p=[1. - 1./K, 1./K])
        self._data_weights /= self._data_weights.sum(axis = 0, keepdims = True)    

    def _init_data_weights_bag_mult(self, opts, data):
        """initialize wegiths with uniform distribution"""
        K = opts['number_of_kGANs']
        train_size = len(data.data)
        bag_size = int(train_size*1./K)
        data_copy = np.copy(data.data)
        for k in range(0,K):
            data_ids = np.random.choice(train_size, bag_size ,replace=True)
            self._kGANs[k]._data.data = np.copy(data_copy[data_ids])
            self._kGANs[k]._data_weights = np.ones(bag_size)*1./bag_size
            self._kGANs[k]._data.num_points = bag_size
        #self._data_weights = np.random.choice([0., 1.], size=(self._data_num, opts['number_of_kGANs']), p=[1. - 1./K, 1./K])
        #self._data_weights /= self._data_weights.sum(axis = 0, keepdims = True)    
    
    def _update_data_weights_soft(self, opts, data):
        """ 
        update the data weights with the soft assignments for the kGAN object
        """
        prob_x_given_gan = self._prob_data_under_gan(opts, data)
        new_weights = np.ones(prob_x_given_gan.shape)*0.
        
        max_values = np.transpose(np.repeat([np.amax(prob_x_given_gan, axis = 1)], opts['number_of_kGANs'], axis = 0))
        new_weights[prob_x_given_gan == max_values] = 1.
        #new_weights[np.arange(len(prob_x_given_gan)), prob_x_given_gan.argmax(1)] = 1.
        self._hard_assigned = new_weights.sum(axis = 0, keepdims = True)
        self.KV._hard_assigned = self._hard_assigned[0] 
        print self._hard_assigned
        self._mixture_weights = np.copy(self._hard_assigned)/self._hard_assigned.sum()
        new_weights /= np.maximum(1., self._hard_assigned)
        # update data weights in eahc gan for the importance sampling 
        self.KV._data_weights = new_weights
        #print plots
        if (opts['dataset'] == 'gmm'):
            print "step done"
        if opts['number_of_steps_made']%opts["plot_every"] ==0:
            wm = self.KV._data_weights

            #sampled = self._sample_from_training(opts,data, wm, 50)
            if opts['dataset'] == 'mnist':
                sampled = self._sample_from_training(opts,data, wm, 64)
                metrics = Metrics()
                metrics._return_plots_pics(opts, opts['number_of_steps_made'], data.data, sampled, 64,  wm, prefix = "train")    
                self._mnist_label_proportions( opts, data, metrics)
        
        
    
    def _sample_from_training(self, opts, data, probs, num_pts):
        """sample num_pts from the training set with importance sampling"""
        data_ids = np.random.choice(self._data_num, num_pts,replace=True, p=probs[:,0])
        sampled = data.data[data_ids].astype(np.float)
        
        for k in range(1,self._number_of_kGANs):
            data_ids = np.random.choice(self._data_num, num_pts,replace=True, p=probs[:,k])
            sampled = np.concatenate((sampled, data.data[data_ids].astype(np.float)),axis = 0)
        return sampled



    def _prob_data_under_gan(self, opts, data):
        """compute p(x_train | gan j) for each gan and store it in self._prob_x_given_gan
        Could be done more efficiently by appending columns to empty array
        Returns:
        (data.num_points, opts['number_of_kGANs']) NumPy array

        """

        num_fake_images = len(data.data)
        fake_pts = np.zeros((opts['number_of_kGANs'],) +  data.data.shape)
        for k in range(opts['number_of_kGANs']):
            fake_pts[k] = self.KV.kVAEs[k].sample(opts, num_fake_images)

        if opts['reinit_class']:
            self.KV.reinit_classifiers(opts)
        D_k = self.KV.train_class(opts, fake_pts)
        p_k = ((1. - D_k)/(D_k + 1e-12)) + 1e-12
        return p_k / p_k.sum(keepdims = True)



    def sample_mixture_separate_color(self, opts, number_of_samples):
        """sample from the mixture of generators and remember from which generator they come from"""
        number_samples = [int(round(number_of_samples * self._mixture_weights.flatten()[0]))]
        initialized = False
        if number_samples[-1] != 0:
            fake_samples = self.KV.kVAEs[0].sample(opts, number_samples[0])
            initialized = True

        for k in range(1,self._number_of_kGANs):
            number_samples.append(int(round(number_of_samples * self._mixture_weights.flatten()[k])))
            if number_samples[-1] != 0:
                if initialized:
                    fake_samples = np.concatenate((fake_samples,self.KV.kVAEs[k].sample(opts, number_samples[k])),axis = 0)
                else:
                    fake_samples = self.KV.kVAEs[k].sample(opts, number_samples[k])
                    initialized = True
        return fake_samples, number_samples

    def sample_mixture(self, opts, number_of_samples):
        """sample from the mixture of generators"""
        number_samples = [int(round(number_of_samples * self._mixture_weights.flatten()[0]))]
        initialized = False
        if number_samples[-1] != 0:
            fake_samples = self.KV.kVAEs[0].sample(opts, number_samples[0])
            initialized = True

        for k in range(1,self._number_of_kGANs):
            number_samples.append(int(round(number_of_samples * self._mixture_weights.flatten()[k])))
            if number_samples[-1] != 0:
                if initialized:
                    fake_samples = np.concatenate((fake_samples,self.KV.kVAEs[k].sample(opts, number_samples[k])),axis = 0)
                else:
                    fake_samples = self.KV.kVAEs[k].sample(opts, number_samples[k])
                    initialized = True
        return fake_samples

    def sample_mixture_separate_uniform(self, opts, number_samples):
        """sample same number of samples from each gan"""
        fake_samples = self.KV.kVAEs[0].sample(opts, number_samples)

        for k in range(1,self._number_of_kGANs):
            fake_samples = np.concatenate((fake_samples,self.KV.kVAEs[k].sample(opts, number_samples)),axis = 0)
        return fake_samples
    
    #def _sample_from_training_gan_label_real_fake(self, opts, data, probs, num_pts):
    #    """labels are the real fake, sample from the training set
    #    uniformly and from the mixture"""
    #    data_ids = np.random.choice(self._data_num, num_pts,replace=False)
    #    sampled = data.data[data_ids].astype(np.float)
    #    labels = np.zeros(num_pts)
    #    fake_points = self.sample_mixture(opts, num_pts)
    #    num_fake = len(fake_points)
    #    sampled = np.concatenate((sampled, fake_points),axis = 0)
    #    labels = np.append(labels, np.ones(num_fake))
    #    return sampled, labels
    #def _sample_from_training_gan_label(self, opts, data, probs, num_pts):
    #    """labels is the gan idx, sample from the training set
    #    with importance sampling using the probability of each gan and
    #    remember from which gan do the samples come from """
    #    data_ids = np.random.choice(self._data_num, num_pts,replace=False, p=probs[:,0])
    #    sampled = data.data[data_ids].astype(np.float)
    #    labels = np.zeros(num_pts)
    #    for k in range(1,self._number_of_kGANs):
    #        data_ids = np.random.choice(self._data_num, num_pts,replace=False, p=probs[:,k])
    #        sampled = np.concatenate((sampled, data.data[data_ids].astype(np.float)),axis = 0)
    #        labels = np.append(labels, np.ones(num_pts)*k)
    #    return sampled, labels
    def _mnist_label_proportions(self, opts, data, metrics):
        rez = np.zeros(10)

        for k in range(0, self._number_of_kGANs):
            num_pts = int(self._hard_assigned[0][k] )
            if num_pts >0:
                data_ids = np.random.choice(self._data_num, num_pts,replace=False, p=self.KV._data_weights[:,k])
                samp =  data.data[data_ids].astype(np.float)
                labels = metrics.evaluate_mnist_classes(opts,samp)
                counts = np.zeros(10)
                for i in labels: counts[i]+=1
                rez = np.vstack((rez,counts))
        #print rez
        rez = np.delete(rez, 0, 0)
        rez/=rez.sum(axis = 0, keepdims = True)
        import matplotlib.pyplot as plt
        fig, ax =  plt.subplots()
        cax = plt.imshow(rez, cmap='hot', interpolation='nearest')
        ax.set_xlabel('digits')
        ax.set_ylabel('model')
        cbar = fig.colorbar(cax, ticks=[0, 1])
        cbar.ax.set_yticklabels(['0', '1']) 
        filename = opts['work_dir'] + '/assignments{:08d}.png'.format(opts['number_of_steps_made'])
        fig.savefig(filename)
        plt.close()


