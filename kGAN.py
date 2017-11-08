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
import gan as GAN
from utils import ArraySaver
from metrics import Metrics
import utils
import classifier as CLASSIFIER

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

    # pylint: disable=too-many-instance-attributes
    # We need this many.
    def __init__(self, opts, data):
        self.steps_total = opts['kGANs_number_rounds']
        self.steps_made = 0
        num = data.num_points
        self._data_num = num
        self._number_of_kGANs = opts['number_of_kGANs']
        self._assignment = opts['assignment']
        if self._assignment == 'soft':
            self._init_data_weights_soft(opts, data)
        elif self._assignment == 'hard':
            self._init_data_weights_hard(opts, data)
        else:
            assert False, 'Only soft and hard assignments are supported'

        self._saver = ArraySaver('disk', workdir=opts['work_dir'])
        self._mixture_weights = np.ones([1, opts['number_of_kGANs']]) / (opts['number_of_kGANs'] + 0.)
        
        # Which GAN architecture should we use?
        pic_datasets = ['mnist']
        gan_class = None
        if opts['dataset'] in ('gmm'):
            gan_class = GAN.ToyGan
            self._classifier = CLASSIFIER.ToyClassifier
        elif opts['dataset'] in pic_datasets:
            gan_class = GAN.ImageGan
            self._classifier = CLASSIFIER.ToyClassifier
        else:
            assert False, "We don't have any other GAN implementation yet..."
        
        self._gan_class = gan_class 
            
        #initialize k-graphs and the kGANs
        self._graphs = []
        self._kGANs = []
        for it in range(0,self._number_of_kGANs):
            # initialize graph k
            self._graphs.append(tf.Graph())
            with self._graphs[it].as_default():
                dev = it%opts['number_of_gpus']
                with tf.device('/device:GPU:%d' %dev):
                    # initialize gan k
                    self._kGANs.append(gan_class(opts, data, self._data_weights[:,it]))
    
    def _kill_a_gan(self, opts, data, idx):
        opts['number_of_kGANs'] -= 1
        self._number_of_kGANs = opts['number_of_kGANs']
        self._mixture_weights = np.delete(self._mixture_weights, idx, 1)
        self._mixture_weights /= np.sum(self._mixture_weights)
        self._data_weights = np.delete(self._data_weights, idx, 1)
        del self._graphs[idx]
        del self._kGANs[idx]

    def _internal_step(self, (opts, data, k)):
        if self._mixture_weights[0][k] != 0:
            with self._graphs[k].as_default():
                self._kGANs[k].train(opts)

    def make_step(self, opts, data):
        """Make step of the kGAN algorithm. First train the GANs on the reweighted training set,
        then update the weights"""
        
        #from joblib import Parallel, delayed
        #import multiprocessing
        #Parallel(n_jobs=5)(delayed(self._internal_step)(opts, data, k) for k in (0,self._number_of_kGANs))
        
        
        #import pdb
        #pdb.set_trace()
        from multiprocessing import Pool
        from multiprocessing.dummy import Pool as ThreadPool

        pool = ThreadPool(opts['number_of_gpus'])
        job_args = [(opts, data, i) for i in range(0,self._number_of_kGANs)] 
        pool.map(self._internal_step, job_args)
        pool.close()
        pool.join()
        # train the gans
        #for k in range (0,self._number_of_kGANs):
        #    if self._mixture_weights[0][k] != 0:
        #        with self._graphs[k].as_default():
        #            self._kGANs[k].train(opts)
        
        # update the weights
        if self._assignment == 'soft':
            self._update_data_weights_soft(opts, data)
        elif self._assignment == 'hard':
            self._update_data_weights_hard(opts, data)
        killed = False
        while (np.where(self._mixture_weights[0] < opts['kill_threshold'])[0].size!= 0):
            idx_to_kill = np.argmin(self._mixture_weights[0])
            self._kill_a_gan(opts, data, idx_to_kill)
            killed = True
        if killed:
            if self._assignment == 'soft':
                self._update_data_weights_soft(opts, data)
            elif self._assignment == 'hard':
                self._update_data_weights_hard(opts, data)
        
        if np.abs(np.sum(self._mixture_weights)-1) > 0.001:
            assert np.abs(np.sum(self._mixture_weights)-1) <= 0.001, 'mixture weights dont sum to 1.. something went wrong'

        
    def _init_data_weights_hard(self, opts, data):
        """randomly assign each point to a gan with uniform probability"""
        K = opts['number_of_kGANs']
        self._data_weights = np.random.choice([0., 1.], size=(self._data_num, opts['number_of_kGANs']), p=[1. - 1./K, 1./K])
        self._data_weights /= self._data_weights.sum(axis = 0, keepdims = True)    

    def _init_data_weights_soft(self, opts, data):
        """initialize wegiths with dirichlet distribution with alpha = 1"""
        alpha = 1./opts['number_of_kGANs']
        self._data_weights = np.random.dirichlet(np.ones(self._data_num)*alpha, opts['number_of_kGANs']).transpose()

    def _init_data_weights_uniform(self, opts, data):
        self._data_weights = np.ones([self._data_num, opts['number_of_kGANs']])
        self._data_weights /= self._data_weights.sum(axis = 0, keepdims = True)

    def _update_data_weights_soft(self, opts, data):
        """ 
        update the data weights with the soft assignments for the kGAN object
        """
        # compute p(x | gan) 
        prob_x_given_gan = self._prob_data_under_gan(opts, data)
        
        #force uniform if no gan is taking the point
        normalization_over_gan = prob_x_given_gan.sum(axis = 1, keepdims = True)
        prob_x_given_gan[np.argwhere(normalization_over_gan == 0)] = 1.
        # compute pi and alpha
        # pi_x(j) = 1/Z alpha_j p(x | g_j)
        annealing = 1.
        if opts['annealed']:
            annealing = min(1.,(1. + opts['number_of_steps_made']*1.)/opts["kGANs_number_rounds"])
        
        
        pi = np.power(prob_x_given_gan * np.repeat(self._mixture_weights,self._data_weights.shape[0],axis = 0), annealing) + np.power((1./self._data_num),2) 
        pi /= pi.sum(axis = 1, keepdims = True)
        
        # Compute Gan probability (Is it the prior I need to use computing the previous line?)
        self._mixture_weights = pi.sum(axis = 0, keepdims = True)
        self._mixture_weights /= np.sum(self._mixture_weights)
        print self._mixture_weights
        
        #_data_weights = pi/alpha
        self._data_weights = pi / np.repeat(self._mixture_weights,self._data_weights.shape[0],axis = 0)
        
        # divide by N
        self._data_weights /= self._data_weights.sum(axis = 0, keepdims = True)
        
        # update data weights in eahc gan for the importance sampling 
        for k in range (0,self._number_of_kGANs):
            self._kGANs[k]._data_weights = self._data_weights[:,k]
        
        #print plots
        if (opts['dataset'] == 'gmm'):
            self._plot_competition_2d(opts)
        elif(opts['dataset'] == 'mnist'):
            sampled = self._sample_from_training(opts,data, self._data_weights, 50)
            metrics = Metrics()
            metrics._return_plots_pics(opts, opts['number_of_steps_made'], data.data, sampled, 50,  self._data_weights, prefix = "train")    
            self.tsne_plotter(opts,  data)
    
    def _update_data_weights_hard(self, opts, data):
        """ 
        update the data weights with the hard assignments for the kGAN object
        """
        # compute p(x | gan) 
        prob_x_given_gan = self._prob_data_under_gan(opts, data) #/ np.repeat(self._mixture_weights,self._data_weights.shape[0],axis = 0)*np.sum(self._data_weights)
        
        #compute the argmax of the prob_x_given_gan
        new_weights = np.ones(prob_x_given_gan.shape)*0.
        new_weights[np.arange(len(prob_x_given_gan)), prob_x_given_gan.argmax(1)] = 1.
        
        #alpha_k = N_k / N 
        self._mixture_weights = new_weights.sum(axis = 0, keepdims = True)/np.sum(new_weights)
        print self._mixture_weights
        sum_new_weights = new_weights.sum(axis = 0, keepdims = True) #N_k
        new_weights /= new_weights.sum(axis = 0, keepdims = True) 
        self._data_weights = np.copy(new_weights)
        
        # update data weights in eahc gan for the importance sampling 
        for k in range (0,self._number_of_kGANs):
            self._kGANs[k]._data_weights = self._data_weights[:,k]
        
        #print plots
        if (opts['dataset'] == 'gmm'):
            self._plot_competition_2d(opts)
        elif(opts['dataset'] == 'mnist'):
            sampled = self._sample_from_training(opts,data, self._data_weights, 50)
            metrics = Metrics()
            metrics._return_plots_pics(opts, opts['number_of_steps_made'], data.data, sampled, 50,  self._data_weights, prefix = "train")
            self.tsne_plotter(opts,  data)

    def _sample_from_training(self, opts, data, probs, num_pts):
        """sample num_pts from the training set with importance sampling"""
        data_ids = np.random.choice(self._data_num, num_pts,replace=False, p=probs[:,0])
        sampled = data.data[data_ids].astype(np.float)
        
        for k in range(1,self._number_of_kGANs):
            data_ids = np.random.choice(self._data_num, num_pts,replace=False, p=probs[:,k])
            sampled = np.concatenate((sampled, data.data[data_ids].astype(np.float)),axis = 0)
        return sampled

    def _sample_from_training_gan_label(self, opts, data, probs, num_pts):
        """labels are the gan idx, sample from the training set
        with importance sampling using the probability of each gan anf
        remember from which gan do the samples come from """
        data_ids = np.random.choice(self._data_num, num_pts,replace=False, p=probs[:,0])
        sampled = data.data[data_ids].astype(np.float)
        labels = np.zeros(num_pts)
        for k in range(1,self._number_of_kGANs):
            data_ids = np.random.choice(self._data_num, num_pts,replace=False, p=probs[:,k])
            sampled = np.concatenate((sampled, data.data[data_ids].astype(np.float)),axis = 0)
            labels = np.append(labels, np.ones(num_pts)*k)
        return sampled, labels
    
    def tsne_plotter(self, opts,  data):
        """tsne plotter, not particularly interesting"""
        #return True 
        sampled,labels = self._sample_from_training_gan_label(opts,data, self._data_weights, 50)
        from sklearn.manifold import TSNE
        import matplotlib 
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig =  plt.figure()
        sampled = np.asarray(sampled).astype('float64')
        sampled = sampled.reshape((sampled.shape[0], -1))
        
        vis_data = TSNE(n_components=2).fit_transform(sampled)
        vis_x = vis_data[:, 0]
        vis_y = vis_data[:, 1]

        plt.scatter(vis_x, vis_y, c=labels.astype(int), cmap=plt.cm.get_cmap("jet", opts['number_of_kGANs']))
        plt.colorbar(ticks=range(opts['number_of_kGANs']))
        #plt.clim(-0.5, 9.5)
        filename = opts['work_dir'] + '/tsne{:02d}.png'.format(opts['number_of_steps_made'])
        fig.savefig(filename)
        plt.close()

    def _prob_data_under_gan_internal(self, (opts, data, k)):
        device = k%opts['number_of_gpus']
        # select gan k
        gan_k = self._kGANs[k]
        # discriminator output for training set 
        D_k = self._get_prob_real_data(opts, gan_k, self._graphs[k], data, device)
        # probability x_i given gan_j
            
        if self._assignment == 'soft':
            p_k = self._data_weights[:,k]*np.transpose((1. - D_k)/(D_k + 1e-8))
        elif self._assignment == 'hard':
            p_k = 1./(self._data_num*self._mixture_weights[0][k])*np.transpose((1. - D_k)/(D_k + 1e-8))
            
        return p_k / p_k.sum(keepdims = True)
    
    def _prob_data_under_gan(self, opts, data):
        """compute p(x_train | gan j) for each gan and store it in self._prob_x_given_gan
        Could be done more efficiently by appending columns to empty array
        Returns:
        (data.num_points, opts['number_of_kGANs']) NumPy array
            
        """
        #prob_x_given_gan = np.empty([data.num_points, opts['number_of_kGANs']])
        #for k in range (0,self._number_of_kGANs):
        from multiprocessing import Pool
        from multiprocessing.dummy import Pool as ThreadPool
        #import pdb
        #pdb.set_trace()
        pool = ThreadPool(opts['number_of_gpus'])
        job_args = [(opts, data, i) for i in range(0,self._number_of_kGANs)] 
        prob_x_given_gan = pool.map(self._prob_data_under_gan_internal, job_args)
        pool.close()
        pool.join()
        #pdb.set_trace()
        return np.transpose(np.squeeze(np.asarray(prob_x_given_gan)))
        #return prob_x_given_gan

    def _get_prob_real_data(self, opts, gan, graph, data, device):
        """Train a classifier, separating true data from the current gan.
        Returns:
        (data.num_points,) NumPy array, containing probabilities of 
        true data. I.e., output of the sigmoid function. Create a graph and train a classifier"""
        g = tf.Graph()
        with g.as_default():
            with tf.device('/device:GPU:%d' %device):
                classifier = CLASSIFIER.ToyClassifier(opts, data)
                num_fake_images = data.num_points
                fake_images = gan.sample(opts, num_fake_images)
                prob_real, prob_fake = classifier.train_mixture_discriminator(opts, fake_images)
        return prob_real


    def _plot_competition_2d(self, opts):
        """ plot assignment for 2d gmm"""
        import matplotlib 
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        max_val = opts['gmm_max_val'] * 2
        plt.clf()
        fig = plt.figure()
        for k in range(0,opts['number_of_kGANs']):
            g1 = fig.add_subplot(opts['number_of_kGANs'],1,k+1)
            g1.axis([-max_val, max_val, -max_val, max_val])
            g1.scatter(self._kGANs[0]._data.data[:, 0], self._kGANs[0]._data.data[:, 1], c=self._data_weights[:,k], s=20, label='real')

        filename = opts['work_dir'] + '/competition_heatmap_step{:02d}.png'.format(opts['number_of_steps_made'])
        fig.savefig(filename)
        plt.close()




    def sample_mixture_separate_color(self, opts, number_of_samples):
        """sample from the mixture of generators and remember from which generator they come from"""
        number_samples = [int(round(number_of_samples * self._mixture_weights.flatten()[0]))]
        initialized = False
        if number_samples[-1] != 0:
            fake_samples = self._kGANs[0].sample(opts, number_samples[0])
            initialized = True

        for k in range(1,self._number_of_kGANs):
            number_samples.append(int(round(number_of_samples * self._mixture_weights.flatten()[k])))
            if number_samples[-1] != 0:
                if initialized:
                    fake_samples = np.concatenate((fake_samples,self._kGANs[k].sample(opts, number_samples[k])),axis = 0)
                else:
                    fake_samples = self._kGANs[k].sample(opts, number_samples[k])
                    initialized = True
        return fake_samples, number_samples

    def sample_mixture(self, opts, number_of_samples):
        """sample from the mixture of generators"""
        number_samples = [int(round(number_of_samples * self._mixture_weights.flatten()[0]))]
        initialized = False
        if number_samples[-1] != 0:
            fake_samples = self._kGANs[0].sample(opts, number_samples[0])
            initialized = True

        for k in range(1,self._number_of_kGANs):
            number_samples.append(int(round(number_of_samples * self._mixture_weights.flatten()[k])))
            if number_samples[-1] != 0:
                if initialized:
                    fake_samples = np.concatenate((fake_samples,self._kGANs[k].sample(opts, number_samples[k])),axis = 0)
                else:
                    fake_samples = self._kGANs[k].sample(opts, number_samples[k])
                    initialized = True
        return fake_samples

    def sample_mixture_separate_uniform(self, opts, number_samples):
        """sample same number of samples from each gan"""
        fake_samples = self._kGANs[0].sample(opts, number_samples)

        for k in range(1,self._number_of_kGANs):
            fake_samples = np.concatenate((fake_samples,self._kGANs[k].sample(opts, number_samples)),axis = 0)
        return fake_samples
