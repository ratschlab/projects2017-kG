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
        self.idx_loss = 0 
        num = data.num_points
        self._data_num = num
        self.data_test = data_test
        self._number_of_kGANs = opts['number_of_kGANs']
        self._assignment = opts['assignment']
        if self._assignment == 'soft':
            self._init_data_weights_uniform(opts, data)
        elif self._assignment == 'hard':
            self._init_data_weights_hard(opts, data)
        else:
            assert False, 'Only soft and hard assignments are supported'

        self._saver = ArraySaver('disk', workdir=opts['work_dir'])
        self._mixture_weights = np.ones([1, opts['number_of_kGANs']]) / (opts['number_of_kGANs'] + 0.)
        self._hard_assigned = np.ones([1, opts['number_of_kGANs']]) / (opts['number_of_kGANs'] + 0.) * self._data_num
        # Which GAN architecture should we use?
        pic_datasets = ['mnist','cifar10']
        gan_class = None
        if opts['dataset'] in ('gmm'):
            gan_class = GAN.ToyVae
            self._classifier = CLASSIFIER.ToyClassifier
        elif opts['dataset'] in pic_datasets:
            gan_class = GAN.ImageVae
            self._classifier = CLASSIFIER.ImageClassifier
            self.loss = np.zeros(opts["kGANs_number_rounds"])
            self.train_loss = np.zeros(2*opts["kGANs_number_rounds"])
        else:
            assert False, "We don't have any other GAN implementation yet..."
        
        self._gan_class = gan_class 
            
        #initialize k-graphs and the kGANs
        self._graphs = []
        self._kGANs = []
        self._k_class = []
        self._k_class_graphs = [] 
        self._ever_dead = []
        for it in range(0,self._number_of_kGANs):
            # initialize graph k
            self._graphs.append(tf.Graph())
            self._ever_dead.append(False)
            with self._graphs[it].as_default():
                dev = it%opts['number_of_gpus']
                with tf.device('/device:GPU:%d' %dev):
                    # initialize gan k
                    self._kGANs.append(gan_class(opts, data, self._data_weights[:,it]))
                    if opts['dataset'] == 'mnist':
                        self._kGANs[it].test_data = data.data
            self._k_class_graphs.append(tf.Graph())
            with self._k_class_graphs[it].as_default():
                dev = it%opts['number_of_gpus']
                with tf.device('/device:GPU:%d' %dev):
                    self._k_class.append(self._classifier(opts, data))


    def _internal_step(self, (opts, data, k)):
        if self._mixture_weights[0][k] != 0:
            with self._graphs[k].as_default():
                if self._hard_assigned[0][k] > opts['batch_size']:
                    self._kGANs[k].train(opts)
    def make_step(self, opts, data):
        """Make step of the kGAN algorithm. First train the GANs on the reweighted training set,
        then update the weights"""
        pool = ThreadPool(opts['number_of_gpus'])
        job_args = [(opts, data, i) for i in range(0,self._number_of_kGANs)] 
        pool.map(self._internal_step, job_args)
        pool.close()
        pool.join()
        if self._assignment == 'soft':
            if opts['number_of_kGANs'] == 1:
                if opts['dataset'] == 'gmm':
                    self._plot_competition_2d(opts)
                    self._plot_distr_2d(opts)
                if opts['dataset'] == 'mnist':
                    #self._kGANs[0].test_data = self.data_test.data
                    #self.loss[opts['number_of_steps_made']], _, _ = self._kGANs[0].test(opts)
                    print 'loss: ' 
                    #print self.loss[opts['number_of_steps_made']]
            else:
                self._update_data_weights_soft(opts, data)
        elif self._assignment == 'hard':
            self._update_data_weights_hard(opts, data)
        
        if np.abs(np.sum(self._mixture_weights)-1) > 0.001:
            assert np.abs(np.sum(self._mixture_weights)-1) <= 0.001, 'mixture weights dont sum to 1.. something went wrong'

        
    def _init_data_weights_uniform(self, opts, data):
        """initialize wegiths with uniform distribution"""
        self._data_weights = np.ones([self._data_num, opts['number_of_kGANs']])
        self._data_weights /= self._data_weights.sum(axis = 0, keepdims = True)
    
    
    def _update_data_weights_soft(self, opts, data):
        """ 
        update the data weights with the soft assignments for the kGAN object
        """
        # compute p(x | gan)
        #if opts['dataset'] == 'mnist' and opts['number_of_steps_made']>=1:
            #tot_loss = 0.
            #for k in range(0,opts['number_of_kGANs']):
            #    self._kGANs[k].test_data = data.data
            #    if self._kGANs[k].test_data.shape[0]  >= 1:
            #        loss, _, _ = self._kGANs[k].test(opts)
            #        tot_loss += loss*self._kGANs[k].test_data.shape[0]/self._data_num
            #        self._kGANs[k].test_data = None
            #print("----- training loss after model training------")
            #print(tot_loss)
            #self.train_loss[self.idx_loss] = tot_loss
            #self.idx_loss += 1 
         #   self.loss[opts['number_of_steps_made']] = self.test_mnist(opts,data, self.data_test)
         #   print 'test loss: ' 
         #   print self.loss[opts['number_of_steps_made']]
        prob_x_given_gan = self._prob_data_under_gan(opts, data)
        # compute pi and alpha
        # pi_x(j) = 1/Z (alpha_j p(x | g_j))^temperature + smoothing
        #annealing = 1.
        #if opts['annealed']:
        #    annealing = min(1.,(1. + opts['number_of_steps_made']*1.)/opts["kGANs_number_rounds"])
        
        #pi = np.power(prob_x_given_gan * np.repeat(self._mixture_weights,self._data_weights.shape[0],axis = 0), annealing)
        #pi = np.minimum(1., pi + opts['smoothing'])
        #pi /= pi.sum(axis = 1, keepdims = True)
        
        # Compute Gan probability
        #self._mixture_weights = pi.sum(axis = 0, keepdims = True)
        #self._mixture_weights /= np.sum(self._mixture_weights)
        #print self._mixture_weights
        
        #_data_weights = pi/(N*alpha) commented these two lines to do uniform weights
        #self._data_weights = pi / np.repeat(self._mixture_weights,self._data_weights.shape[0],axis = 0)
        #self._data_weights /= self._data_weights.sum(axis = 0, keepdims = True)
        new_weights = np.ones(prob_x_given_gan.shape)*0.
        
        #if opts['dataset'] == 'mnist':
        #    new_weights_norepeat = np.ones(prob_x_given_gan.shape)*0.
            #for i in range(0,self._data_num):
            #    j = np.random.choice(opts['number_of_kGANs'], size=1, p=pi[i,:])    
            #    new_weights[i,j] = 1. 
        
            # the new weights is a hard assignment, therefore, we assigne the point to the gan
            # with maximum likelihood p(x|gan) (see report/hard assignment)
        #    new_weights_norepeat[np.arange(len(prob_x_given_gan)), prob_x_given_gan.argmax(1)] = 1.
        #    tot_loss = 0.
        #    for k in range(0,opts['number_of_kGANs']):
        #        assigned = data.data[np.argwhere(new_weights_norepeat[:,k]==1).flatten()]
        #        if assigned.shape[0] >= 1:
        #            self._kGANs[k].test_data = assigned
        #            loss, _, _ = self._kGANs[k].test(opts)
        #            tot_loss += loss*assigned.shape[0]/new_weights.shape[0]
        #        self._kGANs[k].test_data = None
        #    print("----- training loss after assignment ------")
        #    print(tot_loss)
        #    self.train_loss[self.idx_loss] = tot_loss
        #    self.idx_loss += 1
        
        max_values = np.transpose(np.repeat([np.amax(prob_x_given_gan, axis = 1)], opts['number_of_kGANs'], axis = 0))
        new_weights[prob_x_given_gan == max_values] = 1.
        #new_weights[np.arange(len(prob_x_given_gan)), prob_x_given_gan.argmax(1)] = 1.
        self._hard_assigned = new_weights.sum(axis = 0, keepdims = True)
        print self._hard_assigned
        self._mixture_weights = np.copy(self._hard_assigned)/self._hard_assigned.sum()
        new_weights /= np.maximum(1., self._hard_assigned)
        # update data weights in eahc gan for the importance sampling 
        for k in range (0,self._number_of_kGANs):
            self._kGANs[k]._data_weights = new_weights[:,k]
            #self._data_weights[:,k] = self._data_weights[:,k]
        
        #print plots
        if (opts['dataset'] == 'gmm'):
            print "step done"
            #self._plot_competition_2d(opts)
            #self._plot_distr_2d(opts)
        elif(opts['dataset'] == 'mnist') and opts['number_of_steps_made']%opts["plot_every"] ==0:
            wm = np.zeros(self._data_weights.shape)
            for k in range (0,self._number_of_kGANs):
                wm[:,k] = self._kGANs[k]._data_weights

            #sampled = self._sample_from_training(opts,data, wm, 50)
            metrics = Metrics()
            #metrics._return_plots_pics(opts, opts['number_of_steps_made'], data.data, sampled, 50,  wm, prefix = "train")    
            self._mnist_label_proportions( opts, data, metrics)
            #self.tsne_plotter(opts,  data)
            if opts['rotated_mnist']==True:
                self._plot_labels_ratio(opts, new_weights, data.labels)
            #self.tsne_real_fake(opts,  data)
        elif(opts['dataset'] == 'cifar10'):
            sampled = self._sample_from_training(opts,data, self._data_weights, 50)
            metrics = Metrics()
            metrics._return_plots_pics(opts, opts['number_of_steps_made'], data.data, sampled, 50,  self._data_weights, prefix = "train")    
            self.tsne_real_fake(opts,  data)
    def _plot_labels_ratio(self, opts, new_weights, labels):
        import matplotlib.pyplot as plt
        ass0 = np.argwhere(new_weights[:,0]!=0)
        ass1 = np.argwhere(new_weights[:,1]!=0)
        lab_model0 = labels[ass0]
        lab_model1 = labels[ass1]
        self.rotate_balance[0,opts['number_of_steps_made']] = len(np.argwhere(lab_model0 == 0))*1./len(np.argwhere(new_weights[:,0]!=0))
        self.rotate_balance[1,opts['number_of_steps_made']] = len(np.argwhere(lab_model1 == 0))*1./len(np.argwhere(new_weights[:,1]!=0))
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax = plt.plot(np.arange(opts['number_of_steps_made']+1),self.rotate_balance[0,:opts['number_of_steps_made']+1],'r', label = 'model 1')
        ax = plt.plot(np.arange(opts['number_of_steps_made']+1),self.rotate_balance[1,:opts['number_of_steps_made']+1],'b', label = 'model 2')

        filename = opts['work_dir'] + '/balance.png'
        fig.savefig(filename)
        plt.close()
    #def _update_data_weights_hard(self, opts, data):
    #    """ 
    #    update the data weights with the hard assignments for the kGAN object
    #    """
    #    # compute p(x | gan) 
    #    prob_x_given_gan = self._prob_data_under_gan(opts, data) #/ np.repeat(self._mixture_weights,self._data_weights.shape[0],axis = 0)*np.sum(self._data_weights)
    #    
    #    #compute the argmax of the prob_x_given_gan
    #    new_weights = np.ones(prob_x_given_gan.shape)*0.
    #    new_weights[np.arange(len(prob_x_given_gan)), prob_x_given_gan.argmax(1)] = 1.
        
    #    #alpha_k = N_k / N 
    #    self._mixture_weights = new_weights.sum(axis = 0, keepdims = True)/np.sum(new_weights)
    #    print self._mixture_weights
    #    new_weights /= new_weights.sum(axis = 0, keepdims = True) 
    #    self._data_weights = np.copy(new_weights)
    #    
    #    # update data weights in eahc gan for the importance sampling 
    #    for k in range (0,self._number_of_kGANs):
    #        self._kGANs[k]._data_weights = self._data_weights[:,k]
    #    
    #    #print plots
    #    if (opts['dataset'] == 'gmm'):
    #        self._plot_competition_2d(opts)
    #        self._plot_distr_2d(opts)
    #    elif(opts['dataset'] == 'mnist'):
    #        sampled = self._sample_from_training(opts,data, self._data_weights, 50)
    #        metrics = Metrics()
    #        metrics._return_plots_pics(opts, opts['number_of_steps_made'], data.data, sampled, 50,  self._data_weights, prefix = "train")
    #        self.tsne_plotter(opts,  data)

    def _sample_from_training(self, opts, data, probs, num_pts):
        """sample num_pts from the training set with importance sampling"""
        data_ids = np.random.choice(self._data_num, num_pts,replace=True, p=probs[:,0])
        sampled = data.data[data_ids].astype(np.float)
        
        for k in range(1,self._number_of_kGANs):
            data_ids = np.random.choice(self._data_num, num_pts,replace=True, p=probs[:,k])
            sampled = np.concatenate((sampled, data.data[data_ids].astype(np.float)),axis = 0)
        return sampled


    def _prob_data_under_gan_internal(self, (opts, data, k)):
        #return self._kGANs[k].loss_pt(opts)*(-1.)
        device = k%opts['number_of_gpus']
        # select gan k
        gan_k = self._kGANs[k]
        # discriminator output for training set 
        #D_k = self._get_prob_real_data(opts, gan_k, self._k_class_graphs[k], data, device,k)
        D_k = self._get_prob_real_data(opts, gan_k, data, device,k)
        # probability x_i given gan_j
        p_k = (np.transpose((1. - D_k)/(D_k + 1e-12)) + 1e-12) 
            
        return p_k / p_k.sum(keepdims = True)

    def _prob_data_under_gan(self, opts, data):
        """compute p(x_train | gan j) for each gan and store it in self._prob_x_given_gan
        Could be done more efficiently by appending columns to empty array
        Returns:
        (data.num_points, opts['number_of_kGANs']) NumPy array

        """

        #prob_x_given_gan = np.empty([data.num_points, opts['number_of_kGANs']])
        #for k in range (0,self._number_of_kGANs):
        #import pdb
        #pdb.set_trace()
        pool = ThreadPool(opts['number_of_gpus'])
        job_args = [(opts, data, i) for i in range(0,self._number_of_kGANs)]
        prob_x_given_gan = pool.map(self._prob_data_under_gan_internal, job_args)
        pool.close()
        pool.join()
        #pdb.set_trace()
        return np.transpose(np.squeeze(np.asarray(prob_x_given_gan)))
        
    def _get_prob_real_data(self, opts, gan, data, device,k):
    #def _get_prob_real_data(self, opts, gan, data, device,k):
        """Train a classifier, separating weighted true data from the current gan.
        Returns:
        (data.num_points,) NumPy array, containing probabilities of 
        true data. I.e., output of the sigmoid function. Create a graph and train a classifier"""
        if opts['reinit_class'] == False:
            graph = self._k_class_graphs[k]
            with graph.as_default():
                with tf.device('/device:GPU:%d' %device):
                    #self._data_weights[:,k] = np.on
                    if opts['test'] == False:
                        self._k_class[k]._data_weights = self._data_weights[:,k]
                        num_fake_images = data.num_points
                        fake_images = gan.sample(opts, num_fake_images)
                        prob_real, prob_fake = self._k_class[k].train_mixture_discriminator(opts, fake_images)
                    else: 
                        data_weights = np.ones(len(data.data))
                        data_weights /= data_weights.sum(axis = 0, keepdims = True)
                        self._k_class[k]._data_weights = data_weights
                        prob_real = self._k_class[k].classify(opts,self.data_test.data)
            return prob_real
        
        
        else:
            g = tf.Graph()
            with g.as_default():
                with tf.device('/device:GPU:%d' %device):
                    #if opts['test'] == False:
                    classifier = self._classifier(opts, data, self._data_weights[:,k])
                    #else:
                    #    data_weights = np.ones(len(data.data))
                    #    data_weights /= data_weights.sum(axis = 0, keepdims = True)
                    #    classifier = self._classifier(opts, data, data_weights)
                    num_fake_images = data.num_points
                    fake_images = gan.sample(opts, num_fake_images)
                    prob_real, _  = classifier.train_mixture_discriminator(opts, fake_images)
                
                    if opts['test'] == True:
                        prob_real = classifier.classify(opts,self.data_test.data)
            return prob_real
    

    def _plot_distr_2d(self, opts):
        """ plot histograms of the distributions for 2d gmm, plot is a bit ugly"""
        import matplotlib 
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        max_val = opts['gmm_max_val'] * 2
        plt.clf()
        fig = plt.figure()
        for k in range(0,opts['number_of_kGANs']):
            g1 = fig.add_subplot(opts['number_of_kGANs'],1,k+1)
            g1.axis([-max_val, max_val, -max_val, max_val])
            fake = self._kGANs[k].sample(opts, 60000)
            g1.hist2d(fake[:,0].flatten(), fake[:,1].flatten(), bins=(50, 50), cmap=plt.cm.jet)
            g1.set_xlim(-max_val, max_val)
            g1.set_ylim(-max_val, max_val)
        filename = opts['work_dir'] + '/distr_heatmap_step{:05d}.png'.format(opts['number_of_steps_made'])
        fig.savefig(filename)
        plt.close()

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

        filename = opts['work_dir'] + '/competition_heatmap_step{:05d}.png'.format(opts['number_of_steps_made'])
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
    
    def _sample_from_training_gan_label_real_fake(self, opts, data, probs, num_pts):
        """labels are the real fake, sample from the training set
        uniformly and from the mixture"""
        data_ids = np.random.choice(self._data_num, num_pts,replace=False)
        sampled = data.data[data_ids].astype(np.float)
        labels = np.zeros(num_pts)
        fake_points = self.sample_mixture(opts, num_pts)
        num_fake = len(fake_points)
        sampled = np.concatenate((sampled, fake_points),axis = 0)
        labels = np.append(labels, np.ones(num_fake))
        return sampled, labels
    def _sample_from_training_gan_label(self, opts, data, probs, num_pts):
        """labels is the gan idx, sample from the training set
        with importance sampling using the probability of each gan and
        remember from which gan do the samples come from """
        data_ids = np.random.choice(self._data_num, num_pts,replace=False, p=probs[:,0])
        sampled = data.data[data_ids].astype(np.float)
        labels = np.zeros(num_pts)
        for k in range(1,self._number_of_kGANs):
            data_ids = np.random.choice(self._data_num, num_pts,replace=False, p=probs[:,k])
            sampled = np.concatenate((sampled, data.data[data_ids].astype(np.float)),axis = 0)
            labels = np.append(labels, np.ones(num_pts)*k)
        return sampled, labels
    def _mnist_label_proportions(self, opts, data, metrics):
        rez = np.zeros(10)

        for k in range(0, self._number_of_kGANs):
            num_pts = int(self._hard_assigned[0][k] )
            if num_pts >0:
                data_ids = np.random.choice(self._data_num, num_pts,replace=False, p=self._kGANs[k]._data_weights)
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

    def tsne_real_fake(self, opts,  data):
        """tsne plotter, not particularly interesting"""
        #return True 
        sampled,labels = self._sample_from_training_gan_label_real_fake(opts,data, self._data_weights, 500)
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

        plt.scatter(vis_x, vis_y, c=labels.astype(int), cmap=plt.cm.get_cmap("jet", 2))
        plt.colorbar(ticks=range(2))
        #plt.clim(-0.5, 9.5)
        filename = opts['work_dir'] + '/tsne{:04d}.png'.format(opts['number_of_steps_made'])
        fig.savefig(filename)
        plt.close()
    def tsne_plotter(self, opts,  data):
        """tsne plotter, not particularly interesting"""
        #return True 
        sampled,labels = self._sample_from_training_gan_label(opts,data, self._data_weights, 500)
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
    def test_mnist(self, opts, data, data_test):
        opts['test'] = True
        prob_x_given_gan = self._prob_data_under_gan(opts, data)
        opts['test'] = False
        new_weights = np.ones(prob_x_given_gan.shape)*0.
        new_weights[np.arange(len(prob_x_given_gan)), prob_x_given_gan.argmax(1)] = 1.
        tot_loss = 0.
        for k in range(0,opts['number_of_kGANs']):
            assigned = self.data_test.data[np.argwhere(new_weights[:,k]==1).flatten()]
            if assigned.shape[0] >= 1:
                self._kGANs[k].test_data = assigned
                loss, _, _ = self._kGANs[k].test(opts)
                tot_loss += loss*assigned.shape[0]/new_weights.shape[0]
        return tot_loss


