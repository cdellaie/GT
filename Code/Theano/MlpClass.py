import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T

__docformat__ = 'restructedtext en'

##############
##  DATA    ##
##############

#chargement des donnees. dataset : string, param dataset : path to dataset
#utilisation du package zip qui permet de lire directement les images depuis un fichier zip
def load_data(dataset):

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(os.path.split(__file__)[0], "..", "data", dataset)
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = 'https://github.com/cdellaie/GT/tree/master/MNIST/mnist.pkl.gz'
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... chargement des donnees'

    # chargement
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input : numpy.ndarray of 2 dimensions (matrix) 1 colonne = 1 exemple.
    # target : numpy.ndarray of 1 dimensions (vector) de longueur le nombre de colonne de l'output

    def shared_dataset(data_xy, borrow=True):
    #charge les donnees dans des shared variables
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

##########################################
######## CLASSE REGRESSION LOGISTIQUE   ##
##########################################

class LogisticRegression(object):

    def __init__(self, input, n_in, n_out): 
        # classe decrite par W et b , initialisation a 0       
        #input : theano.tensor
        #n_in et n_out : int

        self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX),
                                name='W', borrow=True)
        
        self.b = theano.shared(value=numpy.zeros((n_out,),
                                                 dtype=theano.config.floatX),
                               name='b', borrow=True)

        # calcul de la prediction : classe qui maximise p(y|x)
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parametres
        self.params = [self.W, self.b]

#retourne la logvraisemblance moyenne de la prediction sous une cible donnee
#y : tensor type
    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

#retourne un float = nombre d'erreurs dans le minibatch/nombre total d'exemples dans le minibatch
#y = theano.tensor.Tensortype
    def errors(self, y):
        # verification sur les dimensions et les types
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

##########################################
#######     HIDDEN LAYER                ##
##########################################

#fonction d'activation sigmoide
# matrice des poids W de taille (n_in,n_out)
# vecteur de biais b de taille (n_out,)
# activation de la couche cachee : tanh( input*W + b) 
# rng : numpy.random.RandomState , input : theano.tensor.dmatrix , n_in et n_out : int ,activation : theano.Op ou fonction

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        self.input = input
        # W initialise avec des valeurs uniformes sur sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        # parametres
        self.params = [self.W, self.b]

##########################################
#######     MultiLayerPerceptron    ######
##########################################

class MLP(object):

    # couches intermediaires : definies par la classe HiddenLayer, de fct d'activation tanh ou sigm
    # couches finale : softmax definies par la class LogisticRegression, qui permet d'interpreter 
    # l'output de facon probabiliste

    # rng: numpy.random.RandomState , input: theano.tensor.TensorType , n_in n_hidden n_out: int

    def __init__(self, rng, input, n_in, n_hidden, n_out):

        # data -> hiddenlayer avec tanh activation -> LogisticRegressionLayer
        self.hiddenLayer = HiddenLayer(rng=rng, input=input,
                                       n_in=n_in, n_out=n_hidden,
                                       activation=T.tanh)

        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out)

        # norme L1 et norme L2 au carree 
        self.L1 = abs(self.hiddenLayer.W).sum() \
                + abs(self.logRegressionLayer.W).sum()

        self.L2_sqr = (self.hiddenLayer.W ** 2).sum() \
                    + (self.logRegressionLayer.W ** 2).sum()

        # -LogVraisemblancenegative = LogVraisemblance des output calcules dans la dernier couche
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        self.errors = self.logRegressionLayer.errors
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params