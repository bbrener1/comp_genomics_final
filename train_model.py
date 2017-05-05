#!/usr/bin/env python

from __future__ import division
import sys
import numpy as np
from numpy.linalg import inv
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from sklearn.covariance import GraphLasso
from scipy.stats import pearsonr
# import pyensembl as en

from datareader_server_var import *
from glasso import *

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

imputed = np.loadtxt('imputed_counts.txt', dtype=np.float64)

reduced = count_PCA(imputed)

labels = GMM(reduced, "params_and_bic.txt")

uniq = np.unique(np.asarray(labels))
random_labels = rand1 = np.random.randint(len(uniq), size=(np.asarray(labels).shape[0],))


# labels = GMM(reduced)

# label_net = correlation_matrix(imputed, labels, "correlation_matrix.txt")

corr_label_net = correlation_matrix(imputed, labels)

# glasso_label_net = glasso_net(imputed, labels)

random_corr_label_net = correlation_matrix(imputed, random_labels)

# random_glasso_label_net = glasso_net(imputed, random_labels)

np.save('correlation_network.npy', corr_label_net, allow_pickle=True)
np.save('random_correlation_network.npy', random_corr_label_net, allow_pickle=True)
np.save('gmm_labels.npy', labels, allow_pickle=True)
np.save('random_labels.npy', random_labels, allow_pickle=True)
# np.save('random_glasso_network.npy', random_glasso_label_net, allow_pickle=True)
# np.save('glasso_network.npy', glasso_label_net, allow_pickle=True)
