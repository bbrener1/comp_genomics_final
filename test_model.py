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

header = np.loadtxt('header.txt').tolist()

gold = translate_gold_standard(sys.argv[1], header)

correlation_label_net = np.load('correlation_network.npy')
glasso_label_net = np.load('glasso_network.npy')

output = open('Comparison.txt', 'w')

output.write("For comparison using Correlation Matrix:")
for i, label in enumerate(correlation_label_net):
    for j, network in enumerate(label):
        output.write("Label " + str(i))
        output.write("Network (threshold): " + str(j))
        compare(network, gold, output)

output.write("\n\n\nFor comparison using Graphical Lasso Network:")
for i, label in enumerate(glasso_label_net):
    for j, network in enumerate(label):
        output.write("Label " + str(i))
        output.write("Network (threshold): " + str(j))
        compare(network, gold, output)

output.close()
