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

output1 = open('Correlation.txt', 'w')
output2 = open('Glasso.txt', 'w')
output3 = open('Random_Correlation.txt', 'w')
output4 = open('Random_Glasso.txt', 'w')

output1.write("For comparison using Correlation Matrix:")
for i, label in enumerate(correlation_label_net):
    for j, network in enumerate(label):
        output1.write("Label " + str(i))
        output1.write("Network (threshold): " + str(j))
        compare(network, gold, output1)

output2.write("For comparison using Graphical Lasso Network:")
for i, label in enumerate(glasso_label_net):
    for j, network in enumerate(label):
        output2.write("Label " + str(i))
        output2.write("Network (threshold): " + str(j))
        compare(network, gold, output2)

correlation_label_net = np.load('random_correlation_network.npy')
glasso_label_net = np.load('random_glasso_network.npy')

output3.write("For comparison using Correlation Matrix with random labels:")
for i, label in enumerate(correlation_label_net):
    for j, network in enumerate(label):
        output3.write("Label " + str(i))
        output3.write("Network (threshold): " + str(j))
        compare(network, gold, output3)


output4.write("For comparison using Graphical Lasso Network with random labels:")
for i, label in enumerate(glasso_label_net):
    for j, network in enumerate(label):
        output4.write("Label " + str(i))
        output4.write("Network (threshold): " + str(j))
        compare(network, gold, output4)


output1.close()
output2.close()
output3.close()
output4.close()
