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

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

in_data = open(sys.argv[2])

header = in_data.readline().split()

counts = np.loadtxt(sys.argv[2], skiprows=1, usecols=np.arange(27,stop=len(header)))


print "Samples, Features"
print counts.shape

imputed = impute(counts,"imputed_counts.txt")

# imputed = impute(counts)

in_data.seek(0)
in_data_list = in_data.readlines()[1:]
print "Batch OUTER DEBUG"
print len(in_data_list)
print header[24]

batch_check(imputed, map(lambda x: x.split()[24],in_data_list))


reduced = count_PCA(imputed)

labels = GMM(reduced, "params_and_bic.txt")

# labels = GMM(reduced)

label_net = correlation_matrix(imputed, labels, correlation_presolve = "correlation_matrix.txt")
#
# label_net = correlation_matrix(imputed, labels)

gold = translate_gold_standard(sys.argv[3],header)

print "correlation_matrix debug"
print label_net.shape

gseapy_analysis(label_net[1,12], header)

# for i, label in enumerate(label_net[:-1]):
#     for j, network in enumerate(label):
#         print i
#         print j
#
#
#
# for i, label in enumerate(label_net):
#     for j, network in enumerate(label):
#         print "Label " + str(i)
#         print "Network (threshold): " + str(j)
#         print network.shape
#         print i
#         print j
#
#         compare(network,gold)
