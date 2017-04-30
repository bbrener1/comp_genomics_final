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

reduced = count_PCA(imputed)

labels = GMM(reduced, "params_and_bic.txt")

# labels = GMM(reduced)

label_net = correlation_matrix(imputed, labels)

gold = translate_gold_standard(sys.argv[3],header)
for label in label_net:
    for network in label:

        compare(network,gold)
