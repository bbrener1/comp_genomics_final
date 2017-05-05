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
from impute import *

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

in_data = open(sys.argv[1])

header = np.asarray(in_data.readline().split())

counts = np.loadtxt(sys.argv[1], skiprows=1, usecols=np.arange(27, stop=len(header)))


print "Samples, Features"
print counts.shape

# imputed = impute(counts,"imputed_counts.txt")
imputed = test_and_impute(counts)

batch_check(imputed, map(lambda x: x.split()[25], open(sys.argv[2]).readlines()[1:]))

np.savetxt('data_imputed.txt', imputed, delimiter=' ')
np.savetxt('header.txt', header, delimiter=' ')
