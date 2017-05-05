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

in_data = open(sys.argv[2])

header = in_data.readline().split()

counts = np.loadtxt(sys.argv[2], skiprows=1, usecols=np.arange(27, stop=len(header)))


print "Samples, Features"
print counts.shape

imputed = test_and_impute(counts)

batch_check(imputed, map(lambda x: x.split()[25], open(sys.argv[2]).readlines()[1:]))


in_data.seek(0)
in_data_list = in_data.readlines()[1:]
print "Batch OUTER DEBUG"
print len(in_data_list)
print header[24]

batch_check(imputed, map(lambda x: x.split()[24],in_data_list))


reduced = count_PCA(imputed)

labels = GMM(reduced, "params_and_bic.txt")

# labels = GMM(reduced)

# label_net = correlation_matrix(imputed, labels, correlation_presolve = "correlation_matrix.txt")
#
label_net = correlation_matrix(imputed, labels)

gold = translate_gold_standard(sys.argv[3], header)

print "correlation_matrix debug"
print label_net.shape


# for i, label in enumerate(label_net[:-1]):
#     for j, network in enumerate(label):
#         print i
#         print j
#
#
#
for i in range(label_net.shape[1]):
    compare(map(lambda x: x,label_net[:,i,:,:]))

similar_genes = np.zeros(counts.shape[1],dtype=bool)

sub_pop_counts = np.zeros((len(similar_genes),len(labels)))
for i in range(len(labels)):
    sub_pop_counts[:,i] = (np.mean(imputed[labels == i],axis=0))
print "Sub pop counts"
print sub_pop_counts.shape
print sub_pop_counts[:5,:3]

for i in range(len(similar_genes)):
    print sub_pop_counts[i,:3]
    print np.std(sub_pop_counts[i,:3])
    print (np.mean(sub_pop_counts[i,:3])*.2)
    if np.std(sub_pop_counts[i,:3]) < (np.mean(sub_pop_counts[i,:3])*.2):
        similar_genes[i] = True

print similar_genes
print np.sum(similar_genes)

similar_gene_net = label_net[:,:,:,similar_genes]
similar_gene_net = similar_gene_net[:,:,similar_genes,:]

print similar_gene_net.shape

for i in range(label_net.shape[1]):
    compare(map(lambda x: x,similar_gene_net[:,i,:,:]))
