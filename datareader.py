#!/usr/bin/env python
from __future__ import division
import sys
import numpy as np
from numpy.linalg import inv
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from sklearn.covariance import GraphLasso
from scipy.stats import pearsonr
import pyensembl as en
from sklearn.decomposition import PCA
from sklearn.covariance import GraphLasso

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

in_data = open(sys.argv[1])

header = in_data.readline().split()

print header[:28]

counts = np.loadtxt(sys.argv[1], skiprows=1, usecols=np.arange(27,stop=len(header)))

print counts[:5,:5]

#
# nonzero_means = np.zeros(zero_percents.shape[0])
# for i, column in enumerate(counts.T):
#     nonzero_means[i] = np.mean(column[column != 0])

# print pearsonr(zero_percents,nonzero_means)

def impute(counts):

    imputed_counts = np.zeros(counts.shape)

    for i, column in enumerate(counts):

        imputation_model = LinearRegression()
        imputation_model.fit(np.delete(counts, i, axis=1)[counts[:,i]!=0],counts[:,i][counts[:,i]!=0])
        for j, row in enumerate(counts):
            if counts[i,j] == 0:
                imputed_counts[i,j] = imputation_model.predict(np.delete(row,i))
                print "J = " + str(j)
                print "I = " + str(i)

    imputed_counts[imputed_counts==0] = counts[imputed_counts==0]

    return imputed_counts

    print imputed_counts[:8,:8]

    # twice_imputed = np.zeros(counts.shape)
    #
    # for i, column in enumerate(counts):
    #
    #     imputation_model = LinearRegression()
    #     imputation_model.fit(np.delete(counts, i, axis=1)[imputed_counts[:,i]!=0],imputed_counts[:,i][counts[:,i]!=0])
    #     for j, row in enumerate(imputed_counts):
    #         if counts[i,j] == 0:
    #             twice_imputed[i,j] = imputation_model.predict(np.delete(row,i))
    #             print "J = " + str(j)
    #             print "I = " + str(i)
    #
    # twice_imputed[twice_imputed==0] = imputed_counts[twice_imputed==0]
    #
    # zero_percents = np.zeros(counts.shape[1])
    #
    # for i, column in enumerate(imputed_counts.T):
    #     zero_percents[i] = float(np.sum(column == 0))/float(counts.shape[1])
    #
    # print zero_percents



def translate_gold_standard(in_data,labels):

    ens_obj = en.EnsemblRelease(species='mouse')

    for i in range(len(labels)):
        try:
            labels[i] = ens_obj.gene_name_of_gene_id(labels[i])
        except ValueError:
            labels[i] = "error"

    if not isinstance(in_data, file):
        in_data = open(in_data)

    set_of_interest = set(labels)

    network_matrix = np.zeros((len(labels),len(labels)))

    for line in in_data:
        if line.split()[0] in set_of_interest:
            try:
                network_matrix[labels.index(line.split()[0]),labels.index(line.split()[2])] = 1
            except ValueError:
                continue

    return network_matrix

def count_PCA(imputed):
    model = PCA(n_components=imputed.shape[1])

    reduced = model.fit_transform(imputed)

    plt.figure()
    plt.bar(model.explained_variance_ratio_)
    plt.savefig("PCAexplanatorypower.png")

    return reduced

def GMM(counts):
    param_list = []
    bic_list = []
    for i in range(50):
        model = GaussianMixture(n_components=i)
        model.fit(counts)
        param_list.append(model.get_params())
        bic_list.append(model.bic(counts))

    return param_list[bic_list.index(max(bic_list))], bic_list.index(max(bic_list))

def lasso(counts):
    network_set = np.zeros((20,counts.shape[1],counts.shape[1]))
    for i in map(lambda x: float(x)*.01, range(0,100,5)):
        glasso = GraphLasso()
        covariance = np.cov(counts)
        precision = glasso.fit(covariance)


def compare(network1, network2):
    pass

gold = translate_gold_standard(sys.argv[2],header)

translate_gold_standard(sys.argv[2],header[28:])

plt.figure()
plt.hist(map(lambda x: np.sum(x),gold), bins=20, range=(0,100),log=True)
plt.savefig("degree_histogram.png")


#
# mixture_model_list = []
# bic_list = []
#
# for i in range(1,20):
#     mixture_model_list.append(GaussianMixture(n_components=i))
#     mixture_model_list[-1].fit(imputed_counts)
#     bic_list.append(mixture_model_list[-1].bic(imputed_counts))
