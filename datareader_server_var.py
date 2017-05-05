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
from sklearn.covariance import GraphLassoCV
from sklearn.preprocessing import scale

import warnings
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# in_data = open(sys.argv[1])
#
# header = in_data.readline().split()
#
# print header[:28]
#
# counts = np.loadtxt(sys.argv[1], skiprows=1, usecols=np.arange(27,stop=len(header)))
#
# print counts[:5,:5]

#
# nonzero_means = np.zeros(zero_percents.shape[0])
# for i, column in enumerate(counts.T):
#     nonzero_means[i] = np.mean(column[column != 0])

# print pearsonr(zero_percents,nonzero_means)

def impute(counts,passed_txt=None):

    if passed_txt != None:
        imputed_counts = np.loadtxt(passed_txt)
        print "Debug backup imputation"
        print imputed_counts.shape
        return imputed_counts

    imputed_counts = np.zeros(counts.shape)

    for i, row in enumerate(counts):

        imputation_model = LinearRegression()
        imputation_model.fit(np.delete(counts, i, axis=1)[counts[:,i]!=0],counts[:,i][counts[:,i]!=0])
        for j, column in enumerate(counts.T):
            if counts[i,j] == 0:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    imputed_counts[i,j] = imputation_model.predict(np.delete(row,i))
                if i%100 == 0 and j%100 == 0:
                    print "J = " + str(j)
                    print "I = " + str(i)

    imputed_counts[imputed_counts==0] = counts[imputed_counts==0]

    np.savetxt("imputed_counts.txt", imputed_counts)

    print "Imputed counts"
    print imputed_counts.shape

    return imputed_counts
    #
    # print imputed_counts[:8,:8]
    #
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
    error_count = 0

    for i in range(len(labels)):
        try:
            labels[i] = ens_obj.gene_name_of_gene_id(labels[i])
        except ValueError:
            labels[i] = "error"
            error_count += 1

    if not isinstance(in_data, file):
        in_data = open(in_data)

    set_of_interest = set(labels[27:])

    network_matrix = np.zeros((len(labels[27:]),len(labels[27:])))

    for line in in_data:
        if line.split()[0] in set_of_interest:
            try:
                network_matrix[labels[27:].index(line.split()[0]),labels[27:].index(line.split()[2])] = 1
            except ValueError:
                continue

    np.savetxt("gold_network.txt",network_matrix)

    return network_matrix

def count_PCA(imputed):
    model = PCA(n_components=imputed.shape[1])

    reduced = model.fit_transform(imputed)[:,:8]

    plt.figure()
    # plt.bar(np.arange(model.explained_variance_ratio_.shape[0]),model.explained_variance_ratio_)
    plt.bar(np.arange(20),model.explained_variance_ratio_[:20])
    plt.savefig("PCAexplanatorypower.png")

    print "Reduced feature space"
    print reduced.shape

    return reduced

def GMM(counts, pre_solved=None):

    if pre_solved != None:
        model = GaussianMixture(int(open(pre_solved).readline())+1)
        model.fit(counts)
        return model.predict(counts)

    param_list = []
    bic_list = []
    for i in range(1, 20):
        model = GaussianMixture(n_components=i)
        model.fit(counts)
        param_list.append(model.get_params())
        bic_list.append(model.bic(counts))
        print "GMM debug"
        print i
        print bic_list[-1]


    backup = open("params_and_bic.txt",mode='w')
    print param_list[bic_list.index(min(bic_list))]
#    backup.write(str(param_list[bic_list.index(max(bic_list))]))
    backup.write(str(bic_list.index(min(bic_list))))
    backup.close()


    model = GaussianMixture(n_components=bic_list.index(max(bic_list))+1)
    model.fit(counts)
    return model.predict(counts)


def lasso(counts, labels):
    label_net = []
    for label in range(max(labels.flatten())):
        network_set = np.zeros((20,counts.shape[1],counts.shape[1]))
        for i,j in enumerate(map(lambda x: float(x)*.01, range(20,100,5))):
            print "test lasso"
            print i
            print j
            glasso = GraphLasso(alpha=j)
            # covariance = np.cov(counts)
            standardized_counts = scale(counts)

            precision = glasso.fit(standardized_counts)
            network_set[i] = glasso.get_precision()
        label_net.append(network_set)
    return label_net


def correlation_matrix(counts,labels,correlations=None):
    label_net = []
    print "Correlation Debug"
    print counts.shape

    label_net = np.zeros((max(labels.flatten())+1,16,counts.shape[1],counts.shape[1]))

    for label, _ in enumerate(label_net):
        if correlations == None:

            correlations = np.zeros((counts.shape[1],counts.shape[1]))
            print "Correlation dim"
            print correlations.shape
            for x in range(counts.shape[1]):
                for y in range(counts.shape[1]):
                    # print "test iteration"
                    # print counts.shape

                    print x
                    print y
                    correlations[x,y] = pearsonr(counts[:,x],counts[:,y])[0]

            np.savetxt("correlation_matrix.txt",correlations)

        else:
            correlations = np.loadtxt(correlations)

        network_set = np.zeros((16,counts.shape[1],counts.shape[1]))

        for i,j in enumerate(map(lambda x: float(x)*.01, range(20,100,5))):
            network_set[i] = (correlations > j).astype(dtype=int)
            # print "test correlation"
            print i
            print j
        label_net[label]= network_set


    return label_net

def batch_check(counts, batch_labels):
    batch_label_index = list(set(batch_labels))
    batch_array = np.asarray(map(lambda x: batch_label_index.index(x),batch_labels))
    print "Batch Check Debug"
    print counts.shape
    print batch_array.shape
    for i in range(len(batch_label_index)):
        filter_array = (batch_array == i)
        model = LinearRegression()
        model.fit(counts, filter_array.astype(dtype=int))
        print "Batch label"
        print batch_labels
        print "Score"
        print model.score(counts, filter_array.astype(dtype=int))


def compare(network1, network2, output):

    output.write(str(np.sum((network1 != 0).flatten())) + " edges present in network 1")
    output.write(str(np.sum((network2 != 0).flatten())) + " edges present in network 2")
    output.write(str(np.sum(np.logical_and((network1 != 0),(network2 != 0))).flatten()) + " edges shared between the networks")

# gold = translate_gold_standard(sys.argv[2],header)
#
# translate_gold_standard(sys.argv[2],header[28:])
#
# plt.figure()
# plt.hist(map(lambda x: np.sum(x),gold), bins=20, range=(0,100),log=True)
# plt.savefig("degree_histogram.png")


#
# mixture_model_list = []
# bic_list = []
#
# for i in range(1,20):
#     mixture_model_list.append(GaussianMixture(n_components=i))
#     mixture_model_list[-1].fit(imputed_counts)
#     bic_list.append(mixture_model_list[-1].bic(imputed_counts))
