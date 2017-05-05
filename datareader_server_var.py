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

import gseapy

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
                # print "GOLD ERROR"
                # print line
                # try:
                #     labels[27:].index(line.split()[0])
                # except ValueError:
                #     print "Couldn't find " + line.split()[0]
                # try:
                #     labels[27:].index(line.split()[2])
                # except ValueError:
                #     print "Couldn't find " + line.split()[2]
                # # error_count += 1
                continue

    np.savetxt("gold_network.txt",network_matrix)

    print "Gold Standard Debug"
    print np.sum(network_matrix.flatten())
    print "GOLD STANDARD ERRORS"
    print error_count

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
        np.savetxt('gmm_labels.txt',model.predict(counts))

        return model.predict(counts)

    param_list = []
    bic_list = []
    for i in range(1,20):
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
    np.savetxt('gmm_labels.txt',model.predict(counts))

    return model.predict(counts)

# def agglomerative(counts, pre_solved = None):
#     if presolved != None:
#         pass


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


def correlation_matrix(counts,labels,correlation_presolve=None):
    label_net = []
    print "Correlation Debug"
    print counts.shape
    print set(labels)

    label_net = np.zeros((max(labels.flatten())+1,16,counts.shape[1],counts.shape[1]))

    for label in range(max(labels)+1):
        print "Label size for label " + str(label)
        print sum(labels == label)

    for label, _ in enumerate(label_net):
        if correlation_presolve == None:
            filter_array = labels == label
            correlations = np.zeros((counts[filter_array].shape[1],counts[filter_array].shape[1]))
            print "Correlation dim"
            print correlations.shape
            correlations = np.corrcoef(counts[filter_array].T).T

        network_set = np.zeros((16,counts.shape[1],counts.shape[1]))

        for i,j in enumerate(map(lambda x: float(x)*.01, range(84,100,1))):
            network_set[i] = (correlations > j).astype(dtype=int)
            print "test correlation"
            print i
            print j
            print np.sum(network_set[i].flatten())
        label_net[label] = network_set


    return label_net

def batch_check(counts, batch_labels):
    batch_label_index = list(set(batch_labels))
    batch_array = np.asarray(map(lambda x: batch_label_index.index(x),batch_labels))
    print "Batch Check Debug"
    print counts.shape
    print batch_array.shape

    split = np.random.randint(0,2, size=(counts.shape[0]))

    train_exp = counts[split == 0]
    train_batch = batch_array[split == 0]

    test_exp = counts[split == 1]
    test_batch = batch_array[split == 1]

    for i in range(len(batch_label_index)):
        filter_array = (train_batch == i)
        model = LinearRegression()
        model.fit(train_exp, filter_array.astype(dtype=int))
        print "Batch label index"
        print i
        print "Batch label"
        print batch_label_index[i]
        print "Score"
        print model.score(test_exp, (test_batch == i).astype(dtype=int))

def compare(networks):

    for i, network in enumerate(networks):
        print str(np.sum((network != 0).flatten())) + " edges present in network " + str(i)
        print str(np.sum((network != 0).flatten())/(network.shape[0]*network.shape[1])) + " percent of all possible edges"
    # print str(np.sum((network2 != 0).flatten())) + " edges present in network 2"
    # print str(np.sum((network2 != 0).flatten())/(network2.shape[0]*network2.shape[1])) + " percent of all possible edges"
    shared_net = np.ones(networks[0].shape)
    for i, network in enumerate(networks):
        shared_net = np.logical_and(shared_net,network)

    print "There are " + str(np.sum(shared_net.flatten())) + " universal edges"
    for i, network in enumerate(networks):
        print str(float(np.sum((shared_net != 0).flatten()))/(float(np.sum(network.flatten())))) + " percent of network " + str(i) + " is shared."



def gseapy_analysis(network,header):
    top_20 = network.T[np.argsort(np.sum(network, axis=1)) > network.shape[1]-20]
    top_20_indecies = np.arange(network.shape[1])[np.argsort(np.sum(network, axis=1)) > network.shape[1]-20]
    "GSEAPY DEBUG"
    print top_20.shape
    print np.sum(top_20)
    for gene in top_20:
        genes = []
        for j, edge in enumerate(gene):
            if header[j] != "error" and edge > 0:
                genes.append(header[j])

        print genes
        print len(genes)
        gseapy.enrichr(gene_list=genes, description='pathway', gene_sets='KEGG_2016', outdir='test')


        # gene_list = []
        # for i, edge in enumerate(gene):
        #     if edge > 0:
        #         gene_list.append(header[i])



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
