import numpy as np
from sklearn.covariance import GraphLasso
from joblib import Parallel, delayed
from sklearn import preprocessing
import multiprocessing


def get_data():
    data_file = open('data.txt', 'r')
    data_file.readline()
    line = data_file.readline()
    feature_matrix = []
    while line:
        feature_matrix.append(line.strip().split()[27:])
        line = data_file.readline()
    data_file.close()
    feature_matrix = np.asarray(feature_matrix, dtype=np.float64)
    data_file.close()
    return feature_matrix


def parallelize(tup, feature_matrix_by_label):
    label = tup[0]
    penalty = tup[1]
    count = tup[2]
    graphical_model = GraphLasso(alpha=penalty, tol=(10.00 ** -4))
    # print feature_matrix_by_label[label]
    graphical_model.fit(feature_matrix_by_label)
    precision_matrix = graphical_model.precision_
    membership_matrix = np.full([len(precision_matrix), len(precision_matrix)], 0.0)
    for row in range(0, len(precision_matrix)):
        for col in range(0, len(precision_matrix)):
            if precision_matrix[row][col] != 0:
                membership_matrix[row][col] = 1
    # four_D_matrix[label][count] = membership_matrix
    print "Job Completed!"
    return membership_matrix


def graph_lasso(counts, labels):
    if len(labels) != len(counts):
        print "The number of labels and counts do not match"
    number_of_labels = len(np.unique(labels))
    unique, count = np.unique(labels, return_counts=True)
    # print count
    feature_matrix_by_label = []
    # initializes a list containing feature matrices segregated by label
    for i in range(0, number_of_labels):
        feature_matrix_by_label.append(np.empty([0, len(counts[0])], dtype=np.float64))
    # assuming zero indexing of labels
    for i in range(0, len(labels)):
        feature_matrix_by_label[labels[i]] = np.asarray(np.append(feature_matrix_by_label[labels[i]], counts[i]))
    for i in range(0, number_of_labels):
        feature_matrix_by_label[i] = np.reshape(feature_matrix_by_label[i], (count[i], len(counts[0])))
    penalty_list = []
    for i in range(number_of_labels):
        for j in range(20, 100, 5):
            penalty_list.append((i, float(j) / 100, len(penalty_list)))
    four_D_matrix = np.full((number_of_labels, 16, len(counts[0]), len(counts[0])), 0.0)
    no_of_cpus = multiprocessing.cpu_count()
    if no_of_cpus > len(penalty_list):
        temp_matrix = Parallel(n_jobs=len(penalty_list))(delayed(parallelize)(i, feature_matrix_by_label[i[0]]) for i in penalty_list)
    else:
        temp_matrix = Parallel(n_jobs=no_of_cpus)(delayed(parallelize)(i, feature_matrix_by_label[i[0]]) for i in penalty_list)
    for i in range(len(temp_matrix)):
        four_D_matrix[penalty_list[i][0]][penalty_list[i][2] % 16] = temp_matrix[i]
    return four_D_matrix


def glasso_net(feature_matrix, labels):
    # data_file = open('data_imputed.txt', 'r')
    # feature_matrix = np.loadtxt(data_file, delimiter=' ', dtype=np.float64)
    # model = preprocessing.StandardScaler()
    # feature_matrix = model.fit_transform(feature_matrix)
    # labels = []
    # labels_file = open('gmm_labels.txt', 'r')
    # labels_line = labels_file.readline().strip()
    # while labels_line:
    #     labels.append(int(labels_line.split('.')[0]))
    #     labels_line = labels_file.readline().strip()
    fourdmatrix = graph_lasso(feature_matrix, labels)
    return fourdmatrix

