import numpy as np
# import fancyimpute
from sklearn import preprocessing
from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute, MICE, IterativeSVD


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


def count_missing(feature_matrix, value=0):
    count = 0
    for i in feature_matrix:
        for j in i:
            if j == value:
                count += 1
    return count


def create_synth(feature_matrix):
    rowcount = []
    for i in feature_matrix:
        rowcount.append(0)
        for j in i:
            if j == 0:
                rowcount[len(rowcount) - 1] += 1
    colcount = []
    for i in feature_matrix.T:
        colcount.append(0)
        for j in i:
            if j == 0:
                colcount[len(colcount) - 1] += 1
    rowcount = np.asarray(rowcount)
    colcount = np.asarray(colcount)
    rows = np.argsort(rowcount)
    cols = np.argsort(colcount)
    count = 0
    temp_synth_data = []
    for i in rows:
        if count < 50:
            temp_synth_data.append(feature_matrix[i])
            count += 1
        else:
            break
    temp_synth_data = np.asarray(temp_synth_data).T
    synth_data = []
    count = 0
    for i in cols:
        if count < 50:
            synth_data.append(temp_synth_data[i])
            count += 1
        else:
            break
    synth_data = np.asarray(synth_data).T
    return synth_data


def introduce_miss():
    synth_file = open('synthetic_data.txt', 'r')
    synth_data = np.loadtxt(synth_file, delimiter=' ')
    synth_file.close()
    perc = 100
    while perc < 40 or perc > 42:
        rand1 = np.random.randint(3, size=(50, 50))
        rand2 = np.random.randint(3, size=(50, 50))
        rand = np.multiply(rand1, rand2)
        count = 0
        for i in rand:
            for j in i:
                if j != 0:
                    count += 1
        perc = (count * 100.0)/float(50*50)
    for i in range(rand.shape[0]):
        for j in range(rand.shape[0]):
            if rand[i][j] == 0:
                rand[i][j] = 1
            else:
                rand[i][j] = 0
    synth_miss = np.multiply(synth_data, rand)
    miss_file = open('synthetic_missing.txt', 'w')
    np.savetxt(miss_file, synth_miss, delimiter=' ')
    miss_file.close()


def impute_error(full_data, missing_data):
    return np.abs(np.sum(np.abs(full_data) - np.abs(missing_data)))


def impute(feature_matrix, method='All'):
    feature_matrix[feature_matrix == 0] = np.NaN
    # imputer = preprocessing.Imputer(missing_values=0, strategy='mean')
    if method == 'All':
        feature_matrix_knn = KNN(k=5).complete(feature_matrix)
        feature_matrix_nnm = NuclearNormMinimization().complete(feature_matrix)
        feature_matrix_soft = SoftImpute().complete(BiScaler().fit_transform(feature_matrix))
        feature_matrix_mice = MICE().complete(feature_matrix)
        feature_matrix_isvd = IterativeSVD().complete(feature_matrix)
        return (feature_matrix_knn, 'KNN'), (feature_matrix_nnm, 'NNM'), (feature_matrix_soft, 'SoftImpute'), \
               (feature_matrix_mice, 'MICE'), (feature_matrix_isvd, 'Iterative SVD')
    else:
        if method == 'Iterative SVD':
            return IterativeSVD().complete(feature_matrix)
        elif method == 'NNM':
            return NuclearNormMinimization().complete(feature_matrix)
        elif method == 'SoftImpute':
            return SoftImpute().complete(BiScaler().fit_transform(feature_matrix))
        elif method == 'MICE':
            return MICE().complete(feature_matrix)
        else:
            return KNN(k=10).complete(feature_matrix)
    # feature_matrix_biscalar = BiScaler().fit_transform()
    # feature_matrix_sk = imputer.fit_transform(feature_matrix)


#
# def gmm(feature_matrix):
#


def main():
    feature_matrix = get_data()
    synth_data = create_synth(feature_matrix)
    synth_file = open('synthetic_data.txt', 'w')
    np.savetxt(synth_file, synth_data, delimiter=' ')
    synth_file.close()
    introduce_miss()
    random_file = open('synthetic_missing.txt', 'r')
    synth_missing = np.loadtxt(random_file, dtype=np.float64, delimiter=' ')
    random_file.close()
    imputed_datasets = impute(synth_missing)
    impute_score = np.inf
    method = ''
    impute_score_list = []
    for i in imputed_datasets:
        if i[1] == 'SoftImpute':
            temp_score = impute_error(BiScaler().fit_transform(synth_data), i[0])
        else:
            temp_score = impute_error(synth_data, i[0])
        impute_score_list.append((temp_score, i[1]))
        if temp_score < impute_score:
            method = i[1]
            impute_score = temp_score
    feature_matrix_imputed = impute(feature_matrix, method)
    print "These are the imputation methods considered and their respective scores(The smaller the better):"
    for i in impute_score_list:
        print "Method: " + i[1] + ", Score: " + str(i[0])
    data_file = open('data_imputed.txt', 'w')
    np.savetxt(data_file, feature_matrix_imputed, delimiter=' ')
    data_file.close()


if __name__ == '__main__':
    main()
