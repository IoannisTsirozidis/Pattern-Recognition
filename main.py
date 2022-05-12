import matplotlib
from keras.datasets import mnist
import numpy as np
import math

from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA

# UNNECESSARY
# def eucl_distance(a, b):  # sqrt((x-xo)**2 + (y-yo)**2)
#     return math.sqrt(math.pow((a[0] - b[0]), 2) + math.pow((a[1] - b[1]), 2))
from sklearn.preprocessing import StandardScaler

"""

a is a sample of len(a) dimensions, meaning for V=25 : a(k0,k1,k2,....,k18,k19,..k24)
b is also a sample of dimensions len(b) = len(a), because it is a sample of the same dataset.
returns a manhattan distance, based on the V dimensions. 

for example, for V=2: d = sqrt((x-xo)**2 + (y-yo)**2)

"""


def eucl_distance2(a, b):
    sum = 0
    for i in range(len(a)):
        sum += math.pow((a[i] - b[i]), 2)

    return math.sqrt(sum)


# UNNECESSARY
# works just fine for 2 features
# def maximin(data, clusters):
#     created_clusters = [data[0].tolist()]  # append to the result the first data_sample
#     distances = []
#     for i in range(np.size(data, 0)):
#         distances.append(eucl_distance(data[i], data[0]))
#
#     max_index = distances.index(max(distances))
#     created_clusters.append(data[max_index].tolist())  # append the furthest data_sample from the initial data_sample
#
#     no_cluster = 2
#
#     while no_cluster < clusters:
#         distances = []
#
#         for i in range(len(created_clusters)):
#             dist_i = []
#             for j in range(np.size(data, 0)):
#                 if created_clusters[i][0] != data[j][0] and created_clusters[i][1] != data[j][1]:
#                     dist_i.append(eucl_distance(created_clusters[i], data[j]))
#
#             minimum = min(dist_i)
#             distances.append(minimum)
#
#         max_index = distances.index(max(distances))
#         created_clusters.append(data[max_index].tolist())
#         no_cluster += 1
#
#     return created_clusters


# UNNECESSARY

# def min_non_zero(distances):
#     i=0
#     while distances[i] == 0:
#         i+=1
#
#     min = distances[i]
#     for j in range(len(distances)):
#         if distances[j] < min and distances[j]!=0:
#             min = distances[j]
#
#     return min


"""

Maximin working for samples with more than 2 dimensions

Maximin algorithm accepts as parameters:
                                            (data, ...) : numpy array, data.shape = (samples, 2)  
                                            (..., num_of_centers) : the desirable number of centers to be created

In my implementation I decided to take as first center, the first sample.
As second center, the furthest possible sample, 
and I continued calculating centers based on manhattan distances, 
until the point that I have created as many centers as requested.

"""



def maximin2(data, num_of_centers):
    created_centers = [data[0].tolist()]  # append to the result the first data_sample

    distances = []
    for i in range(np.size(data, 0)):
        distances.append(eucl_distance2(data[i], data[0]))

    max_index = distances.index(max(distances))
    created_centers.append(data[max_index].tolist())  # append the furthest data_sample from the initial data_sample

    no_cluster = 2

    while no_cluster < num_of_centers:
        distances = []
        indexes = []
        for i in range(np.size(data, 0)):
            dist_i = []

            for j in range(len(created_centers)):  # παίρνω τις αποστάσεις του i σημείου από κάθε κέντρο που έχω
                dist_i.append(eucl_distance2(created_centers[j], data[i]))
                # print(eucl_distance2(created_centers[j], data[i]))

            # print('array of distances for each sample: ', dist_i)
            distances.append(min(dist_i))
            # print('min distance for sample ',i, ': ', min(dist_i))

        # print('  ')
        # print('final array with distances for each sample: ', distances)

        max_index = distances.index(max(distances))
        # print('max index: ', max_index)
        created_centers.append(data[max_index].tolist())
        # print('new center is: ', data[max_index].tolist())
        no_cluster += 1

    return created_centers


def is_equal(a, b):
    for i in range(len(a)):
        if a[i] != b[i]:
            return False

    return True


# def new_centroid(data, list1):
#     n = len(list1)
#     sum_x = 0
#     sum_y = 0
#     for i in range(n):
#         sum_x += data[list1[i]][0]
#         sum_y += data[list1[i]][1]
#
#     return [sum_x / n, sum_y / n]

"""
The purpose is to create a new center of V dimensions. 
This center is the average of all the points in a cluster.
"""


def new_centroid2(data, clusteri):
    n = len(clusteri)  # The number of elements in the cluster -> list1
    sums = []

    for i in range(np.size(data, 1)):
        sumi = 0
        for j in range(n):
            sumi += data[clusteri[j]][i]

        sums.append(sumi)

    return [x / n for x in sums]


# def k_means(data, centroids):
#     clusters = []
#     dists_i_from_centroids = []
#     new_centroids = []
#
#     for i in range(len(centroids)):
#         new_centroids.append([0, 0])
#
#     while not is_equal(centroids, new_centroids):
#         clusters = []
#         for i in range(len(centroids)):
#             clusters.append([])
#
#         for i in range(np.size(data, 0)):
#             for j in range(len(centroids)):  # for each point calculate as many distances as there are centroids
#                 dists_i_from_centroids.append(eucl_distance(data[i], centroids[j]))  # array size = len(centroids)
#
#             min_index = dists_i_from_centroids.index(min(dists_i_from_centroids))
#             clusters[min_index].append(i)
#             dists_i_from_centroids = []
#
#         # up until this point I have a list of lists of pointers (clusters) = [ [0, 1, 6, 9], [...], [...], [...] ]
#         #                                                                             0         1      2      3
#         # each subarray representing a cluster
#
#         new_centroids = centroids[:]
#         a = []
#         for i in range(len(clusters)):
#             if len(clusters[i]) != 0:
#                 a.append(new_centroid(data, clusters[i]))
#             else:
#                 a.append(centroids[i])
#
#         centroids = a[:]
#
#     return clusters, centroids

"""
Implemented a variation of the Kmeans algorithm.
This algorithm takes as parameters: 
                                        The samples of dataset  (data, ...)
                                        An array of n floats that represent n centers, 
                                        provided by maximin (..., centroids)
                                        
K_means returned n lists with indexes of data (clusters).
and the n new centers after the iterations.

Condition to terminate Kmeans, was chosen to be when the new centers created 
are equal to the previous centers of the last iteration. 
"""


def k_means2(data, centroids):
    clusters = []
    dists_i_from_centroids = []
    new_centroids = []

    for i in range(len(centroids)):
        new_centroids.append([0] * np.size(data, 1))

    while not is_equal(centroids, new_centroids):
        clusters = []
        for i in range(len(centroids)):
            clusters.append([])

        for i in range(np.size(data, 0)):
            for j in range(len(centroids)):  # for each point calculate as many distances as there are centroids
                dists_i_from_centroids.append(eucl_distance2(data[i], centroids[j]))  # array size = len(centroids)

            min_index = dists_i_from_centroids.index(min(dists_i_from_centroids))
            clusters[min_index].append(i)
            dists_i_from_centroids = []

        # up until this point I have a list of lists of pointers (clusters) = [ [0, 1, 6, 9], [...], [...], [...] ]
        #                                                                             0         1      2      3
        # each subarray representing a cluster

        new_centroids = centroids[:]

        a = []
        for i in range(len(clusters)):
            if len(clusters[i]) != 0:
                a.append(new_centroid2(data, clusters[i]))
            else:
                a.append(centroids[i])

        centroids = a[:]

    return clusters, centroids


"""
Purity in general is a metric that characterizes the quality of the clusters altogether.
For each cluster, find the count of their most populous element.
After that, add the counts that were calculated in the previous step, store that to a variable 'sum'.
Finally, divide the sum by the total number of samples that were clustered.
"""


def clusters_purity(labels, clusters):
    purity = []

    for i in range(len(clusters)):
        temp = []
        cluster_i = clusters[i]
        int1, int3, int7, int9 = 0, 0, 0, 0
        temp.append(int1)
        temp.append(int3)
        temp.append(int7)
        temp.append(int9)

        for j in range(len(cluster_i)):

            if labels[cluster_i[j]] == 1:
                temp[0] += 1

            elif labels[cluster_i[j]] == 3:
                temp[1] += 1

            elif labels[cluster_i[j]] == 7:
                temp[2] += 1

            else:
                temp[3] += 1

        if len(cluster_i) != 0:
            purity.append(max(temp))
        else:
            purity.append(0)

    num_of_samples = sum([len(a) for a in clusters])

    return sum(purity) / num_of_samples


"""
Code that implements Principal Component Analysis
            for Dimensionality Reduction
"""


def pca2(X, num_components):
    # Step-1
    X_meaned = X - np.mean(X, axis=0)

    # Step-2
    cov_mat = np.cov(X_meaned, rowvar=False)

    # Step-3
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)

    # Step-4
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]

    # Step-5
    eigenvector_subset = sorted_eigenvectors[:, 0:num_components]

    # Step-6
    X_reduced = np.dot(eigenvector_subset.transpose(), X_meaned.transpose()).transpose()

    return X_reduced



def pca(X, n_dim):
    means = np.mean(X, axis=0)
    X = X - means
    square_m = np.dot(X.T, X)
    (evals, evecs) = np.linalg.eig(square_m)
    result = np.dot(X, evecs[:, 0:n_dim])
    return result

"""
converts [samples x 28 x 28] arrays to
                [samples x 784] array
"""
# REALYYYY SLOWWW
def multi_to_one(X):
    Y = []
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y.append(X[i][j])

    arr = np.array(Y)
    return arr


def plotting_2_1(samples, oddRows_list, evenColumns_list):
    x_1 = []
    y_1 = []
    x_3 = []
    y_3 = []
    x_7 = []
    y_7 = []
    x_9 = []
    y_9 = []

    for i in range(samples):
        if Y_train[i] == 1:
            x_1.append(np.mean(oddRows_list[i]))
            y_1.append(np.mean(evenColumns_list[i]))
        elif Y_train[i] == 3:
            x_3.append(np.mean(oddRows_list[i]))
            y_3.append(np.mean(evenColumns_list[i]))
        elif Y_train[i] == 7:
            x_7.append(np.mean(oddRows_list[i]))
            y_7.append(np.mean(evenColumns_list[i]))
        else:
            x_9.append(np.mean(oddRows_list[i]))
            y_9.append(np.mean(evenColumns_list[i]))

    plt.scatter(x_1, y_1, color='red')
    plt.scatter(x_3, y_3, color='green')
    plt.scatter(x_7, y_7, color='blue')
    plt.scatter(x_9, y_9, color='yellow')

    plt.show()


def plotting_3_and_4_2(m, clusters):
    cluster_x0 = [m[i][0] for i in clusters[0]]
    cluster_y0 = [m[i][1] for i in clusters[0]]

    cluster_x1 = [m[i][0] for i in clusters[1]]
    cluster_y1 = [m[i][1] for i in clusters[1]]

    cluster_x2 = [m[i][0] for i in clusters[2]]
    cluster_y2 = [m[i][1] for i in clusters[2]]

    cluster_x3 = [m[i][0] for i in clusters[3]]
    cluster_y3 = [m[i][1] for i in clusters[3]]

    plt.scatter(cluster_x0, cluster_y0, color='red')
    plt.scatter(cluster_x1, cluster_y1, color='green')
    plt.scatter(cluster_x2, cluster_y2, color='blue')
    plt.scatter(cluster_x3, cluster_y3, color='yellow')

    plt.show()


def plotting_4_1(samples, X_pca):
    x_1 = []
    y_1 = []
    x_3 = []
    y_3 = []
    x_7 = []
    y_7 = []
    x_9 = []
    y_9 = []

    for i in range(samples):
        if Y_train[i] == 1:
            x_1.append(X_pca[i][0])
            y_1.append(X_pca[i][1])
        elif Y_train[i] == 3:
            x_3.append(X_pca[i][0])
            y_3.append(X_pca[i][1])
        elif Y_train[i] == 7:
            x_7.append(X_pca[i][0])
            y_7.append(X_pca[i][1])
        else:
            x_9.append(X_pca[i][0])
            y_9.append(X_pca[i][1])

    plt.scatter(x_1, y_1, color='red')
    plt.scatter(x_3, y_3, color='green')
    plt.scatter(x_7, y_7, color='blue')
    plt.scatter(x_9, y_9, color='yellow')

    plt.show()


# --------------------------------------------------------------------------------------


print("-EXERCISE 1.   Import the Dataframe. Select classes 1,3,7,9")

"""
X_train:  an Array of Arrays. It has 60.000 array elements. Each element is an 28 x 28 array which represents a number.
Y_train:  an Array of Integers. It represents the label for the corresponding X[i] array. (e.g. y[0]=1)
"""

"""Initially there are images of 10 images, from MNIST"""
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
print()
print('Initial size of MNIST dataset: ', X_train.shape)

train_selected = np.where((Y_train == 1) | (Y_train == 3) | (Y_train == 7) | (Y_train == 9))
test_selected = np.where((Y_test == 1) | (Y_test == 3) | (Y_test == 7) | (Y_test == 9))

"""I have narrowed it to 4 sets of samples. each one of the labels are either 1, 3, 7 or  9"""

X_train, Y_train = X_train[train_selected], Y_train[train_selected]
X_test, Y_test = X_test[test_selected], Y_test[test_selected]

print('Reduced size of MNIST dataset, after selecting classes 1,3,7,9: ', X_train.shape)

"""  m(x,y)    x-> mean pixel value of rows with odd index    y->mean pixel value of columns with even number"""
print()
print()
print("-EXERCISE 2.   Odd Rows    and    Even Columns")

samples = X_train.shape[0]
test_samples = X_test.shape[0]

m = np.zeros((samples, 2))
labels = np.zeros((samples, 1))
test_labels = np.zeros((test_samples, 1))

oddRows_list = []
odd_rows = []
for i in range(samples):
    """oddRows_List.shape = (25087, 14) """
    for j in range(X_train.shape[1]):
        if j % 2 == 1:
            odd_rows.append(np.average(X_train[i][j]))
    # print(i, odd_rows)
    oddRows_list.append(odd_rows)
    odd_rows = []
    # print(oddRows_list)

evenColumns_list = []
even_columns = []
for i in range(samples):
    """evenColumns_List.shape = (25087, 14) """
    for j in range(X_train.shape[1]):
        if j % 2 == 0:
            sum1 = 0
            for k in range(X_train.shape[2]):
                sum1 += X_train[i][k][j]
            even_columns.append(sum1 / X_train.shape[1])
    # print(i, even_columns)
    evenColumns_list.append(even_columns)
    even_columns = []
    # print(evenColumns_list)

for i in range(samples):
    m[i][0] = np.mean(oddRows_list[i])
    m[i][1] = np.mean(evenColumns_list[i])
    labels[i] = Y_train[i]
print()
print('After taking for each sample the Even Columns and the Odd Rows, the dataset now has the shape: ', m.shape)

print("Plotting exercise 2.1 ...")
plotting_2_1(samples, oddRows_list, evenColumns_list)

print()
print()
print("-EXERCISE 3. Implementation of Maximin algorithm and K-Means algorithm  ")
print()
# comment:  m.shape = (samples, 2)
"""maximin algorithm produced 4 centers given the dataset m created with the restrictions provided"""

centers = maximin2(m, 4)

print("Maximin produced the following centers: ", centers)

""" k_means used as a parameter the dataset m, and the 4 centers provided by maximin.
    K_means returned 4 lists with indexes of m (clusters).
    and the 4 new centers after the iterations."""

clusters, centroids = k_means2(m, centers)

print("K_means clustered the dataset in groups of sizes: ", [len(a) for a in clusters])

print("The Clustering Purity for Exercise 3 was found to be: ", clusters_purity(labels, clusters))

# Clustering Purity for Exercise 3.
# 0.46848965599713

print("Plotting exercise 3... ")
plotting_3_and_4_2(m, clusters)

print()
print()

print("-EXERCISE 4. Converting X_train, to X_train2 array, with the dimensions of 25087 x 784. Then, perform  PCA on X_train2, for V=2, 25, 50, 100. ")
print()
# transforming MNIST X_train data from [samples,28,28] format
#                                        to [samples,784] format

print('Initial dimensions of X_train: ', X_train.shape)
print('We want to transform this to [samples, 784]')
print('*this may take a while*')


X_train2 = np.array(multi_to_one(X_train[0]))
labels[0] = Y_train[0]
for i in range(1, samples):
    X_train2 = np.vstack([X_train2, multi_to_one(X_train[i])])
    labels[i] = Y_train[i]
print('X_train2 dimensions: ', X_train2.shape)
print()

# ------ Performing PCA for V=2, 25, 50, 100
# OBSERVATIONS
# Clustering Purity V= 2
# 0.7224458883086857

# Clustering Purity V= 25
# 0.7477578028460956

# Clustering Purity V= 50
# 0.7153107186989277

# Clustering Purity V= 100
# 0.7472794674532627


V = [2, 25, 50, 100]        # V = [2, 25, 50, 100]
for i in V:
    print("Performing PCA for V=", i, "...")
    # pca = PCA(i)
    # X_pca = pca.fit_transform(X_train2)

    X_pca = pca2(X_train2, i)

    print("X_pca shape: ", X_pca.shape)
    centers = maximin2(X_pca, 4)
    clusters, centroids = k_means2(X_pca, centers)
    print("Clustering Purity after PCA for V=", i, ": ", clusters_purity(labels, clusters))
    if i == 2:
        print("Plotting exercise 4 after PCA, for V=2 ...")
        plotting_4_1(samples, X_pca)
        print("Plotting exercise 4 after PCA and K_means for V=2 ...")
        plotting_3_and_4_2(X_pca, clusters)

    print()



print()
print()
print("-EXERCISE 5. Implementing Naive Bayes Classifier for V = Vmax = 25")
print()
print('Initial dimensions of X_test: ', X_test.shape)
# transforming MNIST X_test data from [samples,28,28] format
#                                   to [samples,784] format
X_test2 = np.array(multi_to_one(X_test[0]))
test_labels[0] = Y_test[0]
for i in range(1, test_samples):
    X_test2 = np.vstack([X_test2, multi_to_one(X_test[i])])
    test_labels[i] = Y_test[i]

print('X_test dimensions: ', X_test2.shape)

pca = PCA(25)

X_test_pca = pca.fit_transform(X_test2)
print("xpca_test shape: ", X_test_pca.shape)
print('test-labels shape: ', test_labels.shape)

X_pca = pca.fit_transform(X_train2)
gnb = GaussianNB()
y_pred = gnb.fit(X_pca, labels.ravel()).predict(X_test_pca)

# number of samples that are mislabeled:
count_mislabeled = 0
for i in range(X_test_pca.shape[0]):
    if y_pred[i] != Y_test[i]:
        count_mislabeled += 1

print('mislabeled counts: ', count_mislabeled)

print("model's accuracy is: ", (X_test_pca.shape[0] - count_mislabeled) / X_test_pca.shape[0])

