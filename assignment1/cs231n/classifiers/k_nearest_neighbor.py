from builtins import range
from builtins import object
import numpy as np
from past.builtins import xrange


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """
    """ 使用L2距离的kNN分类器 """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.
        训练分类器时，k近邻分类器只是记住训练数据

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
          一个形状为(num_train, D)的numpy数组，其中有num_train个训练数据，每个数据的维度为D
        - y: A numpy array of shape (N,) containing the training labels, where
          y[i] is the label for X[i].
          一个形状为(N,)的numpy数组，其中y[i]是X[i]的标签
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.
        使用分类器预测测试数据的标签

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
          of num_test samples each of dimension D.
          一个形状为(num_train, D)的numpy数组，其中有num_train个测试数据，每个数据的维度为D
        - k: The number of nearest neighbors that vote for the predicted labels.
          为预测标签进行投票的邻居数
          即k最近邻的k
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.
          计算训练点和测试点之间距离使用到的方法

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
          一个形状为(num_test,)的numpy数组，其中是测试数据的标签
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.
        使用测试数据X和训练数据self.X_train的嵌套两层循环来计算距离

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.
          包含测试数据的一个形状为(num_test, D)的numpy数组

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
          返回一个大小为(num_test, num_train)的numpy数组
          并且dists[i, j]代表的含义是第i个测试数据和第j个训练数据的欧氏距离
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension, nor use np.linalg.norm().          #
                # 计算第i个测试数据和第j个训练数据的L2距离
                # 不可在维度上使用循环，也不可使用np.linalg.norm()
                #####################################################################
                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
                dists[i][j] = np.sqrt(np.sum(np.square(X[i] - self.X_train[j])))
                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.
        使用测试数据X和训练数据self.X_train的单层循环计算距离

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            # Do not use np.linalg.norm().                                        #
            # 计算第i个测试数据和所有训练数据的L2距离
            #######################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            dists[i, :] = np.sqrt(np.sum(np.square(X[i] - self.X_train), axis=1))
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.
        计算L2距离，不使用任何显式循环

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        dists = np.zeros((num_test, num_train))
        # 这里需要按axis=1来求和
        # 因为i为测试集的坐标
        dists = np.sqrt(
            np.sum(X ** 2, axis=1, keepdims=True)
            + np.sum(self.X_train ** 2, axis=1, keepdims=True).T
            - 2 * np.dot(X, self.X_train.T)
        )
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.
        给定一个测试点和训练点之间的距离矩阵
        预测每个测试点的标签

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance between the ith test point and the jth training point.
          一个大小为(num_test, num_train)的矩阵
          其中dists[i, j]为第i个测试点和第j个训练点的距离

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
			# 使用距离矩阵寻找第i个测试数据的k个最近邻
			# 并且使用self.y_train来寻找这些邻居的标签
			# 将这些标签存储在closest_y中
			# 提示：查询numpy.argsort函数
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            max_index = np.argsort(dists[i])
            for j in range(k):
                index = max_index[j]
                closest_y.append(self.y_train[index])
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
			# 现在你已经找到了k个最近邻的标签
			# 你需要找到在closest_y中出现最多的标签
			# 将这个标签存储在y_pred[i]
			# 如果多个标签出现的次数相同，选择最小的标签
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            label_count = {}
            labels = set(closest_y)
            for label in labels:
                count = closest_y.count(label)
                label_count[label] = count
            y_pred[i] = int(max(label_count, key=label_count.get))
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred
