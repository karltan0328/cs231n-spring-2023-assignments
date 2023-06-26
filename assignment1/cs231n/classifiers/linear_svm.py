from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).
    结构化SVM损失函数，使用循环的朴素实现

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.
    输入为D维，共C类，我们在大小为N个样本的minibatches上操作

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
      权重，一个形状为(D, C)的numpy数组
    - X: A numpy array of shape (N, D) containing a minibatch of data.
      数据的一个minibatch，一个形状为(N, D)的numpy数组
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
      训练标签，一个大小为(N,)的numpy数组
      y[i] = c代表X[i]的标签为c，且0 <= c < C
    - reg: (float) regularization strength
      正则化强度

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]    # (D, C)
    num_train = X.shape[0]      # (N, D)
    loss = 0.0
    for i in range(num_train):
        # X[i]  (1, D)
        # W     (D, C)
        # scores(1, C)
        scores = np.dot(X[i], W)
        # y[i]中是一个类别
        # 从scores中取出类别y[i]的分数
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            # hinge损失函数
            # L = max(0, scores[j] - correct_class_score + 1)
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    # 将误差取平均值
    loss /= num_train

    # Add regularization to the loss.
    # 将正则化项加到损失函数中
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    # 计算损失函数的梯度并且存储在dW中
    # 与其先计算损失函数，然后再计算导数
    # 不如在计算损失函数的同时计算导数可能更简单
    # 所以你可能需要修改上面的一些代码来计算梯度
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    for i in range(num_train):
        scores = np.dot(X[i], W)
        correct_class_score = scores[y[i]]

        for j in range(num_classes):
            if j != y[i] and scores[j] - correct_class_score + 1 >= 0:
                dW[:, j] += X[i]
                dW[:, y[i]] -= X[i]

    dW /= num_train
    dW += 2 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.
    结构化SVM损失函数，向量化实现

    Inputs and outputs are the same as svm_loss_naive.
    输入和输出与svm_loss_naive一致
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    # 实现结构化SVM损失函数的向量化版本
    # 并将结果存储在loss中
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    scores = np.dot(X, W)   # (N, C)
    # np.arange(3)的输出为array([0, 1, 2])
    """
    num_train = 3
    num_classes = 4
    scores = np.array([[0.1, 0.2, 0.3, 0.4],
                       [0.5, 0.6, 0.7, 0.8],
                       [0.9, 1.0, 1.1, 1.2]])
    y = np.array([1, 3, 2])

    rightscore = scores[np.arange(num_train), y]
    print(rightscore)

    输出为[0.2 0.8 1.1]
    """
    # 在每一行中找到正确类别的得分
    rightscore = scores[np.arange(num_train), y]
    # hinge损失函数
    # L = max(0, scores[j] - correct_class_score + 1)
    scores = scores - rightscore.reshape(-1, 1) + 1.
    # 将分类正确的损失置为0
    scores[scores <= 0] = 0
    scores[np.arange(num_train), y] = 0
    # 取平均值
    loss = np.sum(scores) / num_train
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    # 实现结构化SVM损失函数的梯度向量化版本，将结果存储在dW中
    #
    # 提示：与其从头开始计算梯度，不如重用用于计算损失的一些中间值
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_classes = W.shape[1]
    mask = np.zeros((num_train, num_classes))
    scores[scores > 0] = 1
    mask += scores
    mask[np.arange(num_train), y] = -np.sum(scores, axis=1)
    dW = np.dot(X.T, mask) / num_train
    dW += 2 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
