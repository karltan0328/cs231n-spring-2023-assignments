from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)
    Softmax损失函数使用循环的朴素实现

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.
    输入的维度为D，并且有C类，我们将N个样例作为一个minibatch

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    # 将loss存储在loss中，将梯度存储在dW中
    # 如果你在这里粗心，你可能会遇到数值不稳定的问题
    # 不要忘了正则化
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # W (D, C)
    # X (N, D)
    num_train, dim = X.shape
    classes = W.shape[1]
    scores = np.dot(X, W)   # (N, C)

    # 数值稳定
    """
    scores = np.array([[0.1, 0.2, 0.3],
                       [0.4, 0.5, 0.6],
                       [0.7, 0.8, 0.9]])
    max_scores = np.max(scores, axis=1, keepdims=True)
    print(max_scores)

    结果为：
    [[0.3]
     [0.6]
     [0.9]]
    就是在行中找出最大值
    """
    sta_scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(sta_scores)
    # np.sum(exp_scores, axis=1, keepdims=True)是按行求和
    # 一行有C个数据，每个数据代表了该类的得分，除以每行的和
    # 实现softmax
    norm_scores = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # 求损失函数
    for i in range(num_train):
        loss -= np.log(norm_scores[i, y[i]])

    loss /= num_train
    loss += reg * np.sum(np.square(W))

    # norm_scores   (N, C)
    # X             (N, D)
    # dW            (D, C)
    for i in range(num_train):
        for d in range(dim):
            for c in range(classes):
                dW[d, c] += (norm_scores[i, c] * X[i, d])
                if c == y[i]:
                    dW[d, c] -= X[i, d]

    dW /= num_train
    dW += 2 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    # 不使用显式循环，计算softmax的损失函数和梯度
    # 将loss存储在loss中，将梯度存储在dW中
    # 如果你在这里粗心，你可能会遇到数值不稳定的问题
    # 不要忘了正则化
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train, dim = X.shape
    classes = W.shape[1]
    scores = np.dot(X, W)

    sta_scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(sta_scores)
    norm_scores = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # 与softmax_loss_naive()函数相比
    # 计算loss和dW时没有使用显式循环
    loss = -np.sum(np.log(norm_scores[np.arange(num_train), y])) / num_train
    loss += reg * np.sum(np.square(W))

    # norm_scores   (N, C)
    # X             (N, D)
    # dW            (D, C)
    # mask          (N, C)
    mask = norm_scores.copy()
    mask[np.arange(num_train), y] -= 1
    dW = np.dot(X.T, mask) / num_train
    dW += 2 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
