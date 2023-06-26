from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.
    采用模块化设计的具有ReLU非线性和softmax损失函数的一个两层全连接神经网络
    我们假定输入的维度为D，隐藏层的维度为H，共有C类

    The architecure should be affine - relu - affine - softmax.
    网络结构为：仿射层-relu层-仿射层-softmax

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.
    注意该类内没有实现梯度下降，相反，它将与负责运行优化的单独规划求解对象进行交互

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    模型中可学习的参数都存储在了self.params字典中
    映射规则为：参数名-numpy数组
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        # 初始化两层网络的权重和偏置
        # 权重应从均值为0，标准差为weight_scale的高斯分布中初始化而来
        # 偏置应被初始化为0
        # 所有的权重和偏置应被存储在字典self.params中
        # 第一层的权重和偏置应使用键'W1'和'b1'
        # 第二层的权重和偏置应使用键'W2'和'b2'
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.params['W1'] = np.random.normal(loc=0.0, scale=weight_scale, size=(input_dim, hidden_dim))
        self.params['b1'] = np.zeros((hidden_dim,))
        self.params['W2'] = np.random.normal(loc=0.0, scale=weight_scale, size=(hidden_dim, num_classes))
        self.params['b2'] = np.zeros((num_classes,))
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.
        为数量为minibatch的数据计算损失和梯度

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        如果y是None，那么运行模型的测试时前向传播
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        如果y不是None，那么运行模型的训练时前向传播和反向传播
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        # 实现两层网络的前向和反向传播，计算X的类得分，并且将他们存储在得分变量中
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']

        affine1_out, affine1_cache = affine_forward(X, W1, b1)
        relu_out, relu_cache = relu_forward(affine1_out)
        affine2_out, affine2_cache = affine_forward(relu_out, W2, b2)
        scores = affine2_out
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        loss, d_affine2_out = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
        d_relu_out,dw2,db2 = affine_backward(d_affine2_out, affine2_cache)
        d_affine1_out = relu_backward(d_relu_out, relu_cache)
        d_X,dw1,db1 = affine_backward(d_affine1_out, affine1_cache)

        # Add regularization part for dw2 and dw1
        dw1 += self.reg * W1
        dw2 += self.reg * W2

        grads['W1'] = dw1
        grads['b1'] = db1
        grads['W2'] = dw2
        grads['b2'] = db2
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
