from builtins import object
import numpy as np

from ..layers import *
from ..fast_layers import *
from ..layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:
    一个三层的卷积神经网络，结构如下：

    conv - relu - 2x2 max pool - affine - relu - affine - softmax
    卷积 - relu - 2x2 最大池化 - 全连接 - relu - 全连接 - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    网络对数据的操作是以小批量的形式，数据的形状是 (N, C, H, W)，
    其中 N 是图像的数量，C 是输入通道的数量，H 和 W 是图像的高度和宽度。
    """

    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        """
        Initialize a new network.
        初始化一个新的网络。

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
          输入维度是 (C, H, W)，C 是通道数，H 和 W 是图像的高度和宽度
        - num_filters: Number of filters to use in the convolutional layer
          卷积层使用的滤波器的数量
        - filter_size: Width/height of filters to use in the convolutional layer
          卷积层使用的滤波器的宽度和高度
        - hidden_dim: Number of units to use in the fully-connected hidden layer
          全连接隐藏层使用的单元数
        - num_classes: Number of scores to produce from the final affine layer.
          最后的全连接层输出的分数的数量
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
          随机初始化权重的标准差
        - reg: Scalar giving L2 regularization strength
          L2正则化强度
        - dtype: numpy datatype to use for computation.
          计算时使用的numpy数据类型
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        # 初始化三层卷积网络的权重和偏置。
        # 权重应该从均值为0.0，标准差为weight_scale的高斯分布中初始化；
        # 偏置应该初始化为0。所有的权重和偏置都应该存储在字典self.params中。
        # 使用键'W1'和'b1'存储卷积层的权重和偏置；
        # 使用键'W2'和'b2'存储隐藏全连接层的权重和偏置；
        # 使用键'W3'和'b3'存储输出全连接层的权重和偏置。
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #
        # 重要提示：在这个作业中，你可以假设第一个卷积层的填充和步长被选择为
        # **输入的宽度和高度被保留**。查看loss()函数的开头部分，看看是怎么做到的。
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        C, H, W = input_dim
        self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
        self.params['b1'] = np.zeros(num_filters)
        self.params['W2'] = weight_scale * np.random.randn(num_filters * H * W // 4, hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b3'] = np.zeros(num_classes)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        计算三层卷积网络的损失和梯度。

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # pass conv_param to the forward pass for the convolutional layer
        # 传递卷积参数给卷积层的前向传播
        # Padding and stride chosen to preserve the input spatial size
        # 选择填充和步长以保持输入的空间大小
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        # 传递池化参数给最大池化层的前向传播
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        # 实现三层卷积网络的前向传播，计算X的类别分数并将它们存储在scores变量中。
        #                                                                          #
        # Remember you can use the functions defined in cs231n/fast_layers.py and  #
        # cs231n/layer_utils.py in your implementation (already imported).         #
        # 你可以在实现中使用cs231n/fast_layers.py和cs231n/layer_utils.py中定义的函数（已经导入了）。
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        out, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        out, cache2 = affine_relu_forward(out, W2, b2)
        scores, cache3 = affine_forward(out, W3, b3)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        # 实现三层卷积网络的反向传播，将损失和梯度存储在损失和梯度变量中。
        # 使用softmax计算数据损失，并确保grads[k]保存self.params[k]的梯度。
        # 不要忘记添加L2正则化！
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        # 为了确保你的实现与我们的匹配，并且你通过了自动测试，
        # 确保你的L2正则化包含一个因子0.5，以简化梯度的表达式。
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        loss, dout = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(W1 ** 2) + np.sum(W2 ** 2) + np.sum(W3 ** 2))
        dout, dW3, db3 = affine_backward(dout, cache3)
        dout, dW2, db2 = affine_relu_backward(dout, cache2)
        dout, dW1, db1 = conv_relu_pool_backward(dout, cache1)

        grads['W1'] = dW1 + self.reg * W1
        grads['b1'] = db1
        grads['W2'] = dW2 + self.reg * W2
        grads['b2'] = db2
        grads['W3'] = dW3 + self.reg * W3
        grads['b3'] = db3
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
