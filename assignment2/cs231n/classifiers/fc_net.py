from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    """
    Class for a multi-layer fully connected neural network.
    为多层全连接神经网络设计的类

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be
    网络包含了若干个隐藏层，ReLU非线性层和softmax损失函数。
    也将实现dropout和批/层归一化作为选项。
    对于一个有L层的网络，其网络结构如下

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.
    批/层归一化和dropout层是可选的，并且{...}模块会重复L - 1次

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    可学习的参数被存放在字典self.params中，并且会使用Solver类进行学习
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
          一个列表，给出了每个隐藏层的大小
        - input_dim: An integer giving the size of the input.
          一个整数，给出了输入的维度
        - num_classes: An integer giving the number of classes to classify.
          一个整数，给出了最后需要分类的类别数
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
          dropout强度
            If dropout_keep_ratio=1 then the network should not use dropout at all.
            如果dropout_keep_ratio = 1，则整个网络不会存在dropout
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
          归一化种类，可选项有"batchnorm", "layernorm", or None for no normalization（默认）
        - reg: Scalar giving L2 regularization strength.
          L2正则化强度
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
          权重随机初始化的标准差
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
          随机种子
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        # 初始化神经网络的参数，并将所有值存放在字典self.params中
        # 将第一层的权重和偏置存储为W1和b1；第二层为W2，b2等等
        # 权重应从均值为0，标准差为weight_scale的正态分布中获取
        # 偏置应该初始化为0
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        # 当使用批归一化时，将第一层的缩放和偏移参数存储为gamma1和beta1；
        # 第二层为gamma2，beta2等等
        # 缩放参数应初始化为1，偏移参数应初始化为0
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        for layer, (input, output) in enumerate(zip([input_dim] + hidden_dims, hidden_dims + [num_classes])):
            self.params['W' + str(layer + 1)] = np.random.normal(0, weight_scale, (input, output))
            self.params['b' + str(layer + 1)] = np.zeros(output)
            if self.normalization and layer < self.num_layers - 1:
                self.params['gamma' + str(layer + 1)] = np.ones(output)
                self.params['beta' + str(layer + 1)] = np.zeros(output)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully connected net.
        为全连接网络计算loss和梯度

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        如果y为空，那么运行测试前向传播，并返回如下返回值
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.
          分数：其形状为(N, C)，其中scores[i, c]是X[i]被分为第c类的分数

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        如果y不是空，那么运行训练时前向传播和反向传播，并返回如下返回值
        - loss: Scalar value giving the loss
          损失：损失的标量值
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
          梯度：与字典self.params的键相同的一个字典，为每个参数提供了一个梯度值
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        # 为全连接网络实现前向传播，为X计算类别分数并且存储在变量scores中
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        # 当使用dropout时，你需要将self.dropout_param传递给每个dropout转发传递
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        # 当使用批归一化时，你需要将self.bn_params[0]传递给第一个批归一化层；
        # 将self.bn_params[1]传递给第二个批归一化层
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        caches = {}
        out = X
        for layer in range(self.num_layers - 1):
            n_cache, drop_cache = None, None
            out, cache = affine_forward(out, self.params['W' + str(layer + 1)], self.params['b' + str(layer + 1)])
            caches[f'affine{layer + 1}'] = cache
            if self.normalization == "batchnorm":
                out, n_cache = batchnorm_forward(out, self.params['gamma' + str(layer + 1)], self.params['beta' + str(layer + 1)], self.bn_params[layer])
                caches[f'normalization{layer + 1}'] = n_cache
            if self.normalization == "layernorm":
                out, n_cache = layernorm_forward(out, self.params['gamma' + str(layer + 1)], self.params['beta' + str(layer + 1)], self.bn_params[layer])
                caches[f'normalization{layer + 1}'] = n_cache
            out, relu_cache = relu_forward(out)
            caches[f'relu{layer + 1}'] = relu_cache
            if self.use_dropout:
                out, drop_cache = dropout_forward(out, self.dropout_param)
                caches[f'dropout{layer + 1}'] = drop_cache
        scores, last_cache = affine_forward(out, self.params['W' + str(self.num_layers)], self.params['b' + str(self.num_layers)])
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        # 为全连接网络实现反向传播，将损失存储在loss变量中，梯度存储在grads字典中
        # 使用softmax计算数据损失，并且确保grads[k]保存了self.params[k]的梯度
        # 不要忘记添加L2正则化
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        # 当使用批/层归一化时，你不需要对缩放和偏移参数进行正则化
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        # 为了确保你的实现与我们的实现相匹配，并且你通过了自动化测试
        # 确保你的L2正则化包含一个因子0.5，以简化梯度表达式
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        loss, dout = softmax_loss(scores, y)
        dout, dW, db = affine_backward(dout, last_cache)
        grads['W' + str(self.num_layers)] = dW + self.reg * self.params['W' + str(self.num_layers)]
        grads['b' + str(self.num_layers)] = db
        for layer in reversed(range(self.num_layers - 1)):
            if self.use_dropout:
                dout = dropout_backward(dout, caches[f'dropout{layer + 1}'])
            dout = relu_backward(dout, caches[f'relu{layer + 1}'])
            if self.normalization == "batchnorm":
                dout, dgamma, dbeta = batchnorm_backward_alt(dout, caches[f'normalization{layer + 1}'])
                grads['gamma' + str(layer + 1)] = dgamma
                grads['beta' + str(layer + 1)] = dbeta
            if self.normalization == "layernorm":
                dout, dgamma, dbeta = layernorm_backward(dout, caches[f'normalization{layer + 1}'])
                grads['gamma' + str(layer + 1)] = dgamma
                grads['beta' + str(layer + 1)] = dbeta
            dout, dW, db = affine_backward(dout, caches[f'affine{layer + 1}'])
            grads['W' + str(layer + 1)] = dW + self.reg * self.params['W' + str(layer + 1)]
            grads['b' + str(layer + 1)] = db
        for layer in range(self.num_layers):
            loss += 0.5 * self.reg * np.sum(self.params['W' + str(layer + 1)] ** 2)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
