from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully connected) layer.
    对仿射层计算前向传播

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.
    输入x的形状为(N, d_1, ..., d_k)，其中N表示这个minibatch中的样本数
    d_1, ..., d_k是样本x[i]的维度
    我们会将每个输入都reshape成一个向量，该向量的维度为D = d_1 * ... * d_k
    然后将这个向量进行处理，处理后的向量长度为M

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # x是一个高维矩阵，第0个维度的大小为N，第1个维度的大小为d_1，第2个维度的大小为d_2，...
    # 第k个维度的大小为d_k
    # 取得第0个维度的大小
    N = x.shape[0]
    # 这里的reshape方式是保留了第0个维度
    # 然后将后面所有维度的大小相乘，作为新的第1个维度的大小
    # 即D = d_1 * ... * d_k
    # 所以x_reshaped的形状为(N, D)
    # 这里数据按行优先的顺序排列，但是由于想象不出高维的数据，所以这里也没办法描述
    # 如果有一个二维矩阵：
    # [[1, 2, 3],
    #  [4, 5, 6]]
    # 其展平后为：[1, 2, 3, 4, 5, 6]
    x_reshaped = x.reshape((N, -1))
    # w (D, M)
    # b (M,)
    # 依照维度相乘
    out = np.dot(x_reshaped, w) + b
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine (fully connected) layer.
    对仿射层计算反向传播

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
      上流梯度，形状为(N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x_reshaped = x.reshape((x.shape[0], -1))
    dx = np.dot(dout, w.T).reshape(x.shape)
    dw = np.dot(x_reshaped.T, dout)
    db = np.sum(dout, axis=0)
    # x_reshaped(N, D)
    # dx        (N, M) x (M, D) = (N, D)
    # dw        (D, N) x (N, M) = (D, M)
    # db        (M,)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs)
    对ReLU层计算前向传播

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    out = np.maximum(0, x)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dx = dout * (x > 0)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = x.shape[0]
    x = np.exp(x - np.max(x))
    x /= np.sum(x, axis=1, keepdims=True)
    loss = -np.sum(np.log(x[np.arange(num_train), y])) / num_train
    x[np.arange(num_train), y] -= 1
    dx = x / num_train
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.
    批量归一化的前向传播

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.
    训练期间，从minibatch统计中计算样本均值和（未校正的）样本方差，并用它们来归一化传入的数据。
    在训练期间，我们还保持每个特征的均值和方差的指数衰减运行均值，这些平均值用于在测试时归一化数据。

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:
    在每个时间步长，我们使用基于动量参数的指数衰减更新均值和方差的运行平均值：

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.
    请注意，批量归一化论文建议不同的测试时间行为：他们使用大量训练图像计算每个特征的样本均值和方差，
    而不是使用运行平均值。对于这个实现，我们选择使用运行平均值，因为它们不需要额外的估计步骤；
    torch7批量归一化实现也使用运行平均值。

    Input:
    - x: Data of shape (N, D)
      形状为(N, D)的数据
    - gamma: Scale parameter of shape (D,)
      形状为(D,)的比例参数
    - beta: Shift paremeter of shape (D,)
      形状为(D,)的移位参数
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
        'train'或'test'；必需
      - eps: Constant for numeric stability
        数值稳定性常数
      - momentum: Constant for running mean / variance.
        运行均值/方差的常数
      - running_mean: Array of shape (D,) giving running mean of features
        形状为(D,)的数组，给出特征的运行均值
      - running_var Array of shape (D,) giving running variance of features
        形状为(D,)的数组，给出特征的运行方差

    Returns a tuple of:
    - out: of shape (N, D)
      形状为(N, D)的输出
    - cache: A tuple of values needed in the backward pass
      反向传播所需的值的元组
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        # 实现在批归一化的训练时前向传播。使用小批量统计来计算均值和方差，
        # 使用这些统计数据来归一化传入的数据，
        # 并使用gamma和beta来缩放和移位归一化的数据。
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        # 你应该将输出存储在变量out中。
        # 你需要反向传播的任何中间值都应存储在cache变量中。
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        # 您还应该使用计算得到的样本均值和方差以及动量变量来更新运行均值和运行方差，
        # 并将结果存储在running_mean和running_var变量中。
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        # 请注意，虽然您应该跟踪运行方差，
        # 但您应该根据标准偏差（方差的平方根）来归一化数据！
        # 参考原始论文（https://arxiv.org/abs/1502.03167）可能会有所帮助。
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        mean, var = np.mean(x, axis=0), np.var(x, axis=0)
        std = np.sqrt(var + eps)
        running_mean = momentum * running_mean + (1 - momentum) * mean
        running_var = momentum * running_var + (1 - momentum) * var
        x_norm = (x - mean) / std
        out = gamma * x_norm + beta
        cache = (gamma, beta, x, x_norm, mean, var, std, eps)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        # 实现批量归一化的测试时前向传播。
        # 使用运行均值和方差来归一化传入的数据，
        # 然后使用gamma和beta来缩放和移位归一化的数据。
        # 将结果存储在out变量中。
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        x_norm = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_norm + beta
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.
    批量归一化的反向传播

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.
    对于这个实现，你应该在纸上写出批量归一化的计算图，并通过中间节点向后传播梯度。

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
      上游导数，形状为(N, D)
    - cache: Variable of intermediates from batchnorm_forward.
      来自batchnorm_forward的中间变量

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
      对于每个输入x的梯度，形状为(N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
      对于比例参数gamma的梯度，形状为(D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
      对于移位参数beta的梯度，形状为(D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    # 实现批量归一化的反向传播。将结果存储在dx，dgamma和dbeta变量中。
    # 参考原始论文（https://arxiv.org/abs/1502.03167）可能会有所帮助。
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    gamma, beta, x, x_norm, mean, var, std, eps = cache
    dgamma = np.sum(dout * x_norm, axis=0)
    dbeta = np.sum(dout, axis=0)
    # Sumk是对第0个维度求和
    Sumk = lambda x: np.sum(x, axis=0)
    dx_norm = dout * gamma
    dvar = Sumk(dx_norm * (x - mean) * (-0.5) * (var + eps) ** (-1.5))
    dmean = Sumk(dx_norm * (-1) / std) + dvar * Sumk(-2 * (x - mean)) / x.shape[0]
    dx = dx_norm / std + dvar * 2 * (x - mean) / x.shape[0] + dmean / x.shape[0]
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.
    批量归一化的替代反向传播

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.
    对于这个实现，你应该在纸上计算批量归一化反向传播的导数，并尽可能简化。
    你应该能够推导出一个简单的表达式来进行反向传播。
    请参阅jupyter笔记本获取更多提示。

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.
    注意：这个实现应该期望接收与batchnorm_backward相同的cache变量，

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # 实现批量归一化的反向传播。将结果存储在dx，dgamma和dbeta变量中。
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    # 在计算了关于中心化输入的梯度之后，您应该能够在一个语句中计算关于输入的梯度；
    # 我们的实现适合一行80个字符。
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    gamma, beta, x, x_norm, mean, var, std, eps = cache
    dgamma = np.sum(dout * x_norm, axis=0)
    dbeta = np.sum(dout, axis=0)

    dx_norm = dout * gamma / (len(dout) * std)
    dx = len(dout) * dx_norm - np.sum(dx_norm * x_norm, axis=0) * x_norm - np.sum(dx_norm, axis=0)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.
    层归一化的前向传播

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    在训练和测试时，传入的数据是每个数据点归一化的，
    然后由与批量归一化相同的gamma和beta参数缩放。

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.
    请注意，与批量归一化相比，层归一化在训练和测试时的行为是相同的，
    我们不需要跟踪任何类型的运行平均值。

    Input:
    - x: Data of shape (N, D)
      形状为(N, D)的数据
    - gamma: Scale parameter of shape (D,)
      形状为(D,)的比例参数
    - beta: Shift paremeter of shape (D,)
      形状为(D,)的移位参数
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability
          数值稳定性常数

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    # using gamma and beta.                                                   #
    # 实现层归一化的训练时前向传播。
    # 归一化传入的数据，然后使用gamma和beta缩放和移位归一化的数据。
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    # 提示：这可以通过稍微修改批量归一化的训练时实现，并插入一行或两行放置良好的代码来完成。
    # 特别是，你能想到任何矩阵变换，可以让你复制批量归一化代码并几乎不改变它吗？
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x = x.T
    mean, var = np.mean(x, axis=0), np.var(x, axis=0)
    std = np.sqrt(var + eps)
    x_norm = (x - mean) / std
    x_norm = x_norm.T
    out = gamma * x_norm + beta
    cache = (gamma, beta, x, x_norm, mean, var, std, eps)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.
    层归一化的反向传播

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.
    对于这个实现，你可以大量依赖你已经完成的批量归一化的工作。

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
      上游导数，形状为(N, D)
    - cache: Variable of intermediates from layernorm_forward.
      来自layernorm_forward的中间变量

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
      对于每个输入x的梯度，形状为(N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
      对于比例参数gamma的梯度，形状为(D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
      对于移位参数beta的梯度，形状为(D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    # 实现层归一化的反向传播。
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    # 提示：这可以通过稍微修改批量归一化的训练时实现来完成。
    # 前向传播的提示仍然适用！
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    gamma, beta, x, x_norm, mean, var, std, eps = cache
    dgamma = np.sum(dout * x_norm, axis=0)
    dbeta = np.sum(dout, axis=0)
    Sumk = lambda x: np.sum(x, axis=0)
    dx_norm = dout * gamma
    dout = dout.T
    dx_norm = dx_norm.T
    dvar = Sumk(dx_norm * (x - mean) * (-0.5) * (var + eps) ** (-1.5))
    dmean = Sumk(dx_norm * (-1) / std) + dvar * Sumk(-2 * (x - mean)) / x.shape[0]
    dx = dx_norm / std + dvar * 2 * (x - mean) / x.shape[0] + dmean / x.shape[0]
    dx = dx.T
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Forward pass for inverted dropout.
    反向dropout的前向传播

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.
    请注意，这与vanilla版本的dropout不同。
    在这里，p是保持神经元输出的概率，而不是丢弃神经元输出的概率。
    有关更多详细信息，请参阅http://cs231n.github.io/neural-networks-2/#reg。

    Inputs:
    - x: Input data, of any shape
      任意形状的输入数据
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
        Dropout参数。我们以概率p保持每个神经元的输出。
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
        'test'或'train'。
        如果模式是train，则执行dropout；如果模式是test，则只返回输入。
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.
        随机数生成器的种子。
        传递种子使该函数确定性，这对于梯度检查是必需的，但在真实网络中不是必需的。

    Outputs:
    - out: Array of the same shape as x.
      与x形状相同的数组。
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
      dropout_param, mask的元组。
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        # 实现反向dropout的训练阶段前向传播。
        # 将dropout掩码存储在变量mask中。
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        # 实现反向dropout的测试阶段前向传播。
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        out = x
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Backward pass for inverted dropout.
    反向dropout的反向传播

    Inputs:
    - dout: Upstream derivatives, of any shape
      上游导数，任意形状
    - cache: (dropout_param, mask) from dropout_forward.
      来自dropout_forward的(dropout_param, mask)
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        # 实现反向dropout的训练阶段反向传播
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        dx = dout * mask
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.
    卷积层的前向传播的naive实现。

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.
    输入由N个数据点组成，每个数据点具有C个通道，高度H和宽度W。
    我们用F个不同的滤波器对每个输入进行卷积，其中每个滤波器跨越所有C个通道，高度HH和宽度WW。

    Input:
    - x: Input data of shape (N, C, H, W)
      形状为(N, C, H, W)的输入数据
    - w: Filter weights of shape (F, C, HH, WW)
      形状为(F, C, HH, WW)的滤波器权重
    - b: Biases, of shape (F,)
      偏置，形状为(F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
        水平和垂直方向上相邻感受野之间的像素数。
      - 'pad': The number of pixels that will be used to zero-pad the input.
        在输入上用于零填充的像素数。

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.
    在填充期间，应该对输入的高度和宽度轴对称地（即两侧相等地）放置“pad”个零。
    注意不要直接修改原始输入x。

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      输出数据，形状为(N, F, H', W')，其中H'和W'由以下公式给出
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    # 实现卷积前向传播。
    # 提示：您可以使用np.pad函数进行填充。
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    stride = conv_param.get('stride', 1)
    pad = conv_param.get('pad', 0)
    H_ = 1 + int((H + 2 * pad - HH) / stride)
    W_ = 1 + int((W + 2 * pad - WW) / stride)
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    out = np.zeros((N, F, H_, W_))
    w_row = w.reshape(F, -1)
    x_col = np.zeros((C * HH * WW, H_ * W_))
    for f in range(N):
        temp = 0
        for h_ in range(0, (H_ - 1) * stride + 1, stride):
            for w_ in range(0, (H_ - 1) * stride + 1, stride):
                x_col[:, temp] = x_pad[f, :, h_:h_ + HH, w_:w_ + WW].reshape(-1)
                temp += 1
        out[f] = (np.dot(w_row, x_col) + b.reshape(-1, 1)).reshape(F, H_, W_)
    x = x_pad
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.
    卷积层的反向传播的naive实现。

    Inputs:
    - dout: Upstream derivatives.
      上游导数。
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive
      与conv_forward_naive中的(x, w, b, conv_param)元组相同

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    # 实现卷积反向传播。
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x, w, b, conv_param = cache
    stride = conv_param.get('stride', 1)
    pad = conv_param.get('pad', 0)
    N, C, H_, W_ = dout.shape
    F, C, HH, WW = w.shape
    dx, dw, db = np.zeros_like(x), np.zeros_like(w), np.zeros_like(b)

    w_row = w.reshape(F, -1)
    x_col = np.zeros((C * HH * WW, H_ * W_))
    for f in range(N):
        d_out = dout[f].reshape(F, -1)
        db += np.sum(d_out, axis=1)
        dz = np.dot(w_row.T, d_out)
        temp = 0
        for h_ in range(0, (H_ - 1) * stride + 1, stride):
            for w_ in range(0, (H_ - 1) * stride + 1, stride):
                x_col[:, temp] = x[f, :, h_:h_ + HH, w_:w_ + WW].reshape(-1)
                dx[f, :, h_:h_ + HH, w_:w_ + WW] += dz[:, temp].reshape(C, HH, WW)
                temp += 1
        dw += np.dot(d_out, x_col.T).reshape(F, C, HH, WW)
    dx = dx[:, :, pad:-pad, pad:-pad]
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.
    最大池化层的前向传播的naive实现。

    Inputs:
    - x: Input data, of shape (N, C, H, W)
      输入数据，形状为(N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
        池化区域的高度
      - 'pool_width': The width of each pooling region
        池化区域的宽度
      - 'stride': The distance between adjacent pooling regions
        相邻池化区域之间的距离

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    # 实现最大池化的前向传播
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = x.shape
    pool_height = pool_param.get('pool_height', 2)
    pool_width = pool_param.get('pool_width', 2)
    stride = pool_param.get('stride', 2)
    H_ = 1 + int((H - pool_height) / stride)
    W_ = 1 + int((W - pool_width) / stride)
    out = np.zeros((N, C, H_, W_))
    for h in range(H_):
        for w in range(W_):
            out[:, :, h, w] = np.max(x[:, :, h * stride:h * stride + pool_height, w * stride:w * stride + pool_width], axis=(2, 3))
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.
    最大池化层的反向传播的naive实现。

    Inputs:
    - dout: Upstream derivatives
      上游导数
    - cache: A tuple of (x, pool_param) as in the forward pass.
      与前向传播中的(x, pool_param)元组相同。

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    # 实现最大池化的反向传播
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x, pool_param = cache
    N, C, H_, W_ = dout.shape
    pool_height = pool_param.get('pool_height', 2)
    pool_width = pool_param.get('pool_width', 2)
    stride = pool_param.get('stride', 2)

    dx = np.zeros_like(x)
    for h in range(H_):
        for w in range(W_):
            x_catch = x[:, :, h * stride:h * stride + pool_height, w * stride:w * stride + pool_width]
            x_max = np.max(x_catch, axis=(2, 3))
            x_catch -= np.tile(x_max[:, :, np.newaxis, np.newaxis], (1, 1, pool_height, pool_width))
            x_catch[x_catch == 0] = 1
            x_catch[x_catch < 0] = 0
            x_catch *= np.tile(dout[:, :, h, w][:, :, np.newaxis, np.newaxis], (1, 1, pool_height, pool_width))
            dx[:, :, h * stride:h * stride + pool_height, w * stride:w * stride + pool_width] = x_catch
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.
    计算空间批量归一化的前向传播。

    Inputs:
    - x: Input data of shape (N, C, H, W)
      形状为(N, C, H, W)的输入数据
    - gamma: Scale parameter, of shape (C,)
      比例参数，形状为(C,)
    - beta: Shift parameter, of shape (C,)
      移位参数，形状为(C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
        'train'或'test'；必需
      - eps: Constant for numeric stability
        数值稳定性常数
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
        运行均值/方差的常数。momentum=0意味着旧信息在每个时间步骤完全被丢弃，
        而momentum=1意味着新信息永远不会被合并。
        默认的momentum=0.9在大多数情况下都很好。
      - running_mean: Array of shape (D,) giving running mean of features
        形状为(D,)的数组，给出特征的运行均值
      - running_var Array of shape (D,) giving running variance of features
        形状为(D,)的数组，给出特征的运行方差

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    # 实现空间批量归一化的前向传播。
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    # 您可以通过调用上面实现的批量归一化的vanilla版本来实现空间批量归一化。
    # 您的实现应该非常简短；我们的代码不到五行。
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = x.shape
    x = x.transpose(0, 2, 3, 1).reshape(N * H * W, C)
    out, cache = batchnorm_forward(x, gamma, beta, bn_param)
    out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.
    计算空间批量归一化的反向传播。

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
      上游导数，形状为(N, C, H, W)
    - cache: Values from the forward pass
      来自前向传播的值

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    # 实现空间批量归一化的反向传播。
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    # 您可以通过调用上面实现的批量归一化的vanilla版本来实现空间批量归一化。
    # 您的实现应该非常简短；我们的代码不到五行。
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = dout.shape
    dout = dout.transpose(0, 2, 3, 1).reshape(N * H * W, C)
    dx, dgamma, dbeta = batchnorm_backward(dout, cache)
    dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    计算空间组归一化的前向传播。

    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.
    与层归一化相反，组归一化将数据中的每个条目分成G个连续的片段，然后独立地对其进行归一化。
    然后以与批量归一化和层归一化完全相同的方式对数据进行移位和缩放。

    Inputs:
    - x: Input data of shape (N, C, H, W)
      形状为(N, C, H, W)的输入数据
    - gamma: Scale parameter, of shape (1, C, 1, 1)
      比例参数，形状为(1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
      移位参数，形状为(1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
      要分割的组数的整数，应该是C的除数
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability
        数值稳定性常数

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    # 实现空间组归一化的前向传播。
    # 这与层归一化实现非常相似。
    # 特别是，想想如何转换矩阵，使得大部分代码与训练时批量归一化和层归一化都相似！
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = x.shape
    x = x.reshape(N * G, -1).T
    mean, var = np.mean(x, axis=0), np.var(x, axis=0)
    std = np.sqrt(var + eps)
    x_norm = (x - mean) / std
    x_norm = x_norm.T.reshape(N, C, H, W)
    out = gamma * x_norm + beta
    cache = (gamma, beta, x, x_norm, mean, var, std, eps, G)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.
    计算空间组归一化的反向传播。

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
      上游导数，形状为(N, C, H, W)
    - cache: Values from the forward pass
      来自前向传播的值

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    # 实现空间组归一化的反向传播。
    # 这与层归一化实现非常相似。
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = dout.shape
    gamma, beta, x, x_norm, mean, var, std, eps, G = cache
    dgamma = np.sum(dout * x_norm, axis=(0, 2, 3)).reshape((1, C, 1, 1))
    dbeta = np.sum(dout, axis=(0, 2, 3)).reshape((1, C, 1, 1))
    Sumk = lambda x: np.sum(x, axis=0)

    dx_norm = dout * gamma
    dout = dout.T
    dx_norm = dx_norm.reshape(N * G, -1).T
    dvar = Sumk(dx_norm * (x - mean) * (-0.5) * (var + eps) ** (-1.5))
    dmean = Sumk(dx_norm * (-1) / std) + dvar * Sumk(-2 * (x - mean)) / x.shape[0]
    dx = dx_norm / std + dvar * 2 * (x - mean) / x.shape[0] + dmean / x.shape[0]
    dx = dx.T.reshape(N, C, H, W)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
