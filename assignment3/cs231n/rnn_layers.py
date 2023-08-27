"""
This file defines layer types that are commonly used for recurrent neural networks.
"""

import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully connected) layer.
    计算全连接层的前向传播

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.
    输入x的形状为(N, d_1, ..., d_k)，包含N个样本，每个样本x[i]的形状为(d_1, ..., d_k)。
    我们将每个输入重塑为一个维度为D = d_1 * ... * d_k的向量，然后将其转换为维度为M的输出向量。

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
      包含输入数据的numpy数组，形状为(N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
      权重的numpy数组，形状为(D, M)
    - b: A numpy array of biases, of shape (M,)
      偏置的numpy数组，形状为(M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = x.reshape(x.shape[0], -1).dot(w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
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
    dx = dout.dot(w.T).reshape(x.shape)
    dw = x.reshape(x.shape[0], -1).T.dot(dout)
    db = np.sum(dout, axis=0)
    return dx, dw, db


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN using a tanh activation function.
    使用tanh激活函数运行vanilla RNN的单个时间步的前向传播。

    The input data has dimension D, the hidden state has dimension H,
    and the minibatch is of size N.
    输入数据的维度为D，隐藏状态的维度为H，小批量的大小为N。

    Inputs:
    - x: Input data for this timestep, of shape (N, D)
      此时间步的输入数据，形状为(N, D)
    - prev_h: Hidden state from previous timestep, of shape (N, H)
      上一个时间步的隐藏状态，形状为(N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
      输入到隐藏连接的权重矩阵，形状为(D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
      隐藏到隐藏连接的权重矩阵，形状为(H, H)
    - b: Biases of shape (H,)
      偏置，形状为(H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    # 实现vanilla RNN的单个前向步骤。
    # 分别将下一个隐藏状态和反向传播所需的任何值存储在next_h和cache变量中。
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    W = np.concatenate([Wh, Wx], axis=0)    # (H+D, H)
    hx = np.concatenate([prev_h, x], axis=1) # (N, H+D)
    next_h = np.tanh(np.dot(hx, W) + b)     # (N, H)
    cache = (x, prev_h, next_h, Wx, Wh)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.
    vanilla RNN的单个时间步的反向传播。

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state, of shape (N, H)
      损失相对于下一个隐藏状态的梯度，形状为(N, H)
    - cache: Cache object from the forward pass
      前向传播的缓存对象

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
      输入数据的梯度，形状为(N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
      先前隐藏状态的梯度，形状为(N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
      输入到隐藏权重的梯度，形状为(D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
      隐藏到隐藏权重的梯度，形状为(H, H)
    - db: Gradients of bias vector, of shape (H,)
      偏置向量的梯度，形状为(H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    # 实现vanilla RNN的单步反向传播。
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    # 对于tanh函数，可以根据tanh的输出值计算局部导数。
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x, prev_h, next_h, Wx, Wh = cache
    dl_tanh = dnext_h * (1 - next_h ** 2) # (N, H)
    dWh = np.dot(prev_h.T, dl_tanh)       # (H, H)
    dWx = np.dot(x.T, dl_tanh)            # (D, H)
    dprev_h = np.dot(dl_tanh, Wh.T)       # (N, H)
    dx = np.dot(dl_tanh, Wx.T)            # (N, D)
    db = np.sum(dl_tanh, axis=0)          # (H,)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data.
    在整个数据序列上运行vanilla RNN前向传播。

    We assume an input sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running the RNN forward,
    we return the hidden states for all timesteps.
    我们假设输入序列由T个维度为D的向量组成。RNN使用隐藏大小为H，我们在包含N个序列的小批量上工作。
    在运行RNN前向传播之后，我们将返回所有时间步的隐藏状态。

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D)
      整个时间序列的输入数据，形状为(N, T, D)
    - h0: Initial hidden state, of shape (N, H)
      初始隐藏状态，形状为(N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
      输入到隐藏连接的权重矩阵，形状为(D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
      隐藏到隐藏连接的权重矩阵，形状为(H, H)
    - b: Biases of shape (H,)
      偏置，形状为(H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H)
      整个时间序列的隐藏状态，形状为(N, T, H)
    - cache: Values needed in the backward pass
      反向传播所需的值
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    # 实现在输入数据序列上运行的vanilla RNN的前向传播。
    # 您应该使用上面定义的rnn_step_forward函数。您可以使用for循环来帮助计算前向传播。
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    (N, T, D), H = x.shape, b.shape[0]
    h_, cache, h = h0, [], np.zeros((N, T, H))
    for t in range(T):
        x_ = x[:, t, :]
        h_, cache_ = rnn_step_forward(x_, h_, Wx, Wh, b)
        h[:, t, :] = h_
        cache.append(cache_)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.
    计算整个数据序列上vanilla RNN的反向传播。

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H)
      所有隐藏状态的上游梯度，形状为(N, T, H)

    NOTE: 'dh' contains the upstream gradients produced by the
    individual loss functions at each timestep, *not* the gradients
    being passed between timesteps (which you'll have to compute yourself
    by calling rnn_step_backward in a loop).
    注意：'dh'包含每个时间步的单个损失函数产生的上游梯度，
    而不是在时间步之间传递的梯度（您将不得不通过在循环中调用rnn_step_backward来自己计算）。


    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
      输入的梯度，形状为(N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
      初始隐藏状态的梯度，形状为(N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
      输入到隐藏权重的梯度，形状为(D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
      隐藏到隐藏权重的梯度，形状为(H, H)
    - db: Gradient of biases, of shape (H,)
      偏置的梯度，形状为(H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    # 实现在整个数据序列上运行vanilla RNN的反向传播。
    # 您应该使用上面定义的rnn_step_backward函数。您可以使用for循环来帮助计算反向传播。
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    Wx = cache[0][3]
    (N, T, H), D = dh.shape, Wx.shape[0]
    dx, dh_2, dWx, dWh, db = np.zeros((N, T, D)), np.zeros((N, H)), np.zeros((D, H)), np.zeros((H, H)), np.zeros((H,))
    for t in reversed(range(T)):
        dh_ = dh[:, t, :]
        dx_, dh_2, dWx_, dWh_, db_ = rnn_step_backward(dh_ + dh_2, cache[t])
        dx[:, t, :] = dx_
        dWx += dWx_
        dWh += dWh_
        db += db_
    dh0 = dh_2
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings.
    词嵌入的前向传播。

    We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    word to a vector of dimension D.
    我们对大小为N的小批量进行操作，其中每个序列的长度为T。
    我们假设有V个单词的词汇表，将每个单词分配给维度为D的向量。

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
      形状为(N, T)的整数数组，给出单词的索引。x的每个元素idx必须在0 <= idx < V的范围内。
    - W: Weight matrix of shape (V, D) giving word vectors for all words.
      形状为(V, D)的权重矩阵，给出所有单词的单词向量。

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
      形状为(N, T, D)的数组，给出所有输入单词的单词向量。
    - cache: Values needed for the backward pass
      反向传播所需的值
    """
    out, cache = None, None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    # 实现词嵌入的前向传播。
    #                                                                            #
    # HINT: This can be done in one line using NumPy's array indexing.           #
    # 这可以在一行中使用NumPy的数组索引完成。
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    out, cache = W[x], (x, W)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings.
    词嵌入的反向传播。

    We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.
    我们不能反向传播到单词中，因为它们是整数，所以我们只返回单词嵌入矩阵的梯度。

    HINT: Look up the function np.add.at
    提示：查找np.add.at函数

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
      上游梯度的形状为(N, T, D)
    - cache: Values from the forward pass
      前向传播的值

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D)
      单词嵌入矩阵的梯度，形状为(V, D)
    """
    dW = None
    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    # 实现词嵌入的反向传播。
    #                                                                            #
    # Note that words can appear more than once in a sequence.                   #
    # HINT: Look up the function np.add.at                                       #
    # 注意，单词可以在序列中出现多次。
    # 提示：查找np.add.at函数
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x, W = cache
    dW = np.zeros_like(W)
    np.add.at(dW, x, dout) # (V, D)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = x >= 0
    neg_mask = x < 0
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.
    LSTM的单个时间步的前向传播。

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.
    输入数据的维度为D，隐藏状态的维度为H，我们使用小批量大小为N。

    Note that a sigmoid() function has already been provided for you in this file.
    请注意，此文件中已为您提供了sigmoid()函数。

    Inputs:
    - x: Input data, of shape (N, D)
      输入数据，形状为(N, D)
    - prev_h: Previous hidden state, of shape (N, H)
      先前的隐藏状态，形状为(N, H)
    - prev_c: previous cell state, of shape (N, H)
      先前的单元状态，形状为(N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
      输入到隐藏权重，形状为(D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
      隐藏到隐藏权重，形状为(H, 4H)
    - b: Biases, of shape (4H,)
      偏置，形状为(4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
      下一个隐藏状态，形状为(N, H)
    - next_c: Next cell state, of shape (N, H)
      下一个单元状态，形状为(N, H)
    - cache: Tuple of values needed for backward pass.
      反向传播所需的值的元组。
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    # 实现LSTM的单个时间步的前向传播。
    # 您可能希望使用上面的数值稳定的sigmoid实现。
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    H = prev_h.shape[1]
    a = np.dot(x, Wx) + np.dot(prev_h, Wh) + b # (N, 4H)
    a_i, a_f, a_o, a_g = a[:, :H], a[:, H:2 * H], a[:, 2 * H:3 * H], a[:, 3 * H:] # (N, H)
    i, f, o, g = sigmoid(a_i), sigmoid(a_f), sigmoid(a_o), np.tanh(a_g) # (N, H)
    next_c = f * prev_c + i * g # (N, H)
    next_h = o * np.tanh(next_c) # (N, H)
    cache = (x, prev_h, next_h, prev_c, next_c, Wx, Wh, i, f, o, g)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.
    LSTM的单个时间步的反向传播。

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
      下一个隐藏状态的梯度，形状为(N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
      下一个单元状态的梯度，形状为(N, H)
    - cache: Values from the forward pass
      前向传播的值

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
      输入数据的梯度，形状为(N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
      先前隐藏状态的梯度，形状为(N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
      先前单元状态的梯度，形状为(N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
      输入到隐藏权重的梯度，形状为(D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
      隐藏到隐藏权重的梯度，形状为(H, 4H)
    - db: Gradient of biases, of shape (4H,)
      偏置的梯度，形状为(4H,)
    """
    dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    # 实现LSTM的单个时间步的反向传播。
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    # 对于sigmoid和tanh，您可以根据非线性的输出值计算局部导数。
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x, prev_h, next_h, prev_c, next_c, Wx, Wh, i, f, o, g = cache
    dnext_c += dnext_h * o * (1 - np.tanh(next_c) ** 2) # (N, H)
    dprev_c = dnext_c * f # (N, H)
    di = dnext_c * g # (N, H)
    df = dnext_c * prev_c # (N, H)
    do = dnext_h * np.tanh(next_c) # (N, H)
    dg = dnext_c * i # (N, H)
    da_i = di * i * (1 - i) # (N, H)
    da_f = df * f * (1 - f) # (N, H)
    da_o = do * o * (1 - o) # (N, H)
    da_g = dg * (1 - g ** 2) # (N, H)
    da = np.concatenate([da_i, da_f, da_o, da_g], axis=1) # (N, 4H)
    dWh = np.dot(prev_h.T, da) # (H, 4H)
    dWx = np.dot(x.T, da) # (D, 4H)
    dprev_h = np.dot(da, Wh.T) # (N, H)
    dx = np.dot(da, Wx.T) # (N, D)
    db = np.sum(da, axis=0) # (4H,)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data.
    在整个数据序列上运行LSTM的前向传播。

    We assume an input sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running the LSTM forward,
    we return the hidden states for all timesteps.
    我们假设输入序列由T个维度为D的向量组成。LSTM使用隐藏大小为H，我们在包含N个序列的小批量上工作。
    在运行LSTM前向传播之后，我们将返回所有时间步的隐藏状态。

    Note that the initial cell state is passed as input, but the initial cell state is set to zero.
    Also note that the cell state is not returned; it is an internal variable to the LSTM and is not
    accessed from outside.
    请注意，初始单元状态作为输入传递，但初始单元状态设置为零。
    还要注意，单元状态不会返回；它是LSTM的内部变量，不会从外部访问。

    Inputs:
    - x: Input data of shape (N, T, D)
      形状为(N, T, D)的输入数据
    - h0: Initial hidden state of shape (N, H)
      初始隐藏状态的形状为(N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
      输入到隐藏连接的权重，形状为(D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
      隐藏到隐藏连接的权重，形状为(H, 4H)
    - b: Biases of shape (4H,)
      偏置，形状为(4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
      所有序列的所有时间步的隐藏状态，形状为(N, T, H)
    - cache: Values needed for the backward pass.
      反向传播所需的值。
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    # 实现在整个时间序列上运行LSTM的前向传播。
    # 您应该使用刚刚定义的lstm_step_forward函数。
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    (N, T, D), H = x.shape, h0.shape[1]
    h_, cache, h, c_ = h0, [], np.zeros((N, T, H)), np.zeros_like(h0)
    for t in range(T):
        x_ = x[:, t, :]
        h_, c_, cache_ = lstm_step_forward(x_, h_, c_, Wx, Wh, b)
        h[:, t, :] = h_
        cache.append(cache_)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.
    在整个数据序列上运行LSTM的反向传播。

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
      隐藏状态的上游梯度，形状为(N, T, H)
    - cache: Values from the forward pass
      前向传播的值

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
      输入数据的梯度，形状为(N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
      初始隐藏状态的梯度，形状为(N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
      输入到隐藏权重矩阵的梯度，形状为(D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
      隐藏到隐藏权重矩阵的梯度，形状为(H, 4H)
    - db: Gradient of biases, of shape (4H,)
      偏置的梯度，形状为(4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    # 实现在整个时间序列上运行LSTM的反向传播。
    # 您应该使用刚刚定义的lstm_step_backward函数。
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x = cache[0][5]
    (N, T, H), D = dh.shape, x.shape[0]
    dx, dh_2, dc, dWx, dWh, db = np.zeros((N, T, D)), np.zeros((N, H)), np.zeros((N, H)), np.zeros((D, 4 * H)), np.zeros((H, 4 * H)), np.zeros((4 * H,))
    for t in reversed(range(T)):
        dh_ = dh[:, t, :]
        dx_, dh_2, dc, dWx_, dWh_, db_ = lstm_step_backward(dh_ + dh_2, dc, cache[t])
        dx[:, t, :] += dx_
        dWx += dWx_
        dWh += dWh_
        db += db_
    dh0 = dh_2
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer.

    The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs.

    We assume that we are making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores for all vocabulary
    elements at all timesteps, and y gives the indices of the ground-truth element at each timestep.
    We use a cross-entropy loss at each timestep, summing the loss over all timesteps and averaging
    across the minibatch.

    As an additional complication, we may want to ignore the model output at some timesteps, since
    sequences of different length may have been combined into a minibatch and padded with NULL
    tokens. The optional mask argument tells us which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose:
        print("dx_flat: ", dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
