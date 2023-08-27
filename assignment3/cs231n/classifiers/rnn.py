import numpy as np

from ..rnn_layers import *


class CaptioningRNN:
    """
    A CaptioningRNN produces captions from image features using a recurrent
    neural network.
    CaptioningRNN使用循环神经网络从图像特征中生成标题。

    The RNN receives input vectors of size D, has a vocab size of V, works on
    sequences of length T, has an RNN hidden dimension of H, uses word vectors
    of dimension W, and operates on minibatches of size N.
    RNN接收大小为D的输入向量，具有词汇量V，适用于长度为T的序列，具有RNN隐藏维度H，使用维度为W的单词向量，并且在大小为N的小批量上运行。

    Note that we don't use any regularization for the CaptioningRNN.
    请注意，我们不使用任何正则化来进行CaptioningRNN。
    """

    def __init__(
        self,
        word_to_idx,
        input_dim=512,
        wordvec_dim=128,
        hidden_dim=128,
        cell_type="rnn",
        dtype=np.float32,
    ):
        """
        Construct a new CaptioningRNN instance.
        构造一个新的CaptioningRNN实例。

        Inputs:
        - word_to_idx: A dictionary giving the vocabulary. It contains V entries,
          and maps each string to a unique integer in the range [0, V).
          一个给出词汇表的字典。它包含V个条目，并将每个字符串映射到范围[0，V）内的唯一整数。
        - input_dim: Dimension D of input image feature vectors.
          输入图像特征向量的维度D
        - wordvec_dim: Dimension W of word vectors.
          单词向量的维度W。
        - hidden_dim: Dimension H for the hidden state of the RNN.
          RNN隐藏状态的维度H。
        - cell_type: What type of RNN to use; either 'rnn' or 'lstm'.
          要使用的RNN类型; 要么是“rnn”或“lstm”。
        - dtype: numpy datatype to use; use float32 for training and float64 for
          numeric gradient checking.
          要使用的numpy数据类型; 使用float32进行训练，使用float64进行数值梯度检查。
        """
        if cell_type not in {"rnn", "lstm"}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.dtype = dtype
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self.params = {}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx.get("<START>", None)
        self._end = word_to_idx.get("<END>", None)

        # Initialize word vectors
        self.params["W_embed"] = np.random.randn(vocab_size, wordvec_dim)
        self.params["W_embed"] /= 100

        # Initialize CNN -> hidden state projection parameters
        self.params["W_proj"] = np.random.randn(input_dim, hidden_dim)
        self.params["W_proj"] /= np.sqrt(input_dim)
        self.params["b_proj"] = np.zeros(hidden_dim)

        # Initialize parameters for the RNN
        dim_mul = {"lstm": 4, "rnn": 1}[cell_type]
        self.params["Wx"] = np.random.randn(wordvec_dim, dim_mul * hidden_dim)
        self.params["Wx"] /= np.sqrt(wordvec_dim)
        self.params["Wh"] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
        self.params["Wh"] /= np.sqrt(hidden_dim)
        self.params["b"] = np.zeros(dim_mul * hidden_dim)

        # Initialize output to vocab weights
        self.params["W_vocab"] = np.random.randn(hidden_dim, vocab_size)
        self.params["W_vocab"] /= np.sqrt(hidden_dim)
        self.params["b_vocab"] = np.zeros(vocab_size)

        # Cast parameters to correct dtype
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)

    def loss(self, features, captions):
        """
        Compute training-time loss for the RNN. We input image features and
        ground-truth captions for those images, and use an RNN (or LSTM) to compute
        loss and gradients on all parameters.
        计算RNN的训练时间损失。我们输入图像特征和这些图像的ground-truth标题，并使用RNN（或LSTM）计算所有参数的损失和梯度。

        Inputs:
        - features: Input image features, of shape (N, D)
          输入图像特征，形状为（N，D）
        - captions: Ground-truth captions; an integer array of shape (N, T + 1) where
          each element is in the range 0 <= y[i, t] < V
          地面实况标题; 形状为（N，T + 1）的整数数组，其中每个元素在范围0 <= y [i，t] <V内

        Returns a tuple of:
        - loss: Scalar loss
          标量损失
        - grads: Dictionary of gradients parallel to self.params、
          损失：标量损失
        """
        # Cut captions into two pieces: captions_in has everything but the last word
        # and will be input to the RNN; captions_out has everything but the first
        # word and this is what we will expect the RNN to generate. These are offset
        # by one relative to each other because the RNN should produce word (t+1)
        # after receiving word t. The first element of captions_in will be the START
        # token, and the first element of captions_out will be the first word.
        # 将标题分为两部分：captions_in除了最后一个单词之外的所有内容，并将其输入到RNN中;
        # captions_out除了第一个单词之外的所有内容，这是我们期望RNN生成的内容。
        # 这些相对于彼此偏移一个，因为RNN应该在接收单词t之后生成单词（t + 1）。
        # captions_in的第一个元素将是START令牌，captions_out的第一个元素将是第一个单词。
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        # You'll need this
        # 你将需要这个
        mask = captions_out != self._null

        # Weight and bias for the affine transform from image features to initial
        # hidden state
        # 从图像特征到初始隐藏状态的仿射变换的权重和偏差
        W_proj, b_proj = self.params["W_proj"], self.params["b_proj"]

        # Word embedding matrix
        # 单词嵌入矩阵
        W_embed = self.params["W_embed"]

        # Input-to-hidden, hidden-to-hidden, and biases for the RNN
        # RNN的输入到隐藏，隐藏到隐藏和偏差
        Wx, Wh, b = self.params["Wx"], self.params["Wh"], self.params["b"]

        # Weight and bias for the hidden-to-vocab transformation.
        # 隐藏到词汇转换的权重和偏差。
        W_vocab, b_vocab = self.params["W_vocab"], self.params["b_vocab"]

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the forward and backward passes for the CaptioningRNN.   #
        # In the forward pass you will need to do the following:                   #
        # (1) Use an affine transformation to compute the initial hidden state     #
        #     from the image features. This should produce an array of shape (N, H)#
        # (2) Use a word embedding layer to transform the words in captions_in     #
        #     from indices to vectors, giving an array of shape (N, T, W).         #
        # (3) Use either a vanilla RNN or LSTM (depending on self.cell_type) to    #
        #     process the sequence of input word vectors and produce hidden state  #
        #     vectors for all timesteps, producing an array of shape (N, T, H).    #
        # (4) Use a (temporal) affine transformation to compute scores over the    #
        #     vocabulary at every timestep using the hidden states, giving an      #
        #     array of shape (N, T, V).                                            #
        # (5) Use (temporal) softmax to compute loss using captions_out, ignoring  #
        #     the points where the output word is <NULL> using the mask above.     #
        # 实现CaptioningRNN的前向和后向传递。
        # 在前向传递中，您需要执行以下操作：
        # （1）使用仿射变换从图像特征计算初始隐藏状态。这应该产生一个形状数组（N，H）
        # （2）使用单词嵌入层将captions_in中的单词从索引转换为向量，从而产生形状数组（N，T，W）。
        # （3）使用vanilla RNN或LSTM（取决于self.cell_type）处理输入单词向量的序列并为所有时间步长产生隐藏状态向量，从而产生形状数组（N，T，H）。
        # （4）使用（时间）仿射变换使用隐藏状态在每个时间步长上计算词汇表的分数，从而产生形状数组（N，T，V）。
        # （5）使用（时间）softmax使用captions_out计算损失，使用上面的掩码忽略输出单词为<NULL>的点。
        #                                                                          #
        #                                                                          #
        # Do not worry about regularizing the weights or their gradients!          #
        # 不要担心正则化权重或它们的梯度！
        #                                                                          #
        # In the backward pass you will need to compute the gradient of the loss   #
        # with respect to all model parameters. Use the loss and grads variables   #
        # defined above to store loss and gradients; grads[k] should give the      #
        # gradients for self.params[k].                                            #
        # 在反向传递中，您需要计算损失相对于所有模型参数的梯度。
        # 使用上面定义的损失和grads变量来存储损失和梯度; grads [k]应该给出self.params [k]的梯度。
        #                                                                          #
        # Note also that you are allowed to make use of functions from layers.py   #
        # in your implementation, if needed.                                       #
        # 还要注意，如果需要，您可以在实现中使用layers.py中的函数。
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        if self.cell_type == 'rnn':
            nn_forward = rnn_forward
            nn_backward = rnn_backward
        elif self.cell_type == 'lstm':
            nn_forward = lstm_forward
            nn_backward = lstm_backward

        h, a_cache = affine_forward(features, W_proj, b_proj)
        x, w_cache = word_embedding_forward(captions_in, W_embed)
        h, r_cache = nn_forward(x, h, Wx, Wh, b)
        h, t_cache = temporal_affine_forward(h, W_vocab, b_vocab)

        loss, dx = temporal_softmax_loss(h, captions_out, mask)

        dx, grads['W_vocab'], grads['b_vocab'] = temporal_affine_backward(dx, t_cache)
        dx, dh, grads['Wx'], grads['Wh'], grads['b'] = nn_backward(dx, r_cache)
        grads['W_embed'] = word_embedding_backward(dx, w_cache)
        dx, grads['W_proj'], grads['b_proj'] = affine_backward(dh, a_cache)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    def sample(self, features, max_length=30):
        """
        Run a test-time forward pass for the model, sampling captions for input
        feature vectors.
        为模型运行测试时间前向传递，对输入特征向量进行采样标题。

        At each timestep, we embed the current word, pass it and the previous hidden
        state to the RNN to get the next hidden state, use the hidden state to get
        scores for all vocab words, and choose the word with the highest score as
        the next word. The initial hidden state is computed by applying an affine
        transform to the input image features, and the initial word is the <START>
        token.
        在每个时间步长，我们嵌入当前单词，将其和先前的隐藏状态传递给RNN以获得下一个隐藏状态，
        使用隐藏状态来获得所有词汇单词的分数，并选择具有最高分数的单词作为下一个单词。
        初始隐藏状态是通过对输入图像特征应用仿射变换来计算的，初始单词是<START>令牌。

        For LSTMs you will also have to keep track of the cell state; in that case
        the initial cell state should be zero.
        对于LSTM，您还必须跟踪单元格状态; 在这种情况下，初始单元格状态应为零。

        Inputs:
        - features: Array of input image features of shape (N, D).
          输入图像特征的数组，形状为（N，D）。
        - max_length: Maximum length T of generated captions.
          生成标题的最大长度T。

        Returns:
        - captions: Array of shape (N, max_length) giving sampled captions,
          where each element is an integer in the range [0, V). The first element
          of captions should be the first sampled word, not the <START> token.
          形状为（N，max_length）的数组，给出采样的标题，其中每个元素是范围[0，V）内的整数。
          captions的第一个元素应该是第一个采样的单词，而不是<START>令牌。
        """
        N = features.shape[0]
        captions = self._null * np.ones((N, max_length), dtype=np.int32)

        # Unpack parameters
        W_proj, b_proj = self.params["W_proj"], self.params["b_proj"]
        W_embed = self.params["W_embed"]
        Wx, Wh, b = self.params["Wx"], self.params["Wh"], self.params["b"]
        W_vocab, b_vocab = self.params["W_vocab"], self.params["b_vocab"]

        ###########################################################################
        # TODO: Implement test-time sampling for the model. You will need to      #
        # initialize the hidden state of the RNN by applying the learned affine   #
        # transform to the input image features. The first word that you feed to  #
        # the RNN should be the <START> token; its value is stored in the         #
        # variable self._start. At each timestep you will need to do to:          #
        # (1) Embed the previous word using the learned word embeddings           #
        # (2) Make an RNN step using the previous hidden state and the embedded   #
        #     current word to get the next hidden state.                          #
        # (3) Apply the learned affine transformation to the next hidden state to #
        #     get scores for all words in the vocabulary                          #
        # (4) Select the word with the highest score as the next word, writing it #
        #     (the word index) to the appropriate slot in the captions variable   #
        # 根据模型实现测试时间采样。您需要通过将学习的仿射变换应用于输入图像特征来初始化RNN的隐藏状态。
        # 您馈送到RNN的第一个单词应该是<START>令牌; 其值存储在变量self._start中。
        # 在每个时间步长，您需要执行以下操作：
        # （1）使用学习的单词嵌入嵌入先前的单词
        # （2）使用先前的隐藏状态和嵌入的当前单词进行RNN步骤，以获得下一个隐藏状态。
        # （3）将学习的仿射变换应用于下一个隐藏状态，以获得词汇表中所有单词的分数
        # （4）选择具有最高分数的单词作为下一个单词，将其（单词索引）写入captions变量中的适当位置
        #                                                                         #
        # For simplicity, you do not need to stop generating after an <END> token #
        # is sampled, but you can if you want to.                                 #
        # 为简单起见，您不需要在采样<END>令牌后停止生成，但如果您愿意，可以这样做。
        #                                                                         #
        # HINT: You will not be able to use the rnn_forward or lstm_forward       #
        # functions; you'll need to call rnn_step_forward or lstm_step_forward in #
        # a loop.                                                                 #
        # 您将无法使用rnn_forward或lstm_forward函数; 您需要在循环中调用rnn_step_forward或lstm_step_forward。
        #                                                                         #
        # NOTE: we are still working over minibatches in this function. Also if   #
        # you are using an LSTM, initialize the first cell state to zeros.        #
        # 注意：我们仍然在这个函数中使用小批量。另外，如果您使用的是LSTM，请将第一个单元格状态初始化为零。
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        if self.cell_type == 'rnn':
            nn_forward = rnn_step_forward
            nn_backward = rnn_step_backward
        elif self.cell_type == 'lstm':
            nn_forward = lstm_step_forward
            nn_backward = lstm_step_backward

        h_, a_cache = affine_forward(features, W_proj, b_proj)
        word = self._start * np.ones((N), dtype=np.int32)
        c_ = np.zeros_like(h_)
        for i in range(max_length):
            word, w_cache = word_embedding_forward(word, W_embed)
            if self.cell_type == 'rnn':
                h_, cache_ = rnn_step_forward(word, h_, Wx, Wh, b)
            elif self.cell_type == 'lstm':
                h_, c_, cache_ = lstm_step_forward(word, h_, c_, Wx, Wh, b)
            out, t_cache = affine_forward(h_, W_vocab, b_vocab)
            word = np.argmax(out, axis=1)
            captions[:, i] = word
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return captions
