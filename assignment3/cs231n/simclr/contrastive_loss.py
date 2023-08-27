import torch
import numpy as np


def sim(z_i, z_j):
    """
    Normalized dot product between two vectors.
    正则化两个向量之间的点积。

    Inputs:
    - z_i: 1xD tensor.
      形状为1xD的张量。
    - z_j: 1xD tensor.
      形状为1xD的张量。

    Returns:
    - A scalar value that is the normalized dot product between z_i and z_j.
      z_i和z_j之间的归一化点积的标量值。
    """
    norm_dot_product = None
    ##############################################################################
    # TODO: Start of your code.                                                  #
    # 从此处开始编写代码。
    #                                                                            #
    # HINT: torch.linalg.norm might be helpful.                                  #
    # 提示：torch.linalg.norm可能有帮助。
    ##############################################################################
    norm = torch.linalg.norm
    norm_dot_product = np.dot(z_i, z_j) / (norm(z_i) * norm(z_j))
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return norm_dot_product


def simclr_loss_naive(out_left, out_right, tau):
    """
    Compute the contrastive loss L over a batch (naive loop version).
    计算批次（naive loop版本）上的对比损失L。

    Input:
    - out_left: NxD tensor; output of the projection head g(), left branch in SimCLR model.
      NxD张量；投影头g()的输出，SimCLR模型中的左分支。
    - out_right: NxD tensor; output of the projection head g(), right branch in SimCLR model.
      NxD张量；投影头g()的输出，SimCLR模型中的右分支。
    Each row is a z-vector for an augmented sample in the batch. The same row in out_left and out_right form a positive pair.
    In other words, (out_left[k], out_right[k]) form a positive pair for all k=0...N-1.
    每一行都是批处理中增强样本的z向量。out_left和out_right中的同一行形成一个正对。
    换句话说，（out_left[k]，out_right[k]）对于所有k=0...N-1形成一个正对。
    - tau: scalar value, temperature parameter that determines how fast the exponential increases.
      标量值，温度参数，决定指数增长的速度。

    Returns:
    - A scalar value; the total loss across all positive pairs in the batch. See notebook for definition.
      标量值；批处理中所有正对的总损失。请参阅笔记本以获取定义。
    """
    N = out_left.shape[0]  # total number of training examples

     # Concatenate out_left and out_right into a 2*N x D tensor.
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]

    total_loss = 0
    for k in range(N):  # loop through each positive pair (k, k+N)
        z_k, z_k_N = out[k], out[k+N]

        ##############################################################################
        # TODO: Start of your code.                                                  #
        # 从此处开始编写代码。
        #                                                                            #
        # Hint: Compute l(k, k+N) and l(k+N, k).                                     #
        # 提示：计算l(k, k+N)和l(k+N, k)。
        ##############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        f = lambda x, y : np.exp(sim(x, y) / tau)
        head = f(z_k, z_k_N)
        down_left = sum([f(z_k, out[j]) for j in range(2 * N) if j != k])
        down_right = sum([f(z_k_N, out[j]) for j in range(2 * N) if j != k + N])
        loss_left = -torch.log(head / down_left)
        loss_right = -torch.log(head / down_right)
        total_loss += loss_left + loss_right
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
         ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

    # In the end, we need to divide the total loss by 2N, the number of samples in the batch.
    total_loss = total_loss / (2*N)
    return total_loss


def sim_positive_pairs(out_left, out_right):
    """
    Normalized dot product between positive pairs.
    正对之间的归一化点积。

    Inputs:
    - out_left: NxD tensor; output of the projection head g(), left branch in SimCLR model.
      NxD张量；投影头g()的输出，SimCLR模型中的左分支。
    - out_right: NxD tensor; output of the projection head g(), right branch in SimCLR model.
      NxD张量；投影头g()的输出，SimCLR模型中的右分支。
    Each row is a z-vector for an augmented sample in the batch.
    The same row in out_left and out_right form a positive pair.
    每一行都是批处理中增强样本的z向量。out_left和out_right中的同一行形成一个正对。

    Returns:
    - A Nx1 tensor; each row k is the normalized dot product between out_left[k] and out_right[k].
      每一行k都是out_left[k]和out_right[k]之间的归一化点积。
    """
    pos_pairs = None

    ##############################################################################
    # TODO: Start of your code.                                                  #
    # 从此处开始编写代码。
    #                                                                            #
    # HINT: torch.linalg.norm might be helpful.                                  #
    # 提示：torch.linalg.norm可能有帮助。
    ##############################################################################

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    norm = lambda x : torch.linalg.norm(x, dim=1, keepdim=True)
    pos_pairs = torch.sum(out_left * out_right, dim=1, keepdim=True) / (norm(out_left) * norm(out_right))
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return pos_pairs


def compute_sim_matrix(out):
    """
    Compute a 2N x 2N matrix of normalized dot products between all pairs of augmented examples in a batch.
    计算批处理中所有增强示例对之间的归一化点积的2N x 2N矩阵。

    Inputs:
    - out: 2N x D tensor; each row is the z-vector (output of projection head) of a single augmented example.
      每一行都是单个增强示例的z向量（投影头的输出）。
    There are a total of 2N augmented examples in the batch.
    批处理中总共有2N个增强示例。

    Returns:
    - sim_matrix: 2N x 2N tensor; each element i, j in the matrix is the normalized dot product between out[i] and out[j].
      2N x 2N张量；矩阵中的每个元素i，j是out[i]和out[j]之间的归一化点积。
    """
    sim_matrix = None

    ##############################################################################
    # TODO: Start of your code.                                                  #
    # 从此处开始编写代码。
    ##############################################################################

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    out_norm = out / torch.linalg.norm(out, dim=1, keepdim=True)
    sim_matrix = torch.matmul(out_norm, out_norm.T)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return sim_matrix


def simclr_loss_vectorized(out_left, out_right, tau, device='cuda'):
    """
    Compute the contrastive loss L over a batch (vectorized version). No loops are allowed.
    计算批次（向量化版本）上的对比损失L。不允许循环。

    Inputs and output are the same as in simclr_loss_naive.
    输入和输出与simclr_loss_naive中的相同。
    """
    N = out_left.shape[0]

    # Concatenate out_left and out_right into a 2*N x D tensor.
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]

    # Compute similarity matrix between all pairs of augmented examples in the batch.
    sim_matrix = compute_sim_matrix(out)  # [2*N, 2*N]

    ##############################################################################
    # TODO: Start of your code. Follow the hints.                                #
    # 从此处开始编写代码。按照提示操作。
    ##############################################################################

    # Step 1: Use sim_matrix to compute the denominator value for all augmented samples.
    # 步骤1：使用sim_matrix计算所有增强样本的分母值。
    # Hint: Compute e^{sim / tau} and store into exponential, which should have shape 2N x 2N.
    # 提示：计算e^{sim / tau}并存储到指数中，其形状应为2N x 2N。
    # exponential = None
    exponential = torch.exp(sim_matrix / tau).to(device)

    # This binary mask zeros out terms where k=i.
    mask = (torch.ones_like(exponential, device=device) - torch.eye(2 * N, device=device)).to(device).bool()

    # We apply the binary mask.
    exponential = exponential.masked_select(mask).view(2 * N, -1)  # [2*N, 2*N-1]

    # Hint: Compute the denominator values for all augmented samples. This should be a 2N x 1 vector.
    # 提示：计算所有增强样本的分母值。这应该是一个2N x 1向量。
    # denom = None
    denom = torch.sum(exponential, dim=1, keepdim=True)

    # Step 2: Compute similarity between positive pairs.
    # 步骤2：计算正对之间的相似性。
    # You can do this in two ways:
    # Option 1: Extract the corresponding indices from sim_matrix.
    # Option 2: Use sim_positive_pairs().
    # 您可以通过两种方式完成此操作：
    # 选项1：从sim_matrix中提取相应的索引。
    # 选项2：使用sim_positive_pairs()。
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    sim_pp = sim_positive_pairs(out_left, out_right)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Step 3: Compute the numerator value for all augmented samples.
    # 步骤3：计算所有增强样本的分子值。
    numerator = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    numerator = torch.exp(torch.cat([sim_pp, sim_pp]) / tau)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Step 4: Now that you have the numerator and denominator for all augmented samples, compute the total loss.
    # 步骤4：现在您已经获得了所有增强样本的分子和分母，计算总损失。
    loss = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    loss = torch.sum(-torch.log(numerator / denom)) / (2 * N)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return loss


def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
