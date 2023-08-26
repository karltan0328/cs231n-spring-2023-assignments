import numpy as np

"""
This file implements various first-order update rules that are commonly used
for training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:
此文件实现了用于训练神经网络的各种一阶更新规则
每个更新规则都接受当前权重和相对于这些权重的损失梯度，并生成下一组权重
每个更新规则具有相同的接口：

def update(w, dw, config=None):

Inputs:
  - w: A numpy array giving the current weights.
    当前权重的numpy数组
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
    相对于w的损失梯度的形状与w相同的numpy数组
  - config: A dictionary containing hyperparameter values such as learning
    rate, momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.
    包含超参数值（如学习率，动量等）的字典

Returns:
  - next_w: The next point after the update.
    更新后的下一个点
  - config: The config dictionary to be passed to the next iteration of the
    update rule.
    传递给更新规则下一次迭代的config字典

NOTE: For most update rules, the default learning rate will probably not
perform well; however the default values of the other hyperparameters should
work well for a variety of different problems.
注意：对于大多数更新规则，缺省的学习率可能不会表现良好
但是其他超参数的默认值应该适用于各种不同的问题

For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.
为了提高效率，更新规则可能会执行就地更新，改变w并将next_w设置为w
"""


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.
    执行普通随机梯度下降

    config format:
    - learning_rate: Scalar learning rate.
      标量学习率
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)

    w -= config["learning_rate"] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.
    执行带动量的随机梯度下降

    config format:
    - learning_rate: Scalar learning rate.
      标量学习率
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
      介于0和1之间的标量，给出动量值
    - velocity: A numpy array of the same shape as w and dw used to store a
      moving average of the gradients.
      与w和dw形状相同的numpy数组，用于存储梯度的移动平均值
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("momentum", 0.9)
    v = config.get("velocity", np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the momentum update formula. Store the updated value in #
    # the next_w variable. You should also use and update the velocity v.     #
    # 实现动量更新公式。将更新后的值存储在next_w变量中。您还应该使用并更新速度v。
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    v = config["momentum"] * v - config["learning_rate"] * dw
    next_w = w + v
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    config["velocity"] = v

    return next_w, config


def rmsprop(w, dw, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.
    使用RMSProp更新规则，该规则使用平方梯度值的移动平均值来设置自适应的每参数学习率

    config format:
    - learning_rate: Scalar learning rate.
      标量学习率
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
      介于0和1之间的标量，给出平方梯度缓存的衰减率
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
      用于平滑以避免除以零的小标量
    - cache: Moving average of second moments of gradients.
      梯度的二阶矩的移动平均值
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("decay_rate", 0.99)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("cache", np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the RMSprop update formula, storing the next value of w #
    # in the next_w variable. Don't forget to update cache value stored in    #
    # config['cache'].                                                        #
    # 实现RMSprop更新公式，将w的下一个值存储在next_w变量中。
    # 不要忘记更新存储在config['cache']中的缓存值。
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    gs = config["decay_rate"] * config["cache"] + (1 - config["decay_rate"]) * dw ** 2
    next_w = w - config["learning_rate"] * dw / (np.sqrt(gs) + config["epsilon"])
    config["cache"] = gs
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_w, config


def adam(w, dw, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.
    使用Adam更新规则，该规则结合了梯度及其平方的移动平均值和偏差校正项

    config format:
    - learning_rate: Scalar learning rate.
      标量学习率
    - beta1: Decay rate for moving average of first moment of gradient.
      梯度一阶矩的移动平均值的衰减率
    - beta2: Decay rate for moving average of second moment of gradient.
      梯度二阶矩的移动平均值的衰减率
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
      用于平滑以避免除以零的小标量
    - m: Moving average of gradient.
      梯度的移动平均值
    - v: Moving average of squared gradient.
      梯度的平方的移动平均值
    - t: Iteration number.
      迭代次数
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", np.zeros_like(w))
    config.setdefault("v", np.zeros_like(w))
    config.setdefault("t", 0)

    next_w = None
    ###########################################################################
    # TODO: Implement the Adam update formula, storing the next value of w in #
    # the next_w variable. Don't forget to update the m, v, and t variables   #
    # stored in config.                                                       #
    # 实现Adam更新公式，将w的下一个值存储在next_w变量中。
    # 不要忘记更新存储在config中的m，v和t变量。
    #                                                                         #
    # NOTE: In order to match the reference output, please modify t _before_  #
    # using it in any calculations.                                           #
    # 为了匹配参考输出，请在任何计算之前修改t。
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    lr, beta1, beta2, eps, m, v, t = config["learning_rate"], config["beta1"], config["beta2"], config["epsilon"], config["m"], config["v"], config["t"]
    m = beta1 * m + (1 - beta1) * dw
    v = beta2 * v + (1 - beta2) * dw ** 2
    t += 1
    first_unbias = m / (1 - beta1 ** t)
    second_unbias = v / (1 - beta2 ** t)
    next_w = w - lr * first_unbias / (np.sqrt(second_unbias) + eps)
    config["m"], config["v"], config["t"] = m, v, t
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_w, config
