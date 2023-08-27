import torch
import random
import torchvision.transforms as T
import numpy as np
from .image_utils import SQUEEZENET_MEAN, SQUEEZENET_STD
from scipy.ndimage.filters import gaussian_filter1d

def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.
    计算使用模型的类显著性图，用于图像X和标签y。

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
      输入图像；形状张量(N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
      X的标签；形状张量(N,)
    - model: A pretrained CNN that will be used to compute the saliency map.
      将用于计算显著性图的预训练CNN。

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    # Make sure the model is in "test" mode
    model.eval()

    # Make input tensor require gradient
    X.requires_grad_()

    saliency = None
    ##############################################################################
    # TODO: Implement this function. Perform a forward and backward pass through #
    # the model to compute the gradient of the correct class score with respect  #
    # to each input image. You first want to compute the loss over the correct   #
    # scores (we'll combine losses across a batch by summing), and then compute  #
    # the gradients with a backward pass.                                        #
    # 实现这个函数。通过模型进行前向和后向传递，计算正确类分数相对于每个输入图像的梯度。
    # 首先要计算正确分数的损失（我们将通过求和来组合批次中的损失），然后进行反向传递计算梯度。
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    y_pred = model(X).gather(1, y.view(-1, 1)).squeeze()
    y_pred.backward(torch.ones(y_pred.shape[0]))
    saliency, _ = torch.max(X.grad.abs(), 1)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return saliency

def make_fooling_image(X, target_y, model):
    """
    Generate a fooling image that is close to X, but that the model classifies
    as target_y.
    生成一个接近X的愚弄图像，但模型将其分类为target_y。

    Inputs:
    - X: Input image; Tensor of shape (1, 3, 224, 224)
      输入图像：形状张量(1, 3, 224, 224)
    - target_y: An integer in the range [0, 1000)
      一个范围在[0, 1000)的整数
    - model: A pretrained CNN
      一个预训练的CNN

    Returns:
    - X_fooling: An image that is close to X, but that is classifed as target_y
    by the model.
    """
    # Initialize our fooling image to the input image, and make it require gradient
    X_fooling = X.clone()
    X_fooling = X_fooling.requires_grad_()

    learning_rate = 1
    ##############################################################################
    # TODO: Generate a fooling image X_fooling that the model will classify as   #
    # the class target_y. You should perform gradient ascent on the score of the #
    # target class, stopping when the model is fooled.                           #
    # When computing an update step, first normalize the gradient:               #
    #   dX = learning_rate * g / ||g||_2                                         #
    # 生成一个愚弄图像X_fooling，模型将其分类为target_y类。
    # 您应该对目标类的分数执行梯度上升，当模型被愚弄时停止。
    # 计算更新步骤时，首先归一化梯度：
    #   dX = learning_rate * g / ||g||_2
    #                                                                            #
    # You should write a training loop.                                          #
    # 您应该编写一个训练循环。
    #                                                                            #
    # HINT: For most examples, you should be able to generate a fooling image    #
    # in fewer than 100 iterations of gradient ascent.                           #
    # You can print your progress over iterations to check your algorithm.       #
    # 大多数情况下，您应该能够在梯度上升的100次迭代中生成愚弄图像。
    # 您可以打印迭代过程中的进度以检查算法。
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    while(True):
        outputs = model(X_fooling)
        y_pred = outputs[:, target_y]
        if outputs.argmax() == target_y:
            break
        y_pred.backward()
        X_fooling.data += learning_rate * X_fooling.grad / X_fooling.grad.norm()
        X_fooling.grad.data.zero_()
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return X_fooling

def class_visualization_update_step(img, model, target_y, l2_reg, learning_rate):
    ########################################################################
    # TODO: Use the model to compute the gradient of the score for the     #
    # class target_y with respect to the pixels of the image, and make a   #
    # gradient step on the image using the learning rate. Don't forget the #
    # L2 regularization term!                                              #
    # Be very careful about the signs of elements in your code.            #
    # 使用模型计算图像像素的类target_y的分数的梯度，并使用学习率对图像进行梯度步骤。
    # 不要忘记L2正则化项！
    # 在代码中非常小心元素的符号。
    ########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    outputs = model(img)
    y_pred = outputs[:, target_y]
    loss = y_pred - l2_reg * img.norm() ** 2
    loss.backward()
    img.data += learning_rate * img.grad / img.grad.norm()
    img.grad.data.zero_()
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ########################################################################
    #                             END OF YOUR CODE                         #
    ########################################################################


def preprocess(img, size=224):
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=SQUEEZENET_MEAN.tolist(),
                    std=SQUEEZENET_STD.tolist()),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(img)

def deprocess(img, should_rescale=True):
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=(1.0 / SQUEEZENET_STD).tolist()),
        T.Normalize(mean=(-SQUEEZENET_MEAN).tolist(), std=[1, 1, 1]),
        T.Lambda(rescale) if should_rescale else T.Lambda(lambda x: x),
        T.ToPILImage(),
    ])
    return transform(img)

def rescale(x):
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled

def blur_image(X, sigma=1):
    X_np = X.cpu().clone().numpy()
    X_np = gaussian_filter1d(X_np, sigma, axis=2)
    X_np = gaussian_filter1d(X_np, sigma, axis=3)
    X.copy_(torch.Tensor(X_np).type_as(X))
    return X

def jitter(X, ox, oy):
    """
    Helper function to randomly jitter an image.

    Inputs
    - X: PyTorch Tensor of shape (N, C, H, W)
    - ox, oy: Integers giving number of pixels to jitter along W and H axes

    Returns: A new PyTorch Tensor of shape (N, C, H, W)
    """
    if ox != 0:
        left = X[:, :, :, :-ox]
        right = X[:, :, :, -ox:]
        X = torch.cat([right, left], dim=3)
    if oy != 0:
        top = X[:, :, :-oy]
        bottom = X[:, :, -oy:]
        X = torch.cat([bottom, top], dim=2)
    return X
