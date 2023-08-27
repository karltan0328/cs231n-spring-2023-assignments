import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import sampler

import PIL

NOISE_DIM = 96

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def sample_noise(batch_size, dim, seed=None):
    """
    Generate a PyTorch Tensor of uniform random noise.
    生成均匀随机噪声的PyTorch张量。

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
      要生成的噪声的批量大小的整数。
    - dim: Integer giving the dimension of noise to generate.
      要生成的噪声的维度的整数。

    Output:
    - A PyTorch Tensor of shape (batch_size, dim) containing uniform
      random noise in the range (-1, 1).
      包含范围（-1,1）内的均匀随机噪声的形状（batch_size，dim）的PyTorch张量。
    """
    if seed is not None:
        torch.manual_seed(seed)

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return torch.rand(batch_size, dim) * 2 - 1 # (0, 1) -> (-1, 1)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

def discriminator(seed=None):
    """
    Build and return a PyTorch model implementing the architecture above.
    """

    if seed is not None:
        torch.manual_seed(seed)

    model = None

    ##############################################################################
    # TODO: Implement architecture                                               #
    # 实现架构
    #                                                                            #
    # HINT: nn.Sequential might be helpful. You'll start by calling Flatten().   #
    # nn.Sequential可能有用。您将首先调用Flatten（）。
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    model = nn.Sequential(
        Flatten(),
        nn.Linear(784, 256),
        nn.LeakyReLU(0.01),
        nn.Linear(256, 256),
        nn.LeakyReLU(0.01),
        nn.Linear(256, 1)
    )
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return model

def generator(noise_dim=NOISE_DIM, seed=None):
    """
    Build and return a PyTorch model implementing the architecture above.
    构建并返回实现上述架构的PyTorch模型。
    """

    if seed is not None:
        torch.manual_seed(seed)

    model = None

    ##############################################################################
    # TODO: Implement architecture                                               #
    # 实现架构
    #                                                                            #
    # HINT: nn.Sequential might be helpful.                                      #
    # nn.Sequential可能有用。
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # Generator
    # Now to build the generator network:
    #  * Fully connected layer from noise_dim to 1024
    #  * `ReLU`
    #  * Fully connected layer with size 1024
    #  * `ReLU`
    #  * Fully connected layer with size 784
    #  * `TanH` (to clip the image to be in the range of [-1,1])

    model = nn.Sequential(
        nn.Linear(noise_dim, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 784),
        nn.Tanh()
    )
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return model

def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function in PyTorch.

    Inputs:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.

    Returns:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of input data.
    """
    bce = nn.BCEWithLogitsLoss()
    return bce(input.squeeze(), target)

def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss described above.
    计算上述鉴别器损失。

    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
      给出真实数据的分数的PyTorch张量。
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
      给出假数据的分数的PyTorch张量。

    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
      包含鉴别器损失的PyTorch张量（标量）。
    """
    loss = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    loss_real = bce_loss(logits_real, torch.ones_like(logits_real).type(dtype).squeeze())
    loss_fake = bce_loss(logits_fake, torch.zeros_like(logits_fake).type(dtype).squeeze())
    loss = loss_real + loss_fake
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss

def generator_loss(logits_fake):
    """
    Computes the generator loss described above.
    计算上述发生器损失。

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
      给出假数据的分数的PyTorch张量。

    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
      包含发生器损失的PyTorch张量（标量）。
    """
    loss = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    loss = bce_loss(logits_fake, torch.ones_like(logits_fake).type(dtype).squeeze())
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss

def get_optimizer(model):
    """
    Construct and return an Adam optimizer for the model with learning rate 1e-3,
    beta1=0.5, and beta2=0.999.
    构造并返回一个Adam优化器，该优化器的学习率为1e-3，beta1 = 0.5，beta2 = 0.999。

    Input:
    - model: A PyTorch model that we want to optimize.
      我们要优化的PyTorch模型。

    Returns:
    - An Adam optimizer for the model with the desired hyperparameters.
      一个Adam优化器，具有所需的超参数。
    """
    optimizer = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999))
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return optimizer

def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.
    计算鉴别器的最小二乘GAN损失。

    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
      给出真实数据的分数的PyTorch张量。
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
      给出假数据的分数的PyTorch张量。

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
      包含损失的PyTorch张量。
    """
    loss = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    loss_real = torch.mean((scores_real - 1) ** 2) / 2
    loss_fake = torch.mean((scores_fake) ** 2) / 2
    loss = loss_real + loss_fake
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss

def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.
    计算发生器的最小二乘GAN损失。

    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
      给出假数据的分数的PyTorch张量。

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
      包含损失的PyTorch张量。
    """
    loss = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    loss = torch.mean((scores_fake - 1) ** 2) / 2
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss

def build_dc_classifier(batch_size):
    """
    Build and return a PyTorch model for the DCGAN discriminator implementing
    the architecture above.
    构建并返回一个PyTorch模型，用于实现上述架构的DCGAN鉴别器。
    """

    ##############################################################################
    # TODO: Implement architecture                                               #
    # 实现架构
    #                                                                            #
    # HINT: nn.Sequential might be helpful.                                      #
    # nn.Sequential可能有用。
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # * Conv2D: 32 Filters, 5x5, Stride 1
    # * Leaky ReLU(alpha=0.01)
    # * Max Pool 2x2, Stride 2
    # * Conv2D: 64 Filters, 5x5, Stride 1
    # * Leaky ReLU(alpha=0.01)
    # * Max Pool 2x2, Stride 2
    # * Flatten
    # * Fully Connected with output size 4 x 4 x 64
    # * Leaky ReLU(alpha=0.01)
    # * Fully Connected with output size 1
    model = nn.Sequential(
        Unflatten(N=batch_size, C=1, H=28, W=28),
        nn.Conv2d(1, 32, kernel_size=5, stride=1),  # 28 -> 24
        nn.LeakyReLU(0.01),
        nn.MaxPool2d(kernel_size=2, stride=2),      # 24 -> 12
        nn.Conv2d(32, 64, kernel_size=5, stride=1), # 12 -> 8
        nn.LeakyReLU(0.01),
        nn.MaxPool2d(kernel_size=2, stride=2),      # 8 -> 4
        Flatten(),
        nn.Linear(4 * 4 * 64, 4 * 4 * 64),
        nn.LeakyReLU(0.01),
        nn.Linear(4 * 4 * 64, 1)
    )
    return model
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################


def build_dc_generator(noise_dim=NOISE_DIM):
    """
    Build and return a PyTorch model implementing the DCGAN generator using
    the architecture described above.
    构建并返回一个PyTorch模型，该模型使用上述描述的架构实现DCGAN发生器。
    """

    ##############################################################################
    # TODO: Implement architecture                                               #
    # 实现架构
    #                                                                            #
    # HINT: nn.Sequential might be helpful.                                      #
    # nn.Sequential可能有用。
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # * Fully connected with output size 1024
    # * `ReLU`
    # * BatchNorm
    # * Fully connected with output size 7 x 7 x 128
    # * ReLU
    # * BatchNorm
    # * Use `Unflatten()` to reshape into Image Tensor of shape 7, 7, 128
    # * ConvTranspose2d: 64 filters of 4x4, stride 2, 'same' padding (use `padding=1`)
    # * `ReLU`
    # * BatchNorm
    # * ConvTranspose2d: 1 filter of 4x4, stride 2, 'same' padding (use `padding=1`)
    # * `TanH`
    # * Should have a 28x28x1 image, reshape back into 784 vector (using `Flatten()`)
    model = nn.Sequential(
        nn.Linear(noise_dim, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Linear(1024, 7 * 7 * 128),
        nn.ReLU(),
        nn.BatchNorm1d(7 * 7 * 128),
        Unflatten(N=-1, C=128, H=7, W=7),
        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
        nn.Tanh(),
        Flatten()
    )
    return model
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

def run_a_gan(D, G, D_solver, G_solver, discriminator_loss, generator_loss, loader_train, show_every=250,
              batch_size=128, noise_size=96, num_epochs=10):
    """
    Train a GAN!

    Inputs:
    - D, G: PyTorch models for the discriminator and generator
    - D_solver, G_solver: torch.optim Optimizers to use for training the
      discriminator and generator.
    - discriminator_loss, generator_loss: Functions to use for computing the generator and
      discriminator loss, respectively.
    - show_every: Show samples after every show_every iterations.
    - batch_size: Batch size to use for training.
    - noise_size: Dimension of the noise to use as input to the generator.
    - num_epochs: Number of epochs over the training dataset to use for training.
    """
    images = []
    iter_count = 0
    for epoch in range(num_epochs):
        for x, _ in loader_train:
            if len(x) != batch_size:
                continue
            D_solver.zero_grad()
            real_data = x.type(dtype)
            logits_real = D(2* (real_data - 0.5)).type(dtype)

            g_fake_seed = sample_noise(batch_size, noise_size).type(dtype)
            fake_images = G(g_fake_seed).detach()
            logits_fake = D(fake_images.view(batch_size, 1, 28, 28))

            d_total_error = discriminator_loss(logits_real, logits_fake)
            d_total_error.backward()
            D_solver.step()

            G_solver.zero_grad()
            g_fake_seed = sample_noise(batch_size, noise_size).type(dtype)
            fake_images = G(g_fake_seed)

            gen_logits_fake = D(fake_images.view(batch_size, 1, 28, 28))
            g_error = generator_loss(gen_logits_fake)
            g_error.backward()
            G_solver.step()

            if (iter_count % show_every == 0):
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count,d_total_error.item(),g_error.item()))
                imgs_numpy = fake_images.data.cpu().numpy()
                images.append(imgs_numpy[0:16])

            iter_count += 1

    return images



class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset.
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples


class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

class Unflatten(nn.Module):
    """
    An Unflatten module receives an input of shape (N, C*H*W) and reshapes it
    to produce an output of shape (N, C, H, W).
    """
    def __init__(self, N=-1, C=128, H=7, W=7):
        super(Unflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W
    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)

def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight.data)

def preprocess_img(x):
    return 2 * x - 1.0

def deprocess_img(x):
    return (x + 1.0) / 2.0

def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def count_params(model):
    """Count the number of parameters in the model. """
    param_count = np.sum([np.prod(p.size()) for p in model.parameters()])
    return param_count
