from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10
import random
import torch

def compute_train_transform(seed=123456):
    """
    This function returns a composition of data augmentations to a single training image.
    Complete the following lines. Hint: look at available functions in torchvision.transforms
    该函数返回对单个训练图像的数据增强组合。
    完成以下行。提示：查看torchvision.transforms中的可用函数
    """
    random.seed(seed)
    torch.random.manual_seed(seed)

    # Transformation that applies color jitter with brightness=0.4, contrast=0.4, saturation=0.4, and hue=0.1
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)

    train_transform = transforms.Compose([
        ##############################################################################
        # TODO: Start of your code.                                                  #
        # 从此处开始编写代码，实现数据增强组合。
        #                                                                            #
        # Hint: Check out transformation functions defined in torchvision.transforms #
        # The first operation is filled out for you as an example.                   #
        # 检查torchvision.transforms中定义的转换函数
        # 第一个操作作为示例填充给你。
        ##############################################################################
        # Step 1: Randomly resize and crop to 32x32.
        # 步骤1：随机调整大小并裁剪为32x32。
        transforms.RandomResizedCrop(32),
        # Step 2: Horizontally flip the image with probability 0.5
        # 步骤2：水平翻转图像的概率为0.5
        transforms.RandomHorizontalFlip(0.5),
        # Step 3: With a probability of 0.8, apply color jitter (you can use "color_jitter" defined above.
        # 步骤3：以0.8的概率应用颜色抖动（可以使用上面定义的“color_jitter”。
        transforms.RandomApply([color_jitter], p=0.8),
        # Step 4: With a probability of 0.2, convert the image to grayscale
        # 步骤4：以0.2的概率将图像转换为灰度
        transforms.RandomGrayscale(p=0.2),
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    return train_transform

def compute_test_transform():
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    return test_transform


class CIFAR10Pair(CIFAR10):
    """
    CIFAR10 Dataset.
    """
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        x_i = None
        x_j = None

        if self.transform is not None:
            ##############################################################################
            # TODO: Start of your code.                                                  #
            # 从此处开始编写代码，实现数据增强组合。
            #                                                                            #
            # Apply self.transform to the image to produce x_i and x_j in the paper      #
            ##############################################################################
            x_i, x_j = self.transform(img), self.transform(img)
            ##############################################################################
            #                               END OF YOUR CODE                             #
            ##############################################################################

        if self.target_transform is not None:
            target = self.target_transform(target)

        return x_i, x_j, target
