### datasets

from __future__ import absolute_import
import numpy as np
import warnings
import torchvision


__factory=['cifar10', 'svhn']



def create(name, root, download=True, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    name : str
        The dataset name. Can be one of 'cifar10', 'mnist', 'cifar100'
    root : str
        The path to the dataset directory.
    download : bool, optional
        If True, will download the dataset. Default: False
    """
    if name == 'cifar10':
        data = {}
        trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=download, transform=None)
        #  data['train'] = [trainset.train_data, np.array(trainset.train_labels)] ### pytorch 1.0
        data['train'] = [trainset.data, np.array(trainset.targets)] ### pytorch 1.3
        testset = torchvision.datasets.CIFAR10(root=root, train=False, download=download, transform=None)
        #  data['test'] = [testset.test_data, np.array(testset.test_labels)] ### pytorch 1.0
        data['test'] = [testset.data, np.array(testset.targets)] ### pytorch 1.3
        return  data
    elif name == 'svhn':
        data = {}
        trainset = torchvision.datasets.SVHN(root=root, split='train', download=download, transform=None)
        data['train'] = [trainset.data, np.array(trainset.labels)] ### pytorch 1.0
        testset = torchvision.datasets.SVHN(root=root, split='test', download=download, transform=None)
        data['test'] = [testset.data, np.array(testset.labels)] ### pytorch 1.0
        return  data
    elif name == 'mnist':
        data = {}
        trainset = torchvision.datasets.MNIST(root=root, train=True, download=download, transform=None)
        data['train'] = [np.array(trainset.train_data), np.array(trainset.train_labels)] ### pytorch 1.0
        testset = torchvision.datasets.MNIST(root=root, train=False, download=download, transform=None)
        data['test'] = [np.array(testset.test_data), np.array(testset.test_labels)] ### pytorch 1.0
        return  data
    else:
        raise KeyError("Unknown dataset:", name)


def get_dataset(name, root, *args, **kwargs):
    warnings.warn("get_dataset is deprecated. Use create instead.")
    return create(name, root, *args, **kwargs)


#### loss

from .soft_cross_entropy_loss import SoftCrossEntropyLoss


__all__ = [
    'SoftCrossEntropyLoss'
]


from torch import nn
import torch.nn.functional as F


class SoftCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftCrossEntropyLoss, self).__init__()

    def forward(self, inputs, targets, weights=None):
        loss = F.cross_entropy(inputs, targets, reduction='none')
        if weights is None:
            return loss.mean()
        loss = loss * weights
        return loss.mean()

    def update_clusters(self,clusters):
        self.clusters = clusters



#### models
'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()


#### util
import numpy as np
from torch.utils.data import DataLoader


def get_augmentation_func_list(aug_list, config):
    if aug_list is None: return []
    assert isinstance(aug_list, list)
    aug_func = []
    for aug in aug_list:
        if aug == 'rf':
            aug_func += [T.RandomHorizontalFlip()]
        elif aug == 'rc':
            aug_func += [T.RandomCrop(config.height, padding=config.padding)]
        elif aug == 're':
            aug_func += [T.RandomErasing(probability=0.5, sh=0.4, r1=0.3)]
        else:
            raise ValueError('wrong augmentation name')
    return aug_func



def get_transformer(config, is_training=False):
    normalizer = T.Normalize(mean=config.mean, std=config.std)
    base_transformer = [T.ToTensor(), normalizer]
    if not is_training:
        return T.Compose(base_transformer)
    aug1 = T.RandomErasing(probability=0.5, sh=0.4, r1=0.3)
    early_aug = get_augmentation_func_list(config.early_transform, config)
    later_aug = get_augmentation_func_list(config.later_transform, config)
    aug_list = early_aug + base_transformer + later_aug
    return T.Compose(aug_list)


def get_dataloader(dataset, config, is_training=False):
    transformer = get_transformer(config, is_training=is_training)
    sampler = None
    if is_training and config.sampler:
        sampler = config.sampler(dataset, config.num_instances)
    data_loader = DataLoader(Preprocessor(dataset, transform=transformer),
                             batch_size=config.batch_size,
                             num_workers=config.workers,
                             shuffle=is_training,
                             sampler=sampler,
                             pin_memory=True,
                             drop_last=is_training)
    return data_loader


def update_train_untrain(sel_idx,
                         train_data,
                         untrain_data,
                         pred_y,
                         weights=None):
    #  assert len(train_data) == len(untrain_data)
    if weights is None:
        weights = np.ones(len(untrain_data[0]), dtype=np.float32)
    add_data = [untrain_data[0][sel_idx], pred_y[sel_idx], weights[sel_idx]]
    new_untrain = [
      untrain_data[0][~sel_idx], pred_y[~sel_idx], weights[~sel_idx]
    ]
    new_train = [
      np.concatenate((d1, d2)) for d1, d2 in zip(train_data, add_data)
    ]
    return new_train, new_untrain




def select_ids(score, train_data, max_add):
    y = train_data[1]
    add_indices = np.zeros(score.shape[0])
    clss = np.unique(y)
    assert score.shape[1] == len(clss)
    pred_y = np.argmax(score, axis=1)
    ratio_per_class = [sum(y == c)/len(y) for c in clss]
    for cls in range(len(clss)):
        indices = np.where(pred_y == cls)[0]
        cls_score = score[indices, cls]
        idx_sort = np.argsort(cls_score)
        add_num = min(int(np.ceil(ratio_per_class[cls] * max_add)),
                      indices.shape[0])
        add_indices[indices[idx_sort[-add_num:]]] = 1
    return add_indices.astype('bool')


def get_lambda_class(score, pred_y, train_data, max_add):
    y = train_data[1]
    lambdas = np.zeros(score.shape[1])
    add_ids = np.zeros(score.shape[0])
    clss = np.unique(y)
    assert score.shape[1] == len(clss)
    ratio_per_class = [sum(y == c)/len(y) for c in clss]
    for cls in range(len(clss)):
        indices = np.where(pred_y == cls)[0]
        if len(indices) == 0:
            continue
        cls_score = score[indices, cls]
        idx_sort = np.argsort(cls_score)
        add_num = min(int(np.ceil(ratio_per_class[cls] * max_add)),
                      indices.shape[0])
        add_ids[indices[idx_sort[-add_num:]]] = 1
        lambdas[cls] = cls_score[idx_sort[-add_num]] - 0.1
    return add_ids.astype('bool'), lambdas


def get_ids_weights(pred_prob, pred_y, train_data, max_add, gamma, regularizer='hard'):
    '''
    pred_prob: predicted probability of all views on untrain data
    pred_y: predicted label for untrain data
    train_data: training data
    max_add: number of selected data
    gamma: view correlation hyper-parameter
    '''

    add_ids, lambdas = get_lambda_class(pred_prob, pred_y, train_data, max_add)
    weight = np.array([(pred_prob[i, l] - lambdas[l]) / (gamma + 1e-5)
                       for i, l in enumerate(pred_y)],
                      dtype='float32')
    weight[~add_ids] = 0
    if regularizer == 'hard' or gamma == 0:
        weight[add_ids] = 1
        return add_ids, weight
    weight[weight < 0] = 0
    weight[weight > 1] = 1
    return add_ids, weight

def update_ids_weights(view, probs, sel_ids, weights, pred_y, train_data,
                       max_add, gamma, regularizer='hard'):
    num_view = len(probs)
    for v in range(num_view):
        if v == view:
            continue
        ov = sel_ids[v]
        probs[view][ov, pred_y[ov]] += gamma * weights[v][ov] / (num_view - 1)
    sel_id, weight = get_ids_weights(probs[view], pred_y, train_data,
                                     max_add, gamma, regularizer)
    return sel_id, weight

def get_weights(pred_prob, pred_y, train_data, max_add, gamma, regularizer):
    lamb = get_lambda_class(pred_prob, pred_y, train_data, max_add)
    weight = np.array([(pred_prob[i, l] - lamb[l]) / gamma
                       for i, l in enumerate(pred_y)],
                      dtype='float32')
    if regularizer is 'hard':
        weight[weight > 0] = 1
        return weight
    weight[weight > 1] = 1
    return weight


def split_dataset(dataset, train_ratio=0.2, seed=0, num_per_class=400):
    """
    split dataset to train_set and untrain_set
    """
    assert 0 <= train_ratio <= 1
    np.random.seed(seed)
    pids = np.array(dataset[1])
    clss = np.unique(pids)
    sel_ids = np.zeros(len(dataset[0]), dtype=bool)
    for cls in clss:
        indices = np.where(pids == cls)[0]
        np.random.shuffle(indices)
        if num_per_class:
            sel_id = indices[:num_per_class]
        else:
            train_num = int(np.ceil((len(indices) * train_ratio)))
            sel_id = indices[:train_num]
        sel_ids[sel_id] = True
    train_set = [d[sel_ids] for d in dataset]
    untrain_set = [d[~sel_ids] for d in dataset]
    ### add sample weight
    train_set += [np.full((len(train_set[0])), 1.0)]
    return train_set, untrain_set



from .augmentation import *
from PIL import ImageOps, ImageEnhance, ImageFilter, Image
import random
import math
import numpy as np
import torch


class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''

    def __init__(self,
                 probability=0.5,
                 sl=0.02,
                 sh=0.4,
                 r1=0.3,
                 mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img

class RandomPolicy(object):
    '''
    Class RandomPolicy for augment data
    -------------------------------------------------------------------------------------
    '''

    def __init__(self, data_name='cifar10'):
      if data_name == 'cifar10':
        self.policies = cifar10_policies()

    def pil_wrap(self, img):
        """Convert the PIL image to RGBA"""

        return img.convert('RGBA')


    def pil_unwrap(self, pil_img, img_shape):
        """Converts the PIL RGBA img to a RGB image."""
        pic_array = np.array(pil_img)
        #  import pdb;pdb.set_trace()
        i1, i2 = np.where(pic_array[:, :, 3] == 0)
        pic_array = pic_array[:,:,:3]
        pic_array[i1, i2] = [0, 0, 0]
        return Image.fromarray(pic_array)

    def __call__(self, img):
        policy_idx = np.random.randint(len(self.policies))
        policy = self.policies[policy_idx]
        img_shape = img.size
        pil_img = self.pil_wrap(img)
        for xform in policy:
            assert len(xform) == 3
            name, probability, level = xform
            xform_fn = NAME_TO_TRANSFORM[name].pil_transformer(
              probability, level, img_shape)
            pil_img = xform_fn(pil_img)
        return self.pil_unwrap(pil_img, img_shape)



from torchvision.transforms import *
from PIL import ImageOps, ImageEnhance, ImageFilter, Image
import random
import math
import numpy as np
import torch


PARAMETER_MAX = 10

def float_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval .

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.

  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
  return float(level) * maxval / PARAMETER_MAX


def int_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval .

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.

  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
  return int(level * maxval / PARAMETER_MAX)

class TransformFunction(object):
    """Wraps the Transform function for pretty printing options."""

    def __init__(self, func, name):
        self.f = func
        self.name = name

    def __repr__(self):
        return '<' + self.name + '>'

    def __call__(self, pil_img):
        return self.f(pil_img)


class TransformT(object):
    """Each instance of this class represents a specific transform."""

    def __init__(self, name, xform_fn):
        self.name = name
        self.xform = xform_fn

    def pil_transformer(self, probability, level, img_shape):
        def return_function(im):
            if random.random() < probability:
                im = self.xform(im, level, img_shape)
            return im

        name = self.name + '({:.1f},{})'.format(probability, level)
        return TransformFunction(return_function, name)


identity = TransformT('identity', lambda pil_img, level, _: pil_img)
flip_lr = TransformT(
  'FlipLR', lambda pil_img, level, _: pil_img.transpose(Image.FLIP_LEFT_RIGHT))
flip_ud = TransformT(
  'FlipUD', lambda pil_img, level, _: pil_img.transpose(Image.FLIP_TOP_BOTTOM))
# pylint:disable=g-long-lambda
auto_contrast = TransformT(
  'AutoContrast', lambda pil_img, level, _: ImageOps.autocontrast(
    pil_img.convert('RGB')).convert('RGBA'))
equalize = TransformT(
  'Equalize', lambda pil_img, level, _: ImageOps.equalize(
    pil_img.convert('RGB')).convert('RGBA'))
invert = TransformT(
  'Invert', lambda pil_img, level, _: ImageOps.invert(pil_img.convert('RGB')).
  convert('RGBA'))
# pylint:enable=g-long-lambda
blur = TransformT(
  'Blur', lambda pil_img, level, _: pil_img.filter(ImageFilter.BLUR))
smooth = TransformT(
  'Smooth', lambda pil_img, level, _: pil_img.filter(ImageFilter.SMOOTH))


def _rotate_impl(pil_img, level, _):
    """Rotates `pil_img` from -30 to 30 degrees depending on `level`."""
    degrees = int_parameter(level, 30)
    if random.random() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees)


rotate = TransformT('Rotate', _rotate_impl)


def _posterize_impl(pil_img, level, _):
    """Applies PIL Posterize to `pil_img`."""
    level = int_parameter(level, 4)
    return ImageOps.posterize(pil_img.convert('RGB'),
                              4 - level).convert('RGBA')


posterize = TransformT('Posterize', _posterize_impl)


def _shear_x_impl(pil_img, level, img_shape):
    """Applies PIL ShearX to `pil_img`.

  The ShearX operation shears the image along the horizontal axis with `level`
  magnitude.

  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].

  Returns:
    A PIL Image that has had ShearX applied to it.
  """
    level = float_parameter(level, 0.3)
    if random.random() > 0.5:
        level = -level
    return pil_img.transform((img_shape[0], img_shape[1]), Image.AFFINE,
                             (1, level, 0, 0, 1, 0))


shear_x = TransformT('ShearX', _shear_x_impl)


def _shear_y_impl(pil_img, level, img_shape):
    """Applies PIL ShearY to `pil_img`.

  The ShearY operation shears the image along the vertical axis with `level`
  magnitude.

  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].

  Returns:
    A PIL Image that has had ShearX applied to it.
  """
    level = float_parameter(level, 0.3)
    if random.random() > 0.5:
        level = -level
    return pil_img.transform((img_shape[0], img_shape[1]), Image.AFFINE,
                             (1, 0, 0, level, 1, 0))


shear_y = TransformT('ShearY', _shear_y_impl)


def _translate_x_impl(pil_img, level, img_shape):
    """Applies PIL TranslateX to `pil_img`.

  Translate the image in the horizontal direction by `level`
  number of pixels.

  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].

  Returns:
    A PIL Image that has had TranslateX applied to it.
  """
    level = int_parameter(level, 10)
    if random.random() > 0.5:
        level = -level
    return pil_img.transform((img_shape[0], img_shape[1]), Image.AFFINE,
                             (1, 0, level, 0, 1, 0))


translate_x = TransformT('TranslateX', _translate_x_impl)


def _translate_y_impl(pil_img, level, img_shape):
    """Applies PIL TranslateY to `pil_img`.

  Translate the image in the vertical direction by `level`
  number of pixels.

  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].

  Returns:
    A PIL Image that has had TranslateY applied to it.
  """
    level = int_parameter(level, 10)
    if random.random() > 0.5:
        level = -level
    return pil_img.transform((img_shape[0], img_shape[1]), Image.AFFINE,
                             (1, 0, 0, 0, 1, level))


translate_y = TransformT('TranslateY', _translate_y_impl)


def _crop_impl(pil_img, level, img_shape, interpolation=Image.BILINEAR):
    """Applies a crop to `pil_img` with the size depending on the `level`."""
    cropped = pil_img.crop(
      (level, level, img_shape[0] - level, img_shape[1] - level))
    resized = cropped.resize((img_shape[0], img_shape[1]), interpolation)
    return resized


crop_bilinear = TransformT('CropBilinear', _crop_impl)


def _solarize_impl(pil_img, level, _):
    """Applies PIL Solarize to `pil_img`.

  Translate the image in the vertical direction by `level`
  number of pixels.

  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].

  Returns:
    A PIL Image that has had Solarize applied to it.
  """
    level = int_parameter(level, 256)
    return ImageOps.solarize(pil_img.convert('RGB'),
                             256 - level).convert('RGBA')


solarize = TransformT('Solarize', _solarize_impl)


def _enhancer_impl(enhancer):
    """Sets level to be between 0.1 and 1.8 for ImageEnhance transforms of PIL."""

    def impl(pil_img, level, _):
        v = float_parameter(level, 1.8) + .1  # going to 0 just destroys it
        return enhancer(pil_img).enhance(v)

    return impl


color = TransformT('Color', _enhancer_impl(ImageEnhance.Color))
contrast = TransformT('Contrast', _enhancer_impl(ImageEnhance.Contrast))
brightness = TransformT('Brightness', _enhancer_impl(ImageEnhance.Brightness))
sharpness = TransformT('Sharpness', _enhancer_impl(ImageEnhance.Sharpness))

ALL_TRANSFORMS = [
  flip_lr, flip_ud, auto_contrast, equalize, invert, rotate, posterize,
  crop_bilinear, solarize, color, contrast, brightness, sharpness, shear_x,
  shear_y, translate_x, translate_y, blur, smooth
]

NAME_TO_TRANSFORM = {t.name: t for t in ALL_TRANSFORMS}
TRANSFORM_NAMES = NAME_TO_TRANSFORM.keys()


def cifar10_policies():
  """AutoAugment policies found on CIFAR-10."""
  exp0_0 = [[("Invert", 0.1, 7), ("Contrast", 0.2, 6)],
            [("Rotate", 0.7, 2), ("TranslateX", 0.3, 9)],
            [("Sharpness", 0.8, 1), ("Sharpness", 0.9, 3)],
            [("ShearY", 0.5, 8), ("TranslateY", 0.7, 9)],
            [("AutoContrast", 0.5, 8), ("Equalize", 0.9, 2)]]
  exp0_1 = [[("Solarize", 0.4, 5), ("AutoContrast", 0.9, 3)],
            [("TranslateY", 0.9, 9), ("TranslateY", 0.7, 9)],
            [("AutoContrast", 0.9, 2), ("Solarize", 0.8, 3)],
            [("Equalize", 0.8, 8), ("Invert", 0.1, 3)],
            [("TranslateY", 0.7, 9), ("AutoContrast", 0.9, 1)]]
  exp0_2 = [[("Solarize", 0.4, 5), ("AutoContrast", 0.0, 2)],
            [("TranslateY", 0.7, 9), ("TranslateY", 0.7, 9)],
            [("AutoContrast", 0.9, 0), ("Solarize", 0.4, 3)],
            [("Equalize", 0.7, 5), ("Invert", 0.1, 3)],
            [("TranslateY", 0.7, 9), ("TranslateY", 0.7, 9)]]
  exp0_3 = [[("Solarize", 0.4, 5), ("AutoContrast", 0.9, 1)],
            [("TranslateY", 0.8, 9), ("TranslateY", 0.9, 9)],
            [("AutoContrast", 0.8, 0), ("TranslateY", 0.7, 9)],
            [("TranslateY", 0.2, 7), ("Color", 0.9, 6)],
            [("Equalize", 0.7, 6), ("Color", 0.4, 9)]]
  exp1_0 = [[("ShearY", 0.2, 7), ("Posterize", 0.3, 7)],
            [("Color", 0.4, 3), ("Brightness", 0.6, 7)],
            [("Sharpness", 0.3, 9), ("Brightness", 0.7, 9)],
            [("Equalize", 0.6, 5), ("Equalize", 0.5, 1)],
            [("Contrast", 0.6, 7), ("Sharpness", 0.6, 5)]]
  exp1_1 = [[("Brightness", 0.3, 7), ("AutoContrast", 0.5, 8)],
            [("AutoContrast", 0.9, 4), ("AutoContrast", 0.5, 6)],
            [("Solarize", 0.3, 5), ("Equalize", 0.6, 5)],
            [("TranslateY", 0.2, 4), ("Sharpness", 0.3, 3)],
            [("Brightness", 0.0, 8), ("Color", 0.8, 8)]]
  exp1_2 = [[("Solarize", 0.2, 6), ("Color", 0.8, 6)],
            [("Solarize", 0.2, 6), ("AutoContrast", 0.8, 1)],
            [("Solarize", 0.4, 1), ("Equalize", 0.6, 5)],
            [("Brightness", 0.0, 0), ("Solarize", 0.5, 2)],
            [("AutoContrast", 0.9, 5), ("Brightness", 0.5, 3)]]
  exp1_3 = [[("Contrast", 0.7, 5), ("Brightness", 0.0, 2)],
            [("Solarize", 0.2, 8), ("Solarize", 0.1, 5)],
            [("Contrast", 0.5, 1), ("TranslateY", 0.2, 9)],
            [("AutoContrast", 0.6, 5), ("TranslateY", 0.0, 9)],
            [("AutoContrast", 0.9, 4), ("Equalize", 0.8, 4)]]
  exp1_4 = [[("Brightness", 0.0, 7), ("Equalize", 0.4, 7)],
            [("Solarize", 0.2, 5), ("Equalize", 0.7, 5)],
            [("Equalize", 0.6, 8), ("Color", 0.6, 2)],
            [("Color", 0.3, 7), ("Color", 0.2, 4)],
            [("AutoContrast", 0.5, 2), ("Solarize", 0.7, 2)]]
  exp1_5 = [[("AutoContrast", 0.2, 0), ("Equalize", 0.1, 0)],
            [("ShearY", 0.6, 5), ("Equalize", 0.6, 5)],
            [("Brightness", 0.9, 3), ("AutoContrast", 0.4, 1)],
            [("Equalize", 0.8, 8), ("Equalize", 0.7, 7)],
            [("Equalize", 0.7, 7), ("Solarize", 0.5, 0)]]
  exp1_6 = [[("Equalize", 0.8, 4), ("TranslateY", 0.8, 9)],
            [("TranslateY", 0.8, 9), ("TranslateY", 0.6, 9)],
            [("TranslateY", 0.9, 0), ("TranslateY", 0.5, 9)],
            [("AutoContrast", 0.5, 3), ("Solarize", 0.3, 4)],
            [("Solarize", 0.5, 3), ("Equalize", 0.4, 4)]]
  exp2_0 = [[("Color", 0.7, 7), ("TranslateX", 0.5, 8)],
            [("Equalize", 0.3, 7), ("AutoContrast", 0.4, 8)],
            [("TranslateY", 0.4, 3), ("Sharpness", 0.2, 6)],
            [("Brightness", 0.9, 6), ("Color", 0.2, 8)],
            [("Solarize", 0.5, 2), ("Invert", 0.0, 3)]]
  exp2_1 = [[("AutoContrast", 0.1, 5), ("Brightness", 0.0, 0)],
            [("Equalize", 0.7, 7), ("AutoContrast", 0.6, 4)],
            [("Color", 0.1, 8), ("ShearY", 0.2, 3)],
            [("ShearY", 0.4, 2), ("Rotate", 0.7, 0)]]
  exp2_2 = [[("ShearY", 0.1, 3), ("AutoContrast", 0.9, 5)],
            [("Equalize", 0.5, 0), ("Solarize", 0.6, 6)],
            [("AutoContrast", 0.3, 5), ("Rotate", 0.2, 7)],
            [("Equalize", 0.8, 2), ("Invert", 0.4, 0)]]
  exp2_3 = [[("Equalize", 0.9, 5), ("Color", 0.7, 0)],
            [("Equalize", 0.1, 1), ("ShearY", 0.1, 3)],
            [("AutoContrast", 0.7, 3), ("Equalize", 0.7, 0)],
            [("Brightness", 0.5, 1), ("Contrast", 0.1, 7)],
            [("Contrast", 0.1, 4), ("Solarize", 0.6, 5)]]
  exp2_4 = [[("Solarize", 0.2, 3), ("ShearX", 0.0, 0)],
            [("TranslateX", 0.3, 0), ("TranslateX", 0.6, 0)],
            [("Equalize", 0.5, 9), ("TranslateY", 0.6, 7)],
            [("ShearX", 0.1, 0), ("Sharpness", 0.5, 1)],
            [("Equalize", 0.8, 6), ("Invert", 0.3, 6)]]
  exp2_5 = [[("ShearX", 0.4, 4), ("AutoContrast", 0.9, 2)],
            [("ShearX", 0.0, 3), ("Posterize", 0.0, 3)],
            [("Solarize", 0.4, 3), ("Color", 0.2, 4)],
            [("Equalize", 0.1, 4), ("Equalize", 0.7, 6)]]
  exp2_6 = [[("Equalize", 0.3, 8), ("AutoContrast", 0.4, 3)],
            [("Solarize", 0.6, 4), ("AutoContrast", 0.7, 6)],
            [("AutoContrast", 0.2, 9), ("Brightness", 0.4, 8)],
            [("Equalize", 0.1, 0), ("Equalize", 0.0, 6)],
            [("Equalize", 0.8, 4), ("Equalize", 0.0, 4)]]
  exp2_7 = [[("Equalize", 0.5, 5), ("AutoContrast", 0.1, 2)],
            [("Solarize", 0.5, 5), ("AutoContrast", 0.9, 5)],
            [("AutoContrast", 0.6, 1), ("AutoContrast", 0.7, 8)],
            [("Equalize", 0.2, 0), ("AutoContrast", 0.1, 2)],
            [("Equalize", 0.6, 9), ("Equalize", 0.4, 4)]]
  exp0s = exp0_0 + exp0_1 + exp0_2 + exp0_3
  exp1s = exp1_0 + exp1_1 + exp1_2 + exp1_3 + exp1_4 + exp1_5 + exp1_6
  exp2s = exp2_0 + exp2_1 + exp2_2 + exp2_3 + exp2_4 + exp2_5 + exp2_6 + exp2_7
  return exp0s + exp1s + exp2s


class Config(object):

    # logs dir
    logs_dir = 'logs'
    # model training parameters
    workers = 8
    dropout = 0.5
    lr = 0.1
    momentum = 0.9
    weight_decay = 1e-4
    step_size = 20
    sampler = None
    print_freq = 40
    padding = 4

    def __init__(self,
                 model_name='resnet18',
                 loss_name='softmax',
                 num_classes=10,
                 height=32,
                 width=32,
                 batch_size=128,
                 epochs=50,
                 checkpoint=None,
                 mean=[0.4914, 0.4822, 0.4465],
                 std=[0.2023, 0.1994, 0.2010],
                 early_transform=['rc', 'rf'],
                 later_transform=['re']):
        self.model_name = model_name
        self.loss_name = loss_name
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.epochs = epochs
        self.checkpoint = checkpoint
        self.early_transform = early_transform
        self.later_transform = later_transform
        self.mean = mean
        self.std = std


### model utils
import torch
from torch import nn
import models
from util.data import data_process as dp
import numpy as np
from collections import OrderedDict
from loss import SoftCrossEntropyLoss


def get_optim_params(config, params):

    if config.loss_name not in ['softmax', 'weight_softmax']:
        raise ValueError('wrong loss name')
    optimizer = torch.optim.SGD(params,
                                lr=config.lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay,
                                nesterov=True)
    if config.loss_name == 'softmax':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = SoftCrossEntropyLoss()
    return criterion, optimizer


def train_model(model, dataloader, config, device):
    """
    train model given the dataloader the criterion,
    stop when epochs are reached
    params:
        model: model for training
        dataloader: training data
        config: training config
        criterion
    """
    param_groups = model.parameters()
    criterion, optimizer = get_optim_params(config, param_groups)
    criterion = criterion.to(device)

    def adjust_lr(epoch, step_size=20):
        lr = 0.1 * (0.1**(epoch // step_size))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    for epoch in range(config.epochs):
        print('\nEpoch: %d' % epoch)
        model.train()
        adjust_lr(epoch, config.step_size)
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets, weights) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            weights = weights.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets, weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('Loss: %.3f | Acc: %.3f%% (%d/%d)' %
              (train_loss /
               (batch_idx + 1), 100. * correct / total, correct, total))


def train(model, train_data, config, device):
    #  model = models.create(config.model_name)
    #  model = nn.DataParallel(model).cuda()
    dataloader = dp.get_dataloader(train_data, config, is_training=True)
    train_model(model, dataloader, config, device)
    #  return model


def predict_prob(model, data, config, device):
    model.eval()
    dataloader = dp.get_dataloader(data, config)
    probs = []
    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            prob = nn.functional.softmax(output, dim=1)
            probs += [prob.data.cpu().numpy()]
    return np.concatenate(probs)


def evaluate(model, data, config, device):
    model.eval()
    correct = 0
    total = 0
    dataloader = dp.get_dataloader(data, config)
    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    print('Accuracy on Test data: %0.5f' % acc)
    return acc



### spamco
import os
import torch
import argparse
import model_utils as mu
from util.data import data_process as dp
from config import Config
from util.serialization import load_checkpoint, save_checkpoint
import datasets
import models
import numpy as np
import torch.multiprocessing as mp

parser = argparse.ArgumentParser(description='soft_spaco')
parser.add_argument('-s', '--seed', type=int, default=0)
parser.add_argument('-r', '--regularizer', type=str, default='hard')
parser.add_argument('-d', '--dataset', type=str, default='cifar10')
parser.add_argument('--gamma', type=float, default=0.3)
parser.add_argument('--iter-steps', type=int, default=5)
parser.add_argument('--num-per-class', type=int, default=400)
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)


def train_predict(net, train_data, untrain_data, test_data, config, device, pred_probs):
    mu.train(net, train_data, config, device)
    pred_probs.append(mu.predict_prob(net, untrain_data, configs[view], view))


def parallel_train(nets, train_data, data_dir, configs):
    processes = []
    for view, net in enumerate(nets):
        p = mp.Process(target=mu.train, args=(net, train_data, config, view))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()




def adjust_config(config, num_examples, iter_step):
    repeat = 20 * (1.1 ** iter_step)
    #  epochs = list(range(300, 20, -20))
    #  config.epochs = epochs[iter_step]
    #  config.epochs = int((50000 * repeat) // num_examples)
    config.epochs = 200
    config.step_size = max(int(config.epochs // 3), 1)
    return config

def spaco(configs,
          data,
          iter_steps=1,
          gamma=0,
          train_ratio=0.2,
          regularizer='soft'):
    """
    self-paced co-training model implementation based on Pytroch
    params:
    model_names: model names for spaco, such as ['resnet50','densenet121']
    data: dataset for spaco model
    save_pathts: save paths for two models
    iter_step: iteration round for spaco
    gamma: spaco hyperparameter
    train_ratio: initiate training dataset ratio
    """
    num_view = len(configs)
    train_data, untrain_data = dp.split_dataset(
      data['train'], seed=args.seed, num_per_class=args.num_per_class)
    add_num = 4000
    pred_probs = []
    test_preds = []
    sel_ids = []
    weights = []
    start_step = 0
    ###########
    # initiate classifier to get preidctions
    ###########
    for view in range(num_view):
        configs[view] = adjust_config(configs[view], len(train_data[0]), 0)
        net = models.create(configs[view].model_name).to(view)
        mu.train(net, train_data, configs[view], device=view)
        pred_probs.append(mu.predict_prob(net, untrain_data, configs[view], view))
        test_preds.append(mu.predict_prob(net, data['test'], configs[view], view))
        acc = mu.evaluate(net, data['test'], configs[view], view)
        save_checkpoint(
          {
            'state_dict': net.state_dict(),
            'epoch': 0,
          },
          False,
          fpath=os.path.join(
            'spaco/%s.epoch%d' % (configs[view].model_name, 0)))
    pred_y = np.argmax(sum(pred_probs), axis=1)

    # initiate weights for unlabled examples
    for view in range(num_view):
        sel_id, weight = dp.get_ids_weights(pred_probs[view], pred_y,
                                            train_data, add_num, gamma,
                                            regularizer)
        import pdb;pdb.set_trace()
        sel_ids.append(sel_id)
        weights.append(weight)

    # start iterative training
    gt_y = data['test'][1]
    for step in range(start_step, iter_steps):
        for view in range(num_view):
            print('Iter step: %d, view: %d, model name: %s' % (step+1,view,configs[view].model_name))

            # update sample weights
            sel_ids[view], weights[view] = dp.update_ids_weights(
              view, pred_probs, sel_ids, weights, pred_y, train_data,
              add_num, gamma, regularizer)
            # update model parameter
            new_train_data, _ = dp.update_train_untrain(
              sel_ids[view], train_data, untrain_data, pred_y, weights[view])
            configs[view] = adjust_config(configs[view], len(train_data[0]), step)
            net = models.create(configs[view].model_name).cuda()
            mu.train(net, new_train_data, configs[view], device=view)

            # update y
            pred_probs[view] = mu.predict_prob(model, untrain_data,
                                               configs[view])

            # evaluation current model and save it
            acc = mu.evaluate(net, data['test'], configs[view], device=view)
            predictions = mu.predict_prob(net, data['train'], configs[view], device=view)
            save_checkpoint(
              {
                'state_dict': net.state_dict(),
                'epoch': step + 1,
                'predictions': predictions,
                'accuracy': acc
              },
              False,
              fpath=os.path.join(
                'spaco/%s.epoch%d' % (configs[view].model_name, step + 1)))
            test_preds[view] = mu.predict_prob(model, data['test'], configs[view], device=view)
        add_num +=  4000 * num_view
        fuse_y = np.argmax(sum(test_preds), axis=1)
        print('Acc:%0.4f' % np.mean(fuse_y== gt_y))
    #  print(acc)


config1 = Config(model_name='shake_drop2', loss_name='weight_softmax')
config2 = Config(model_name='wrn', loss_name='weight_softmax')

dataset = args.dataset
cur_path = os.getcwd()
logs_dir = os.path.join(cur_path, 'logs')
data_dir = os.path.join(cur_path, 'data', dataset)
data = datasets.create(dataset, data_dir)

spaco([config1, config2],
      data,
      gamma=args.gamma,
      iter_steps=args.iter_steps,
      regularizer=args.regularizer)
