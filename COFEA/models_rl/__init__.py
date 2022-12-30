from .vgg import *
from .densenet import *
from .resnet import *
from .q_learn import *
from .q_learn_osi import *
from .q_learn_fea import *
from .sarsa import *
from .sarsa_osi import *
from .sarsa_fea import *
from .e_sarsa import *
from .e_sarsa_osi import *
from .e_sarsa_fea import *


__factory = {
    'resnet18': ResNet18,
    'resnet34': ResNet34,
    'resnet50': ResNet50,
    'resnet101': ResNet101,
    'resnet152': ResNet152,
    'vgg': VGG,
    'densenet121': DenseNet121,
    'sarsa': sarsa_agent,
    'sarsa_osi': sarsa_osi_agent,
    'sarsa_fea': sarsa_fea_agent,
    'e_sarsa': expected_sarsa_agent,
    'e_sarsa_osi': expected_sarsa_osi_agent,
    'e_sarsa_fea': expected_sarsa_fea_agent,
    'q_learn': q_learn_agent,
    'q_learn_osi': q_learn_osi_agent,
    'q_learn_fea': q_learn_fea_agent,
}


def names():
    return sorted(__factory.keys())


def create(name, env, *args, **kwargs):
    """
    Create a model instance.
    Parameters
    ----------
    name : str
        Model name. Can be one of 'inception', 'resnet18', 'resnet34',
        'resnet50', 'resnet101', and 'resnet152'.
    pretrained : bool, optional
        Only applied for 'resnet*' models. If True, will use ImageNet pretrained
        model. Default: True
    cut_at_pooling : bool, optional
        If True, will cut the model before the last global pooling layer and
        ignore the remaining kwargs. Default: False
    num_features : int, optional
        If positive, will append a Linear layer after the global pooling layer,
        with this number of output units, followed by a BatchNorm layer.
        Otherwise these layers will not be appended. Default: 256 for
        'inception', 0 for 'resnet*'
    norm : bool, optional
        If True, will normalize the feature to be unit L2-norm for each sample.
        Otherwise will append a ReLU layer after the above Linear layer if
        num_features > 0. Default: False
    dropout : float, optional
        If positive, will append a Dropout layer with this dropout rate.
        Default: 0
    num_classes : int, optional
        If positive, will append a Linear layer at the end as the classifier
        with this number of output units. Default: 0
    """
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](env, *args, **kwargs)
