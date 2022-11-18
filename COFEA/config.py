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




import gym

env = gym.make('CliffWalking-v0')

"""
self.epsilon = epsilon
self.alpha = alpha
self.gamma = gamma
self.num_state = num_state
self.num_actions = num_actions

self.Q = np.zeros((self.num_state, self.num_actions))
self.action_space = action_space
"""
class ConfigRL(object):

    # logs dir
    logs_dir = 'logs'
    # model training parameters
    epsilon = 0.5
    alpha = 0.5
    gamma = 0.8
    workers = 8
    sampler = None
    print_freq = 40
    epsilon_decay_factor = 0.999

    def __init__(self,
                 model_name='e_sarsa_osi',
                 epochs=50,
                 max_steps=300,
                 envi=env,
                 checkpoint=None
                 ):
        self.model_name = model_name
        self.epochs = epochs
        self.max_steps = max_steps
        self.envi = envi
        self.checkpoint = checkpoint
