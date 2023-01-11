import sys
import numpy as np
from contextlib import closing
from io import StringIO
from typing import Optional
from gym import Env, spaces
from gym.envs.toy_text.utils import categorical_sample

from CoFEA.environments import env_utils


UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
SIZE: str = "large"
TRACK_FILE: str = "R-track"
# TERMINAL_STATE: tuple = (2, 34)
# TERMINAL_STATE: tuple = (27, 28)
TERMINAL_STATE: tuple = (28, 8)

def build_cliff_L(shape):
    _cliff = np.zeros(shape, dtype=bool)
    _cliff[0, :] = True
    for i in range(1,6):
        _cliff[i, :-5] = True
        _cliff[i, -1:] = True
    for i in range(6,10):
        _cliff[i, :1] = True
        _cliff[i, -1:] = True
    _cliff[10, :] = True
    return _cliff


def build_cliff_R(shape):
    _cliff = np.zeros(shape, dtype=bool)
    _cliff[:1, :] = True
    for i in range(2,5):
        _cliff[i, :1] = True
        _cliff[i, -1:] = True
    for i in range(5,15):
        _cliff[i, :1] = True
        _cliff[i, 5:-6] = True
        _cliff[i, -1:] = True
    for i in range(15,20):
        _cliff[i, :1] = True
        _cliff[i, 5:10] = True
        _cliff[i, -1:] = True
    for i in range(20,shape[0] - 1):
        _cliff[i, :1] = True
        _cliff[i, 5:20] = True
        _cliff[i, -1:] = True
    _cliff[-1, :] = True
    return _cliff


def build_cliff_P(shape):
    _cliff = np.zeros(shape, dtype=bool)
    _cliff[:1, :] = True
    for i in range(2,6):
        _cliff[i, :1] = True
        _cliff[i, -1:] = True
    for i in range(6,24):
        _cliff[i, :1] = True
        _cliff[i, 5:-6] = True
        _cliff[i, -1:] = True
    for i in range(24,shape[0] - 1):
        _cliff[i, :1] = True
        _cliff[i, 5:7] = True # uncomment this to test policy sensitivity
        _cliff[i, -1:] = True
    _cliff[-1, :] = True
    return _cliff



class Racetrack(Env):
    """
    This is a simple implementation of the Gridworld Cliff
    reinforcement learning task.

    Adapted from Example 6.6 (page 106) from [Reinforcement Learning: An Introduction
    by Sutton and Barto](http://incompleteideas.net/book/bookdraft2018jan1.pdf).

    With inspiration from:
    https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/cliff_walking.py

    ### Description
    The board is a 4x12 matrix, with (using NumPy matrix indexing):
    - [3, 0] as the start at bottom-left
    - [3, 11] as the goal at bottom-right
    - [3, 1..10] as the cliff at bottom-center

    If the agent steps on the cliff it returns to the start.
    An episode terminates when the agent reaches the goal.

    ### Actions
    There are 4 discrete deterministic actions:
    - 0: move up
    - 1: move right
    - 2: move down
    - 3: move left

    ### Observations
    There are 3x12 + 1 possible states. In fact, the agent cannot be at the cliff, nor at the goal
    (as this results the end of episode). They remain all the positions of the first 3 rows plus the bottom-left cell.
    The observation is simply the current position encoded as
    [flattened index](https://numpy.org/doc/stable/reference/generated/numpy.unravel_index.html).

    ### Reward
    Each time step incurs -1 reward, and stepping into the cliff incurs -100 reward.

    ### Arguments

    ```
    gym.make('CliffWalking-v0')
    ```

    ### Version History
    - v0: Initial version release
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self):
        # self.shape = env_utils.get_track_shape(input_file=TRACK_FILE)
        # self.start_state_index = env_utils.get_new_initial_state(input_file=TRACK_FILE)
        # print(self.start_state_index)
        global TERMINAL_STATE
        if TRACK_FILE == "L-track":
            self.shape = (11, 37)
            # self.start_state_index = np.ravel_multi_index((10, 0), self.shape
            self.start_state_index = 297
            TERMINAL_STATE = (2, 35)
        elif TRACK_FILE == "R-track":
            self.shape = (28, 30)
            # self.start_state_index = np.ravel_multi_index((10, 0), self.shape
            self.start_state_index = 782
            TERMINAL_STATE = (26, 28)
        elif TRACK_FILE == "P-track":
            self.shape = (30, 30)
            # self.start_state_index = np.ravel_multi_index((10, 0), self.shape
            self.start_state_index = 782
            TERMINAL_STATE = (28, 8)

        # print(np.ravel_multi_index((3, 0), self.shape))
        self.nS = np.prod(self.shape)
        self.nA = 4

        # Cliff Location
        # self._cliff = np.zeros(self.shape, dtype=bool)
        # self._cliff[env_utils.get_track_boundaries(input_file=TRACK_FILE)] = True
        # if SIZE == "small":
        self._cliff = np.zeros(self.shape, dtype=bool)
        if TRACK_FILE == "L-track":
        # self._cliff[-1, 1:-1] = True
            self._cliff = build_cliff_L(self.shape)
        elif TRACK_FILE == "R-track":
        # self._cliff[-1, 1:-1] = True
            self._cliff = build_cliff_R(self.shape)
        if TRACK_FILE == "P-track":
        # self._cliff[-1, 1:-1] = True
            self._cliff = build_cliff_P(self.shape)
        # print(self._cliff)
        # elif SIZE == "large":
        #     self._cliff = np.zeros(self.shape, dtype=bool)
        #     self._cliff[-2:, 2:-2] = True
        # elif SIZE == "mega":
        #     self._cliff = np.zeros(self.shape, dtype=bool)
        #     self._cliff[-4:, 4:-4] = True
        # elif SIZE == "giga":
        #     self._cliff = np.zeros(self.shape, dtype=bool)
        #     self._cliff[-8:, 8:-8] = True

        # Calculate transition probabilities and rewards
        self.P = {}
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            self.P[s] = {a: [] for a in range(self.nA)}
            self.P[s][UP] = self._calculate_transition_prob(position, [-1, 0])
            self.P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1])
            self.P[s][DOWN] = self._calculate_transition_prob(position, [1, 0])
            self.P[s][LEFT] = self._calculate_transition_prob(position, [0, -1])

        # Calculate initial state distribution
        # We always start in state (3, 0)
        self.initial_state_distrib = np.zeros(self.nS)
        self.initial_state_distrib[self.start_state_index] = 1.0

        self.observation_space = spaces.Discrete(self.nS)
        self.action_space = spaces.Discrete(self.nA)

    def _limit_coordinates(self, coord: np.ndarray) -> np.ndarray:
        """Prevent the agent from falling out of the grid world."""
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta):
        """Determine the outcome for an action. Transition Prob is always 1.0.

        Args:
            current: Current position on the grid as (row, col)
            delta: Change in position for transition

        Returns:
            Tuple of ``(1.0, new_state, reward, done)``
        """
        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        if self._cliff[tuple(new_position)]:
            return [(1.0, self.start_state_index, -100, False)]

        # terminal_state = (self.shape[0] - 1, self.shape[1] - 1)
        is_done = tuple(new_position) == TERMINAL_STATE
        return [(1.0, new_state, -1, is_done)]

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[i]
        self.s = s
        self.lastaction = a
        return (int(s), r, d, {"prob": p})

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None
    ):
        super().reset(seed=seed)
        self.s = categorical_sample(self.initial_state_distrib, self.np_random)
        self.lastaction = None
        if not return_info:
            return int(self.s)
        else:
            return int(self.s), {"prob": 1}

    def render(self, mode="human"):
        outfile = StringIO() if mode == "ansi" else sys.stdout
        out = []
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            if self.s == s:
                output = " x "
            # Print terminal state
            elif position == (3, 11) and SIZE == "small":
                output = " T "
            elif position == (7, 23) and SIZE == "large":
                output = " T "
            elif position == (15, 47) and SIZE == "mega":
                output = " T "
            elif position == (31, 95) and SIZE == "giga":
                output = " T "
            elif self._cliff[position]:
                output = " C "
            else:
                output = " o "

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += "\n"

            outfile.write(output)
            out.append(output)
        outfile.write("\n")
        print(out)

        # No need to return anything for human
        if mode != "human":
            with closing(outfile):
                return outfile.getvalue()
