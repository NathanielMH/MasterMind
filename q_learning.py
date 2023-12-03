from collections import defaultdict
import random, dill
import numpy as np
from mastermind import MasterMind


class QLearningAgent:
    def __init__(self, alpha, epsilon, discount, mastermind: MasterMind, qValues=None):
        """
        Q-Learning Agent.
            alpha: Learning rate
            epsilon: Exploration rate
            discount: Discount factor for past rewards
            mastermind: Mastermind object
        """
        self.mastermind = mastermind
        self._q_values = (
            defaultdict(lambda: defaultdict(lambda: 0)) if qValues is None else qValues
        )
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount

    @staticmethod
    def arrayToTuple(array):
        return tuple(array)

    @staticmethod
    def tupleToArray(tup):
        return np.array(tup)

    def getQValue(self, state, action):
        """
        Returns Q(state,action)

        Args:
            state: state of the game
            action: action taken in state

        Returns:
            Q(state,action)
        """
        return self._q_values[self.arrayToTuple(state)][self.arrayToTuple(action)]

    def setQValue(self, state, action, value):
        """
        Sets the Qvalue for [state,action] to the given value

        Args:
            state: state of the game
            action: action taken in state
            value: value to set Q(state,action) to

        Returns:
            None
        """
        self._q_values[self.arrayToTuple(state)][self.arrayToTuple(action)] = value

    def getValueAndPossibleActions(self, state):
        """
        Returns max Q(state,action)
        where the max is over legal actions.

        Args:
            state: state of the game

        Returns:
            tuple of (max Q(state,action), possible actions)
        """
        possible_actions = self.mastermind.getLegalActions(state)
        if len(possible_actions) == 0:
            return 0.0, possible_actions
        return (
            max([self.getQValue(state, action) for action in possible_actions]),
            possible_actions,
        )

    def getPolicy(self, state):
        """
        Compute the best action to take in a state.

        Args:
            state: state of the game

        Returns:
            best action to take in state taken randomly if multiple actions have the same Q-value
        """
        max_V, possible_actions = self.getValueAndPossibleActions(state)
        chosen_action = random.choice(
            [
                action
                for action in possible_actions
                if self.getQValue(state, action) == max_V
            ]
        )
        return chosen_action

    def act(self, state):
        """
        Compute the action to take in the current state, including exploration.
        With probability self.epsilon, we should take a random action.
            otherwise - the best policy action (self.getPolicy).
        Note: To pick randomly from a list, use random.choice(list).
              To pick True or False with a given probablity, generate uniform number in [0, 1]

        Args:
            state: state of the game

        Returns:
            action to take in state
        """
        possible_actions = self.mastermind.getLegalActions(state)
        if len(possible_actions) == 0:
            return None
        if np.random.uniform(0, 1) < self.epsilon:
            chosen_action = possible_actions[
                np.random.randint(0, len(possible_actions))
            ]
        else:
            chosen_action = self.getPolicy(state)
        return chosen_action

    def learn(self, state, action, next_state, reward):
        """
        You should do your Q-Value update here:
           Q(s,a) := (1 - alpha) * Q(s,a) + alpha * (r + gamma * V(s'))

        Args:
            state: state of the game
            action: action taken in state
            next_state: next state of the game
            reward: reward received after taking action in state

        Returns:
            None
        """
        q_value = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * (
            reward + self.discount * self.getValueAndPossibleActions(next_state)[0]
        )
        self.setQValue(state, action, q_value)

    def saveAgent(self, path):
        """
        Saves the agent to a file.

        Args:
            path: path to save the agent to

        Returns:
            None
        """
        with open(path, "wb") as f:
            dill.dump(self, f)
