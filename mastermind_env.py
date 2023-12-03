import gymnasium as gym
from mastermind import MasterMind
from q_learning import QLearningAgent


class MasterMindEnv(gym.Env):
    def __init__(
        self, nColors: int, nTurns: int, lenCode: int, agent: QLearningAgent = None
    ):
        """
        Args:
            nColors (int): Number of colors in the code.
            nTurns (int): Number of turns in the game.
            lenCode (int): Length of the code.
            agent (QLearningAgent): Agent to train.
        """
        super(MasterMindEnv, self).__init__()
        self.mastermind = MasterMind(nColors, nTurns, lenCode)
        self.action_space = gym.spaces.MultiDiscrete(
            [self.mastermind.nColors] * self.mastermind.lenCode
        )
        self.code = self.mastermind.code
        turn_space = [self.mastermind.nColors + 1] * self.mastermind.lenCode + [
            self.mastermind.nColors + 1
        ] * 2
        self.observation_space = gym.spaces.MultiDiscrete(
            turn_space * self.mastermind.nTurns
        )
        if agent is None:
            self.agent = QLearningAgent(
                alpha=0.9,
                epsilon=0.9,
                discount=0.99,
                mastermind=self.mastermind,
            )
        else:
            self.agent = agent

        self.reset()

    def reset(self):
        """
        Returns:
            observation (object): the initial observation.
        """

        super(MasterMindEnv, self).reset(seed=None)
        self.agent.mastermind = MasterMind(
            self.mastermind.nColors, self.mastermind.nTurns, self.mastermind.lenCode
        )
        self.state = self.agent.mastermind.getGameStateInRewardFormat(
            self.agent.mastermind.turns
        )
        return self.state

    def step(self, action):
        """
        Args:
            action: action taken in state

        Returns:
            tuple of (observation, reward, terminated, info)
        """
        prev_state = self.agent.mastermind.getGameStateInRewardFormat(
            self.agent.mastermind.turns
        )
        nCorrect, nMisplaced = self.agent.mastermind.play(action)
        terminated = (
            nCorrect == self.agent.mastermind.lenCode
            or len(self.agent.mastermind.turns) == self.agent.mastermind.nTurns
        )
        reward = 1 if nCorrect == self.agent.mastermind.lenCode else -1
        observation = self.agent.mastermind.getGameStateInRewardFormat(
            self.agent.mastermind.turns
        )
        self.agent.learn(prev_state, action, observation, reward)
        return observation, reward, terminated, None

    def render(self):
        """
        Render the environment.
        """
        self.agent.mastermind.render()

    def close(self):
        pass
