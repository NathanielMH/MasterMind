import numpy as np
import matplotlib.pyplot as plt
import dill
from q_learning import QLearningAgent
from mastermind_env import MasterMindEnv


def getAgent(path: str) -> QLearningAgent:
    with open(path, "rb") as f:
        agent = dill.load(f)
    return agent


def trainAgent(
    nColors: int,
    nTurns: int,
    lenCode: int,
    nGames: int,
    nGamesPerSave: int,
    agent: QLearningAgent,
    savePath: str,
    runName: str,
):
    """
    Train an agent to play MasterMind.

    Args:
        nColors (int): Number of colors in the code.
        nTurns (int): Number of turns in the game.
        lenCode (int): Length of the code.
        nGames (int): Number of games to play.
        nGamesPerSave (int): Number of games to play before saving the agent.
        agent (QLearningAgent): Agent to train.
        savePath (str): Path to save the agent.
        runName (str): Name of the run.

    Returns:
        QLearningAgent: Trained agent.
    """
    env = MasterMindEnv(nColors=nColors, nTurns=nTurns, lenCode=lenCode, agent=agent)
    wins = []
    for i in range(nGames):
        env.reset()
        terminated = False
        while not terminated:
            action = env.agent.act(
                env.agent.mastermind.getGameStateInRewardFormat(
                    env.agent.mastermind.turns
                )
            )
            env.render()
            observation, reward, terminated, _ = env.step(action)

        wins.append(reward == 1)
        if i % nGamesPerSave == 0:
            with open(savePath, "wb") as f:
                dill.dump(agent, f)
    winrate = np.cumsum(wins) / np.arange(1, nGames + 1)
    plt.plot(winrate)
    plt.savefig(f"{runName}.png")
    return agent
