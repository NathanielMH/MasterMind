from utils import getAgent, trainAgent
from q_learning import QLearningAgent
from mastermind import MasterMind
import os

# Run parameters
nRuns = 10


# Game parameters
lenCode = 4
nColors = 6
nTurns = 5

# Agent parameters
alpha = 0.9
epsilon = 0.9
discount = 0.99

mastermind = MasterMind(nColors=nColors, nTurns=nTurns, lenCode=lenCode)
agent = (
    getAgent("agent.pkl")
    if os.path.exists("agent.pkl")
    else QLearningAgent(
        alpha=alpha, epsilon=epsilon, discount=discount, mastermind=mastermind
    )
)

for j in range(1, nRuns + 1):
    agent = trainAgent(
        nColors=nColors,
        nTurns=nTurns,
        lenCode=lenCode,
        nGames=10000,
        nGamesPerSave=1000,
        agent=agent,
        savePath="agent.pkl",
        runName=f"Trial{j}",
    )
