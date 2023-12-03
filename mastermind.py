from collections import Counter
import numpy as np
from termcolor import colored


class MasterMind:
    def __init__(self, nColors: int, nTurns: int, lenCode: int) -> None:
        self.nColors = nColors
        self.nTurns = nTurns
        self.lenCode = lenCode
        self.turns = []
        self.evals = []
        self._code = self.generateCode()
        self.numberToColor = {
            1: "green",
            2: "yellow",
            3: "blue",
            4: "magenta",
            5: "cyan",
            6: "white",
            7: "red",
        }

    @property
    def code(self) -> list[int]:
        return self._code

    def generateCode(self) -> list[int]:
        """
        Returns a list of random integers between 1 and nColors.
        """
        return [np.random.randint(1, self.nColors + 1) for _ in range(self.lenCode)]

    def checkGuess(self, guess: np.ndarray, code: np.ndarray) -> tuple[int, int]:
        """
        Returns a tuple of the number of correct and misplaced colors in the guess.

        Args:
            guess (np.ndarray): Guess of the code.
            code (np.ndarray): Code to guess.

        Returns:
            tuple[int, int]: Number of correct and misplaced colors in the guess.
        """

        nMisplaced = 0
        nCorrect = sum([1 if i == j else 0 for i, j in zip(guess, code)])
        nMisplaced = sum((Counter(guess) & Counter(code)).values()) - nCorrect

        return nCorrect, nMisplaced

    def play(self, guess: np.ndarray) -> tuple[int, int]:
        """
        Returns a tuple of the number of correct and misplaced colors in the guess.

        Args:
            guess (np.ndarray): Guess of the code.

        Returns:
            tuple[int, int]: Number of correct and misplaced colors in the guess.
        """
        self.turns.append(guess)
        self.evals.append(self.checkGuess(guess, self.code))
        return self.evals[-1]

    def getGameStateInRewardFormat(self, turns) -> list[int]:
        """
        Returns the game state in the reward format: list of guesses and their evaluations in a flattened list.

        Args:
            turns (list[np.ndarray]): List of guesses.

        Returns:
            list[int]: Game state in the reward format.
        """

        # Change this code to numpy array
        obs = np.zeros((self.nTurns * (self.lenCode + 2)), dtype=np.int32)

        # Fill in the code
        for i in range(len(turns)):
            obs[i * (self.lenCode + 2) : i * (self.lenCode + 2) + self.lenCode] = turns[
                i
            ]
            obs[i * (self.lenCode + 2) + self.lenCode] = self.checkGuess(
                turns[i], self.code
            )[0]
            obs[i * (self.lenCode + 2) + self.lenCode + 1] = self.checkGuess(
                turns[i], self.code
            )[1]
        return obs

    def render(self) -> None:
        """
        Prints the code and the turns.
        """
        self.render_hidden_code()
        for i, turn in enumerate(self.turns):
            self.render_turn(turn, i)

    def render_turn(self, turn: np.ndarray, i: int) -> None:
        """
        Prints the turn.
        """
        piece = "O"
        turnStr = f"Turn {i+1}: "
        turnStr += " ".join([colored(piece, self.numberToColor[j]) for j in turn])
        checkGuess = self.evals[i]
        evaluation = "".join(
            [colored("C", "white") for _ in range(checkGuess[0])]
        ) + "".join([colored("M", "white") for _ in range(checkGuess[1])])
        print(turnStr, evaluation)

    def render_hidden_code(self) -> None:
        """
        Prints the code.
        """
        piece = "O"
        render_code = " ".join(
            [colored(piece, self.numberToColor[i]) for i in self.code]
        )
        print("Code:  ", render_code)

    def getLegalActions(self, state) -> list[np.ndarray]:
        """
        Returns a list of all legal actions.

        Args:
            state (np.ndarray): Game state.

        Returns:
            list[np.ndarray]: List of all legal actions.
        """
        return [
            np.array(i) + 1
            for i in np.ndindex(tuple([self.nColors] * self.lenCode))
            if self.coherentGuess(state, np.array(i) + 1)
            and self.notRepeated(np.array(i) + 1)
        ]

    def coherentGuess(self, state, action) -> bool:
        """
        Returns whether a guess is coherent with the game state.

        Args:
            state (np.ndarray): Game state.

        Returns:
            bool: Whether a guess is coherent with the game state.
        """
        return all(
            [
                self.checkGuess(
                    action,
                    state[
                        i * (self.lenCode + 2) : i * (self.lenCode + 2) + self.lenCode
                    ],
                )
                == tuple(
                    state[
                        i * (self.lenCode + 2)
                        + self.lenCode : i * (self.lenCode + 2)
                        + self.lenCode
                        + 2
                    ]
                )
                for i in range(len(self.turns) + 1)
            ]
        )

    def notRepeated(self, action) -> bool:
        """
        Returns whether a guess has already been made.

        Args:
            action (np.ndarray): Guess.

        Returns:
            bool: Whether a guess has already been made.
        """
        return all([not np.array_equal(action, turn) for turn in self.turns])
