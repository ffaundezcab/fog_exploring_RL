"""Tabular Q-learning agent.

The agent maintains a Q-table of shape (n_states, n_actions) and updates it
on each environment transition with the standard Q-learning rule

    Q[s, a] <- Q[s, a] + alpha * (r + gamma * (1 - terminated) * max Q[s', .] - Q[s, a]).

Action selection is epsilon-greedy with an exponential decay of epsilon over
episodes. The interface (select_action, update, end_episode) is shared with
the SARSA and Monte Carlo agents introduced in Lecture 2, so that the same
training loop drives all three algorithms unchanged.
"""

from __future__ import annotations

import numpy as np

from collections import defaultdict
from typing import Hashable, Optional
        
class LocalQLearningAgent:
    """
    """

    def __init__(
        self,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 1.0,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        initial_q: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}.")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}.")
        if not 0.0 <= epsilon_min <= epsilon_start <= 1.0:
            raise ValueError(
                f"epsilon range must satisfy 0 <= epsilon_min <= epsilon_start <= 1, "
                f"got start={epsilon_start}, min={epsilon_min}."
            )
        if not 0.0 < epsilon_decay <= 1.0:
            raise ValueError(f"epsilon_decay must be in (0, 1], got {epsilon_decay}.")

        self.n_actions = int(n_actions)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.initial_q = float(initial_q)

        # Current epsilon (decayed across episodes).
        self.epsilon = self.epsilon_start

        # Internal RNG, separate from the environment's RNG.
        self._rng = np.random.default_rng(seed)
        
        self.Q = defaultdict(self._new_q_values)
        
    def _new_q_values(self) -> np.ndarray:
        return np.full(
            self.n_actions,
            self.initial_q,
            dtype=np.float64,
        )

    # ---------------------------------------------------------------------
    # Action selection
    # ---------------------------------------------------------------------

    def select_action(self, state: Hashable, *, greedy: bool = False) -> int:
        """
        """
        if not greedy and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.n_actions))
    
        # if evaluation happens, it is possible to register a new state that wasn't seen during training
        # this is to only check the values, but not put new values into the defaultdict we created before
        # idea: measure the number of cases that were in Q and total cases
        # this percentage could be sueful to show why this approach is not the best
        if greedy and state not in self.Q:
            q_values = self._new_q_values()
        else:
            q_values = self.Q[state]
            
        return self._argmax_random_tiebreak(q_values)

    def _argmax_random_tiebreak(self, values: np.ndarray) -> int:
        """Argmax with uniform random tie-breaking among maximisers."""
        max_value = values.max()
        candidates = np.flatnonzero(values == max_value)
        return int(self._rng.choice(candidates))

    # ---------------------------------------------------------------------
    # Learning
    # ---------------------------------------------------------------------

    # mod: state now is Hashable because of the format on storing pos+local view
    # it is essentially an index
    
    def update(
        self,
        state: Hashable,
        action: int,
        reward: float,
        next_state: Hashable,
        terminated: bool,
        next_action: Optional[int] = None,  # this is made for compatibility with SARSA, ignored in Q-Learning
    ) -> None:
        """
        """
        bootstrap = 0.0 if terminated else float(self.Q[next_state].max())
        
        td_target = reward + self.gamma * bootstrap
        
        td_error = td_target - self.Q[state][action]
        
        self.Q[state][action] += self.alpha * td_error

    def end_episode(self) -> None:
        """Hook called by the training loop at the end of every episode.

        For Q-learning the only per-episode bookkeeping is the epsilon decay.
        SARSA and Monte Carlo will reuse this hook for their own bookkeeping.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ---------------------------------------------------------------------
    # Convenience views
    # ---------------------------------------------------------------------

    def greedy_policy(self) -> dict[Hashable, int]:
        """"""
        return {
            state: self._argmax_random_tiebreak(q_values)
            for state, q_values in self.Q.items()
        }

    def state_values(self) -> dict[Hashable, float]:
        """"""
        return {
        state: float(q_values.max())
        for state, q_values in self.Q.items()
        }
        
    @property
    def n_visited_states(self) -> int:
        return len(self.Q)