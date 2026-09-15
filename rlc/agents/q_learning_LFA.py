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
from rlc.utils.features import LocalQuadrantFeatures
        
class LFAQLearningAgent:
    """
    Q-learning agent using linear function approximation (LFA).

    Agent represents each action with a vector of weights. A feature extractor maps
    the current state to a feature vector, and Q-values are computed as the
    dot product between those features and each action's weights.

    Parameters
    ----------
    n_actions : int
        Number of actions available to the agent.
    feature_extractor : LocalQuadrantFeatures
        Object that converts a state into a fixed-length feature vector
    alpha : float
        Learning rate used for the update.
    gamma : float
        Discount factor for future rewards.
    epsilon_start : float
        Initial exploration probability.
    epsilon_min : float
        Minimum value to which epsilon may decay.
    epsilon_decay : float
        Epsilon decay applied after each episode.
    seed : int, optional
        random seed
        
    """

    def __init__(
        self,
        n_actions: int,
        feature_extractor: LocalQuadrantFeatures,
        alpha: float = 0.1,
        gamma: float = 1.0,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
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
        self.feature_extractor = feature_extractor
        
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)

        # Current epsilon (decayed across episodes).
        self.epsilon = self.epsilon_start

        # Internal RNG, separate from the environment's RNG.
        self._rng = np.random.default_rng(seed)
        
        # instead of a tabular state-action, we record action x feature weights
        self.weights = np.zeros((
            self.n_actions,
            self.feature_extractor.n_features
        ), dtype = np.float64)
        
    def _q_values(self, state) -> np.ndarray:
        phi = self.feature_extractor(state)
        
        return self.weights @ phi

    # ---------------------------------------------------------------------
    # Action selection
    # ---------------------------------------------------------------------

    def select_action(self, state: Hashable, *, greedy: bool = False) -> int:
        """
        Choose an action using an epsilon-greedy policy

        Parameters
        ----------
        state
            Current environment state.
        greedy : bool
            If True, disable exploration and choose only among best actions.

        Returns
        -------
        int
            Index of the selected action.
        
        """
        if not greedy and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.n_actions))
        
        q_values = self._q_values(state)
            
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
        Computation of the update and consequent weights
        """
        
        phi = self.feature_extractor(state)
        pred_q = float( self.weights[action] @ phi)
        
        if terminated:
            bootstrap = 0.0
        else:
            next_phi = self.feature_extractor(next_state)
            next_q_values = (self.weights @ next_phi)
            
            bootstrap = float(next_q_values.max())
            
        # td 
        td_target = reward + self.gamma*bootstrap
        td_error = td_target-pred_q
        
        # gradient
        self.weights[action] += self.alpha*td_error*phi

    def end_episode(self) -> None:
        """Hook called by the training loop at the end of every episode.

        For Q-learning the only per-episode bookkeeping is the epsilon decay.
        SARSA and Monte Carlo will reuse this hook for their own bookkeeping.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ---------------------------------------------------------------------
    # Convenience views
    # ---------------------------------------------------------------------

    def q_values(self, state) -> np.ndarray:
        """
        Return Q-values for every action for a specific state
        """
        
        return self._q_values(state)

    def state_values(self, state) -> float:
        """
        Return the estimated state value (in Q-learning is the max among all actions)
        """
        return float(self._q_values(state).max())
        
    @property
    def n_features(self) -> int:
        """
        Number of features for the current feature class
        """
        
        return self.feature_extractor.n_features