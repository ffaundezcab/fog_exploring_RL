"""Feature extractors for linear value function approximation.

Two families of feature maps for continuous 2D state spaces:

- ``TileCoder``: one or more offset rectangular grids partition the state
  space into tiles; the feature vector is a binary indicator of the tile
  the state falls into, concatenated across tilings.
- ``RBFFeatures``: a regular grid of Gaussian centres; the feature vector
  is the Gaussian-evaluated similarity of the state to each centre.

Both classes implement the ``FeatureExtractor`` protocol below: they expose
a public ``n_features`` attribute and a ``__call__`` method that maps a
state vector to a feature vector.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

# Tile types
EMPTY = 0
OBSTACLE = 1
TRAP = 2
GOAL = 3
FOG = 4


# ---------------------------------------------------------------------------
# Structural interface
# ---------------------------------------------------------------------------

class FeatureExtractor(Protocol):
    """Structural interface for feature maps.

    Any object exposing an integer ``n_features`` and a callable that maps
    a state vector ``s`` of shape ``(state_dim,)`` to a feature vector of
    shape ``(n_features,)`` satisfies this protocol.
    """

    n_features: int

    def __call__(self, state: np.ndarray) -> np.ndarray: ...

# ---------------------------------------------------------------------------
# Tile coding
# ---------------------------------------------------------------------------

class TileCoder:
    """Tile coding for 2D continuous state spaces.

    The state space is partitioned into ``n_tilings`` rectangular grids, each
    of size ``n_tiles_x x n_tiles_y``, mutually offset by fractions of a tile
    so that nearby states tend to share *some* but not *all* of their active
    tiles. The feature vector is the concatenation across tilings of one-hot
    indicators of the active tile in each.

    Total feature dimension: ``n_tilings * n_tiles_x * n_tiles_y``.
    Of these, exactly ``n_tilings`` are non-zero on any given state (sparse
    binary representation).

    Parameters
    ----------
    low, high : array-like of shape (2,)
        Lower and upper bounds of the state space.
    n_tiles_x, n_tiles_y : int
        Number of tiles per axis in each tiling.
    n_tilings : int
        Number of overlapping offset grids. Default 4. With one tiling
        the encoding reduces to plain binary discretization.
    seed : int, optional
        Seed used for sampling the tiling offsets, for reproducibility.
    """

    def __init__(
        self,
        low: np.ndarray,
        high: np.ndarray,
        n_tiles_x: int,
        n_tiles_y: int,
        n_tilings: int = 4,
        seed: int = 0,
    ) -> None:
        self.low = np.asarray(low, dtype=np.float64).copy()
        self.high = np.asarray(high, dtype=np.float64).copy()
        if self.low.shape != (2,) or self.high.shape != (2,):
            raise ValueError("low and high must have shape (2,).")
        if np.any(self.high <= self.low):
            raise ValueError("high must be strictly greater than low component-wise.")
        if n_tiles_x < 1 or n_tiles_y < 1:
            raise ValueError("n_tiles_x and n_tiles_y must be positive.")
        if n_tilings < 1:
            raise ValueError("n_tilings must be positive.")

        self.n_tiles_x = int(n_tiles_x)
        self.n_tiles_y = int(n_tiles_y)
        self.n_tilings = int(n_tilings)

        # Tile sizes (uniform across tilings).
        self.tile_size_x = (self.high[0] - self.low[0]) / self.n_tiles_x
        self.tile_size_y = (self.high[1] - self.low[1]) / self.n_tiles_y

        # Per-tiling offsets, in [0, tile_size). With a single tiling the
        # offset is zero. With multiple tilings the offsets are uniformly
        # spaced fractions of a tile, deterministic given the seed.
        rng = np.random.default_rng(seed)
        if self.n_tilings == 1:
            offs_x = np.zeros(1, dtype=np.float64)
            offs_y = np.zeros(1, dtype=np.float64)
        else:
            # Symmetric, deterministic spread; jittered slightly to break
            # any degeneracy when the user later changes resolution.
            base = np.linspace(0.0, 1.0, self.n_tilings, endpoint=False)
            jitter = rng.uniform(-0.05, 0.05, size=self.n_tilings)
            frac = (base + jitter) % 1.0
            offs_x = frac * self.tile_size_x
            offs_y = frac * self.tile_size_y
        self._offsets_x = offs_x
        self._offsets_y = offs_y

        self.n_features = self.n_tilings * self.n_tiles_x * self.n_tiles_y

    def __call__(self, state: np.ndarray) -> np.ndarray:
        x = float(state[0])
        y = float(state[1])
        feat = np.zeros(self.n_features, dtype=np.float64)
        for t in range(self.n_tilings):
            # Shift the state by the tiling offset, then bin.
            sx = (x - self.low[0] + self._offsets_x[t]) / self.tile_size_x
            sy = (y - self.low[1] + self._offsets_y[t]) / self.tile_size_y
            ix = int(np.clip(np.floor(sx), 0, self.n_tiles_x - 1))
            iy = int(np.clip(np.floor(sy), 0, self.n_tiles_y - 1))
            # Index of this tiling's block in the overall feature vector.
            block_start = t * self.n_tiles_x * self.n_tiles_y
            tile_index = iy * self.n_tiles_x + ix
            feat[block_start + tile_index] = 1.0
        return feat


# ---------------------------------------------------------------------------
# Radial basis functions
# ---------------------------------------------------------------------------

class RBFFeatures:
    """Gaussian RBF features on a regular grid of centres.

    Centres are placed on a ``n_centers_x x n_centers_y`` regular grid
    inside the bounding box. Each feature is a Gaussian evaluated at the
    state, with isotropic standard deviation ``sigma``.

    Total feature dimension: ``n_centers_x * n_centers_y``.
    All features are positive on every state (dense, non-sparse).

    Parameters
    ----------
    low, high : array-like of shape (2,)
        Lower and upper bounds of the state space.
    n_centers_x, n_centers_y : int
        Grid dimensions for the RBF centres.
    sigma : float
        Standard deviation of the Gaussian kernels, in the same units as
        the state. Controls the width of each RBF and the smoothness of
        the resulting feature map.
    normalize : bool
        If True, divide the feature vector by its sum so that the features
        sum to 1 at every state (a partition-of-unity-like normalization).
        Useful for stable interpolation. Default False.
    """

    def __init__(
        self,
        low: np.ndarray,
        high: np.ndarray,
        n_centers_x: int,
        n_centers_y: int,
        sigma: float,
        normalize: bool = False,
    ) -> None:
        low = np.asarray(low, dtype=np.float64)
        high = np.asarray(high, dtype=np.float64)
        if low.shape != (2,) or high.shape != (2,):
            raise ValueError("low and high must have shape (2,).")
        if np.any(high <= low):
            raise ValueError("high must be strictly greater than low component-wise.")
        if n_centers_x < 1 or n_centers_y < 1:
            raise ValueError("n_centers_x and n_centers_y must be positive.")
        if sigma <= 0:
            raise ValueError("sigma must be positive.")

        self.low = low.copy()
        self.high = high.copy()
        self.n_centers_x = int(n_centers_x)
        self.n_centers_y = int(n_centers_y)
        self.sigma = float(sigma)
        self.normalize = bool(normalize)

        # Place centres at the cell centres of a regular grid covering [low, high].
        cx = self.low[0] + (np.arange(self.n_centers_x) + 0.5) * \
             (self.high[0] - self.low[0]) / self.n_centers_x
        cy = self.low[1] + (np.arange(self.n_centers_y) + 0.5) * \
             (self.high[1] - self.low[1]) / self.n_centers_y
        # Grid of (n_centers_y, n_centers_x, 2), flattened to (n_features, 2)
        # in row-major order (matching our state-indexing convention).
        cy_grid, cx_grid = np.meshgrid(cy, cx, indexing="ij")
        self.centers = np.stack([cx_grid.ravel(), cy_grid.ravel()], axis=1)

        self.n_features = self.n_centers_x * self.n_centers_y

    def __call__(self, state: np.ndarray) -> np.ndarray:
        s = np.asarray(state, dtype=np.float64).reshape(2)
        # Squared distances from state to each centre.
        d2 = np.sum((self.centers - s[None, :]) ** 2, axis=1)
        feat = np.exp(-d2 / (2.0 * self.sigma ** 2))
        if self.normalize:
            total = feat.sum()
            if total > 0:
                feat = feat / total
        return feat

class LocalQuadrantFeatures:
    """
    """
    
    def __init__(self,
                 height: int,
                 width: int,
                 vision_radius: int = 2,
                 exp_decay: float = 1.0) -> None:
        if vision_radius != 2:
            raise ValueError("vision radius must start at 2 tiles")
        
        self.height = height
        self.width = width
        self.vision_radius = vision_radius
        self.exp_decay = exp_decay
        
        # bias term, agent pos, features per quadrant (6 x 4)
        self.n_features = 27
        
        # quadrants relative to the position of the agent
        
        self.quadrants = {
            "north": [(-2,-1), (-2,0), (-2, 1), (-2,2),
                                        (-1, 0), (-1, 1)],
            "east": [(-1, 2), (0, 2), (1, 2), (2, 2),
                                        (0,1), (1,1)],
            "south": [(2,1), (2,0), (2,-1), (2,-2),
                                    (1,0), (1,-1)],
            "west": [(1, -2), (0, -2), (-1, -2), (-2,-2),
                                        (0, -1), (-1,-1)]            
        }
        
    def __call__(self, state) -> np.ndarray:
        """
        """
        
        r = state[0]
        c = state[1]
        
        norm_r = r/(self.height - 1)
        norm_c = c/(self.width - 1)
        
        # reconstruct the local view from a flattened matrix
        local_view = np.asarray(state[2:], dtype = np.int64).reshape(5,5) 
        
        # bias + everything else
        features = [
            1.0,
            norm_r,
            norm_c
        ]
        
        # quadrant features
        
        for list_offsets in self.quadrants.values():
            
            tiles = []
            
            trap_distances = []
            obstacle_distances = []
            
            # depending on which direction (e.g. nort, east, etc...)
            for dr, dc in list_offsets:
                
                # we assume agent is in the middle of the 5x5 local view (coordinates (2,2))
                # since we stored the delta distance to the agent, we sum up to have the true position
                # in the local view
                
                r = 2+dr
                c = 2+ dc
                
                tile = local_view[r, c]
                tiles.append(tile)
                
                # number of tiles relative to the agent to arrive there
                distance = abs(dr) + abs(dc)
                
                if tile == TRAP:
                    trap_distances.append(distance)
                    
                elif tile == OBSTACLE:
                    obstacle_distances.append(distance)
                    
            tiles = np.asarray(tiles)

            trap_density = np.mean(tiles == TRAP)
            obstacle_density= np.mean(tiles == OBSTACLE)
            
            trap_proximity_index = self._proximity_index(trap_distances)
            obstacle_proximity_index = self._proximity_index(obstacle_distances)
            
            clear_path = 0.0
            
            features.extend([trap_density,
                             obstacle_density,
                             trap_proximity_index,
                             obstacle_proximity_index,
                             clear_path])
            
        # goal in view
        
        goal_positions = np.argwhere(local_view == GOAL)
        
        if len(goal_positions) == 0:
            goal_in_view = 0.0
            goal_r = 0,0
            goal_c = 0,0
                    
                    
    def _proximity_index(self, distances: list[int]) -> float:
        """
        """
        
        if not distances:
            return 0.0
        
        product = 1.0
        
        for d in distances:
            contribution = np.exp(-self.exp_decay*(d-1))
            add_product *= (1.0 - contribution)
            
        return 1.0 - product
                
        
        
    
    
