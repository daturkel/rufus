from typing import Literal

import numpy as np
import numpy.typing as npt
from annoy import AnnoyIndex

from .. import ResultSet
from ._base import NearestNeighborsIndex


class AnnoyNearestNeighborsIndex(NearestNeighborsIndex):
    """A nearest neighbors index that uses Spotify's Annoy index under the hood."""

    def __init__(
        self,
        vectors: npt.NDArray,
        metric: Literal["cosine", "euclidean", "manhattan", "hamming", "dot"] = "dot",
        n_trees: int = 100,
    ):
        """Create a new `ExactNearestNeighbors` index.

        Args:
            vectors: A 2D numpy array, where the 0th dimension corresponds to items and the first dimension is each item's vector.
            metric: The metric that is used to determine how close vectors are to each other. Options are `cosine`, `euclidean`,
                `manhattan`, `hamming`, and `dot`.
        """
        super().__init__(
            vectors,
            metric,
            available_metrics=["cosine", "euclidean", "manhattan", "hamming", "dot"],
            n_trees=n_trees,
        )
        self.index: AnnoyIndex
        """An `AnnoyIndex` of the indexed vectors."""
        self._vectors = vectors

    def _index(
        self,
        vectors: npt.NDArray,
        metric: str,
        n_trees: int,
    ) -> AnnoyIndex:
        if metric == "cosine":
            annoy_metric = "angular"
        else:
            annoy_metric = metric
        index = AnnoyIndex(vectors.shape[1], annoy_metric)  # type: ignore
        for i, vector in enumerate(vectors):
            index.add_item(i, vector)
        index.build(n_trees, -1)
        return index

    def get_nearest_neighbors_from_existing(
        self, index: int, top_k: int | None = 100, search_k: int | None = None
    ) -> ResultSet:
        """Same as `nearest`, but takes an `index` argument indicating to look up an existing known vector.

        Args:
            index: The index of the vector to query.
            top_k: Number of results to return. If `None`, return all results. Defaults to 100.
            search_k: Number of nodes to inspect. Higher number is more accurate but slower. Defaults to `top_k * n_trees`.

        Returns:
            An instance of `ResultSet` containing the nearest neighbors to the query.
        """
        if top_k is None:
            top_k = self.index.get_n_items()

        if search_k is None:
            search_k = -1

        indices, scores = self.index.get_nns_by_item(
            index, top_k, search_k=search_k, include_distances=True
        )

        return ResultSet(np.asarray(indices), np.asarray(scores))

    def get_nearest_neighbors(
        self, vector: npt.NDArray, top_k: int | None = 100, search_k: int | None = None
    ) -> ResultSet:
        """Return the `top_k` nearest vectors to the query vector.

        Args:
            vector: The query vector for which we want the nearest neighbors.
            top_k: Number of results to return. If `None`, return all results. Defaults to 100.
            search_k: Number of nodes of the index to search. Higher makes results more accurate, but slower.
                Defaults to `top_k` * `n_trees`.

        Returns:
            An instance of `ResultSet` containing the nearest neighbors to the query.
        """
        if top_k is None:
            top_k = self.index.get_n_items()

        if search_k is None:
            search_k = -1

        indices, scores = self.index.get_nns_by_vector(
            vector, top_k, search_k=search_k, include_distances=True
        )

        if self.metric == "cosine":
            # annoy does sqrt(2 - 2 * cosine similarity) by default
            scores = (2 - np.asarray(scores) ** 2) / 2  # type: ignore

        return ResultSet(np.asarray(indices), np.asarray(scores))
