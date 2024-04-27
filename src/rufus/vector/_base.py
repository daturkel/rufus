import numpy as np
import numpy.typing as npt

from .. import ResultSet


class NearestNeighborsIndex:
    """Base class for a vector index. Supports `nearest`, `nearest_from_existing`, and `add_to_index` methods."""

    def __init__(
        self,
        vectors: npt.NDArray,
        metric: str,
        available_metrics: list[str] | None = None,
        **index_kwargs,
    ):
        """Create a new `NearestNeighbors` index.

        Args:
            vectors: A 2D numpy array, where the 0th dimension corresponds to items and the first dimension is each item's vector.
            metric: The metric that is used to determine how close vectors are to each other.
            available_metrics: A list of metrics supported by the index. Defaults to None.

        Raises:
            ValueError: The `metric` supplied is not one of the `available_metrics`.
        """
        self._available_metrics = available_metrics or []
        if metric not in self._available_metrics:
            raise ValueError(
                f"Unkown metric: {metric}. Must be one of {self._available_metrics}."
            )
        self.metric = metric
        """The metric that is used to determine how close vectors are to each other."""
        self.index = self._index(vectors, self.metric, **index_kwargs)
        """A representation of the indexed vectors."""

    def get_nearest_neighbors(
        self, vector: npt.NDArray, top_k: int | None = 100
    ) -> ResultSet:
        """Return the `top_k` nearest vectors to the query vector.

        Args:
            vector: The query vector for which we want the nearest neighbors.
            top_k: Number of results to return. If `None`, return all results. Defaults to 100.

        Raises:
            NotImplementedError: Not implemented for the base class.

        Returns:
            An instance of `NearestNeighborsResult` containing the nearest neighbors to the query.
        """
        raise NotImplementedError

    def get_nearest_neighbors_from_existing(
        self, index: int, top_k: int | None = 100
    ) -> ResultSet:
        """Same as `get_nearest_neighbors`, but takes an `index` argument indicating to
        look up an existing known vector.

        Args:
            index: The index of the vector to query.
            top_k: Number of results to return. If `None`, return all results. Defaults to 100.

        Raises:
            NotImplementedError: Not implemented for the base class.

        Returns:
            An instance of `ResultSet` containing the nearest neighbors to the query.
        """
        raise NotImplementedError

    def _index(self, vectors: npt.NDArray, metric: str, *args, **kwargs):
        raise NotImplementedError

    def add_to_index(self, vectors: npt.NDArray):
        """Add additional vectors to the existing index.

        Args:
            vectors: Additional vectors to add to the existing index.

        Raises:
            NotImplementedError: Not implemented for the base class.
        """
        raise NotImplementedError


class ExactNearestNeighborsIndex(NearestNeighborsIndex):
    """A vector index that returns exact nearest neighbor results by calculating scores for all vectors."""

    def __init__(self, vectors: npt.NDArray, metric: str = "dot"):
        """Create a new `ExactNearestNeighbors` index.

        Args:
            vectors: A 2D numpy array, where the 0th dimension corresponds to items and the first dimension is each item's vector.
            metric: The metric that is used to determine how close vectors are to each other. Options are `dot`, `cosine`, and `euclidean`.
        """
        super().__init__(
            vectors, metric, available_metrics=["dot", "cosine", "euclidean"]
        )
        self._scorer = {
            "dot": self._score_nearest_dot,
            "cosine": self._score_nearest_cosine,
            "euclidean": self._score_nearest_euclidean,
        }[self.metric]
        self.index: npt.NDArray
        """A numpy array of the "indexed" vectors."""

    def _score_nearest_dot(self, vector: npt.NDArray) -> tuple[npt.NDArray, int]:
        return self.index @ vector, -1

    def _score_nearest_cosine(self, vector: npt.NDArray) -> tuple[npt.NDArray, int]:
        dot = self._score_nearest_dot(vector)[0]
        magnitudes = np.linalg.norm(self.index, axis=1) * np.linalg.norm(vector)
        return dot / magnitudes, -1

    def _score_nearest_euclidean(self, vector: npt.NDArray) -> tuple[npt.NDArray, int]:
        return np.linalg.norm(self.index - vector, axis=1), 1

    def get_nearest_neighbors(
        self, vector: npt.NDArray, top_k: int | None = 100
    ) -> ResultSet:
        """Return the `top_k` nearest vectors to the query vector.

        Args:
            vector: The query vector for which we want the nearest neighbors.
            top_k: Number of results to return. If `None`, return all results. Defaults to 100.

        Returns:
            An instance of `ResultSet` containing the nearest neighbors to the query.
        """
        scores, sort_direction = self._scorer(vector)
        index = np.argsort(sort_direction * scores)
        return ResultSet(
            indices=index[:top_k],
            scores=scores[index][:top_k],
        )

    def get_nearest_neighbors_from_existing(
        self, index: int, top_k: int | None = 100
    ) -> ResultSet:
        """Same as `get_nearest_neighbors`, but takes an `index` argument indicating to
        look up an existing known vector.

        Args:
            index: The index of the vector to query.
            top_k: Number of results to return. If `None`, return all results. Defaults to 100.

        Returns:
            An instance of `NearestNeighborsResult` containing the nearest neighbors to the query.
        """
        vector = self.index[index]
        results = self.get_nearest_neighbors(vector, top_k)
        return ResultSet(results.indices, results.scores)

    def _index(self, vectors: npt.NDArray, metric: str, **kwargs):
        return vectors

    def add_to_index(self, vectors: npt.NDArray):
        """Add additional vectors to the existing index.

        Args:
            vectors: Additional vectors to add to the existing index.
        """
        self.index = np.concatenate((self.index, vectors))
