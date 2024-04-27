from typing import Literal

import numpy as np
import numpy.typing as npt
from voyager import Index, Space, StorageDataType

from .. import ResultSet
from ._base import NearestNeighborsIndex


class VoyagerNearestNeighborsIndex(NearestNeighborsIndex):
    """A nearest neighbors index that uses Spotify's Voyager index under the hood."""

    def __init__(
        self,
        vectors: npt.NDArray,
        metric: Literal["euclidean", "dot", "cosine"],
        M: int = 12,
        ef_construction: int = 200,
        random_seed: int = 1,
        deterministic: bool = False,
        storage_data_type: StorageDataType = StorageDataType.Float32,
    ):
        """Create a Voyager nearest neighbors index.

        Args:
            vectors: The vectors to be indexed, with shape (num_vectors, num_dimensions).
            metric: The metric that is used to determine how close vectors are to each other.
            M: The number of connections between nodes in the tree's internal data structure.
                Larger values give better recall but use more memory. Defaults to 12.
            ef_construction: The number of vectors to search through when inserting a new
                vector into the index. Higher values make construction slower, but give
                better recall. Defaults to 200.
            random_seed: The seed of the RNG used when constructing the index. This can
                be used to ensure determinism (in combination with the `deterministic` argument.
                Defaults to 1.
            deterministic: If `True`, vectors will always be inserted one at a time, guaranteeing
                deterministic index creation. If `False`, multithreaded vector insertion
                may lead to slightly different indexes. Defaults to `False`.
            storage_data_type: The data type (from `voyager.StorageDataType`) used to store
                vectors in memory and on-disk. Defaults to `voyager.StorageDataType.Float32`.
        """
        super().__init__(
            vectors,
            metric,
            available_metrics=["euclidean", "dot", "cosine"],
            M=M,
            ef_construction=ef_construction,
            random_seed=random_seed,
            storage_data_type=storage_data_type,
            deterministic=deterministic,
        )
        self.index: Index
        """A Voyager `Index` of the indexed vectors."""
        self._vectors = self.index.get_vectors(np.arange(len(vectors)))
        self.deterministic = deterministic

    def _index(
        self,
        vectors: npt.NDArray,
        metric: str,
        M: int,
        ef_construction: int,
        random_seed: int,
        deterministic: bool,
        storage_data_type: StorageDataType,
    ) -> Index:
        space = {
            "euclidean": Space.Euclidean,
            "dot": Space.InnerProduct,
            "cosine": Space.Cosine,
        }
        index = Index(
            space=space[metric],
            num_dimensions=vectors.shape[1],
            M=M,
            ef_construction=ef_construction,
            random_seed=random_seed,
            max_elements=vectors.shape[0],
            storage_data_type=storage_data_type,
        )
        if deterministic:
            for vector in vectors:
                index.add_item(vector)
        else:
            index.add_items(vectors=vectors)
        return index

    def get_nearest_neighbors_from_existing(
        self, index: int, top_k: int | None = 100, query_ef: int = -1
    ) -> ResultSet:
        """Same as `get_nearest_neighbors`, but takes an `index` argument indicating to
        look up an existing known vector.

        Args:
            index: The index of the vector to query.
            top_k: Number of results to return. If `None`, return all results. Defaults to 100.

        Returns:
            An `ResultSet` containing the nearest neighbors to the query.
        """
        vector = self.index.get_vector(index)
        return self.get_nearest_neighbors(vector=vector, top_k=top_k, query_ef=-1)

    def get_nearest_neighbors(
        self, vector: npt.NDArray, top_k: int | None = 100, query_ef: int = -1
    ) -> ResultSet:
        """Return the `top_k` nearest vectors to the query vector.

        Args:
            vector: The query vector for which we want the nearest neighbors.
            top_k: Number of results to return. If `None`, return all results. Defaults to 100.
            query_ef: The depth of search to perform for this query. Up to `query_ef`
                candidates will be searched through to try to find up the k nearest neighbors
                per query vector. Defaults to -1.

        Returns:
            An `ResultSet` containing the nearest neighbors to the query.
        """
        if top_k is None:
            top_k = self._vectors.shape[0]

        indices, scores = self.index.query(vector, k=top_k, query_ef=query_ef)
        if self.metric == "euclidean":
            # voyager returns square of euclidean distances by default
            scores = np.sqrt(scores)
        elif self.metric in ("dot", "cosine"):
            # voyager returns 1 - dot_product and 1 - cosine_similairt by default
            scores = 1 - scores

        return ResultSet(np.asarray(indices), np.asarray(scores))

    def add_to_index(self, vectors: npt.NDArray):
        """Add additional vectors to the Voyager index.

        Args:
            vectors: Additional vectors to add to the existing Voyager index.
        """
        if self.deterministic:
            for vector in vectors:
                self.index.add_item(vector)
        else:
            self.index.add_items(vectors=vectors)
