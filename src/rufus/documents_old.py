from typing import Type

import numpy as np
import pandas as pd
from numpy import typing as npt

from rufus import ResultSet
from rufus.vector import NearestNeighborsIndex

from .models import ModelWrapper


class EmbeddedError(Exception):
    pass


class NotEmbeddedError(Exception):
    pass


class PandasIndex:
    def __init__(
        self,
        items: pd.DataFrame,
        key_column: str,
        model: ModelWrapper,
        nearest_neighbors: NearestNeighborsIndex | None = None,
        nearest_neighbors_class: Type[NearestNeighborsIndex] | None = None,
    ):
        self.items = items
        self.key_column = key_column
        self.model = model
        if nearest_neighbors is not None:
            self.nearest_neighbors = nearest_neighbors
            self._embeddings = self.nearest_neighbors.vectors
        elif nearest_neighbors_class is not None:
            self._nearest_neighbors_class = nearest_neighbors_class
            self._embeddings = None
        else:
            raise ValueError(
                "Either nearest_neighbors or nearest_neighbors_class must be not `None`"
            )

    def _raise_if_not_embedded(self):
        if (self._embeddings is None) or (self.nearest_neighbors is None):
            raise NotEmbeddedError("No embeddings available. Run embed() first.")

    def embed_and_index(self, batch_size: int = 1, metric: str = "dot"):
        if self._embeddings is not None:
            raise EmbeddedError("Already embedded!")

        embeddings = np.asarray(
            self.model.__call__(
                list(self.items[self.key_column]), batch_size=batch_size
            )
        )
        self.nearest_neighbors = self._nearest_neighbors_class(
            embeddings, metric=metric
        )
        self._embeddings = self.nearest_neighbors.vectors

    def embed_query(self, query: str) -> npt.NDArray[np.float_]:
        return np.asarray(self.model.__call__([query]))[0]

    @property
    def embeddings(self) -> npt.NDArray[np.float_]:
        self._raise_if_not_embedded()
        return self._embeddings

    def nn(
        self, vector: npt.NDArray[np.float_], top_k: int | None = None, **kwargs
    ) -> ResultSet:
        self._raise_if_not_embedded()
        return self.nearest_neighbors.get_nearest_neighbors(
            vector, top_k=top_k, **kwargs
        )

    def nn_from_existing(
        self, index: int, top_k: int | None = None, **kwargs
    ) -> ResultSet:
        self._raise_if_not_embedded()
        return self.nearest_neighbors.get_nearest_neighbors_from_existing(
            index, top_k=top_k, **kwargs
        )

    def _nn_items(
        self,
        results: ResultSet,
        item_cols_to_return: list[str] | None = None,
        return_score_col: bool = True,
        score_col_name: str = "score",
    ) -> pd.DataFrame:
        self._raise_if_not_embedded()
        columns = (
            list(self.items.columns)
            if item_cols_to_return is None
            else item_cols_to_return
        )

        if return_score_col:
            columns += [score_col_name]
            return (
                self.items.loc[results.indices, :]
                .assign(**{score_col_name: results.scores})
                .loc[:, columns]
            )

        return self.items.loc[results.indices, columns]

    def nn_items(
        self,
        query: str,
        item_cols_to_return: list[str] | None = None,
        return_score_col: bool = True,
        score_col_name: str = "score",
        **kwargs,
    ):
        results = self.nn(self.embed_query(query))
        return self._nn_items(
            results, item_cols_to_return, return_score_col, score_col_name, **kwargs
        )

    def nn_items_from_existing(
        self,
        index: int,
        item_cols_to_return: list[str] | None = None,
        return_score_col: bool = True,
        score_col_name: str = "score",
        **kwargs,
    ) -> pd.DataFrame:
        self._raise_if_not_embedded()
        results = self.nn_from_existing(index)
        return self._nn_items(
            results, item_cols_to_return, return_score_col, score_col_name, **kwargs
        )
