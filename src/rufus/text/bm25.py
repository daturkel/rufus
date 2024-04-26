import numpy as np
import numpy.typing as npt
from rank_bm25 import BM25, BM25L, BM25Okapi, BM25Plus

from .. import ResultSet


class BM25Index:
    def __init__(
        self,
        items: list[list[str]],
        **kwargs,
    ):
        self._index: BM25 = self._create_index(items, **kwargs)

        self.n_items = len(items)

    def _create_index(self, corpus: list[list[str]], *args, **kwargs):
        raise NotImplementedError

    def get_scores_for_query(
        self, query: list[str] | str, subset: npt.ArrayLike | None = None
    ):
        if subset is not None:
            subset = np.asarray(subset)
            scores = np.array(self._index.get_batch_scores(query, subset))
            indices = subset
        else:
            scores = self._index.get_scores(query)
            indices = np.arange(self.n_items)

        return ResultSet(indices, scores)


class OkapiBM25Index(BM25Index):
    def __init__(
        self,
        items: list[list[str]],
        k1: float = 1.5,
        b: float = 0.75,
        eps: float = 0.25,
    ):
        super().__init__(items, k1=k1, b=b, eps=eps)
        self.k1 = k1
        self.b = b
        self.eps = eps

    def _create_index(
        self, corpus: list[list[str]], k1: float, b: float, eps: float
    ) -> BM25Okapi:
        return BM25Okapi(corpus, k1=k1, b=b, epsilon=eps)


class BM25LIndex(BM25Index):
    def __init__(
        self,
        items: list[list[str]],
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 0.5,
    ):
        super().__init__(items, k1=k1, b=b, delta=delta)
        self.k1 = k1
        self.b = b
        self.delta = delta

    def _create_index(
        self, corpus: list[list[str]], k1: float, b: float, delta: float
    ) -> BM25L:
        return BM25L(corpus, k1=k1, b=b, delta=delta)


class BM25PlusIndex(BM25Index):
    def __init__(
        self,
        items: list[list[str]],
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 0.5,
    ):
        super().__init__(items, k1=k1, b=b, delta=delta)
        self.k1 = k1
        self.b = b
        self.delta = delta

    def _create_index(
        self, corpus: list[list[str]], k1: float, b: float, delta: float
    ) -> BM25Plus:
        # type checker thinks BM25Plus delta is supposed to be an int for some reason
        return BM25Plus(corpus, k1=k1, b=b, delta=delta)  # type: ignore
