from __future__ import annotations

from typing import Iterator

import numpy as np
import numpy.typing as npt
import pandas as pd


class ResultSet:
    """A container for document indices and and scores."""

    def __init__(self, indices: npt.ArrayLike, scores: npt.ArrayLike):
        """Create an object that stores indices and scores corresponding to a query.

        Args:
            indices: A list or numpy array of integers.
            scores: The scores corresponding to the indexed items.
        """
        self.indices = np.asarray(indices)
        # The indices that correspond to the results.
        self.scores = np.asarray(scores)
        # The scores that correspond to the results.

    def __getitem__(self, key) -> ResultSet:
        """Get a `ResultSet` resulting from passing `key` to __getitem__ of `indices` and `scores`.

        Args:
            key: An indexing object (integer, slice, mask).

        Returns:
            A `ResultSet` with indices and scores indexed from this one by `key`.
        """
        return ResultSet(self.indices[key], self.scores[key])

    def __iter__(self) -> Iterator[tuple[int, float]]:
        """Create an iterable containing (index, score) tuples from the `ResultSet`.

        Returns:
            An iterable of (index, score) tuples.

        Yields:
            An (index, score) tuple.
        """
        return zip(self.indices, self.scores)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the `ResultSet` to a Pandas dataframe with a `scores` column and an index set from `indices`.

        Returns:
            A Pandas dataframe with a `scores` column and an index set from `indices`.
        """
        return pd.DataFrame({"scores": self.scores}, index=self.indices)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> ResultSet:
        """Create a `ResultSet` from a Pandas dataframe with a column called `scores` and an index.

        Args:
            df: The Pandas dataframe. Must have a `scores` column and an integer index. All other columns are ignored.

        Returns:
            A `ResultSet` with `indices` and `scores` derived from the dataframe.
        """
        return ResultSet(df.index.values, df["scores"].values)  # type: ignore

    def first(self, k: int) -> ResultSet:
        """Get the first `k` items from the `ResultSet`.

        Args:
            k: How many results to include in the `ResultSet`.

        Returns:
            A `ResultSet` with the first `k` indices and scores from this `ResultSet`.
        """
        return self[:k]

    def sort(self, ascending: bool = False) -> ResultSet:
        """Sort the `ResultSet` by scores.

        Args:
            ascending: If `True`, sort from lowest to highest scores. Defaults to False.

        Returns:
            A `ResultSet` with the same contents as this one, but with contents sorted by score.
        """
        return ResultSet.from_dataframe(
            self.to_dataframe().sort_values("scores", ascending=ascending)
        )

    def max_scale(self) -> ResultSet:
        """Divide all scores by the largest score in the `ResultSet`.

        Returns:
            A `ResultSet` with a maximum score of 1.
        """
        max_score = self.scores.max()
        return self / max_score

    def lt(self, val: float = 0) -> ResultSet:
        """Filter the result set to items with scores less than `val`.

        Args:
            val: Scores in the result must be less than `val`. Defaults to 0.

        Returns:
            A `ResultSet` where all scores are less than `val`.
        """
        index = self.scores < val
        return self[index]

    def gt(self, val: float = 0) -> ResultSet:
        """Filter the result set to items with scores greater than `val`.

        Args:
            val: Scores in the result must be greater than `val`. Defaults to 0.

        Returns:
            A `ResultSet` where all scores are greater than `val`.
        """
        index = self.scores > val
        return self[index]

    def leq(self, val: float = 0) -> ResultSet:
        """Filter the result set to items with scores less than or equal to `val`.

        Args:
            val: Scores in the result must be less than or equal to `val`. Defaults to 0.

        Returns:
            A `ResultSet` where all scores are less than or equal to `val`.
        """
        index = self.scores <= val
        return self[index]

    def geq(self, val: float = 0) -> ResultSet:
        """Filter the result set to items with scores greater than or equal to `val`.

        Args:
            val: Scores in the result must be greater than or equal to `val`. Defaults to 0.

        Returns:
            A `ResultSet` where all scores are greater than or equal to `val`.
        """
        index = self.scores >= val
        return self[index]

    def reverse(self) -> ResultSet:
        """Reverse the order of the items in the `ResultSet`.

        Returns:
            A `ResultSet` with the same contents but in reversed order.
        """
        return ResultSet(self.indices[::-1], self.scores[::-1])

    def __mul__(self, other: float) -> ResultSet:
        """Multiply the scores in the `ResultSet` by `other`.

        Args:
            other: A float to multiply the scores by.

        Returns:
            A new `ResultSet` with scores multiplied by `other`.
        """
        return ResultSet(self.indices, self.scores * other)

    def __rmul__(self, other: float) -> ResultSet:
        return self.__mul__(other)

    def __truediv__(self, other: float) -> ResultSet:
        """Divide the scores in the `ResultSet` by `other`.

        Args:
            other: A float to divide the scores by.

        Returns:
            A new `ResultSet` with scores divided by `other`.
        """
        return ResultSet(self.indices, self.scores / other)

    def __rtruediv__(self, other: float) -> ResultSet:
        return ResultSet(self.indices, other / self.scores)

    def __neg__(self) -> ResultSet:
        """Negate the scores in the `ResultSet`.

        Returns:
            A new `ResultSet` with scores negated.
        """
        return ResultSet(self.indices, -1 * self.scores)

    def __add__(self, other: ResultSet | float) -> ResultSet:
        """Add a constant or another `ResultSet`'s scores to this `ResultSet`. If adding another `ResultSet`, missing items will be coalesced to 0.

        Args:
            other: A float or another `ResultSet` to add to this one.

        Returns:
            A `ResultSet` with scores that are the sum of the original `ResultSet` and `other`.
        """
        df_self = self.to_dataframe()
        if isinstance(other, ResultSet):
            df_other = other.to_dataframe()
            df_both = df_self.add(df_other, fill_value=0)
        else:
            df_both = df_self + other
        return ResultSet.from_dataframe(df_both)

    def __radd__(self, other: ResultSet | float) -> ResultSet:
        return self.__add__(other)

    def __sub__(self, other: ResultSet | npt.ArrayLike) -> ResultSet:
        """Subtract a constant or another `ResultSet`'s scores to this `ResultSet`. If adding another `ResultSet`, missing items will be coalesced to 0.

        Args:
            other: A float or another `ResultSet` to subtract from this one.

        Returns:
            A `ResultSet` with scores that are the difference of the original `ResultSet` and `other`.
        """
        if isinstance(other, ResultSet):
            index_difference = ~np.in1d(self.indices, other.indices)
        else:
            index_difference = ~np.in1d(self.indices, other)
        return ResultSet(self.indices[index_difference], self.scores[index_difference])

    def __contains__(self, item: int) -> bool:
        """Check if an integer is in `indices`.

        Args:
            item: An integer index.

        Returns:
            A boolean indicating if the integer is in `indices`.
        """
        return item in self.indices

    def __len__(self) -> int:
        """Get the number of items in the `ResultSet`.

        Returns:
            An integer indicating the number of items in the `ResultSet`.
        """
        return len(self.indices)

    def __pow__(self, exp: float) -> ResultSet:
        """Raise the scores of this `ResultSet` to the `exp` power.

        Args:
            exp: The power to raise the scores to.

        Returns:
            A `ResultSet` with scores equal to the current scores raised to the `exp` power.
        """
        return ResultSet(self.indices, self.scores**exp)

    def __or__(self, other: ResultSet) -> ResultSet:
        """Create a `ResultSet` that is the union of this and `other`. The score of this `ResultSet` will be preferred over the score from `Other`.

        Args:
            other: Another `ResultSet`.

        Returns:
            A new `ResultSet` with all the `indices` of this and `other`, and the scores from this (if they exist) or else from `other`.
        """
        index_difference = ~np.in1d(other.indices, self.indices)
        indices = np.concatenate([self.indices, other.indices[index_difference]])
        scores = np.concatenate([self.scores, other.scores[index_difference]])
        return ResultSet(indices, scores)

    def __and__(self, other: ResultSet) -> ResultSet:
        """Create a `ResultSet` that is the intersection of this and `other`. The score of this `ResultSet` will be preferred over the score from `Other`.

        Args:
            other: Another `ResultSet`.

        Returns:
            A new `ResultSet` with all the `indices` in both this and `other`, and the scores from this (if they exist) or else from `other`.
        """
        index_intersection = np.in1d(self.indices, other.indices)
        return ResultSet(
            self.indices[index_intersection], self.scores[index_intersection]
        )

    def __eq__(self, other) -> bool:
        """Check whether the indices and scores in this `ResultSet` are the same, in the same order, as those in `other`.

        Args:
            other: Another object. If not a `ResultSet`, then it will always be unequal.

        Returns:
            A boolean indicating equality.
        """
        if not isinstance(other, ResultSet):
            return False
        else:
            return bool(
                np.array(self.scores == other.scores).all()
                and np.array(self.indices == other.indices).all()
            )
