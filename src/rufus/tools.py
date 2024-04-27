from collections import defaultdict
from functools import reduce
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from . import ResultSet

__all__ = ["reciprocal_rank_fusion", "union", "intersection"]


def reciprocal_rank_fusion(
    result_sets: list[ResultSet],
    rank_type: Literal["dense", "average", "min", "max", "dense", "ordinal"] = "dense",
    rank_ascending: bool = False,
    k: int = 60,
) -> ResultSet:
    ascending = 1 if rank_ascending else -1
    scores_dict: defaultdict[int, float] = defaultdict(float)
    for result_set in result_sets:
        for rank, index in zip(
            rankdata(ascending * result_set.scores, rank_type), result_set.indices
        ):
            scores_dict[index] += 1 / (k + rank)
    indices = np.array(list(scores_dict.keys()), dtype=int)
    scores = np.array(list(scores_dict.values()))

    return ResultSet(indices=indices, scores=scores).sort()


def _merge(
    result_sets: list[ResultSet],
    how: Literal["inner", "outer"],
    score: Literal["first", "max", "min"],
) -> ResultSet:
    # inspiration from https://stackoverflow.com/a/65941720
    dfs = [rs.to_dataframe().add_suffix(f"_{i}") for i, rs in enumerate(result_sets)]
    merged_df = reduce(
        lambda x, y: pd.merge(x, y, how=how, left_index=True, right_index=True),
        dfs,
    )
    if score == "max":
        scores = merged_df.max(axis=1).values
    elif score == "min":
        scores = merged_df.min(axis=1).values
    elif score == "first":
        scores = merged_df.bfill(axis=1).iloc[:, 0].values
    else:
        raise ValueError("Argument `how` must be one of 'first', 'max', 'min'")

    return ResultSet(merged_df.index.values, scores)  # type: ignore


def union(
    result_sets: list[ResultSet], score: Literal["first", "max", "min"] = "first"
) -> ResultSet:
    return _merge(result_sets, how="outer", score=score)


def intersection(
    result_sets: list[ResultSet], score: Literal["first", "max", "min"] = "first"
) -> ResultSet:
    return _merge(result_sets, how="inner", score=score)
