from __future__ import annotations

import pickle
from collections.abc import Sequence
from typing import Any, Self

# from sqlite_utils import Database
import duckdb
import numpy as np
import numpy.typing as npt
import pandas as pd

from . import ResultSet

__all__ = ["DocumentIndexBase", "PandasIndex", "SequenceIndex", "DuckDBIndex"]


class DocumentIndexBase:
    def __init__(self):
        pass

    def get_documents(
        self,
        indices: ResultSet | npt.NDArray[np.int_] | list[int],
        with_scores: bool = True,
    ) -> Any:
        raise NotImplementedError()

    def save(self, filename: str):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename: str) -> Self:
        with open(filename, "rb") as f:
            obj = pickle.load(f)
        return obj


class PandasIndex(DocumentIndexBase):
    def __init__(self, dataframe: pd.DataFrame):
        super().__init__()
        self.index = dataframe

    def get_documents(
        self,
        indices: ResultSet | npt.NDArray[np.int_] | list[int],
        with_scores: bool = True,
        score_name: str = "score",
    ) -> pd.DataFrame:
        if isinstance(indices, ResultSet):
            indices_, scores = indices.indices, indices.scores
        else:
            indices_ = np.asarray(indices)
            scores = None

        documents = self.index.loc[indices_, :]

        if (scores is not None) and with_scores:
            documents = documents.assign(**{score_name: scores})

        return documents


class SequenceIndex:
    def __init__(self, sequence: Sequence):
        self.index = sequence

    def get_documents(
        self,
        indices: ResultSet | npt.NDArray[np.int_] | list[int],
        with_scores: bool = True,
    ) -> list:
        if isinstance(indices, ResultSet):
            indices_, scores = indices.indices, indices.scores
        else:
            indices_ = np.asarray(indices)
            scores = None

        if (scores is not None) and with_scores:
            documents = [
                (self.index[i], scores[score_index])
                for score_index, i in enumerate(indices_)
            ]
        else:
            documents = [self.index[i] for i in indices_]

        return documents


# class SqliteIndex:
#     def __init__(self, db_path: str, index_col: str, table_name: str):
#         self.database = Database(filename_or_conn=db_path)
#         self.table_name = table_name
#         self._index_col = index_col

#     @classmethod
#     def from_dataframe(
#         cls, dataframe: pd.DataFrame, db_path: str, index_col: str, table_name: str
#     ) -> SqliteIndex:
#         database = Database(filename_or_conn=db_path)
#         database[table_name].insert_all(dataframe.to_dict("records"), pk=index_col)  # type: ignore
#         return cls(db_path, table_name, index_col)

#     def get_documents(
#         self,
#         indices: ResultSet | npt.NDArray[np.int_] | list[int],
#         with_scores: bool = True,
#         score_name: str = "score",
#     ) -> list[dict[str, Any]]:
#         if isinstance(indices, ResultSet):
#             indices_, scores = indices.indices, indices.scores
#         else:
#             indices_ = np.asarray(indices)
#             scores = None

#         condition = f"{self._index_col} IN {tuple(indices_)}"

#         documents = {
#             item[self._index_col]: item
#             for item in self.database[self.table_name].rows_where(condition)
#         }

#         if (scores is not None) and with_scores:
#             sorted_documents = [
#                 dict(documents[i], **{score_name: scores[score_index]})
#                 for score_index, i in enumerate(indices)
#             ]
#         else:
#             sorted_documents = [documents[i] for i in indices]

#         return sorted_documents


class DuckDBIndex(DocumentIndexBase):
    def __init__(self, db_path: str, index_column: str, table_name: str):
        self._db_path = db_path
        self._index_column = index_column
        self._table_name = table_name

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, index_column: str, db_path: str):
        ddb_index = cls(
            db_path=db_path, index_column=index_column, table_name="documents"
        )
        with duckdb.connect(db_path) as conn:
            relation = conn.from_df(df)  #   # noqa: F841
            conn.sql("CREATE TABLE documents AS SELECT * FROM relation")

        return ddb_index

    def get_documents(
        self,
        indices: ResultSet | npt.NDArray[np.int_] | list[int],
        with_scores: bool = True,
        score_name: str = "score",
    ) -> pd.DataFrame:
        if isinstance(indices, ResultSet):
            _index_df = indices.to_dataframe().reset_index()
            scores_available = True
        else:
            _index_df = pd.DataFrame({"index": indices})
            scores_available = False
        _index_df.loc[:, "row_number"] = range(0, len(_index_df))  # type: ignore

        with duckdb.connect(self._db_path, read_only=True) as conn:
            qry = f"""SELECT t.*
                      {', i.score as ' + score_name if with_scores and scores_available else ''}
                      FROM {self._table_name} t
                      INNER JOIN _index_df i
                        ON i.index = t.{self._index_column}
                      ORDER BY i.row_number ASC
                    """
            res = conn.sql(qry).df()
        return res
