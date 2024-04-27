import pandas as pd
import pytest
from rufus import ResultSet
from rufus.documents import DocumentIndexBase, PandasIndex, SequenceIndex

DF = pd.DataFrame({"foo": ["bar", "baz", "bal"], "fee": ["fi", "fo", "fum"]})
SEQUENCE = [("a", 123), ("b", 456), ("c", 789)]


def test_document_index_not_implemented():
    di = DocumentIndexBase()
    with pytest.raises(NotImplementedError):
        di.get_documents([1])


def test_pandas_index_result_set():
    pi = PandasIndex(DF)
    assert all(
        pd.DataFrame(
            {"foo": ["bar", "bal"], "fee": ["fi", "fum"], "score": [1, 3]}, index=[0, 2]
        )
        == pi.get_documents(ResultSet([0, 2], [1, 3]))
    )


def test_pandas_index_result_set_score_col():
    pi = PandasIndex(DF)
    assert all(
        pd.DataFrame(
            {"foo": ["bar", "bal"], "fee": ["fi", "fum"], "test": [1, 3]}, index=[0, 2]
        )
        == pi.get_documents(ResultSet([0, 2], [1, 3]), score_name="test")
    )


def test_pandas_index_result_no_score():
    pi = PandasIndex(DF)
    assert all(
        pd.DataFrame({"foo": ["bar", "bal"], "fee": ["fi", "fum"]}, index=[0, 2])
        == pi.get_documents(ResultSet([0, 2], [1, 3]), with_scores=False)
    )


def test_pandas_index_arraylike():
    pi = PandasIndex(DF)
    assert all(
        pd.DataFrame({"foo": ["bar", "bal"], "fee": ["fi", "fum"]}, index=[0, 2])
        == pi.get_documents([0, 2])
    )


def test_sequence_index_result_set():
    si = SequenceIndex(SEQUENCE)
    assert [(("a", 123), 1), (("c", 789), 3)] == si.get_documents(
        ResultSet([0, 2], [1, 3])
    )


def test_sequence_index_result_set_no_score():
    si = SequenceIndex(SEQUENCE)
    assert [("a", 123), ("c", 789)] == si.get_documents(
        ResultSet([0, 2], [1, 3]), with_scores=False
    )


def test_sequence_index_arraylike():
    si = SequenceIndex(SEQUENCE)
    assert [("a", 123), ("c", 789)] == si.get_documents([0, 2], with_scores=False)
