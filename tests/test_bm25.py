import numpy as np
import pytest
import rufus.text as text
from rufus.text.bm25 import BM25IndexBase, BM25LIndex, BM25PlusIndex, OkapiBM25Index


@pytest.fixture()
def sample_items() -> list[str]:
    return ["hello world", "my name is Dan!", "Let's take a trip to Berlin"]


@pytest.fixture()
def text_pipeline() -> text.TextPipeline:
    return text.TextPipeline(
        [text.lowercase, text.remove_punctuation, text.split_on_whitespace]
    )


def test_okapi_bm25_index(text_pipeline: text.TextPipeline, sample_items: list[str]):
    okapi_index = OkapiBM25Index(
        text_pipeline.batch(sample_items), k1=1.6, b=0.8, eps=0.3
    )
    result = okapi_index.get_scores_for_query(text_pipeline("hello berlin"))
    assert np.allclose(result.scores, np.array([0.67762583, 0.0, 0.4099218]))
    assert np.all(result.indices == [0, 1, 2])


def test_bm25l_index(text_pipeline: text.TextPipeline, sample_items: list[str]):
    bm25l_index = BM25LIndex(
        text_pipeline.batch(sample_items), k1=1.6, b=0.8, delta=0.6
    )
    result = bm25l_index.get_scores_for_query(text_pipeline("hello berlin"))
    assert np.allclose(result.scores, [1.49491907, 0.0, 1.15007038])
    assert np.all(result.indices == [0, 1, 2])


def test_bm25plus_index(text_pipeline: text.TextPipeline, sample_items: list[str]):
    bm25plus_index = BM25PlusIndex(
        text_pipeline.batch(sample_items), k1=1.6, b=0.8, delta=0.6
    )
    result = bm25plus_index.get_scores_for_query(text_pipeline("hello berlin"))
    assert np.allclose(result.scores, [3.50251514, 1.66355323, 2.77601167])
    assert np.all(result.indices == [0, 1, 2])


def test_bm25_subset(text_pipeline: text.TextPipeline, sample_items: list[str]):
    okapi_index = OkapiBM25Index(text_pipeline.batch(sample_items))
    result = okapi_index.get_scores_for_query(
        text_pipeline("hello berlin"), subset=[0, 2]
    )
    assert np.allclose(result.scores, [0.65912984, 0.41700051])
    assert np.all(result.indices == [0, 2])


def test_bm25_base_notimplementederror():
    with pytest.raises(NotImplementedError):
        _ = BM25IndexBase([[]])
