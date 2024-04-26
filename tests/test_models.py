import numpy as np
import pytest
from rufus.models import CrossEncoderWrapper, ModelWrapper
from rufus.models.huggingface import HFAveragePooled
from rufus.models.sentence_transformers import (
    SentenceTransformerCrossEncoderWrapper,
    SentenceTransformerModelWrapper,
)


def test_model_wrapper_embed_not_implemented():
    mw = ModelWrapper("foo")
    with pytest.raises(NotImplementedError):
        mw(["foo"])


def test_cross_encoder_wrapper_not_implemented():
    ce = CrossEncoderWrapper("foo")
    with pytest.raises(NotImplementedError):
        ce([["a", "b"]])


def test_sentence_transformer_model_wrapper_embed():
    from .data import SENTENCE_TRANSFORMER_TEST_DATA

    st = SentenceTransformerModelWrapper("sentence-transformers/all-minilm-l6-v2")
    embeddings = st(["hello world", "foo bar"])
    assert np.allclose(
        embeddings,
        SENTENCE_TRANSFORMER_TEST_DATA,
        atol=1e-7,
    )


def test_sentence_transformers_cross_encoder_wrapper_predict():
    st = SentenceTransformerCrossEncoderWrapper(
        "cross-encoder/ms-marco-TinyBert-L-2-v2"
    )
    assert np.allclose(
        st(
            [
                ["Berlin is nice this time of year", "Berlin is a tourist trap"],
                ["Berlin is nice this time of year", "Bob is a nice man"],
            ]
        ),
        np.array([-8.354256, -11.128227]),
    )


def test_sentence_transformers_cross_encoder_wrapper_predict_sigmoid():
    st = SentenceTransformerCrossEncoderWrapper(
        "cross-encoder/ms-marco-TinyBert-L-2-v2", apply_sigmoid=True
    )
    assert np.allclose(
        st(
            [
                ["Berlin is nice this time of year", "Berlin is a tourist trap"],
                ["Berlin is nice this time of year", "Bob is a nice man"],
            ]
        ),
        np.array([2.3533724e-04, 1.4691493e-05]),
    )


def test_sentence_transformers_cross_encoder_wrapper_predict_one_to_many():
    st = SentenceTransformerCrossEncoderWrapper(
        "cross-encoder/ms-marco-TinyBert-L-2-v2"
    )
    assert np.allclose(
        st.predict_one_to_many(
            "Berlin is nice this time of year",
            ["Berlin is a tourist trap", "Bob is a nice man"],
        ),
        np.array([-8.354256, -11.128227]),
    )


def test_sentence_transformers_cross_encoder_wrapper_predict_one_to_many_results():
    st = SentenceTransformerCrossEncoderWrapper(
        "cross-encoder/ms-marco-TinyBert-L-2-v2"
    )
    results = st.predict_one_to_many_results(
        "Berlin is nice this time of year",
        items=["Berlin is a tourist trap", "Bob is a nice man"],
        ids=np.array([3, 4]),
    )
    assert np.allclose(
        results.scores,
        np.array([-8.354256, -11.128227]),
    )
    assert all(results.indices == np.array([3, 4]))


def test_hf_average_pooled_wrapper_embed():
    from .data import HF_AVERAGE_POOLED_TEST_DATA

    hf = HFAveragePooled("intfloat/e5-small-v2")
    embeddings = hf(["hello world", "foo bar"])
    assert np.allclose(embeddings, HF_AVERAGE_POOLED_TEST_DATA, atol=0.01)
