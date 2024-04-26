import numpy as np
import pytest
from rufus import ResultSet
from rufus.tools import intersection, reciprocal_rank_fusion, union

RS_A = ResultSet(np.array([0, 1, 2]), np.array([3, 2, 4]))
RS_B = ResultSet(np.array([1, 2, 3, 5]), np.array([4, 5, 1, 4]))

RS_1 = ResultSet(np.array([0, 1, 2]), np.array([0, 1, 2]))
RS_2 = ResultSet(np.array([1, 3, 4]), np.array([9, 3, 4]))
RS_3 = ResultSet(np.array([0, 5]), np.array([-1, 5]))

RS_4 = ResultSet(np.array([1, 2, 3]), np.array([-5, 10, 3]))
RS_5 = ResultSet(np.array([1, 2, 4]), np.array([11, -6, 4]))


def test_reciprocal_rank_fusion():
    result = reciprocal_rank_fusion([RS_A, RS_B])
    assert (result.indices == np.array([2, 1, 0, 5, 3])).all()
    assert np.allclose(
        result.scores,
        np.array([[0.03278689, 0.03200205, 0.01612903, 0.01612903, 0.01587302]]),
    )


def test_reciprocal_rank_fusion_ascending():
    result = reciprocal_rank_fusion([RS_A, RS_B], rank_ascending=True)
    assert (result.indices == np.array([1, 2, 3, 0, 5])).all()
    assert np.allclose(
        result.scores,
        np.array([[0.03252247, 0.03174603, 0.01639344, 0.01612903, 0.01612903]]),
    )


def test_reciprocal_rank_fusion_with_k():
    result = reciprocal_rank_fusion([RS_A, RS_B], k=100)
    assert (result.indices == np.array([2, 1, 0, 5, 3])).all()
    assert np.allclose(
        result.scores,
        np.array([0.01980198, 0.01951266, 0.00980392, 0.00980392, 0.00970874]),
    )


def test_reciprocal_rank_fusion_with_alt_ranking():
    result = reciprocal_rank_fusion([RS_A, RS_B], rank_type="min")
    assert (result.indices == np.array([2, 1, 0, 5, 3])).all()
    assert np.allclose(
        result.scores,
        np.array([0.03278689, 0.03200205, 0.01612903, 0.01612903, 0.015625]),
    )


def test_merge_invalid_score():
    with pytest.raises(ValueError):
        union([RS_1, RS_2], "foo")  # type: ignore


def test_union_first():
    res = union([RS_1, RS_2, RS_3], "first")
    assert (res.indices == np.array([0, 1, 2, 3, 4, 5])).all()
    assert (res.scores == np.array([0, 1, 2, 3, 4, 5])).all()


def test_union_max():
    res = union([RS_1, RS_2, RS_3], "max")
    assert (res.indices == np.array([0, 1, 2, 3, 4, 5])).all()
    assert (res.scores == np.array([0, 9, 2, 3, 4, 5])).all()


def test_union_min():
    res = union([RS_1, RS_2, RS_3], "min")
    assert (res.indices == np.array([0, 1, 2, 3, 4, 5])).all()
    assert (res.scores == np.array([-1, 1, 2, 3, 4, 5])).all()


def test_intersection_first():
    res = intersection([RS_1, RS_4, RS_5], "first")
    assert (res.indices == np.array([1, 2])).all()
    assert (res.scores == np.array([1, 2])).all()


def test_intersection_max():
    res = intersection([RS_1, RS_4, RS_5], "max")
    assert (res.indices == np.array([1, 2])).all()
    assert (res.scores == np.array([11, 10])).all()


def test_intersection_min():
    res = intersection([RS_1, RS_4, RS_5], "min")
    assert (res.indices == np.array([1, 2])).all()
    assert (res.scores == np.array([-5, -6])).all()
