import numpy as np
import numpy.typing as npt
import pandas as pd
from rufus import ResultSet

RS_A = ResultSet(np.array([0, 1, 2, 3]), np.array([0.0, 0.1, 0.2, 0.3]))
RS_B = ResultSet(np.array([5, 3, 4]), np.array([0.5, 0.6, 0.7]))


def _match_indices_and_scores(
    source: ResultSet, indices: npt.NDArray, scores: npt.NDArray, close: bool = False
):
    indices_match = (source.indices == indices).all()
    if close:
        scores_match = np.allclose(source.scores, scores)
    else:
        scores_match = (source.scores == scores).all()
    return bool(indices_match and scores_match)


def test_resultset_to_dataframe():
    df = RS_A.to_dataframe()
    assert np.array(df.index.values == RS_A.indices).all()
    assert np.array(df["scores"].values == RS_A.scores).all()
    assert df.columns == ["scores"]


def test_resultset_top():
    top = RS_A.first(2)
    assert _match_indices_and_scores(top, np.array([0, 1]), np.array([0, 0.1]))


def test_resultset_from_dataframe():
    df = pd.DataFrame({"scores": [0.1, 0.2, 0.3]}, index=[3, 1, 5])
    rs = ResultSet.from_dataframe(df)
    assert _match_indices_and_scores(rs, np.array([3, 1, 5]), np.array([0.1, 0.2, 0.3]))


def test_resultset_mul():
    rs_mul = RS_A * 2.3
    assert _match_indices_and_scores(
        rs_mul, RS_A.indices, np.array([0.0, 0.23, 0.46, 0.69]), close=True
    )


def test_resultset_rmul():
    rs_mul = 2.3 * RS_A
    assert _match_indices_and_scores(
        rs_mul, RS_A.indices, np.array([0.0, 0.23, 0.46, 0.69]), close=True
    )


def test_resultset_truediv():
    rs_div = RS_A / 2.3
    assert _match_indices_and_scores(
        rs_div,
        RS_A.indices,
        np.array([0, 0.04347826, 0.08695652, 0.13043478]),
        close=True,
    )


def test_resultset_rtruediv():
    rs_div = 2.3 / RS_B
    assert _match_indices_and_scores(
        rs_div, RS_B.indices, np.array([2.3 / 0.5, 2.3 / 0.6, 2.3 / 0.7]), close=True
    )


def test_resultset_neg():
    rs_neg = -RS_A
    assert _match_indices_and_scores(
        rs_neg, RS_A.indices, np.array([0, -0.1, -0.2, -0.3])
    )


def test_resultset_add_float():
    rs_add = RS_A + 0.1
    assert _match_indices_and_scores(
        rs_add, RS_A.indices, np.array([0.1, 0.2, 0.3, 0.4]), close=True
    )


def test_resultset_radd_float():
    rs_add = 0.1 + RS_A
    assert _match_indices_and_scores(
        rs_add, RS_A.indices, np.array([0.1, 0.2, 0.3, 0.4]), close=True
    )


def test_resultset_add_rs():
    rs_add = RS_A + RS_B
    assert _match_indices_and_scores(
        rs_add,
        np.array([0, 1, 2, 3, 4, 5]),
        np.array([0, 0.1, 0.2, 0.9, 0.7, 0.5]),
        close=True,
    )


def test_resultset_pow():
    rs_pow = RS_A**2
    assert _match_indices_and_scores(
        rs_pow, RS_A.indices, np.array([0, 0.01, 0.04, 0.09]), close=True
    )


def test_resultset_contains():
    assert 2 in RS_A
    assert 5 not in RS_A


def test_resultset_len():
    assert len(RS_A) == 4


def test_resultset_sub_rs():
    rs_sub = RS_A - RS_B
    assert _match_indices_and_scores(
        rs_sub, np.array([0, 1, 2]), np.array([0, 0.1, 0.2])
    )


def test_resultset_sub_list():
    rs_sub = RS_A - [0, 5]
    assert _match_indices_and_scores(
        rs_sub, np.array([1, 2, 3]), np.array([0.1, 0.2, 0.3])
    )


def test_resultset_sub_array():
    rs_sub = RS_A - np.array([0, 5])
    assert _match_indices_and_scores(
        rs_sub, np.array([1, 2, 3]), np.array([0.1, 0.2, 0.3])
    )


def test_resultset_or():
    rs_or = RS_A | RS_B
    assert _match_indices_and_scores(
        rs_or, np.array([0, 1, 2, 3, 5, 4]), np.array([0, 0.1, 0.2, 0.3, 0.5, 0.7])
    )

    rs_or_b = RS_B | RS_A
    assert _match_indices_and_scores(
        rs_or_b, np.array([5, 3, 4, 0, 1, 2]), np.array([0.5, 0.6, 0.7, 0, 0.1, 0.2])
    )


def test_resultset_and():
    rs_and = RS_A & RS_B
    assert _match_indices_and_scores(rs_and, np.array([3]), np.array([0.3]))
    rs_and_b = RS_B & RS_A
    assert _match_indices_and_scores(rs_and_b, np.array([3]), np.array([0.6]))


def test_resultset_eq():
    assert RS_A == RS_A
    assert RS_A == ResultSet(np.array([0, 1, 2, 3]), np.array([0.0, 0.1, 0.2, 0.3]))
    assert RS_A != ResultSet(np.array([0, 1, 2, 3]), np.array([0.0, 0.1, 0.2, 0.4]))
    assert RS_A != ResultSet(np.array([0, 1, 2, 4]), np.array([0.0, 0.1, 0.2, 0.3]))
    assert RS_A != 1


def test_resultset_sort():
    to_sort = ResultSet(np.array([0, 1, 2]), np.array([5, 1, 2]))
    sort_1 = to_sort.sort()
    assert (sort_1.indices == np.array([0, 2, 1])).all()
    assert (sort_1.scores == np.array([5, 2, 1])).all()
    sort_2 = to_sort.sort(ascending=True)
    assert (sort_2.indices == np.array([1, 2, 0])).all()
    assert (sort_2.scores == np.array([1, 2, 5])).all()


def test_resultset_reverse():
    reversed_ = RS_A.reverse()
    assert (reversed_.indices == np.array([3, 2, 1, 0])).all()
    assert (reversed_.scores == np.array([0.3, 0.2, 0.1, 0])).all()


def test_resultset_iter():
    iterated = list(iter(RS_A))
    assert iterated == [(0, 0.0), (1, 0.1), (2, 0.2), (3, 0.3)]


def test_resultset_max_scale():
    max_scaled = RS_A.max_scale()
    assert _match_indices_and_scores(
        max_scaled, RS_A.indices, np.array([0, 1 / 3, 2 / 3, 1]), close=True
    )


def test_resultset_lt():
    lt = RS_A.lt(0.2)
    assert lt == ResultSet([0, 1], [0, 0.1])


def test_resultset_leq():
    lt = RS_A.leq(0.2)
    assert lt == ResultSet([0, 1, 2], [0, 0.1, 0.2])


def test_resultset_gt():
    lt = RS_A.gt(0.2)
    assert lt == ResultSet([3], [0.3])


def test_resultset_geq():
    lt = RS_A.geq(0.2)
    assert lt == ResultSet([2, 3], [0.2, 0.3])
