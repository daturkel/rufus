import numpy as np
import pytest
from rufus.vector import ExactNearestNeighborsIndex, NearestNeighborsIndex
from rufus.vector.annoy import AnnoyNearestNeighborsIndex
from rufus.vector.voyager import VoyagerNearestNeighborsIndex
from voyager import StorageDataType

VECTORS = np.array(
    [
        [2, 6, 2, 5, 2, 9, 9, 5, 0, 6],
        [7, 3, 0, 6, 9, 0, 8, 3, 0, 3],
        [8, 8, 0, 0, 2, 2, 9, 3, 7, 6],
        [8, 9, 4, 9, 4, 8, 8, 5, 2, 7],
        [2, 1, 4, 6, 8, 9, 5, 9, 9, 6],
        [9, 7, 2, 3, 5, 9, 4, 3, 2, 5],
        [1, 3, 4, 4, 0, 0, 8, 6, 3, 4],
        [3, 4, 4, 7, 6, 1, 8, 0, 7, 0],
        [7, 4, 8, 4, 8, 1, 8, 2, 1, 9],
        [4, 4, 8, 8, 6, 5, 0, 3, 6, 4],
        [3, 6, 7, 2, 6, 4, 9, 2, 6, 0],
        [1, 7, 2, 8, 9, 1, 3, 1, 6, 2],
        [0, 3, 9, 0, 6, 2, 1, 4, 9, 6],
        [4, 0, 4, 2, 9, 6, 5, 6, 3, 8],
        [5, 8, 6, 3, 4, 3, 9, 1, 2, 5],
        [5, 9, 5, 7, 7, 4, 3, 7, 9, 4],
        [2, 8, 9, 8, 7, 7, 5, 6, 5, 7],
        [5, 9, 9, 3, 9, 4, 2, 8, 4, 7],
        [1, 9, 7, 4, 1, 6, 9, 6, 5, 5],
        [5, 8, 1, 9, 1, 7, 6, 5, 7, 4],
        [3, 9, 5, 4, 0, 2, 9, 9, 8, 3],
        [1, 0, 6, 3, 0, 3, 5, 9, 0, 6],
        [9, 8, 5, 2, 7, 3, 5, 4, 4, 6],
        [7, 1, 5, 3, 7, 7, 1, 3, 7, 4],
        [2, 5, 7, 2, 4, 1, 7, 2, 5, 7],
        [1, 9, 3, 5, 2, 4, 9, 6, 8, 1],
        [1, 9, 3, 1, 3, 1, 7, 0, 5, 2],
        [6, 3, 4, 7, 8, 7, 1, 0, 4, 8],
        [9, 6, 1, 2, 5, 9, 4, 8, 2, 6],
        [3, 6, 7, 7, 1, 8, 7, 6, 9, 6],
        [2, 8, 6, 8, 8, 3, 8, 6, 8, 7],
        [7, 9, 8, 5, 6, 7, 6, 7, 4, 4],
        [9, 8, 7, 7, 6, 7, 0, 3, 5, 8],
        [0, 9, 6, 0, 0, 0, 0, 2, 7, 8],
        [6, 1, 1, 5, 4, 7, 1, 0, 0, 4],
        [4, 4, 0, 4, 9, 5, 8, 8, 4, 0],
        [8, 7, 9, 4, 4, 6, 7, 2, 5, 8],
        [9, 6, 8, 6, 2, 6, 9, 5, 2, 0],
        [1, 7, 9, 6, 5, 0, 7, 1, 0, 2],
        [6, 8, 7, 0, 4, 7, 0, 4, 6, 0],
        [3, 2, 2, 5, 6, 3, 7, 5, 6, 2],
        [4, 5, 2, 5, 8, 5, 2, 4, 0, 0],
        [4, 7, 0, 5, 6, 6, 7, 6, 2, 1],
        [3, 8, 1, 2, 4, 4, 9, 3, 7, 4],
        [6, 3, 2, 5, 9, 2, 8, 6, 0, 0],
        [4, 4, 8, 0, 0, 7, 2, 0, 0, 1],
        [4, 4, 1, 9, 8, 3, 0, 6, 2, 1],
        [0, 2, 0, 0, 4, 8, 8, 8, 3, 1],
        [5, 0, 5, 8, 3, 0, 3, 7, 6, 0],
        [4, 6, 3, 6, 9, 5, 8, 1, 8, 5],
        [9, 7, 0, 0, 4, 8, 9, 1, 4, 7],
        [8, 3, 2, 7, 4, 3, 5, 3, 5, 3],
        [8, 4, 0, 5, 6, 5, 3, 3, 4, 5],
        [7, 7, 1, 4, 9, 3, 4, 2, 2, 3],
        [2, 4, 1, 7, 5, 0, 5, 3, 7, 8],
        [7, 4, 1, 0, 9, 9, 5, 9, 9, 5],
        [4, 0, 1, 5, 2, 1, 6, 4, 9, 0],
        [1, 1, 7, 3, 3, 4, 6, 0, 6, 4],
        [3, 7, 2, 9, 6, 9, 6, 5, 5, 9],
        [0, 2, 2, 5, 7, 1, 6, 9, 5, 5],
        [0, 3, 8, 1, 6, 8, 2, 5, 6, 2],
        [2, 2, 2, 2, 2, 6, 2, 4, 8, 0],
        [9, 5, 4, 2, 4, 0, 4, 6, 7, 4],
        [2, 3, 1, 8, 0, 1, 0, 4, 5, 7],
        [7, 8, 1, 4, 9, 0, 5, 1, 9, 3],
        [8, 5, 6, 4, 9, 2, 5, 7, 3, 8],
        [1, 1, 4, 2, 1, 3, 7, 0, 1, 0],
        [2, 2, 2, 3, 8, 4, 4, 6, 1, 3],
        [0, 0, 7, 5, 4, 8, 9, 6, 0, 4],
        [8, 3, 4, 9, 2, 9, 6, 6, 3, 0],
        [0, 4, 5, 2, 6, 9, 1, 1, 9, 4],
        [9, 4, 5, 5, 1, 0, 5, 5, 2, 5],
        [7, 4, 0, 3, 8, 6, 1, 6, 5, 3],
        [5, 1, 4, 3, 6, 2, 1, 7, 5, 0],
        [9, 3, 9, 0, 9, 3, 3, 0, 5, 8],
        [7, 7, 1, 5, 0, 0, 1, 8, 0, 2],
        [7, 5, 0, 8, 2, 8, 7, 3, 8, 7],
        [6, 1, 9, 1, 6, 7, 7, 5, 5, 5],
        [4, 1, 1, 7, 3, 1, 4, 8, 6, 9],
        [6, 7, 8, 2, 1, 8, 6, 3, 8, 8],
        [5, 0, 4, 2, 6, 2, 8, 5, 8, 4],
        [0, 9, 9, 7, 2, 0, 7, 5, 1, 1],
        [6, 6, 3, 5, 3, 5, 5, 8, 5, 9],
        [5, 1, 4, 5, 3, 9, 9, 4, 4, 3],
        [9, 3, 8, 8, 5, 2, 4, 3, 0, 6],
        [0, 8, 7, 3, 7, 4, 4, 3, 2, 3],
        [6, 8, 2, 5, 7, 3, 2, 0, 5, 2],
        [6, 9, 8, 0, 2, 3, 8, 1, 9, 9],
        [4, 4, 3, 9, 7, 4, 6, 1, 8, 0],
        [0, 0, 1, 2, 1, 9, 1, 1, 5, 1],
        [6, 2, 0, 1, 1, 4, 7, 3, 8, 9],
        [4, 3, 0, 3, 0, 0, 2, 8, 7, 7],
        [1, 1, 2, 0, 8, 4, 3, 2, 3, 5],
        [7, 6, 2, 6, 6, 2, 4, 3, 7, 9],
        [1, 8, 7, 2, 1, 7, 8, 5, 0, 2],
        [6, 3, 3, 4, 7, 7, 6, 7, 6, 9],
        [2, 6, 7, 9, 8, 4, 1, 5, 1, 7],
        [5, 5, 7, 2, 9, 8, 6, 5, 4, 9],
        [0, 8, 4, 1, 6, 2, 7, 6, 0, 8],
        [4, 9, 2, 5, 8, 6, 7, 0, 1, 3],
        [6, 2, 6, 6, 2, 4, 1, 2, 2, 8],
        [2, 2, 1, 0, 9, 1, 9, 2, 7, 6],
        [1, 7, 6, 1, 7, 9, 2, 7, 6, 2],
        [7, 4, 8, 9, 0, 6, 7, 7, 5, 5],
        [4, 5, 6, 7, 3, 6, 4, 8, 1, 5],
    ]
)


def test_nearest_neighbors_unknown_metric():
    with pytest.raises(ValueError):
        _ = NearestNeighborsIndex(VECTORS, "foo", ["bar", "baz"])


def test_nearest_neighbors_nearest_not_implemented(mocker):
    mocker.patch.object(NearestNeighborsIndex, "_index")
    nn = NearestNeighborsIndex(VECTORS, "bar", ["bar", "baz"])
    with pytest.raises(NotImplementedError):
        nn.get_nearest_neighbors(VECTORS[0])


def test_nearest_neighbors_from_existing_not_implemented(mocker):
    mocker.patch.object(NearestNeighborsIndex, "_index")
    nn = NearestNeighborsIndex(VECTORS, "bar", ["bar", "baz"])
    with pytest.raises(NotImplementedError):
        nn.get_nearest_neighbors_from_existing(1)


def test_nearest_neighbors_index_not_implemented():
    with pytest.raises(NotImplementedError):
        NearestNeighborsIndex(VECTORS, "bar", ["bar", "baz"])


def test_nearest_neighbors_add_to_index_not_implemented(mocker):
    mocker.patch.object(NearestNeighborsIndex, "_index")
    nn = NearestNeighborsIndex(VECTORS, "bar", ["bar", "baz"])
    with pytest.raises(NotImplementedError):
        nn.add_to_index(VECTORS)


def test_exact_nearest_neighbors_dot():
    enn = ExactNearestNeighborsIndex(VECTORS, metric="dot")
    top_2 = enn.get_nearest_neighbors(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), top_k=2)
    assert np.all(top_2.indices == np.array([4, 30]))
    assert np.all(top_2.scores == np.array([323, 308]))


def test_exact_nearest_neighbors_cosine():
    enn = ExactNearestNeighborsIndex(VECTORS, metric="cosine")
    top_2 = enn.get_nearest_neighbors(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), top_k=2)
    assert np.all(top_2.indices == np.array([4, 95]))
    assert np.allclose(top_2.scores, np.array([0.92808046, 0.9176836]))


def test_exact_nearest_neighbors_euclidean():
    enn = ExactNearestNeighborsIndex(VECTORS, metric="euclidean")
    top_2 = enn.get_nearest_neighbors(np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2]), top_k=2)
    assert np.all(top_2.indices == np.array([66, 92]))
    assert np.allclose(top_2.scores, np.array([np.sqrt(42), np.sqrt(57)]))


def test_exact_nearest_neighbors_add_to_index():
    enn = ExactNearestNeighborsIndex(VECTORS[:100], metric="dot")
    enn.add_to_index(VECTORS[100:])
    assert np.all(enn.index == VECTORS)


def test_exact_nearest_neighbors_from_existing():
    enn = ExactNearestNeighborsIndex(VECTORS, metric="euclidean")
    top_2_existing = enn.get_nearest_neighbors_from_existing(1, top_k=2)
    top_2 = enn.get_nearest_neighbors(VECTORS[1], top_k=2)
    assert top_2_existing == top_2


def test_annoy_nearest_neighbors_dot():
    ann = AnnoyNearestNeighborsIndex(VECTORS, metric="dot", n_trees=100)
    top_2 = ann.get_nearest_neighbors(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), top_k=2)
    assert np.all(top_2.indices == np.array([4, 30]))
    assert np.all(top_2.scores == np.array([323, 308]))


def test_annoy_nearest_neighbors_cosine():
    ann = AnnoyNearestNeighborsIndex(VECTORS, metric="cosine", n_trees=100)
    top_2 = ann.get_nearest_neighbors(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), top_k=2)
    assert np.all(top_2.indices == np.array([4, 95]))
    assert np.allclose(
        top_2.scores,
        np.array([0.92808046, 0.9176836]),
    )


def test_annoy_nearest_neighbors_euclidean():
    ann = AnnoyNearestNeighborsIndex(VECTORS, metric="euclidean", n_trees=100)
    top_2 = ann.get_nearest_neighbors(np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2]), top_k=2)
    assert np.all(top_2.indices == np.array([66, 92]))
    assert np.allclose(top_2.scores, np.array([np.sqrt(42), np.sqrt(57)]))


def test_annoy_nearest_neighbors_hamming():
    VECTORS_B = (VECTORS == 2).astype(int)
    v = (np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) == 2).astype(int)
    ann = AnnoyNearestNeighborsIndex(VECTORS_B, metric="hamming", n_trees=100)
    top_2 = ann.get_nearest_neighbors(v, top_k=2)
    assert set(top_2.indices) == set(np.array([51, 58]))
    assert np.allclose(top_2.scores, np.array([0, 0]))


def test_annoy_nearest_neighbors_manhattan():
    ann = AnnoyNearestNeighborsIndex(VECTORS, metric="manhattan", n_trees=100)
    top_2 = ann.get_nearest_neighbors(
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), top_k=2
    )
    assert np.all(top_2.indices == np.array([95, 4]))
    assert np.allclose(top_2.scores, np.array([15, 18]))


def test_annoy_nearest_neighbors_from_existing():
    ann = AnnoyNearestNeighborsIndex(VECTORS, metric="euclidean", n_trees=100)
    top_2_existing = ann.get_nearest_neighbors_from_existing(1, top_k=2)
    top_2 = ann.get_nearest_neighbors(VECTORS[1], top_k=2)
    assert top_2 == top_2_existing


def test_annoy_nearest_neighbors_no_top_k():
    ann = AnnoyNearestNeighborsIndex(VECTORS, metric="euclidean", n_trees=100)
    top_k = ann.get_nearest_neighbors(VECTORS[0], top_k=None)
    assert len(top_k) == VECTORS.shape[0]


def test_annoy_nearest_neighbors_from_existing_no_top_k():
    ann = AnnoyNearestNeighborsIndex(VECTORS, metric="euclidean", n_trees=100)
    top_k = ann.get_nearest_neighbors_from_existing(0, top_k=None)
    assert len(top_k) == VECTORS.shape[0]


def test_voyager_nearest_neighbors_dot():
    ann = VoyagerNearestNeighborsIndex(
        VECTORS,
        metric="dot",
        M=12,
        ef_construction=200,
        random_seed=1,
        deterministic=True,  # needed for testing
        storage_data_type=StorageDataType.Float32,
    )
    top_2 = ann.get_nearest_neighbors(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), top_k=2)
    assert np.all(top_2.indices == np.array([4, 30]))
    assert np.all(top_2.scores == np.array([323, 308]))


def test_voyager_nearest_neighbors_cosine():
    ann = VoyagerNearestNeighborsIndex(
        VECTORS,
        metric="cosine",
        M=12,
        ef_construction=200,
        random_seed=1,
        storage_data_type=StorageDataType.Float32,
    )
    top_2 = ann.get_nearest_neighbors(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), top_k=2)
    assert np.all(top_2.indices == np.array([4, 95]))
    assert np.allclose(
        top_2.scores,
        np.array([0.92808046, 0.9176836]),
    )


def test_voyager_nearest_neighbors_euclidean():
    ann = VoyagerNearestNeighborsIndex(
        VECTORS,
        metric="euclidean",
        M=12,
        ef_construction=200,
        random_seed=1,
        storage_data_type=StorageDataType.Float32,
    )
    top_2 = ann.get_nearest_neighbors(np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2]), top_k=2)
    assert np.all(top_2.indices == np.array([66, 92]))
    assert np.allclose(top_2.scores, np.array([np.sqrt(42), np.sqrt(57)]))


def test_voyager_nearest_neighbors_from_existing():
    ann = VoyagerNearestNeighborsIndex(
        VECTORS,
        metric="dot",
        M=12,
        ef_construction=200,
        random_seed=1,
        storage_data_type=StorageDataType.Float32,
    )
    top_2_existing = ann.get_nearest_neighbors_from_existing(1, top_k=2)
    top_2 = ann.get_nearest_neighbors(VECTORS[1], top_k=2)
    assert top_2 == top_2_existing


def test_voyager_nearest_neighbors_no_top_k():
    ann = VoyagerNearestNeighborsIndex(
        VECTORS,
        metric="dot",
        M=12,
        ef_construction=200,
        random_seed=1,
        storage_data_type=StorageDataType.Float32,
    )
    top_k = ann.get_nearest_neighbors(VECTORS[0], top_k=None)
    assert len(top_k) == VECTORS.shape[0]


def test_voyager_nearest_neighbors_from_existing_no_top_k():
    ann = VoyagerNearestNeighborsIndex(
        VECTORS,
        metric="dot",
        M=12,
        ef_construction=200,
        random_seed=1,
        storage_data_type=StorageDataType.Float32,
    )
    top_k = ann.get_nearest_neighbors_from_existing(0, top_k=None)
    assert len(top_k) == VECTORS.shape[0]
