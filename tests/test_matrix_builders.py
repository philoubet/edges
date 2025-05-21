# test_matrix_builders.py

import pytest
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from unittest.mock import MagicMock

from edges.matrix_builders import (
    initialize_lcia_matrix,
    build_technosphere_edges_matrix,
)


def test_initialize_lcia_matrix_inventory():
    mock_lca = MagicMock()
    mock_lca.inventory = csr_matrix((3, 3))
    result = initialize_lcia_matrix(mock_lca, matrix_type="biosphere")
    assert isinstance(result, lil_matrix)
    assert result.shape == (3, 3)


def test_initialize_lcia_matrix_technosphere():
    mock_lca = MagicMock()
    mock_lca.technosphere_matrix = csr_matrix((4, 2))
    result = initialize_lcia_matrix(mock_lca, matrix_type="technosphere")
    assert isinstance(result, lil_matrix)
    assert result.shape == (4, 2)


def test_build_technosphere_edges_matrix():
    data = np.array([1.0, -2.0, 3.0, -4.0])
    row = np.array([0, 1, 2, 3])
    col = np.array([0, 1, 1, 0])
    matrix = csr_matrix((data, (row, col)), shape=(4, 2))
    supply_array = np.array([10.0, 5.0])  # columns 0 and 1

    result = build_technosphere_edges_matrix(matrix, supply_array)

    assert isinstance(result, csr_matrix)
    assert result.shape == (4, 2)

    expected = np.array(
        [
            [0.0, 0.0],
            [0.0, 10.0],  # -2.0 * 5.0 = 10
            [0.0, 0.0],
            [40.0, 0.0],  # -4.0 * 10.0 = 40
        ]
    )
    np.testing.assert_array_almost_equal(result.toarray(), expected)
