from bw2calc import LCA
from scipy.sparse import lil_matrix, coo_matrix, csr_matrix
import numpy as np


def initialize_lcia_matrix(lca: LCA, matrix_type="biosphere") -> lil_matrix:
    """
    Initialize the LCIA matrix. It is a sparse matrix with the
    dimensions of the `inventory` matrix of the LCA object.
    :param lca: The LCA object.
    :param matrix_type: The type of the matrix.
    :return: An empty LCIA matrix with the dimensions of the `inventory` matrix.
    """
    if matrix_type == "biosphere":
        return lil_matrix(lca.inventory.shape)
    return lil_matrix(lca.technosphere_matrix.shape)


def build_technosphere_edges_matrix(
    technosphere_matrix: csr_matrix,
    supply_array: np.ndarray,
    preserve_diagonal: bool = False,
) -> csr_matrix:
    """
    Generate a matrix showing scaled technosphere flows needed for the LCA solution.

    Args:
        technosphere_matrix: Sparse CSR matrix of the technosphere.
        supply_array: The solved supply array from LCA.
        preserve_diagonal: If True, includes outputs (diagonal) as positive values.
                           If False, excludes diagonal (outputs).

    Returns:
        csr_matrix: Matrix of scaled technosphere flows.
    """
    coo = technosphere_matrix.tocoo()

    rows = coo.row
    cols = coo.col
    data = coo.data

    is_diag = rows == cols
    is_input = data < 0
    is_output = is_diag & preserve_diagonal
    is_valid = is_input | is_output

    scaled_data = np.zeros_like(data)
    scaled_data[is_input] = -data[is_input] * supply_array[cols[is_input]]
    scaled_data[is_output] = data[is_output] * supply_array[cols[is_output]]

    return coo_matrix(
        (scaled_data[is_valid], (rows[is_valid], cols[is_valid])),
        shape=technosphere_matrix.shape,
    ).tocsr()
