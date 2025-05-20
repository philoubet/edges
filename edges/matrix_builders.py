from bw2calc import LCA
from scipy.sparse import lil_matrix, coo_matrix


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

def build_technosphere_edges_matrix(technosphere_matrix, supply_array):
    """
    Generate a matrix with the technosphere flows.
    """

    print(type(technosphere_matrix))
    print(type(supply_array))
    # Convert CSR to COO format for easier manipulation
    coo = technosphere_matrix.tocoo()

    # Extract negative values
    rows, cols, data = coo.row, coo.col, coo.data
    negative_data = -data * (data < 0)  # Keep only negatives and make them positive

    # Scale columns by supply_array
    scaled_data = negative_data * supply_array[cols]

    # Create the flow matrix in sparse format
    flow_matrix_sparse = coo_matrix(
        (scaled_data, (rows, cols)), shape=technosphere_matrix.shape
    )

    return flow_matrix_sparse.tocsr()
