"""
Module that implements the base class for country-specific life-cycle
impact assessments, and the AWARE class, which is a subclass of the
LCIA class.
"""

from .edgelcia import *


class CostLCIA(EdgeLCIA):
    """
    Class that implements the calculation of the regionalized life cycle impact assessment (LCIA) results.
    Relies on bw2data.LCA class for inventory calculations and matrices.
    """

    def __init__(
        self,
        demand: dict,
        method: tuple,
        filepath: Optional[str] = None,
        parameters: Optional[dict] = None,
        weight: Optional[str] = "population",
        use_distributions: Optional[bool] = False,
        random_seed: Optional[int] = None,
        iterations: Optional[int] = 100,
    ):
        super().__init__(
            demand=demand,
            method=method,
            filepath=filepath,
            parameters=parameters,
            weight=weight,
            use_distributions=use_distributions,
            random_seed=random_seed,
            iterations=iterations,
        )
        self.logger.info(f"Initialized CostLCIA with method {self.method}")

    def build_technosphere_edges_matrix(self):
        """
        Generate a matrix of actual exchanges between activities for the solved system.
        - Diagonal values: activity output (always positive).
        - Off-diagonal: inputs (always negative).
        - Waste treatment activities (negative diagonal in solved system):
            - Their diagonal is flipped to positive.
            - Their own inputs are flipped to be positive (reverse of normal logic).
            - Their use as inputs to other activities remains negative.
        """
        A = self.lca.technosphere_matrix.tocoo()
        supply = self.lca.supply_array

        rows, cols, data = A.row, A.col, A.data
        scaled_data = data * supply[cols]

        # First pass: build base matrix with diagonal untouched, inputs as -abs
        adjusted_data = []
        for i, j, val in zip(rows, cols, scaled_data):
            if i == j:
                # Diagonal values (to fix later)
                adjusted_data.append(val)
            else:
                adjusted_data.append(-abs(val))

        # Build temporary matrix to check diagonals
        temp_matrix = coo_matrix((adjusted_data, (rows, cols)), shape=A.shape).tocsc()
        diag = temp_matrix.diagonal()

        # Identify waste treatment activities by negative production
        waste_cols = set(np.where(diag < 0)[0])

        # Second pass: flip signs inside waste activities (column j = waste)
        corrected_data = []
        for i, j, val in zip(rows, cols, adjusted_data):
            if j in waste_cols:
                if i == j:
                    # Flip negative diagonal to positive
                    corrected_data.append(abs(val))
                elif i != j:
                    # Flip internal inputs of waste treatment to positive
                    corrected_data.append(abs(val))
            else:
                corrected_data.append(val)

        final_matrix = coo_matrix((corrected_data, (rows, cols)), shape=A.shape)
        return final_matrix.tocsr()
