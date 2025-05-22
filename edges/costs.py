"""
Module that implements the base class for country-specific life-cycle
impact assessments, and the AWARE class, which is a subclass of the
LCIA class.
"""

import numpy as np

from .edgelcia import *
from scipy import sparse
from scipy.optimize import linprog
from scipy.sparse import diags, csr_matrix


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
        self.technosphere_matrix_star = None
        self.price_vector = None
        self.logger.info(f"Initialized CostLCIA with method {self.method}")

    def lci(self) -> None:

        self.lca.lci()

        self.technosphere_flow_matrix = build_technosphere_edges_matrix(
            self.lca.technosphere_matrix, self.lca.supply_array, preserve_diagonal=True
        )
        self.technosphere_edges = set(
            list(zip(*self.technosphere_flow_matrix.nonzero()))
        )

        self.technosphere_flows = get_flow_matrix_positions(
            {k: v for k, v in self.lca.activity_dict.items()}
        )

        self.reversed_activity, _, self.reversed_biosphere = self.lca.reverse_dict()

        # Build technosphere flow lookups as in the original implementation.
        self.position_to_technosphere_flows_lookup = {
            i["position"]: {k: i[k] for k in i if k != "position"}
            for i in self.technosphere_flows
        }

    def build_price_vector(self):
        """
        Generate a vector of prices for each (involved) activity in the technosphere matrix.
        """
        print("Build price vector")

        self.price_vector = np.zeros_like(self.lca.supply_array)

        if len(self.cfs_mapping) == 0:
            raise ValueError("No CFs found in the mapping. Cannot build price vector.")

        # Iterate over the CFs mapping to fill the price vector
        for cf in self.cfs_mapping:
            for position in cf.get("positions", []):
                # if we find a pair of positions along the diagonal
                if position[0] == position[1]:
                    # we take the value of the first one
                    self.price_vector[position[0]] = cf["value"]

    def infer_missing_costs(self):
        """
        Infers missing costs in the technosphere matrix.
        Aligned with the work by Moreau et al. (2015), we do not want activities
        with negative value added. Thus, the zero price of the reference product
        and the resulting negative value added of the use activity is considered a
        conceptual error. Furthermore, we want to eliminate cutoff errors by
        inferring missing costs.

        Hence, we modify the set of prices so that:

        1. Each product’s price covers the cost of its inputs,
        propagated through the system. Additionally:
        2. Anchored prices are respected as lower bounds.
        3. The solution yields minimum total cost, consistent with the system structure.
        """
        print("Infer missing costs in the technosphere matrix")

        # --- Part 1: Build normalized technosphere matrix (A*) ---
        T = self.lca.technosphere_matrix.tocsc()
        data = []
        rows = []
        cols = []

        for j in range(T.shape[1]):
            output = T[j, j]
            if output == 0:
                continue
            col = T.getcol(j)
            r = col.nonzero()[0]
            for i in r:
                if i != j:
                    rows.append(i)
                    cols.append(j)
                    data.append(-T[i, j] / output)

        self.technosphere_matrix_star = sparse.csr_matrix(
            (data, (rows, cols)), shape=T.shape
        )

        # --- Part 2: Solve LP to infer missing costs ---
        n = self.technosphere_matrix_star.shape[0]
        A_star = self.technosphere_matrix_star
        original_prices = self.price_vector.copy()

        bounds = []
        A_ub = []
        b_ub = []

        for j in range(n):
            col = self.lca.technosphere_matrix[:, j]
            is_independent = col.nnz == 1 and col[j, 0] != 0
            price = original_prices[j]

            if is_independent:
                bounds.append((price, price))  # Fix price
                continue
            else:
                # Constraint: p_j ≥ sum of A*_ij * p_i
                row = np.zeros(n)
                row[j] = -1
                row += A_star[:, j].toarray().flatten()
                A_ub.append(row)
                b_ub.append(0)

                if price > 0:
                    bounds.append((price, None))
                else:
                    bounds.append((None, None))

        if len(A_ub) == 0:
            raise RuntimeError("No constraints generated; check matrix content.")

        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        c = np.ones(n)  # Minimize total price sum

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

        if result.success:
            self.price_vector = result.x
            print("Inferred missing prices successfully.")
        else:
            raise RuntimeError(
                "Linear program failed to find a consistent price vector."
            )

    def evaluate_cfs(self, scenario_idx: str | int = 0, scenario=None):
        if self.use_distributions and self.iterations > 1:
            coords_i, coords_j, coords_k = [], [], []
            data = []

            for cf in self.cfs_mapping:
                samples = sample_cf_distribution(
                    cf=cf,
                    n=self.iterations,
                    parameters=self.parameters,
                    random_state=self.random_state,
                    use_distributions=self.use_distributions,
                    SAFE_GLOBALS=self.SAFE_GLOBALS,
                )
                for i, j in cf["positions"]:
                    for k in range(self.iterations):
                        coords_i.append(i)
                        coords_j.append(j)
                        coords_k.append(k)
                        data.append(samples[k])

            n_rows, n_cols = self.lca.technosphere_matrix.shape

            self.characterization_matrix = sparse.COO(
                coords=[coords_i, coords_j, coords_k],
                data=data,
                shape=(n_rows, n_cols, self.iterations),
            )

            self.scenario_cfs = [{"positions": [], "value": 0}]  # dummy

        else:
            # Fallback to 2D
            self.scenario_cfs = []
            scenario_name = None

            if scenario is not None:
                scenario_name = scenario
            elif self.scenario is not None:
                scenario_name = self.scenario

            if scenario_name is None:
                if isinstance(self.parameters, dict):
                    if len(self.parameters) > 0:
                        scenario_name = list(self.parameters.keys())[0]

            resolved_params = self.resolve_parameters_for_scenario(
                scenario_idx, scenario_name
            )

            for cf in self.cfs_mapping:
                if isinstance(cf["value"], str):

                    value = safe_eval_cached(
                        cf["value"],
                        parameters=resolved_params,
                        scenario_idx=scenario_idx,
                        SAFE_GLOBALS=self.SAFE_GLOBALS,
                    )
                else:
                    value = cf["value"]

                self.scenario_cfs.append(
                    {
                        "supplier": cf["supplier"],
                        "consumer": cf["consumer"],
                        "positions": cf["positions"],
                        "value": value,
                    }
                )

            matrix_type = "technosphere"

            self.characterization_matrix = initialize_lcia_matrix(
                self.lca, matrix_type=matrix_type
            )

            for cf in self.scenario_cfs:
                for i, j in cf["positions"]:
                    self.characterization_matrix[i, j] = cf["value"]

            self.characterization_matrix = self.characterization_matrix.tocsr()

            self.post_process_characterization_matrix()

    def post_process_characterization_matrix(self):
        """
        Post-process the characterization matrix to ensure that:
        1. The diagonal is overwritten with the price vector.
        2. The off-diagonal technosphere flow values are flipped.
        3. The price vector is added to the characterization matrix
              wherever the technosphere flow matrix is non-zero.
        4. The characterization matrix is converted to CSR format.

        This method is called after the characterization matrix has been
        generated and the price vector has been built.
        It modifies the characterization matrix in place.

        """

        # Step 1: Overwrite diagonal of characterization matrix with price vector
        n = self.characterization_matrix.shape[0]
        new_diag = diags(self.price_vector, offsets=0, shape=(n, n), format="csr")
        self.characterization_matrix.setdiag(0)
        self.characterization_matrix = self.characterization_matrix + new_diag

        # Step 2: Flip signs of off-diagonal technosphere flow values
        A_coo = self.technosphere_flow_matrix.tocoo()
        rows, cols, data = A_coo.row, A_coo.col, A_coo.data.copy()
        data[rows != cols] *= -1
        self.technosphere_flow_matrix = csr_matrix(
            (data, (rows, cols)), shape=A_coo.shape
        )

        # Step 3: Add price_vector[i] (not j!) to characterization_matrix[i, j]
        char_lil = self.characterization_matrix.tolil()
        for i, j in zip(rows, cols):
            char_lil[i, j] = self.price_vector[
                i
            ]  # <-- correct direction: add row price
        self.characterization_matrix = char_lil.tocsr()

        self.characterization_matrix = char_lil.tocsr()
