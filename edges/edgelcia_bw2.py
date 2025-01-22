"""
Module that implements the base class for country-specific life-cycle
impact assessments, and the AWARE class, which is a subclass of the
LCIA class.
"""

from collections import defaultdict
import logging
import json
from typing import Iterable, Optional
import numpy as np

from scipy.sparse import coo_matrix
from constructive_geometries import Geomatcher
import pandas as pd
from prettytable import PrettyTable
from bw2calc import LCA
import bw2data
from bw2calc.utils import clean_databases, wrap_functional_unit, get_filepaths
import warnings

try:
    from pypardiso import factorized, spsolve
except ImportError:
    from scipy.sparse.linalg import factorized, spsolve

    from scipy.sparse.linalg._dsolve import linsolve

    if not linsolve.useUmfpack:
        logging.warn("""
        Did not findPypardisio or Umfpack. Matrix computation may be very slow.

        If you are on an Intel architecture, please install pypardiso as explained in the docs :
        https://docs.brightway.dev/en/latest/content/installation/index.html

        > pip install pypardiso
        or 
        > conda install pypardiso
        """)

from .utils import (
    format_data,
    initialize_lcia_matrix,
    get_flow_matrix_positions,
    preprocess_cfs,
    check_database_references,
)
from .filesystem_constants import DATA_DIR

try:
    import pandas
except ImportError:
    pandas = None
try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping
try:
    from presamples import PackagesDataLoader
except ImportError:
    PackagesDataLoader = None

# delete the logs
with open("edgelcia.log", "w", encoding="utf-8"):
    pass

# initiate the logger
logging.basicConfig(
    filename="edgelcia.log",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


geo = Geomatcher()


def compute_average_cf(
    constituents: list,
    supplier_info: dict,
    weight: dict,
    cfs_lookup: dict,
    region: str,
):
    """
    Compute the average characterization factors for the region.
    :param constituents: List of constituent regions.
    :param supplier_info: Information about the supplier.
    :param weight: Weights for the constituents.
    :param cfs_lookup: Lookup dictionary for characterization factors.
    :param region: The region being evaluated.
    :return: The weighted average CF value.
    """

    # Filter constituents with valid weights
    valid_constituents = [(c, weight[c]) for c in constituents if c in weight]
    if not valid_constituents:
        return 0

    # Separate constituents and weights
    constituents, weight_array = zip(*valid_constituents)
    weight_array = np.array(weight_array)

    # Normalize weights
    weight_sum = weight_array.sum()
    if weight_sum == 0:
        return 0
    shares = weight_array / weight_sum

    # Pre-filter supplier keys for filtering
    supplier_keys = supplier_info.keys()

    def match_supplier(cf: dict) -> bool:
        """
        Match a supplier based on operator logic.
        """
        for key in supplier_keys:
            supplier_value = supplier_info.get(key)
            cf_value = cf["supplier"].get(key)
            operator = cf["supplier"].get("operator", "equals")

            if not match_operator(cf_value, supplier_value, operator):
                return False
        return True

    # Compute the weighted average CF value
    value = 0
    for loc, share in zip(constituents, shares):
        loc_cfs = cfs_lookup.get(loc, [])

        # Filter CFs based on supplier info using the operator logic
        filtered_cfs = [cf["value"] for cf in loc_cfs if match_supplier(cf)]

        value += share * sum(filtered_cfs)

    # Log if shares don't sum to 1 due to precision issues
    if not np.isclose(shares.sum(), 1):
        logger.info(
            f"Shares for {region} do not sum to 1 " f"but {shares.sum()}: {shares}"
        )

    return value


def get_str(x):
    return x if isinstance(x, str) else x[-1]


def find_region_constituents(
    region: str, supplier_info: dict, cfs: dict, weight: dict
) -> float:
    """
    Find the constituents of the region.
    :param region: The region to evaluate.
    :param supplier_info: Information about the supplier.
    :param cfs: Lookup dictionary for characterization factors.
    :param weight: Weights for the constituents.
    :return: The new CF value.
    """

    try:
        constituents = [
            get_str(g) for g in geo.contained(region) if get_str(g) in weight
        ]
    except KeyError:
        logger.info(f"Region: {region}. No geometry found.")
        return 0

    new_cfs = compute_average_cf(constituents, supplier_info, weight, cfs, region)

    if len(constituents) == 0:
        logger.info(f"Region: {region}. No constituents found.")
        constituents = list(weight.keys())
        new_cfs = compute_average_cf(constituents, supplier_info, weight, cfs, region)

    logger.info(f"Region: {region}. New CF: {new_cfs}. Constituents: {constituents}")

    return new_cfs


def preprocess_flows(flows_list: list, mandatory_fields: set) -> dict:
    """
    Preprocess flows into a lookup dictionary.
    :param flows_list: List of flows.
    :param mandatory_fields: Fields that must be included in the lookup key.
    :return: A dictionary for flow lookups.
    """
    lookup = {}
    for flow in flows_list:
        key = tuple(
            (k, v) for k, v in flow.items() if k in mandatory_fields and v is not None
        )
        lookup.setdefault(key, []).append(flow["position"])
    return lookup


def match_operator(value: str, target: str, operator: str) -> bool:
    """
    Match a value against a target using the specified operator.
    :param value: The value to match.
    :param target: The target value to compare against.
    :param operator: The matching operator (equals, contains, startswith).
    :return: True if the value matches the target based on the operator.
    """
    if operator == "equals":
        return value == target
    if operator == "contains":
        return value in target
    if operator == "startswith":
        return target.startswith(value)
    raise ValueError(f"Unsupported operator: {operator}")



class EdgeLCIA(LCA):
    """
    Subclass of the `bw2io.lca.LCA` class that implements the calculation
    of the life cycle impact assessment (LCIA) results.
    """

    def __init__(
            self,
            demand,
            method=None,
            weighting=None,
            normalization=None,
            database_filepath=None,
            presamples=None,
            seed=None,
            override_presamples_seed=False,
            lcia_weight = "population",
    ):
        """Create a new LCA calculation.

        Args:
            * *demand* (dict): The demand or functional unit. Needs to be a dictionary to indicate amounts, e.g. ``{("my database", "my process"): 2.5}``.
            * *method* (tuple, optional): LCIA Method tuple, e.g. ``("My", "great", "LCIA", "method")``. Can be omitted if only interested in calculating the life cycle inventory.

        Returns:
            A new LCA object

        """
        if not isinstance(demand, Mapping):
            raise ValueError("Demand must be a dictionary")
        for key in demand:
            if not key:
                raise ValueError("Invalid demand dictionary")


        clean_databases()
        self._fixed = False

        self.demand = demand
        self.method = method
        self.normalization = normalization
        self.weighting = weighting
        self.database_filepath = database_filepath
        self.seed = seed
        self.position_to_technosphere_flows_lookup = None
        self.technosphere_flows_lookup = None
        self.technosphere_edges = None
        self.technosphere_flow_matrix = None
        self.biosphere_edges = None
        self.technosphere_flows = None
        self.biosphere_flows = None
        self.characterized_inventory = None
        self.biosphere_characterization_matrix = None
        self.ignored_flows = set()
        self.ignored_locations = set()
        self.cfs_data = None
        self.weight = lcia_weight

        if presamples and PackagesDataLoader is None:
            warnings.warn("Skipping presamples; `presamples` not installed")
            self.presamples = None
        elif presamples:
            # Iterating over a `Campaign` object will also return the presample filepaths
            self.presamples = PackagesDataLoader(
                dirpaths=presamples,
                seed=self.seed if override_presamples_seed else None,
                lca=self
            )
        else:
            self.presamples = None

        self.database_filepath, _, _, _ = self.get_array_filepaths()

    def get_array_filepaths(self):
        """Use utility functions to get all array filepaths"""
        return (
            get_filepaths(self.demand, "demand"),
            None,
            None,
            None,
        )

    def lci(self, demand: Optional[dict] = None, factorize: bool = False) -> None:

        if not hasattr(self, "technosphere_matrix"):
            self.load_lci_data()
        if demand is not None:
            self.check_demand(demand)
            self.build_demand_array(demand)
            self.demand = demand
        else:
            self.build_demand_array()
        if factorize and not PYPARDISO:
            self.decompose_technosphere()
        self.lci_calculation()

        self.technosphere_flow_matrix = self.build_technosphere_edges_matrix()
        self.technosphere_edges = set(
            list(zip(*self.technosphere_flow_matrix.nonzero()))
        )
        self.biosphere_edges = set(list(zip(*self.inventory.nonzero())))

        unique_biosphere_flows = set(x[0] for x in self.biosphere_edges)

        self.biosphere_flows = get_flow_matrix_positions(
            {
                k: v
                for k, v in self.biosphere_dict.items()
                if v in unique_biosphere_flows
            }
        )

        self.technosphere_flows = get_flow_matrix_positions(
            {k: v for k, v in self.activity_dict.items()}
        )

        self.reversed_activity, _, self.reversed_biosphere = self.reverse_dict()

    def build_technosphere_edges_matrix(self):
        """
        Generate a matrix with the technosphere flows.
        """

        # Convert CSR to COO format for easier manipulation
        coo = self.technosphere_matrix.tocoo()

        # Extract negative values
        rows, cols, data = coo.row, coo.col, coo.data
        negative_data = -data * (data < 0)  # Keep only negatives and make them positive

        # Scale columns by supply_array
        scaled_data = negative_data * self.supply_array[cols]

        # Create the flow matrix in sparse format
        flow_matrix_sparse = coo_matrix(
            (scaled_data, (rows, cols)), shape=self.technosphere_matrix.shape
        )

        return flow_matrix_sparse.tocsr()

    def load_lcia_data(self, data_objs=None):
        """
        Load the data for the LCIA method.
        """
        data_file = DATA_DIR / f"{'_'.join(self.method)}.json"

        if not data_file.is_file():
            raise FileNotFoundError(f"Data file not found: {data_file}")

        with open(data_file, "r", encoding="utf-8") as f:
            self.cfs_data = format_data(json.load(f))
            self.cfs_data = check_database_references(
                self.cfs_data, self.technosphere_flows, self.biosphere_flows
            )

    def identify_exchanges(self):
        """
        Based on search criteria under `supplier` and `consumer` keys,
        identify the exchanges in the inventory matrix.
        """

        def match_with_operator(
            flow_to_match: dict, lookup: dict, required_fields: set
        ) -> list:
            """
            Match a flow against a lookup dictionary considering the operator.
            :param flow_to_match: The flow to match.
            :param lookup: The lookup dictionary.
            :param required_fields: The required fields for matching.
            :return: A list of matching positions.
            """
            matches = []
            for key, positions in lookup.items():
                if all(
                    match_operator(
                        value=flow_to_match.get(k),
                        target=v,
                        operator=flow_to_match.get("operator", "equals"),
                    )
                    for (k, v) in key
                    if k in required_fields
                ):
                    matches.extend(positions)
            return matches

        def preprocess_lookups():
            """
            Preprocess supplier and consumer flows into lookup dictionaries.
            """
            supplier_lookup = preprocess_flows(
                (
                    self.biosphere_flows
                    if all(
                        cf["supplier"].get("matrix") == "biosphere"
                        for cf in self.cfs_data
                    )
                    else self.technosphere_flows
                ),
                required_supplier_fields,
            )

            consumer_lookup = preprocess_flows(
                self.technosphere_flows, required_consumer_fields
            )

            reversed_supplier_lookup = {
                pos: key
                for key, positions in supplier_lookup.items()
                for pos in positions
            }
            reversed_consumer_lookup = {
                pos: key
                for key, positions in consumer_lookup.items()
                for pos in positions
            }

            return (
                supplier_lookup,
                consumer_lookup,
                reversed_supplier_lookup,
                reversed_consumer_lookup,
            )

        def handle_static_regions(
            direction: str,
            unprocessed_edges: list,
            cfs_lookup: dict,
            unprocessed_locations_cache: dict,
        ) -> None:
            """
            Handle static regions and update CF data (e.g., RER, GLO, ENTSOE, etc.).
            CFs are obtained by averaging the CFs of the constituents of the region.

            :param direction: The direction of the flow.
            :param unprocessed_edges: The unprocessed edges.
            :param cfs_lookup: The lookup dictionary for CFs.
            :param unprocessed_locations_cache: The cache for unprocessed locations
            :return: None
            """
            for supplier_idx, consumer_idx in unprocessed_edges:
                supplier_info = dict(reversed_supplier_lookup[supplier_idx])
                consumer_info = dict(reversed_consumer_lookup[consumer_idx])
                location = consumer_info.get("location")

                if location not in ("RoW", "RoE"):
                    # Resolve from cache or compute new CF
                    new_cf = unprocessed_locations_cache.get(location, {}).get(
                        supplier_idx
                    ) or find_region_constituents(
                        region=location,
                        supplier_info=supplier_info,
                        cfs=cfs_lookup,
                        weight=weight,
                    )
                    if new_cf != 0:
                        unprocessed_locations_cache[location][supplier_idx] = new_cf
                        self.cfs_data.append(
                            {
                                "supplier": supplier_info,
                                "consumer": consumer_info,
                                direction: [(supplier_idx, consumer_idx)],
                                "value": new_cf,
                            }
                        )

        def handle_dynamic_regions(
            direction: str, unprocessed_edges: list, cfs_lookup: dict
        ) -> None:
            """
            Handle dynamic regions like RoW and RoE and update CF data.

            :param direction: The direction of the flow.
            :param unprocessed_edges: The unprocessed edges.
            :param cfs_lookup: The lookup dictionary for CFs.
            :return: None
            """
            for supplier_idx, consumer_idx in unprocessed_edges:
                supplier_info = dict(reversed_supplier_lookup[supplier_idx])
                consumer_info = dict(reversed_consumer_lookup[consumer_idx])
                location = consumer_info.get("location")

                if location in ("RoW", "RoE"):
                    consumer_act = self.position_to_technosphere_flows_lookup[
                        consumer_idx
                    ]
                    name, reference_product = consumer_act["name"], consumer_act.get(
                        "reference product"
                    )

                    # Use preprocessed lookup
                    other_than_RoW_RoE = [
                        loc
                        for loc in self.technosphere_flows_lookup.get(
                            (name, reference_product), []
                        )
                        if loc not in ["RoW", "RoE"] and loc in weight
                    ]

                    # constituents are all the candidates in the World (or in Europe)
                    # minus those in other_than_RoW_RoE

                    if location == "RoW":
                        constituents = list(
                            set(list(weight.keys())) - set(other_than_RoW_RoE)
                        )
                    else:
                        # RoE
                        # redefine other_than_RoW_RoE to limit to EU candidates
                        other_than_RoW_RoE = [
                            loc for loc in other_than_RoW_RoE if geo.contained("RER")
                        ]
                        constituents = list(
                            set(geo.contained("RER")) - set(other_than_RoW_RoE)
                        )

                    extra_constituents = []
                    for constituent in constituents:
                        if constituent not in weight:
                            extras = [
                                get_str(e)
                                for e in geo_cache.get(
                                    constituent, geo.contained(constituent)
                                )
                                if get_str(e) in weight and e != constituent
                            ]
                            extra_constituents.extend(extras)
                            geo_cache[constituent] = extras

                    constituents.extend(extra_constituents)

                    new_cf = compute_average_cf(
                        constituents=constituents,
                        supplier_info=supplier_info,
                        weight=weight,
                        cfs_lookup=cfs_lookup,
                        region=location,
                    )

                    # logger.info(
                    #     f"Region: {location}. Activity: {name, reference_product} "
                    #     f"New CF: {new_cf}. "
                    #     f"Candidates other than Row/RoE: {other_than_RoW_RoE} "
                    #     f"Constituents: {constituents}"
                    # )

                    if new_cf:
                        self.cfs_data.append(
                            {
                                "supplier": supplier_info,
                                "consumer": consumer_info,
                                direction: [(supplier_idx, consumer_idx)],
                                "value": new_cf,
                            }
                        )
                    else:
                        self.ignored_locations.add(location)

        # Constants for ignored fields
        IGNORED_FIELDS = {"matrix", "population", "gdp", "operator"}

        # Precompute required fields for faster access
        required_supplier_fields = {
            k
            for cf in self.cfs_data
            for k in cf["supplier"].keys()
            if k not in IGNORED_FIELDS
        }
        required_consumer_fields = {
            k
            for cf in self.cfs_data
            for k in cf["consumer"].keys()
            if k not in IGNORED_FIELDS
        }

        # Preprocess flows and lookups
        (
            supplier_lookup,
            consumer_lookup,
            reversed_supplier_lookup,
            reversed_consumer_lookup,
        ) = preprocess_lookups()

        edges = (
            self.biosphere_edges
            if all(cf["supplier"].get("matrix") == "biosphere" for cf in self.cfs_data)
            else self.technosphere_edges
        )

        for cf in self.cfs_data:
            # Generate supplier candidates
            supplier_candidates = match_with_operator(
                cf["supplier"], supplier_lookup, required_supplier_fields
            )

            # Generate consumer candidates
            consumer_candidates = match_with_operator(
                cf["consumer"], consumer_lookup, required_consumer_fields
            )

            # Create pairs of supplier and consumer candidates
            cf[f"{cf['supplier']['matrix']}-{cf['consumer']['matrix']}"] = [
                (supplier, consumer)
                for supplier in supplier_candidates
                for consumer in consumer_candidates
                if (supplier, consumer) in edges
            ]

        # Preprocess `self.technosphere_flows` once
        if not self.technosphere_flows_lookup:
            self.technosphere_flows_lookup = defaultdict(list)
            for flow in self.technosphere_flows:
                key = (flow["name"], flow.get("reference product"))
                self.technosphere_flows_lookup[key].append(flow["location"])
            self.position_to_technosphere_flows_lookup = {
                i["position"]: {k: i[k] for k in i if k != "position"}
                for i in self.technosphere_flows
            }

        cfs_lookup = preprocess_cfs(self.cfs_data)

        # Process edges
        supplier_bio_tech_edges = {
            f[0] for cf in self.cfs_data for f in cf.get("biosphere-technosphere", [])
        }
        consumer_bio_tech_edges = {
            f[1] for cf in self.cfs_data for f in cf.get("biosphere-technosphere", [])
        }
        supplier_tech_tech_edges = {
            f[0]
            for cf in self.cfs_data
            for f in cf.get("technosphere-technosphere", [])
        }
        consumer_tech_tech_edges = {
            f[1]
            for cf in self.cfs_data
            for f in cf.get("technosphere-technosphere", [])
        }

        unprocessed_biosphere_edges = [
            edge
            for edge in self.biosphere_edges
            if edge[0] in supplier_bio_tech_edges
            and edge[1] not in consumer_bio_tech_edges
        ]
        unprocessed_technosphere_edges = [
            edge
            for edge in self.technosphere_edges
            if edge[0] in supplier_tech_tech_edges
            and edge[1] not in consumer_tech_tech_edges
        ]

        weight = {
            i.get("consumer").get("location"): i.get("consumer").get(self.weight)
            for i in self.cfs_data
            if i.get("consumer").get("location")
        }
        geo_cache = {}

        handle_static_regions(
            "biosphere-technosphere",
            unprocessed_biosphere_edges,
            cfs_lookup,
            defaultdict(dict),
        )
        handle_static_regions(
            "technosphere-technosphere",
            unprocessed_technosphere_edges,
            cfs_lookup,
            defaultdict(dict),
        )

        handle_dynamic_regions(
            "biosphere-technosphere", unprocessed_biosphere_edges, cfs_lookup
        )
        handle_dynamic_regions(
            "technosphere-technosphere", unprocessed_technosphere_edges, cfs_lookup
        )

        self.cfs_data = [
            cf
            for cf in self.cfs_data
            if any(
                [cf.get("biosphere-technosphere"), cf.get("technosphere-technosphere")]
            )
        ]

        # figure out remaining unprocessed edges for information
        processed_biosphere_edges = {
            f for cf in self.cfs_data for f in cf.get("biosphere-technosphere", [])
        }
        processed_technosphere_edges = {
            f for cf in self.cfs_data for f in cf.get("technosphere-technosphere", [])
        }

        unprocessed_biosphere_edges = (
            set(unprocessed_biosphere_edges) - processed_biosphere_edges
        )
        unprocessed_technosphere_edges = (
            set(unprocessed_technosphere_edges) - processed_technosphere_edges
        )

        if unprocessed_biosphere_edges:
            # print a pretty table and list the flows which have not been characterized
            print(
                f"{len(unprocessed_biosphere_edges)} supplying biosphere flows have not been characterized:"
            )
            table = PrettyTable()
            table.field_names = ["Name", "Categories", "Amount"]
            for i, j in unprocessed_biosphere_edges:
                flow = bw2data.get_activity(self.reversed_biosphere[i])
                table.add_row(
                    [flow["name"], flow.get("categories"), self.inventory[i, j]]
                )
            print(table)

        if unprocessed_technosphere_edges:
            # print a pretty table and list the flows which have not been characterized
            print(
                f"{len(unprocessed_technosphere_edges)} supplying technosphere flows have not been characterized:"
            )
            table = PrettyTable()
            table.field_names = ["Name", "Reference product", "Location", "Amount"]
            for i, j in unprocessed_technosphere_edges:
                flow = bw2data.get_activity(self.reversed_activity[i])
                table.add_row(
                    [
                        flow["name"],
                        flow.get("reference product"),
                        flow.get("location"),
                        self.technosphere_flow_matrix[i, j],
                    ]
                )
            print(table)

    def fill_in_lcia_matrix(self) -> None:
        """
        Translate the data to indices in the inventory matrix.
        """

        if all(cf["supplier"].get("matrix") == "biosphere" for cf in self.cfs_data):
            self.characterization_matrix = initialize_lcia_matrix(self)
        else:
            self.characterization_matrix = initialize_lcia_matrix(
                self, matrix_type="technosphere"
            )

        self.identify_exchanges()

        for cf in self.cfs_data:
            for supplier, consumer in cf.get("biosphere-technosphere", []):
                self.characterization_matrix[supplier, consumer] = cf["value"]

            for supplier, consumer in cf.get("technosphere-technosphere", []):
                self.characterization_matrix[supplier, consumer] = cf["value"]

        self.characterization_matrix = self.characterization_matrix.tocsr()

    def lcia_calculation(self) -> None:
        """
        Calculate the LCIA score.
        """
        self.load_lcia_data()
        self.fill_in_lcia_matrix()

        try:
            self.characterized_inventory = self.characterization_matrix.multiply(
                self.inventory
            )
        except ValueError:
            self.characterized_inventory = self.characterization_matrix.multiply(
                self.technosphere_flow_matrix
            )

        if self.ignored_locations:
            print(
                f"{len(self.ignored_locations)} locations were ignored. Check .ignored_locations attribute."
            )

    def generate_cf_table(self) -> pd.DataFrame:
        """
        Generate a pandas DataFrame with the characterization factors,
        from self.characterization_matrix.
        """
        # Determine the matrix type
        is_biosphere = all("biosphere-technosphere" in cf for cf in self.cfs_data)
        print(f"Matrix type: {'biosphere' if is_biosphere else 'technosphere'}")
        inventory = self.inventory if is_biosphere else self.technosphere_flow_matrix

        data = []
        non_zero_indices = self.characterization_matrix.nonzero()

        for i, j in zip(*non_zero_indices):
            consumer = bw2data.get_activity(self.reversed_activity[j])
            supplier = bw2data.get_activity(
                self.reversed_biosphere[i]
                if is_biosphere
                else self.reversed_activity[i]
            )

            entry = {
                "supplier name": supplier["name"],
                "consumer name": consumer["name"],
                "consumer reference product": consumer.get("reference product"),
                "consumer location": consumer.get("location"),
                "amount": inventory[i, j],
                "CF": self.characterization_matrix[i, j],
                "impact": self.characterized_inventory[i, j],
            }

            # Add supplier-specific fields based on matrix type
            if is_biosphere:
                entry.update({"supplier categories": supplier.get("categories")})
            else:
                entry.update(
                    {
                        "supplier reference product": supplier.get("reference product"),
                        "supplier location": supplier.get("location"),
                    }
                )

            data.append(entry)

        # Create the DataFrame
        df = pd.DataFrame(data)

        # Specify the desired column order
        column_order = [
            "supplier name",
            "supplier categories",
            "supplier reference product",
            "supplier location",
            "consumer name",
            "consumer reference product",
            "consumer location",
            "amount",
            "CF",
            "impact",
        ]

        # Reorder columns based on the desired order, ignoring missing columns
        df = df[[col for col in column_order if col in df.columns]]

        return df
