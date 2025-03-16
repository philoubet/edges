"""
Module that implements the base class for country-specific life-cycle
impact assessments, and the AWARE class, which is a subclass of the
LCIA class.
"""

from collections import defaultdict
import logging
import json
from typing import Optional
from pathlib import Path
import bw2calc
import numpy as np

from scipy.sparse import coo_matrix
from constructive_geometries import Geomatcher
import pandas as pd
from prettytable import PrettyTable
import bw2data
from tqdm import tqdm
from textwrap import fill
from functools import lru_cache

from .utils import (
    format_data,
    initialize_lcia_matrix,
    get_flow_matrix_positions,
    preprocess_cfs,
    check_database_references,
    load_missing_geographies,
    get_str,
)
from .filesystem_constants import DATA_DIR

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
missing_geographies = load_missing_geographies()


def compute_average_cf(
    candidates: list,
    supplier_info: dict,
    weight: dict,
    cfs_lookup: dict,
    location: str,
):
    """
    Compute the average characterization factors for the region.
    :param candidates: List of constituent regions.
    :param supplier_info: Information about the supplier.
    :param weight: Weights for the constituents.
    :param cfs_lookup: Lookup dictionary for characterization factors.
    :param location: The region being evaluated.
    :return: The weighted average CF value.
    """

    # Filter constituents with valid weights
    if len(candidates) == 0:
        return 0, None
    elif len(candidates) == 1:
        valid_candidates = [(candidates[0], 1)]
    else:
        valid_candidates = [(c, weight[c]) for c in candidates if c in weight]

    if not valid_candidates:
        return 0, None

    # Separate constituents and weights
    candidates, weight_array = zip(*valid_candidates)
    weight_array = np.array(weight_array)

    # Normalize weights
    weight_sum = weight_array.sum()
    if weight_sum == 0:
        return 0, None
    shares = weight_array / weight_sum

    # Pre-filter supplier keys for filtering
    supplier_keys = supplier_info.keys()

    def match_supplier(cf: dict) -> bool:
        """
        Match a supplier based on operator logic.
        """
        for key in supplier_keys:
            supplier_value = supplier_info.get(key)
            match_target = cf["supplier"].get(key)
            operator = cf["supplier"].get("operator", "equals")

            if not match_operator(
                value=supplier_value, target=match_target, operator=operator
            ):
                return False
        return True

    # Compute the weighted average CF value
    value = 0

    for loc, share in zip(candidates, shares):
        loc_cfs = cfs_lookup.get(loc, [])

        # Filter CFs based on supplier info using the operator logic
        filtered_cfs = [cf["value"] for cf in loc_cfs if match_supplier(cf)]

        if len(filtered_cfs) == 0:
            raise ValueError(f"No CFs found for {supplier_info} for {loc} in {loc_cfs}")
        if len(filtered_cfs) > 1:
            raise ValueError(f"Multiple CFs found for {supplier_info} in {loc}")

        value += share * filtered_cfs[0]

    # Log if shares don't sum to 1 due to precision issues
    if not np.isclose(shares.sum(), 1):
        logger.info(
            f"Shares for {location} do not sum to 1 but {shares.sum()}: {shares}"
        )

    return value


@lru_cache(maxsize=100000)
def find_locations(
    location: str,
    weights_available: tuple,
    containing: bool = True,
    exceptions: [tuple, None] = None,
) -> list:
    """
    Find the locations containing or contained by a given location.
    :param location: The location to evaluate.
    :param supplier_info: Information about the supplier.
    :param weights_available: List of locations for which a weight is available.
    :param cfs: Lookup dictionary for characterization factors.
    :param containing: If True, find containing locations; otherwise, find contained locations.
    :return: The new characterization factor.
    """

    geo_func = geo.contained if containing is True else geo.within
    results = []

    if location in missing_geographies:
        for e in missing_geographies[location]:
            e_str = get_str(e)
            if e_str in weights_available and e_str != location:
                results.append(e_str)
    else:
        try:
            for e in geo_func(
                key=location,
                biggest_first=False,
                exclusive=True if containing is True else False,
                include_self=False,
            ):
                e_str = get_str(e)
                if (
                    e_str in weights_available
                    and e_str != location
                    and (exceptions is None or e_str not in exceptions)
                ):
                    results.append(e_str)

        except KeyError:
            logger.info(f"Region: {location}. No geometry found.")

    if containing is True:
        logger.info(f"Region: {location} minus {exceptions} contains: {results}")
    else:
        logger.info(f"Region: {location} minus {exceptions} is contained by: {results}")

    return results


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
        # add key to lookup if fields and values match requirements

        lookup.setdefault(key, []).append(flow["position"])
    return lookup


def build_index(lookup: dict, required_fields: set) -> dict:
    """
    Build an inverted index from the lookup dictionary.
    The index maps each required field to a dict, whose keys are the values
    from the lookup entries and whose values are lists of tuples:
    (lookup_key, positions), where lookup_key is the original key from lookup.

    :param lookup: The original lookup dictionary.
    :param required_fields: The fields to index.
    :return: A dictionary index.
    """
    index = {field: {} for field in required_fields}
    for key, positions in lookup.items():
        # Each key is assumed to be an iterable of (field, value) pairs.
        for k, v in key:
            if k in required_fields:
                index[k].setdefault(v, []).append((key, positions))
    return index


@lru_cache(maxsize=100000)
def match_operator(value: str, target: str, operator: str) -> bool:
    """
    Implements matching for three operator types:
      - "equals": value == target
      - "startswith": value starts with target (if both are strings)
      - "contains": target is contained in value (if both are strings)

    :param value: The flow's value.
    :param target: The lookup's candidate value.
    :param operator: The operator type ("equals", "startswith", "contains").
    :return: True if the condition is met, False otherwise.
    """
    if operator == "equals":
        return value == target
    elif operator == "startswith":
        if isinstance(value, str) and isinstance(target, str):
            return value.startswith(target)
        return False
    elif operator == "contains":
        if isinstance(value, str) and isinstance(target, str):
            return target in value
        return False
    return False


def match_with_index(
    flow_to_match: dict, index: dict, lookup_mapping: dict, required_fields: set
) -> list:
    """
    Match a flow against the lookup using the inverted index.
    Supports "equals", "startswith", and "contains" operators.

    :param flow_to_match: The flow to match.
    :param index: The inverted index produced by build_index().
    :param lookup_mapping: The original lookup dictionary mapping keys to positions.
    :param required_fields: The required fields for matching.
    :return: A list of matching positions.
    """
    operator_value = flow_to_match.get("operator", "equals")
    candidate_keys = None

    for field in required_fields:
        match_target = flow_to_match.get(field)
        field_index = index.get(field, {})
        field_candidates = set()

        if operator_value == "equals":
            # Fast direct lookup.
            for candidate in field_index.get(match_target, []):
                candidate_key, _ = candidate
                field_candidates.add(candidate_key)
        else:
            # For "startswith" or "contains", we iterate over all candidate values.
            for candidate_value, candidate_list in field_index.items():
                if match_operator(
                    value=candidate_value, target=match_target, operator=operator_value
                ):
                    for candidate in candidate_list:
                        candidate_key, _ = candidate
                        field_candidates.add(candidate_key)

        # Initialize or intersect candidate sets.
        if candidate_keys is None:
            candidate_keys = field_candidates
        else:
            candidate_keys &= field_candidates

        # Early exit if no candidates remain.
        if not candidate_keys:
            return []

    # Gather positions from the original lookup mapping for all candidate keys.
    matches = []
    for key in candidate_keys:
        matches.extend(lookup_mapping.get(key, []))
    return matches


def add_cf_entry(cf_data, supplier_info, consumer_info, direction, indices, value):
    cf_data.append(
        {
            "supplier": supplier_info,
            "consumer": consumer_info,
            direction: indices,
            "value": value,
        }
    )


def default_geo(constituent):
    return [get_str(e) for e in geo.contained(constituent)]


def nested_dict():
    return defaultdict(dict)


class EdgeLCIA:
    """
    Class that implements the calculation of the regionalized life cycle impact assessment (LCIA) results.
    Relies on bw2data.LCA class for inventory calculations and matrices.
    """

    def __init__(
        self,
        demand: dict,
        method: Optional[tuple] = None,
        weight: Optional[str] = "population",
        filepath: Optional[str] = None,
    ):
        """
        Initialize the SpatialLCA class, ensuring `method` is not passed to
        `prepare_lca_inputs` while still being available in the class.
        """

        self.unprocessed_technosphere_edges = []
        self.unprocessed_biosphere_edges = []
        self.score = None
        self.cfs_number = None
        self.filepath = Path(filepath) if filepath else None
        self.reversed_biosphere = None
        self.reversed_activity = None
        self.characterization_matrix = None
        self.method = method  # Store the method argument in the instance
        self.position_to_technosphere_flows_lookup = None
        self.technosphere_flows_lookup = defaultdict(list)
        self.technosphere_edges = None
        self.technosphere_flow_matrix = None
        self.biosphere_edges = None
        self.technosphere_flows = None
        self.biosphere_flows = None
        self.characterized_inventory = None
        self.biosphere_characterization_matrix = None
        self.ignored_flows = set()
        self.ignored_locations = set()
        self.ignored_method_exchanges = list()
        self.cfs_data = None
        self.weight: str = weight

        self.lca = bw2calc.LCA(demand=demand)
        self.load_lcia_data()

    def lci(self) -> None:

        self.lca.lci()

        self.technosphere_flow_matrix = self.build_technosphere_edges_matrix()
        self.technosphere_edges = set(
            list(zip(*self.technosphere_flow_matrix.nonzero()))
        )
        self.biosphere_edges = set(list(zip(*self.lca.inventory.nonzero())))
        unique_biosphere_flows = set(x[0] for x in self.biosphere_edges)
        self.biosphere_flows = get_flow_matrix_positions(
            {
                k: v
                for k, v in self.lca.biosphere_dict.items()
                if v in unique_biosphere_flows
            }
        )

        self.technosphere_flows = get_flow_matrix_positions(
            {k: v for k, v in self.lca.activity_dict.items()}
        )

        self.reversed_activity, _, self.reversed_biosphere = self.lca.reverse_dict()

    def build_technosphere_edges_matrix(self):
        """
        Generate a matrix with the technosphere flows.
        """

        # Convert CSR to COO format for easier manipulation
        coo = self.lca.technosphere_matrix.tocoo()

        # Extract negative values
        rows, cols, data = coo.row, coo.col, coo.data
        negative_data = -data * (data < 0)  # Keep only negatives and make them positive

        # Scale columns by supply_array
        scaled_data = negative_data * self.lca.supply_array[cols]

        # Create the flow matrix in sparse format
        flow_matrix_sparse = coo_matrix(
            (scaled_data, (rows, cols)), shape=self.lca.technosphere_matrix.shape
        )

        return flow_matrix_sparse.tocsr()

    def load_lcia_data(self, data_objs=None):
        """
        Load the data for the LCIA method.
        """
        if self.filepath is None:
            self.filepath = DATA_DIR / f"{'_'.join(self.method)}.json"

        if not self.filepath.is_file():
            raise FileNotFoundError(f"Data file not found: {self.filepath}")

        with open(self.filepath, "r", encoding="utf-8") as f:
            self.cfs_data = format_data(data=json.load(f), weight=self.weight)
            self.cfs_number = len(set([x.get("value") for x in self.cfs_data]))

    def update_unprocessed_edges(self):

        # Process edges

        edges = {
            (f[0], f[1])
            for cf in self.cfs_data
            for f in cf.get("biosphere-technosphere", [])
            if f
        }

        if edges:
            supplier_bio_tech_edges, consumer_bio_tech_edges = map(set, zip(*edges))
        else:
            supplier_bio_tech_edges, consumer_bio_tech_edges = set(), set()

        edges = {
            (f[0], f[1])
            for cf in self.cfs_data
            for f in cf.get("technosphere-technosphere", [])
            if f
        }

        if edges:
            supplier_tech_tech_edges, consumer_tech_tech_edges = map(set, zip(*edges))
        else:
            supplier_tech_tech_edges, consumer_tech_tech_edges = set(), set()

        self.unprocessed_biosphere_edges = [
            edge
            for edge in self.biosphere_edges
            if edge[0] in supplier_bio_tech_edges
            and edge[1] not in consumer_bio_tech_edges
        ]
        self.unprocessed_technosphere_edges = [
            edge
            for edge in self.technosphere_edges
            if edge[0] in supplier_tech_tech_edges
            and edge[1] not in consumer_tech_tech_edges
        ]

    def map_aggregate_locations(self) -> None:
        """
        Handle static regions and update CF data (e.g., RER, GLO, ENTSOE, etc.).
        CFs are obtained by averaging the CFs of the constituents of the region.

        :return: None
        """

        weight = {
            i.get("consumer").get("location"): i.get("consumer").get("weight")
            for i in self.cfs_data
            if i.get("consumer").get("location")
        }

        cfs_lookup = preprocess_cfs(self.cfs_data)

        print("Handling static regions...")

        for direction in ["biosphere-technosphere", "technosphere-technosphere"]:
            unprocessed_edges = (
                self.unprocessed_biosphere_edges
                if direction == "biosphere-technosphere"
                else self.unprocessed_technosphere_edges
            )
            for supplier_idx, consumer_idx in tqdm(unprocessed_edges):
                supplier_info = dict(self.reversed_supplier_lookup[supplier_idx])
                consumer_info = dict(self.reversed_consumer_lookup[consumer_idx])
                location = consumer_info.get("location")

                if location not in ["RoW", "RoE"]:
                    # Resolve from cache or compute new CF
                    locations = find_locations(
                        location=location,
                        weights_available=tuple(weight.keys()),
                    )

                    new_cf = compute_average_cf(
                        candidates=locations,
                        supplier_info=supplier_info,
                        weight=weight,
                        cfs_lookup=cfs_lookup,
                        location=location,
                    )

                    if new_cf != 0:
                        add_cf_entry(
                            cf_data=self.cfs_data,
                            supplier_info=supplier_info,
                            consumer_info=consumer_info,
                            direction=direction,
                            indices=[(supplier_idx, consumer_idx)],
                            value=new_cf,
                        )

        self.update_unprocessed_edges()

    def map_dynamic_locations(self) -> None:
        """
        Handle dynamic regions like RoW and RoE and update CF data.

        :return: None
        """

        weight = {
            i.get("consumer").get("location"): i.get("consumer").get("weight")
            for i in self.cfs_data
            if i.get("consumer").get("location")
        }

        cfs_lookup = preprocess_cfs(self.cfs_data)

        self.position_to_technosphere_flows_lookup = {
            i["position"]: {k: i[k] for k in i if k != "position"}
            for i in self.technosphere_flows
        }
        for flow in self.technosphere_flows:
            key = (flow["name"], flow.get("reference product"))
            self.technosphere_flows_lookup[key].append(flow["location"])

        print("Handling dynamic regions...")

        for direction in ["biosphere-technosphere", "technosphere-technosphere"]:
            unprocessed_edges = (
                self.unprocessed_biosphere_edges
                if direction == "biosphere-technosphere"
                else self.unprocessed_technosphere_edges
            )

            for supplier_idx, consumer_idx in tqdm(unprocessed_edges):
                supplier_info = dict(self.reversed_supplier_lookup[supplier_idx])
                consumer_info = dict(self.reversed_consumer_lookup[consumer_idx])
                location = consumer_info.get("location")

                if location in ["RoW", "RoE"]:
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

                    locations = find_locations(
                        location="GLO",
                        weights_available=tuple(weight.keys()),
                        exceptions=tuple(other_than_RoW_RoE),
                    )

                    new_cf = compute_average_cf(
                        candidates=locations,
                        supplier_info=supplier_info,
                        weight=weight,
                        cfs_lookup=cfs_lookup,
                        location=location,
                    )

                    if new_cf:
                        add_cf_entry(
                            cf_data=self.cfs_data,
                            supplier_info=supplier_info,
                            consumer_info=consumer_info,
                            direction=direction,
                            indices=[(supplier_idx, consumer_idx)],
                            value=new_cf,
                        )

        self.update_unprocessed_edges()

    def map_disaggregate_locations(
        self,
    ) -> None:

        weight = {
            i.get("consumer").get("location"): i.get("consumer").get("weight")
            for i in self.cfs_data
            if i.get("consumer").get("location")
        }
        cfs_lookup = preprocess_cfs(self.cfs_data)

        print("Handling unmatched locations...")

        for direction in ["biosphere-technosphere", "technosphere-technosphere"]:
            unprocessed_edges = (
                self.unprocessed_biosphere_edges
                if direction == "biosphere-technosphere"
                else self.unprocessed_technosphere_edges
            )

            for supplier_idx, consumer_idx in tqdm(unprocessed_edges):
                supplier_info = dict(self.reversed_supplier_lookup[supplier_idx])
                consumer_info = dict(self.reversed_consumer_lookup[consumer_idx])
                location = consumer_info.get("location")

                locations = find_locations(
                    location=location,
                    weights_available=tuple(weight.keys()),
                    containing=False,
                )
                new_cf = compute_average_cf(
                    candidates=locations,
                    supplier_info=supplier_info,
                    weight=weight,
                    cfs_lookup=cfs_lookup,
                    location=location,
                )

                if new_cf != 0:
                    add_cf_entry(
                        cf_data=self.cfs_data,
                        supplier_info=supplier_info,
                        consumer_info=consumer_info,
                        direction=direction,
                        indices=[(supplier_idx, consumer_idx)],
                        value=new_cf,
                    )
        self.update_unprocessed_edges()

    def map_remaining_locations_to_global(self) -> None:

        weight = {
            i.get("consumer").get("location"): i.get("consumer").get("weight")
            for i in self.cfs_data
            if i.get("consumer").get("location")
        }
        cfs_lookup = preprocess_cfs(self.cfs_data)

        # if still unprocessed, we give them global CFs
        print("Handling remaining exchanges...")

        for direction in ["biosphere-technosphere", "technosphere-technosphere"]:
            unprocessed_edges = (
                self.unprocessed_biosphere_edges
                if direction == "biosphere-technosphere"
                else self.unprocessed_technosphere_edges
            )

            for supplier_idx, consumer_idx in tqdm(unprocessed_edges):
                supplier_info = dict(self.reversed_supplier_lookup[supplier_idx])
                consumer_info = dict(self.reversed_consumer_lookup[consumer_idx])
                location = consumer_info.get("location")

                locations = find_locations(
                    location="GLO",
                    weights_available=tuple(weight.keys()),
                )
                new_cf = compute_average_cf(
                    candidates=locations,
                    supplier_info=supplier_info,
                    weight=weight,
                    cfs_lookup=cfs_lookup,
                    location=location,
                )

                if new_cf:
                    add_cf_entry(
                        cf_data=self.cfs_data,
                        supplier_info=supplier_info,
                        consumer_info=consumer_info,
                        direction=direction,
                        indices=[(supplier_idx, consumer_idx)],
                        value=new_cf,
                    )
        self.update_unprocessed_edges()

    def preprocess_lookups(self):
        """
        Preprocess supplier and consumer flows into lookup dictionaries.
        """

        # Constants for ignored fields
        IGNORED_FIELDS = {"matrix", "operator", "weight"}

        # Precompute required fields for faster access
        self.required_supplier_fields = {
            k
            for cf in self.cfs_data
            for k in cf["supplier"].keys()
            if k not in IGNORED_FIELDS
        }
        self.required_consumer_fields = {
            k
            for cf in self.cfs_data
            for k in cf["consumer"].keys()
            if k not in IGNORED_FIELDS
        }

        self.supplier_lookup = preprocess_flows(
            flows_list=(
                self.biosphere_flows
                if all(
                    cf["supplier"].get("matrix") == "biosphere" for cf in self.cfs_data
                )
                else self.technosphere_flows
            ),
            mandatory_fields=self.required_supplier_fields,
        )

        self.consumer_lookup = preprocess_flows(
            flows_list=self.technosphere_flows,
            mandatory_fields=self.required_consumer_fields,
        )

        self.reversed_supplier_lookup = {
            pos: key
            for key, positions in self.supplier_lookup.items()
            for pos in positions
        }
        self.reversed_consumer_lookup = {
            pos: key
            for key, positions in self.consumer_lookup.items()
            for pos in positions
        }

    def identify_exchanges(self):
        """
        Based on search criteria under `supplier` and `consumer` keys,
        identify the exchanges in the inventory matrix.
        """

        # Preprocess flows and lookups
        self.preprocess_lookups()

        edges = (
            self.biosphere_edges
            if all(cf["supplier"].get("matrix") == "biosphere" for cf in self.cfs_data)
            else self.technosphere_edges
        )

        supplier_index = build_index(
            self.supplier_lookup, self.required_supplier_fields
        )
        consumer_index = build_index(
            self.consumer_lookup, self.required_consumer_fields
        )

        # tdqm progress bar
        print("Identifying eligible exchanges...")
        for cf in tqdm(self.cfs_data):
            # Generate supplier candidates
            supplier_candidates = match_with_index(
                cf["supplier"],
                supplier_index,
                self.supplier_lookup,
                self.required_supplier_fields,
            )

            # Generate consumer candidates
            consumer_candidates = match_with_index(
                cf["consumer"],
                consumer_index,
                self.consumer_lookup,
                self.required_consumer_fields,
            )

            # Create pairs of supplier and consumer candidates
            cf[f"{cf['supplier']['matrix']}-{cf['consumer']['matrix']}"] = [
                (supplier, consumer)
                for supplier in supplier_candidates
                for consumer in consumer_candidates
                if (supplier, consumer) in edges
            ]

        self.update_unprocessed_edges()

    def statistics(self):
        """
        Build a table that summarize the method name, data file,
        number of CF, number of CFs used, number of exchanges characterized,
        number of exchanged for which a CF could not be obtained.
        """

        processed_biosphere_edges = {
            f for cf in self.cfs_data for f in cf.get("biosphere-technosphere", [])
        }
        processed_technosphere_edges = {
            f for cf in self.cfs_data for f in cf.get("technosphere-technosphere", [])
        }

        self.unprocessed_biosphere_edges = (
            set(self.unprocessed_biosphere_edges) - processed_biosphere_edges
        )
        self.unprocessed_technosphere_edges = (
            set(self.unprocessed_technosphere_edges) - processed_technosphere_edges
        )

        # build PrettyTable
        table = PrettyTable()
        table.header = False
        rows = []
        rows.append(
            [
                "Activity",
                fill(
                    list(self.lca.demand.keys())[0]["name"],
                    width=45,
                ),
            ]
        )
        rows.append(["Method name", fill(str(self.method), width=45)])
        rows.append(["Data file", fill(self.filepath.stem, width=45)])
        rows.append(["Unique CFs in method", self.cfs_number])
        rows.append(
            [
                "Unique CFs used",
                len(
                    list(
                        set(
                            [
                                x["value"]
                                for x in self.cfs_data
                                if any(
                                    x.get(direction)
                                    for direction in [
                                        "biosphere-technosphere",
                                        "technosphere-technosphere",
                                    ]
                                )
                            ]
                        )
                    )
                ),
            ]
        )

        if self.ignored_method_exchanges:
            rows.append(
                ["CFs without eligible exc.", len(self.ignored_method_exchanges)]
            )

        if self.ignored_locations:
            rows.append(["Product system locations ignored", self.ignored_locations])

        if len(processed_biosphere_edges) > 0:
            rows.append(
                [
                    "Exc. characterized",
                    len(processed_biosphere_edges),
                ]
            )
            rows.append(
                [
                    "Exc. uncharacterized",
                    len(self.unprocessed_biosphere_edges),
                ]
            )

        if len(processed_technosphere_edges) > 0:
            rows.append(
                [
                    "Exc. characterized",
                    len(processed_technosphere_edges),
                ]
            )
            rows.append(
                [
                    "Exc. uncharacterized",
                    len(self.unprocessed_technosphere_edges),
                ]
            )

        for row in rows:
            table.add_row(row)

        print(table)

    def fill_in_lcia_matrix(self) -> None:
        """
        Translate the data to indices in the inventory matrix.
        """

        if all(cf["supplier"].get("matrix") == "biosphere" for cf in self.cfs_data):
            self.characterization_matrix = initialize_lcia_matrix(self.lca)
        else:
            self.characterization_matrix = initialize_lcia_matrix(
                self.lca, matrix_type="technosphere"
            )

        for cf in self.cfs_data:
            for supplier, consumer in cf.get("biosphere-technosphere", []):
                self.characterization_matrix[supplier, consumer] = cf["value"]

            for supplier, consumer in cf.get("technosphere-technosphere", []):
                self.characterization_matrix[supplier, consumer] = cf["value"]

        self.characterization_matrix = self.characterization_matrix.tocsr()

    def lcia(self) -> None:
        """
        Calculate the LCIA score.
        """

        self.fill_in_lcia_matrix()

        try:
            self.characterized_inventory = self.characterization_matrix.multiply(
                self.lca.inventory
            )
        except ValueError:
            self.characterized_inventory = self.characterization_matrix.multiply(
                self.technosphere_flow_matrix
            )

        self.score = self.characterized_inventory.sum()

    def generate_cf_table(self) -> pd.DataFrame:
        """
        Generate a pandas DataFrame with the characterization factors,
        from self.characterization_matrix.
        """
        # Determine the matrix type
        is_biosphere = all("biosphere-technosphere" in cf for cf in self.cfs_data)
        print(f"Matrix type: {'biosphere' if is_biosphere else 'technosphere'}")
        inventory = (
            self.lca.inventory
            if is_biosphere is True
            else self.technosphere_flow_matrix
        )

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
            if is_biosphere is True:
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
