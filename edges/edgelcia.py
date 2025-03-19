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
from functools import cache

from .utils import (
    format_data,
    initialize_lcia_matrix,
    get_flow_matrix_positions,
    preprocess_cfs,
    load_missing_geographies,
    get_str,
    safe_eval,
    validate_parameter_lengths,
    add_population_and_gdp_data,
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


def make_hashable(flow_to_match):
    return tuple(sorted(flow_to_match.items()))


@cache
def cached_match_with_index(flow_to_match_hashable, required_fields_tuple):
    flow_to_match = dict(flow_to_match_hashable)
    required_fields = set(required_fields_tuple)
    return match_with_index(
        flow_to_match,
        cached_match_with_index.index,
        cached_match_with_index.lookup_mapping,
        required_fields,
    )

@cache
def get_shares(candidates: tuple):
    """
    Get the shares of each candidate location based on the weight of each candidate.

    :param candidates: A tuple of (location, weight) pairs.
    :return: A tuple of locations and their shares.
    """
    cand_locs, weights = zip(*candidates)
    weight_array = np.array(weights)
    total_weight = weight_array.sum()
    if total_weight == 0:
        return 0
    return cand_locs, weight_array / total_weight

def compute_average_cf(
    candidates: list,
    supplier_info: dict,
    weight: dict,
    cfs_lookup: dict,
):
    if not candidates:
        return 0
    if len(candidates) == 1:
        valid_candidates = [(candidates[0], 1)]
    else:
        valid_candidates = []
        for c in candidates:
            w = weight.get(c)
            if w is not None:
                valid_candidates.append((c, w))
        if not valid_candidates:
            return 0

    candidates, shares = get_shares(tuple(valid_candidates))

    IGNORED_FIELDS = {"matrix", "operator", "weight"}
    # Precompute the supplier keys and their values once
    supplier_keys = [k for k in supplier_info if k not in IGNORED_FIELDS]
    supplier_items = {k: supplier_info.get(k) for k in supplier_keys}

    expressions = []
    # Alias the match_operator function locally to reduce attribute lookups.
    _match_operator = match_operator

    for loc, share in zip(candidates, shares):
        loc_cfs = cfs_lookup.get(loc, [])
        matched_cf = None

        # Iterate over CFs for the candidate location
        for cf in loc_cfs:
            cf_supplier = cf.get("supplier")
            if not cf_supplier:
                continue

            # Get operator once (default "equals")
            operator = cf_supplier.get("operator", "equals")
            candidate_matches = True

            # Inline the matching logic for each key
            for key in supplier_keys:
                s_val = supplier_items.get(key)
                t_val = cf_supplier.get(key)
                if not _match_operator(s_val, t_val, operator):
                    candidate_matches = False
                    break

            if candidate_matches:
                if matched_cf is not None:
                    raise ValueError(
                        f"Multiple CFs found for {supplier_info} in {loc}: {cf} and {matched_cf}"
                    )
                matched_cf = cf

        if matched_cf is not None:
            cf_expr = matched_cf["value"]
            expressions.append(f"({share:.6f} * ({cf_expr}))")

    if not expressions:
        return 0
    return " + ".join(expressions)




@cache
def find_locations(
    location: str,
    weights_available: tuple,
    containing: bool = True,
    exceptions: [tuple, None] = None,
) -> list:
    """
    Find the locations containing or contained by a given location.
    :param location: The location to evaluate.
    :param weights_available: List of locations for which a weight is available.
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


@cache
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
        return value.startswith(target)
    elif operator == "contains":
        return target in value
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


def add_cf_entry(cfs_mapping, supplier_info, consumer_info, direction, indices, value):

    if direction == "biosphere-technosphere":
        supplier_info["matrix"] = "biosphere"
    else:
        supplier_info["matrix"] = "technosphere"
    consumer_info["matrix"] = "technosphere"

    entry = {
        "supplier": supplier_info,
        "consumer": consumer_info,
        "positions": indices,
        "direction": direction,
        "value": value,
    }
    cfs_mapping.append(entry)


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
        parameters: Optional[dict] = None,
        filepath: Optional[str] = None,
    ):
        """
        Initialize the SpatialLCA class, ensuring `method` is not passed to
        `prepare_lca_inputs` while still being available in the class.
        """

        self.weights = None
        self.consumer_lookup = None
        self.required_supplier_fields = None
        self.reversed_consumer_lookup = None
        self.reversed_supplier_lookup = None
        self.processed_technosphere_edges = None
        self.processed_biosphere_edges = None
        self.raw_cfs_data = None
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
        self.technosphere_edges = []
        self.technosphere_flow_matrix = None
        self.biosphere_edges = []
        self.technosphere_flows = None
        self.biosphere_flows = None
        self.characterized_inventory = None
        self.biosphere_characterization_matrix = None
        self.ignored_flows = set()
        self.ignored_locations = set()
        self.ignored_method_exchanges = list()
        self.cfs_data = None
        self.weight: str = weight
        self.parameters = parameters or {}
        self.scenario_length = validate_parameter_lengths(parameters=self.parameters)

        self.lca = bw2calc.LCA(demand=demand)
        self.load_raw_lcia_data()
        self.cfs_mapping = []

    def lci(self) -> None:

        self.lca.lci()

        if all(
            cf["supplier"].get("matrix") == "technosphere" for cf in self.raw_cfs_data
        ):
            self.technosphere_flow_matrix = self.build_technosphere_edges_matrix()
            self.technosphere_edges = set(
                list(zip(*self.technosphere_flow_matrix.nonzero()))
            )
        else:
            self.biosphere_edges = set(list(zip(*self.lca.inventory.nonzero())))

        unique_biosphere_flows = set(x[0] for x in self.biosphere_edges)
        unique_technosphere_flows = set(x[0] for x in self.technosphere_edges)

        if len(unique_biosphere_flows) > 0:
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

    def load_raw_lcia_data(self):
        if self.filepath is None:
            self.filepath = DATA_DIR / f"{'_'.join(self.method)}.json"
        if not self.filepath.is_file():
            raise FileNotFoundError(f"Data file not found: {self.filepath}")

        with open(self.filepath, "r", encoding="utf-8") as f:
            self.raw_cfs_data = format_data(data=json.load(f), weight=self.weight)

    def evaluate_cfs(self, scenario_idx=0):
        self.scenario_cfs = []

        for cf in self.cfs_mapping:
            symbolic_expr = cf["value"]
            numeric_value = safe_eval(symbolic_expr, self.parameters, scenario_idx)

            scenario_cf = {
                "supplier": cf["supplier"],
                "consumer": cf["consumer"],
                "positions": cf["positions"],
                "value": numeric_value,
            }
            self.scenario_cfs.append(scenario_cf)

        self.scenario_cfs = format_data(self.scenario_cfs, self.weight)
        self.cfs_number = len({x["value"] for x in self.scenario_cfs})

    def update_unprocessed_edges(self):
        self.processed_biosphere_edges = {
            pos
            for cf in self.cfs_mapping
            if cf["direction"] == "biosphere-technosphere"
            for pos in cf["positions"]
        }

        self.processed_technosphere_edges = {
            pos
            for cf in self.cfs_mapping
            if cf["direction"] == "technosphere-technosphere"
            for pos in cf["positions"]
        }

        self.unprocessed_biosphere_edges = [
            edge
            for edge in self.biosphere_edges
            if edge not in self.processed_biosphere_edges
        ]

        self.unprocessed_technosphere_edges = [
            edge
            for edge in self.technosphere_edges
            if edge not in self.processed_technosphere_edges
        ]

    def map_aggregate_locations(self) -> None:
        """
        Handle static regions (e.g., RER, GLO, ENTSOE) while pre-filtering unprocessed edges.
        Edges whose supplier info (based on required fields other than location) doesn't appear in the CF lookup are skipped.
        """
        # Ensure weights are set.
        if self.weights is None:
            self.weights = {
                cf["consumer"]["location"]: cf["consumer"]["weight"]
                for cf in self.cfs_mapping
                if cf["consumer"].get("location")
                   and cf["consumer"].get("weight") is not None
            }

        cfs_lookup = preprocess_cfs(self.raw_cfs_data)

        # Precompute the required supplier fields if not already done.
        # Here we mimic what preprocess_lookups() does.
        self.required_supplier_fields = {
            k
            for cf in self.raw_cfs_data
            for k in cf["supplier"].keys()
            if k not in {"matrix", "operator", "weight"}
        }

        # Build a set of candidate supplier signatures from cfs_lookup.
        # A signature is a tuple of (field, value) for all required supplier fields, sorted by field name.
        candidate_supplier_keys = set()
        for candidate, cf_list in cfs_lookup.items():
            for cf in cf_list:
                key = tuple(
                    (k, cf["supplier"].get(k))
                    for k in sorted(self.required_supplier_fields)
                )
                candidate_supplier_keys.add(key)

        print("Handling static regions...")

        # Helper to compute supplier signature from supplier_info.
        def supplier_signature(supplier_info: dict) -> tuple:
            return tuple((k, supplier_info.get(k)) for k in sorted(self.required_supplier_fields))

        # Process for each exchange direction.
        for direction in ["biosphere-technosphere", "technosphere-technosphere"]:
            unprocessed_edges = (
                self.unprocessed_biosphere_edges
                if direction == "biosphere-technosphere"
                else self.unprocessed_technosphere_edges
            )
            processed_flows = (
                self.processed_biosphere_edges
                if direction == "biosphere-technosphere"
                else self.processed_technosphere_edges
            )

            # Build an inverted index of unprocessed edges keyed by consumer location.
            from collections import defaultdict
            edges_index = defaultdict(list)
            for supplier_idx, consumer_idx in unprocessed_edges:
                # Skip edges already processed (if any)
                if (supplier_idx, consumer_idx) in self.processed_technosphere_edges:
                    continue
                # Note: We use the consumer's location from the consumer lookup.
                location = dict(self.reversed_consumer_lookup[consumer_idx]).get("location")
                edges_index[location].append((supplier_idx, consumer_idx))

            # Iterate over each consumer location group.
            for location, edges in tqdm(edges_index.items(), desc="Processing locations"):
                # Skip dynamic regions.
                if location in ["RoW", "RoE"]:
                    continue
                try:
                    if len(geo.contained(location, biggest_first=False, include_self=False)) == 0:
                        # the location does not contain any sub-locations
                        continue
                except KeyError:
                    continue

                # Compute candidate locations for this consumer location.
                candidate_locations = find_locations(
                    location=location,
                    weights_available=tuple(self.weights.keys()),
                )
                if not candidate_locations:
                    continue

                # Group edges by supplier signature so that we compute the CF only once per group.
                supplier_groups = defaultdict(list)
                for supplier_idx, consumer_idx in edges:
                    if (supplier_idx, consumer_idx) in processed_flows:
                        continue
                    supplier_info = dict(self.reversed_supplier_lookup[supplier_idx])
                    sig = supplier_signature(supplier_info)
                    supplier_groups[sig].append((supplier_idx, consumer_idx, supplier_info))

                # Process each supplier group.
                for sig, group_edges in supplier_groups.items():
                    rep_supplier = group_edges[0][2]  # All edges share the same supplier signature.
                    new_cf = compute_average_cf(
                        candidates=candidate_locations,
                        supplier_info=rep_supplier,
                        weight=self.weights,
                        cfs_lookup=cfs_lookup,
                    )
                    if new_cf != 0:
                        # "Broadcast" the computed CF to all edges in the group.
                        for supplier_idx, consumer_idx, supplier_info in group_edges:
                            consumer_info = dict(self.reversed_consumer_lookup[consumer_idx])
                            add_cf_entry(
                                cfs_mapping=self.cfs_mapping,
                                supplier_info=supplier_info,
                                consumer_info=consumer_info,
                                direction=direction,
                                indices=[(supplier_idx, consumer_idx)],
                                value=new_cf,
                            )

        # Finally, update the list of unprocessed edges.
        self.update_unprocessed_edges()

    def map_dynamic_locations(self) -> None:
        """
        Handle dynamic regions (RoW and RoE) and update CF data, pre-filtering edges
        by supplier signature (using all required supplier fields except location).
        """
        # Build weight dictionary from CF mapping.
        weight = {
            cf["consumer"]["location"]: cf["consumer"]["weight"]
            for cf in self.cfs_mapping
            if cf["consumer"].get("location") and cf["consumer"].get("weight") is not None
        }

        cfs_lookup = preprocess_cfs(self.raw_cfs_data)

        # Precompute required supplier fields (excluding non-matching fields)
        self.required_supplier_fields = {
            k
            for cf in self.raw_cfs_data
            for k in cf["supplier"].keys()
            if k not in {"matrix", "operator", "weight"}
        }

        # Build a set of candidate supplier signatures from cfs_lookup.
        candidate_supplier_keys = set()
        for candidate, cf_list in cfs_lookup.items():
            for cf in cf_list:
                key = tuple(
                    (k, cf["supplier"].get(k))
                    for k in sorted(self.required_supplier_fields)
                )
                candidate_supplier_keys.add(key)

        # Build technosphere flow lookups as in the original implementation.
        self.position_to_technosphere_flows_lookup = {
            i["position"]: {k: i[k] for k in i if k != "position"}
            for i in self.technosphere_flows
        }
        for flow in self.technosphere_flows:
            key = (flow["name"], flow.get("reference product"))
            self.technosphere_flows_lookup[key].append(flow["location"])

        print("Handling dynamic regions...")

        # We'll group edges by a composite key: (supplier_signature, consumer_activity_signature)
        # where supplier_signature is computed from the supplier info,
        # and consumer_activity_signature is computed from the consumer's technosphere flow.
        from collections import defaultdict

        # Helper: compute supplier signature using required supplier fields.
        def supplier_signature(supplier_info: dict) -> tuple:
            return tuple((k, supplier_info.get(k)) for k in sorted(self.required_supplier_fields))

        # Helper: compute consumer activity signature from the technosphere flow.
        # We use the "name" and "reference product" from the consumer's technosphere flow.
        def consumer_act_signature(consumer_idx) -> tuple:
            consumer_act = self.position_to_technosphere_flows_lookup.get(consumer_idx, {})
            return (consumer_act.get("name"), consumer_act.get("reference product"))

        # Process for each exchange direction.
        for direction in ["biosphere-technosphere", "technosphere-technosphere"]:
            unprocessed_edges = (
                self.unprocessed_biosphere_edges
                if direction == "biosphere-technosphere"
                else self.unprocessed_technosphere_edges
            )
            processed_flows = (
                self.processed_biosphere_edges
                if direction == "biosphere-technosphere"
                else self.processed_technosphere_edges
            )

            # Group dynamic edges: only process edges whose consumer location is in ["RoW", "RoE"].
            dynamic_groups = defaultdict(list)
            for supplier_idx, consumer_idx in unprocessed_edges:
                if (supplier_idx, consumer_idx) in processed_flows:
                    continue

                consumer_info = dict(self.reversed_consumer_lookup[consumer_idx])
                location = consumer_info.get("location")
                if location not in ["RoW", "RoE"]:
                    continue

                supplier_info = dict(self.reversed_supplier_lookup[supplier_idx])
                supp_sig = supplier_signature(supplier_info)
                cons_act_sig = consumer_act_signature(consumer_idx)
                key = (supp_sig, cons_act_sig)
                dynamic_groups[key].append((supplier_idx, consumer_idx, supplier_info))

            # Process each group
            for (supp_sig, cons_act_sig), group_edges in tqdm(dynamic_groups.items(), desc="Processing dynamic groups"):
                # Use the supplier info from the first edge in the group.
                rep_supplier = group_edges[0][2]
                # From the consumer activity signature, extract name and reference product.
                name, reference_product = cons_act_sig

                # Get locations (excluding RoW/RoE) from the technosphere flows that have a weight.
                other_than_RoW_RoE = [
                    loc
                    for loc in self.technosphere_flows_lookup.get((name, reference_product), [])
                    if loc not in ["RoW", "RoE"] and loc in weight
                ]

                # Compute candidate locations for "GLO", excluding the locations above.
                candidate_locations = find_locations(
                    location="GLO",
                    weights_available=tuple(weight.keys()),
                    exceptions=tuple(other_than_RoW_RoE),
                )
                if not candidate_locations:
                    continue

                # Compute the CF expression once for this group.
                new_cf = compute_average_cf(
                    candidates=candidate_locations,
                    supplier_info=rep_supplier,
                    weight=weight,
                    cfs_lookup=cfs_lookup,
                )

                if new_cf:
                    # Broadcast the computed CF to all edges in this group.
                    for supplier_idx, consumer_idx, supplier_info in group_edges:
                        consumer_info = dict(self.reversed_consumer_lookup[consumer_idx])
                        add_cf_entry(
                            cfs_mapping=self.cfs_mapping,
                            supplier_info=supplier_info,
                            consumer_info=consumer_info,
                            direction=direction,
                            indices=[(supplier_idx, consumer_idx)],
                            value=new_cf,
                        )

        # Update unprocessed edges after processing.
        self.update_unprocessed_edges()

    def map_contained_locations(self) -> None:
        """
        Handle contained locations (static regions) while pre-filtering unprocessed edges.
        Edges whose supplier info (based on required supplier fields excluding location)
        doesn't appear in the CF lookup are skipped.
        """
        # Ensure weights are set.
        if self.weights is None:
            self.weights = {
                cf["consumer"]["location"]: cf["consumer"]["weight"]
                for cf in self.cfs_mapping
                if cf["consumer"].get("location")
                   and cf["consumer"].get("weight") is not None
            }

        cfs_lookup = preprocess_cfs(self.raw_cfs_data)

        # Precompute required supplier fields if not already done.
        self.required_supplier_fields = {
            k
            for cf in self.raw_cfs_data
            for k in cf["supplier"].keys()
            if k not in {"matrix", "operator", "weight"}
        }

        # Build a set of candidate supplier signatures from cfs_lookup.
        candidate_supplier_keys = set()
        for candidate, cf_list in cfs_lookup.items():
            for cf in cf_list:
                key = tuple(
                    (k, cf["supplier"].get(k))
                    for k in sorted(self.required_supplier_fields)
                )
                candidate_supplier_keys.add(key)

        print("Handling contained locations...")

        # We'll group edges by a composite key: (supplier_signature, consumer_location)
        from collections import defaultdict

        def supplier_signature(supplier_info: dict) -> tuple:
            """Compute a signature from supplier info using required fields."""
            return tuple((k, supplier_info.get(k)) for k in sorted(self.required_supplier_fields))

        contained_groups = defaultdict(list)
        # Process for each exchange direction.
        for direction in ["biosphere-technosphere", "technosphere-technosphere"]:
            unprocessed_edges = (
                self.unprocessed_biosphere_edges
                if direction == "biosphere-technosphere"
                else self.unprocessed_technosphere_edges
            )
            processed_flows = (
                self.processed_biosphere_edges
                if direction == "biosphere-technosphere"
                else self.processed_technosphere_edges
            )

            # Group edges by (supplier_signature, consumer_location)
            for supplier_idx, consumer_idx in unprocessed_edges:
                if (supplier_idx, consumer_idx) in processed_flows:
                    continue

                supplier_info = dict(self.reversed_supplier_lookup[supplier_idx])
                consumer_info = dict(self.reversed_consumer_lookup[consumer_idx])
                consumer_location = consumer_info.get("location")
                # Use a composite key: supplier signature and consumer location.
                key = (supplier_signature(supplier_info), consumer_location)
                contained_groups[(direction, key)].append((supplier_idx, consumer_idx, supplier_info, consumer_info))

            # Process each group.
            for (dir_key, (supp_sig, cons_loc)), group_edges in tqdm(contained_groups.items(),
                                                                     desc="Processing contained groups"):
                if dir_key != direction:
                    continue

                # Compute candidate locations using "contained" logic.
                locations = find_locations(
                    location=cons_loc,
                    weights_available=tuple(self.weights.keys()),
                    containing=False,
                )
                if not locations:
                    continue

                # Use the representative supplier info from the group.
                rep_supplier = group_edges[0][2]
                new_cf = compute_average_cf(
                    candidates=locations,
                    supplier_info=rep_supplier,
                    weight=self.weights,
                    cfs_lookup=cfs_lookup,
                )
                if new_cf != 0:
                    # Broadcast the computed CF to every edge in the group.
                    for supplier_idx, consumer_idx, supplier_info, consumer_info in group_edges:
                        add_cf_entry(
                            cfs_mapping=self.cfs_mapping,
                            supplier_info=supplier_info,
                            consumer_info=consumer_info,
                            direction=direction,
                            indices=[(supplier_idx, consumer_idx)],
                            value=new_cf,
                        )

        # Finally, update the list of unprocessed edges.
        self.update_unprocessed_edges()

    def map_remaining_locations_to_global(self) -> None:
        """
        Handle remaining exchanges by assigning global CFs while pre-filtering
        unprocessed edges based on supplier signature. Only edges whose supplier's
        required attributes (excluding location) match one from the CF lookup are processed.
        """
        # Ensure weights are set.
        if self.weights is None:
            self.weights = {
                cf["consumer"]["location"]: cf["consumer"]["weight"]
                for cf in self.cfs_mapping
                if cf["consumer"].get("location")
                   and cf["consumer"].get("weight") is not None
            }

        cfs_lookup = preprocess_cfs(self.raw_cfs_data)

        # Precompute required supplier fields (excluding non-matching ones).
        self.required_supplier_fields = {
            k for cf in self.raw_cfs_data for k in cf["supplier"].keys() if k not in {"matrix", "operator", "weight"}
        }

        # (Optional) Build a set of candidate supplier signatures from cfs_lookup.
        candidate_supplier_keys = set()
        for candidate, cf_list in cfs_lookup.items():
            for cf in cf_list:
                key = tuple(
                    (k, cf["supplier"].get(k))
                    for k in sorted(self.required_supplier_fields)
                )
                candidate_supplier_keys.add(key)

        print("Handling remaining exchanges...")

        from collections import defaultdict

        # Helper: compute supplier signature from supplier_info.
        def supplier_signature(supplier_info: dict) -> tuple:
            return tuple((k, supplier_info.get(k)) for k in sorted(self.required_supplier_fields))

        # Process each direction separately.
        for direction in ["biosphere-technosphere", "technosphere-technosphere"]:
            unprocessed_edges = (
                self.unprocessed_biosphere_edges
                if direction == "biosphere-technosphere"
                else self.unprocessed_technosphere_edges
            )
            processed_flows = (
                self.processed_biosphere_edges
                if direction == "biosphere-technosphere"
                else self.processed_technosphere_edges
            )

            # Group edges by supplier signature.
            global_groups = defaultdict(list)
            for supplier_idx, consumer_idx in unprocessed_edges:
                if (supplier_idx, consumer_idx) in processed_flows:
                    continue

                supplier_info = dict(self.reversed_supplier_lookup[supplier_idx])
                sig = supplier_signature(supplier_info)
                global_groups[sig].append((supplier_idx, consumer_idx, supplier_info))

            # Process each group.
            for sig, group_edges in tqdm(global_groups.items(), desc="Processing global groups"):
                # For global CFs, candidate locations are determined for "GLO".
                locations = find_locations(
                    location="GLO",
                    weights_available=tuple(self.weights.keys()),
                )
                if not locations:
                    continue

                # Use the supplier info from the first edge in the group.
                rep_supplier = group_edges[0][2]
                new_cf = compute_average_cf(
                    candidates=locations,
                    supplier_info=rep_supplier,
                    weight=self.weights,
                    cfs_lookup=cfs_lookup,
                )
                if new_cf != 0:
                    # Broadcast the computed CF to every edge in the group.
                    for supplier_idx, consumer_idx, supplier_info in group_edges:
                        consumer_info = dict(self.reversed_consumer_lookup[consumer_idx])
                        add_cf_entry(
                            cfs_mapping=self.cfs_mapping,
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
            for cf in self.raw_cfs_data
            for k in cf["supplier"].keys()
            if k not in IGNORED_FIELDS
        }
        self.required_consumer_fields = {
            k
            for cf in self.raw_cfs_data
            for k in cf["consumer"].keys()
            if k not in IGNORED_FIELDS
        }

        self.supplier_lookup = preprocess_flows(
            flows_list=(
                self.biosphere_flows
                if all(
                    cf["supplier"].get("matrix") == "biosphere"
                    for cf in self.raw_cfs_data
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

    def map_exchanges(self):
        self.preprocess_lookups()

        edges = (
            self.biosphere_edges
            if all(
                cf["supplier"].get("matrix") == "biosphere" for cf in self.raw_cfs_data
            )
            else self.technosphere_edges
        )

        supplier_index = build_index(
            self.supplier_lookup, self.required_supplier_fields
        )
        consumer_index = build_index(
            self.consumer_lookup, self.required_consumer_fields
        )

        print("Identifying eligible exchanges...")

        for cf in tqdm(self.raw_cfs_data):
            cached_match_with_index.index = supplier_index
            cached_match_with_index.lookup_mapping = self.supplier_lookup
            supplier_candidates = cached_match_with_index(
                make_hashable(cf["supplier"]),
                tuple(sorted(self.required_supplier_fields)),
            )

            cached_match_with_index.index = consumer_index
            cached_match_with_index.lookup_mapping = self.consumer_lookup
            consumer_candidates = cached_match_with_index(
                make_hashable(cf["consumer"]),
                tuple(sorted(self.required_consumer_fields)),
            )

            positions = [
                (supplier, consumer)
                for supplier in supplier_candidates
                for consumer in consumer_candidates
                if (supplier, consumer) in edges
            ]

            cf_entry = {
                "supplier": cf["supplier"],
                "consumer": cf["consumer"],
                "direction": f"{cf['supplier']['matrix']}-{cf['consumer']['matrix']}",
                "positions": positions,
                "value": cf["value"],
            }

            self.cfs_mapping.append(cf_entry)

        self.update_unprocessed_edges()

    def statistics(self):
        """
        Build a table that summarize the method name, data file,
        number of CF, number of CFs used, number of exchanges characterized,
        number of exchanged for which a CF could not be obtained.
        """

        # build PrettyTable
        table = PrettyTable()
        table.header = False
        rows = []
        try:
            rows.append(
                [
                    "Activity",
                    fill(
                        list(self.lca.demand.keys())[0]["name"],
                        width=45,
                    ),
                ]
            )
        except TypeError:
            rows.append(
                [
                    "Activity",
                    fill(
                        bw2data.get_activity(id=list(self.lca.demand.keys())[0])[
                            "name"
                        ],
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
                                for x in self.cfs_mapping
                                if len(x["positions"]) > 0
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

        if len(self.processed_biosphere_edges) > 0:
            rows.append(
                [
                    "Exc. characterized",
                    len(self.processed_biosphere_edges),
                ]
            )
            rows.append(
                [
                    "Exc. uncharacterized",
                    len(self.unprocessed_biosphere_edges),
                ]
            )

        if len(self.processed_technosphere_edges) > 0:
            rows.append(
                [
                    "Exc. characterized",
                    len(self.processed_technosphere_edges),
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

    def fill_characterization_matrix(self):
        matrix_type = (
            "biosphere"
            if all(cf["supplier"]["matrix"] == "biosphere" for cf in self.scenario_cfs)
            else "technosphere"
        )

        self.characterization_matrix = initialize_lcia_matrix(
            self.lca, matrix_type=matrix_type
        )

        for cf in self.scenario_cfs:
            for supplier, consumer in cf["positions"]:
                self.characterization_matrix[supplier, consumer] = cf["value"]

        self.characterization_matrix = self.characterization_matrix.tocsr()

    def lcia(self) -> None:
        """
        Calculate the LCIA score.
        """

        self.fill_characterization_matrix()

        try:
            self.characterized_inventory = self.characterization_matrix.multiply(
                self.lca.inventory
            )
        except ValueError as err:
            self.characterized_inventory = self.characterization_matrix.multiply(
                self.technosphere_flow_matrix
            )

        self.score = self.characterized_inventory.sum()

    def generate_cf_table(self) -> pd.DataFrame:
        """
        Generate a pandas DataFrame with the evaluated characterization factors
        used in the current scenario.
        """

        # Ensure evaluated CFs are available
        if not self.scenario_cfs:
            raise ValueError("You must run evaluate_cfs_for_scenario() first.")

        # Determine matrix type clearly
        is_biosphere = all(
            cf["supplier"]["matrix"] == "biosphere" for cf in self.scenario_cfs
        )

        inventory = (
            self.lca.inventory if is_biosphere else self.technosphere_flow_matrix
        )

        data = []

        for cf in self.scenario_cfs:
            for supplier_idx, consumer_idx in cf["positions"]:
                consumer = bw2data.get_activity(self.reversed_activity[consumer_idx])

                supplier = bw2data.get_activity(
                    self.reversed_biosphere[supplier_idx]
                    if is_biosphere
                    else self.reversed_activity[supplier_idx]
                )

                amount = inventory[supplier_idx, consumer_idx]
                cf_value = cf["value"]
                impact = amount * cf_value

                entry = {
                    "supplier name": supplier["name"],
                    "consumer name": consumer["name"],
                    "consumer reference product": consumer.get("reference product"),
                    "consumer location": consumer.get("location"),
                    "amount": amount,
                    "CF": cf_value,
                    "impact": impact,
                }

                if is_biosphere:
                    entry.update({"supplier categories": supplier.get("categories")})
                else:
                    entry.update(
                        {
                            "supplier reference product": supplier.get(
                                "reference product"
                            ),
                            "supplier location": supplier.get("location"),
                        }
                    )

                data.append(entry)

        # Create the DataFrame
        df = pd.DataFrame(data)

        # Specify desired column order
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

        # Reorder columns, ignoring missing columns gracefully
        df = df[[col for col in column_order if col in df.columns]]

        return df
