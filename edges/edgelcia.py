"""
Module that implements the base class for country-specific life-cycle
impact assessments, and the AWARE class, which is a subclass of the
LCIA class.
"""

import math
from collections import defaultdict
import logging
import json
from typing import Optional
from pathlib import Path
import bw2calc
import numpy as np
import sparse

from scipy.sparse import coo_matrix
import pandas as pd
from prettytable import PrettyTable
import bw2data
from tqdm import tqdm
from textwrap import fill
from functools import cache
from scipy import stats
from functools import lru_cache

from .utils import (
    format_data,
    initialize_lcia_matrix,
    get_flow_matrix_positions,
    preprocess_cfs,
    safe_eval,
    safe_eval_cached,
    validate_parameter_lengths,
    get_str,
)
from .georesolver import GeoResolver
from .filesystem_constants import DATA_DIR

# delete the logs
with open("edgelcia.log", "w", encoding="utf-8"):
    pass

# initiate the logger
logging.basicConfig(
    filename="edgelcia.log",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(funcName)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


def make_hashable(flow_to_match):
    def convert(value):
        if isinstance(value, list):
            return tuple(value)
        elif isinstance(value, dict):
            return tuple(sorted((k, convert(v)) for k, v in value.items()))
        return value

    return tuple(sorted((k, convert(v)) for k, v in flow_to_match.items()))


@cache
def cached_match_with_index(flow_to_match_hashable, required_fields_tuple):
    flow_to_match = dict(flow_to_match_hashable)
    required_fields = set(required_fields_tuple)
    return match_with_index(
        flow_to_match,
        cached_match_with_index.index,
        cached_match_with_index.lookup_mapping,
        required_fields,
        cached_match_with_index.reversed_lookup,
    )


@cache
def get_shares(candidates: tuple):
    """
    Get the shares of each candidate location based on weights.
    """
    if not candidates:
        return [], np.array([])

    cand_locs, weights = zip(*candidates)
    weight_array = np.array(weights, dtype=float)
    total_weight = weight_array.sum()
    if total_weight == 0:
        return list(cand_locs), np.zeros_like(weight_array)
    return list(cand_locs), weight_array / total_weight


def compute_average_cf(
    candidate_suppliers: list,
    candidate_consumers: list,
    supplier_info: dict,
    consumer_info: dict,
    weight: dict,
    cf_index: dict,
    required_supplier_fields: set = None,
    required_consumer_fields: set = None,
    logger=None,
) -> tuple[str | float, Optional[dict]]:
    """
    Compute the weighted average characterization factor (CF) for a given supplier-consumer pair.
    Supports disaggregated regional matching on both supplier and consumer sides.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if not candidate_suppliers and not candidate_consumers:
        return 0, None

    valid_candidates = [
        (loc, weight[loc])
        for loc in set(candidate_suppliers + candidate_consumers)
        if loc in weight and weight[loc] > 0
    ]
    if not valid_candidates:
        return 0, None

    cand_locs, shares = get_shares(tuple(valid_candidates))
    if not shares.any():
        return 0, None

    filtered_supplier = {
        k: supplier_info[k] for k in required_supplier_fields if k in supplier_info
    }
    filtered_consumer = {
        k: consumer_info[k] for k in required_consumer_fields if k in consumer_info
    }

    if "classifications" in filtered_supplier:
        c = filtered_supplier["classifications"]
        if isinstance(c, dict):
            filtered_supplier["classifications"] = tuple(
                (scheme, tuple(sorted(vals))) for scheme, vals in c.items()
            )
        elif isinstance(c, (list, tuple)):
            filtered_supplier["classifications"] = tuple(sorted(c))

    supplier_sig = make_hashable(filtered_supplier)
    matched_cfs = []

    for loc, share in zip(cand_locs, shares):
        loc_cfs_dict = cf_index.get(loc, {})
        loc_cfs = loc_cfs_dict.get(supplier_sig, [])

        matched_cfs.extend(
            process_cf_list(
                loc_cfs,
                filtered_supplier,
                filtered_consumer,
                supplier_info,
                consumer_info,
                candidate_suppliers,
                candidate_consumers,
                share,
            )
        )

    if not matched_cfs:
        return 0, None

    expressions = [f"({share:.6f} * ({cf['value']}))" for cf, share in matched_cfs]
    expr = " + ".join(expressions)

    return (expr, matched_cfs[0][0]) if len(matched_cfs) == 1 else (expr, None)


def process_cf_list(
    cf_list: list,
    filtered_supplier: dict,
    filtered_consumer: dict,
    supplier_info: dict,
    consumer_info: dict,
    candidate_suppliers: list,
    candidate_consumers: list,
    share: float,
) -> list:
    results = []
    best_score = -1
    best_cf = None

    for cf in cf_list:
        supplier_cf = cf.get("supplier", {})
        consumer_cf = cf.get("consumer", {})
        operator = supplier_cf.get("operator", "equals")

        supplier_match = True
        for k in filtered_supplier:
            if k == "classifications":
                continue
            val = filtered_supplier.get(k, "")
            expected = supplier_cf.get(k, "")

            if k == "location" and candidate_suppliers:
                match = any(
                    match_operator(loc_val, expected, operator)
                    for loc_val in candidate_suppliers
                )
            else:
                match = match_operator(val, expected, operator)

            if not match:
                supplier_match = False
                break

        if not supplier_match:
            continue

        consumer_match = True
        for k in filtered_consumer:
            if k == "classifications":
                continue
            val = filtered_consumer.get(k, "")
            expected = consumer_cf.get(k, "")

            if k == "location" and candidate_consumers:
                match = any(
                    match_operator(loc_val, expected, operator)
                    for loc_val in candidate_consumers
                )
            else:
                match = match_operator(val, expected, operator)

            if not match:
                consumer_match = False
                break

        if not consumer_match:
            continue

        match_score = 0
        cf_class = supplier_cf.get("classifications")
        ds_class = supplier_info.get("classifications")
        if cf_class and ds_class and matches_classifications(cf_class, ds_class):
            match_score += 1

        cf_cons_class = consumer_cf.get("classifications")
        ds_cons_class = consumer_info.get("classifications")
        if (
            cf_cons_class
            and ds_cons_class
            and matches_classifications(cf_cons_class, ds_cons_class)
        ):
            match_score += 1

        if match_score > best_score:
            best_score = match_score
            best_cf = cf

    if best_cf:
        results.append((best_cf, share))

    return results


def preprocess_flows(flows_list: list, mandatory_fields: set) -> dict:
    """
    Preprocess flows into a lookup dictionary.
    Each flow is keyed by a tuple of selected metadata fields.
    If no fields are present, falls back to using its position as key.

    :param flows_list: List of flows (dicts with metadata + 'position')
    :param mandatory_fields: Fields that must be included in the lookup key.
    :return: A dictionary mapping keys -> list of flow positions
    """
    lookup = {}

    for flow in flows_list:

        def make_value_hashable(v):
            if isinstance(v, list):
                return tuple(v)
            if isinstance(v, dict):
                return tuple(
                    sorted((k, make_value_hashable(val)) for k, val in v.items())
                )
            return v

        # Build a hashable key from mandatory fields (if any are present)
        key_elements = [
            (k, make_value_hashable(flow[k]))
            for k in mandatory_fields
            if k in flow and flow[k] is not None
        ]

        if key_elements:
            key = tuple(sorted(key_elements))
        else:
            # Fallback: use the position as a unique key
            key = (("position", flow["position"]),)

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


def matches_classifications(cf_classifications, dataset_classifications):
    """Match CF classification codes to dataset classifications."""
    if isinstance(cf_classifications, dict):
        cf_classifications = [
            (scheme, code)
            for scheme, codes in cf_classifications.items()
            for code in codes
        ]
    elif isinstance(cf_classifications, (list, tuple)):
        if all(
            isinstance(x, tuple) and isinstance(x[1], (list, tuple))
            for x in cf_classifications
        ):
            # Convert from tuple of tuples like (('cpc', ('01.1',)),) -> [('cpc', '01.1')]
            cf_classifications = [
                (scheme, code) for scheme, codes in cf_classifications for code in codes
            ]

    for scheme, value in dataset_classifications:
        code = value.split(":")[0].strip()
        if any(
            code.startswith(cf_code) and scheme == cf_scheme
            for cf_scheme, cf_code in cf_classifications
        ):
            return True
    return False


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
    flow_to_match: dict,
    index: dict,
    lookup_mapping: dict,
    required_fields: set,
    reversed_lookup: dict,
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

    if not required_fields:
        # Match all flows if no fields to match
        return [pos for positions in lookup_mapping.values() for pos in positions]

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

    if "classifications" in flow_to_match:
        cf_classifications_by_scheme = flow_to_match["classifications"]

        if isinstance(cf_classifications_by_scheme, tuple):
            cf_classifications_by_scheme = dict(cf_classifications_by_scheme)

        classified_matches = []

        for pos in matches:
            flow = reversed_lookup.get(pos)
            flow = dict(flow)
            if not flow:
                continue
            try:
                dataset_classifications = flow.get("classifications", [])
            except:
                print(flow)
                raise

            if dataset_classifications:
                for scheme, cf_codes in cf_classifications_by_scheme.items():
                    relevant_codes = [
                        code.split(":")[0].strip()
                        for s, code in dataset_classifications
                        if s.lower() == scheme.lower()
                    ]
                    if any(
                        code.startswith(prefix)
                        for prefix in cf_codes
                        for code in relevant_codes
                    ):
                        classified_matches.append(pos)
                        break

        if classified_matches:
            return classified_matches

    return matches


def add_cf_entry(
    cfs_mapping, supplier_info, consumer_info, direction, indices, value, uncertainty
):

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

    if uncertainty is not None:
        entry["uncertainty"] = uncertainty

    cfs_mapping.append(entry)


def nested_dict():
    return defaultdict(dict)


def normalize_cf_entries(cf_list: list[dict]) -> list[dict]:
    for cf in cf_list:
        supplier = cf.get("supplier", {})
        classifications = supplier.get("classifications")
        if isinstance(classifications, dict):
            # Normalize from dict
            supplier["classifications"] = tuple(
                (scheme, val)
                for scheme, values in sorted(classifications.items())
                for val in values
            )
        elif isinstance(classifications, list):
            # Already list of (scheme, code), just ensure it's a tuple
            supplier["classifications"] = tuple(classifications)
        elif isinstance(classifications, tuple):
            # Handle legacy format like: (('cpc', ('01.1',)),)
            new_classifications = []
            for scheme, maybe_codes in classifications:
                if isinstance(maybe_codes, (tuple, list)):
                    for code in maybe_codes:
                        new_classifications.append((scheme, code))
                else:
                    new_classifications.append((scheme, maybe_codes))
            supplier["classifications"] = tuple(new_classifications)
    return cf_list


def count_matching_fields(supplier_info, consumer_info, cf, required_supplier_fields):
    score = 0
    cf_supplier = cf.get("supplier", {})
    cf_consumer = cf.get("consumer", {})

    for key in required_supplier_fields:
        val1 = supplier_info.get(key)
        val2 = cf_supplier.get(key)

        if key == "classifications":
            if matches_classifications(val2 or [], val1 or []):
                score += 1
        elif val1 == val2:
            score += 1

    # Optional: add consumer-specific fields
    if "classifications" in cf_consumer and "classifications" in consumer_info:
        if matches_classifications(
            cf_consumer["classifications"], consumer_info["classifications"]
        ):
            score += 1

    return score


def hashable_dict(d):
    """Wrapper to hash a dictionary using make_hashable."""
    return make_hashable(d)


def build_cf_index(
    raw_cfs: list[dict], required_supplier_fields: set
) -> dict[str, dict[tuple, list[dict]]]:
    """
    Build a nested CF index:
        cf_index[consumer_location][supplier_signature] → list of CFs
    """
    index = defaultdict(lambda: defaultdict(list))

    for cf in raw_cfs:
        consumer_loc = cf.get("consumer", {}).get("location")
        if not consumer_loc:
            continue

        supplier = cf.get("supplier", {})
        # Create supplier signature
        sig = tuple(
            sorted((k, supplier[k]) for k in required_supplier_fields if k in supplier)
        )

        index[consumer_loc][sig].append(cf)

    return index


def is_aggregate_location(self, loc: str) -> bool:
    try:
        subregions = [
            g for g in self.geo.resolve(loc, containing=True) if g in self.weights
        ]
        return len(subregions) > 0
    except KeyError:
        return False


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
        scenario: Optional[str] = None,
        filepath: Optional[str] = None,
        allowed_functions: Optional[dict] = None,
        use_distributions: Optional[bool] = False,
        random_seed: Optional[int] = None,
        iterations: Optional[int] = 100,
    ):
        """
        Initialize the SpatialLCA class, ensuring `method` is not passed to
        `prepare_lca_inputs` while still being available in the class.
        """
        self.cf_index = None
        self.scenario_cfs = None
        self.method_metadata = None
        self.demand = demand
        self.weights = None
        self.consumer_lookup = None
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
        self.weight: str = weight

        # Accept both "parameters" and "scenarios" for flexibility
        self.parameters = parameters or {}

        self.scenario = scenario  # New: store default scenario
        self.scenario_length = validate_parameter_lengths(parameters=self.parameters)
        self.use_distributions = use_distributions
        self.iterations = iterations
        self.random_seed = random_seed if random_seed is not None else 42
        self.random_state = np.random.default_rng(self.random_seed)

        self.lca = bw2calc.LCA(demand=self.demand)
        self.load_raw_lcia_data()
        self.cfs_mapping = []

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            fh = logging.FileHandler("edgelcia.log")
            formatter = logging.Formatter(
                "%(asctime)s,%(msecs)d %(name)s %(funcName)s %(levelname)s %(message)s",
                datefmt="%H:%M:%S",
            )
            fh.setFormatter(formatter)
            fh.setLevel(logging.DEBUG)
            self.logger.addHandler(fh)
            self.logger.propagate = False

        self.SAFE_GLOBALS = {
            "__builtins__": None,
            "abs": abs,
            "max": max,
            "min": min,
            "round": round,
            "pow": pow,
            "sqrt": math.sqrt,
            "exp": math.exp,
            "log10": math.log10,
        }

        # Allow user-defined trusted functions explicitly
        if allowed_functions:
            self.SAFE_GLOBALS.update(allowed_functions)

        self.required_supplier_fields = {
            k
            for cf in self.raw_cfs_data
            for k in cf["supplier"].keys()
            if k not in {"matrix", "operator", "weight", "position"}
        }
        self._cached_supplier_keys = self.get_candidate_supplier_keys()

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

        # Build technosphere flow lookups as in the original implementation.
        self.position_to_technosphere_flows_lookup = {
            i["position"]: {k: i[k] for k in i if k != "position"}
            for i in self.technosphere_flows
        }

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
            raw = json.load(f)

        # Store full method metadata except exchanges and parameters
        self.raw_cfs_data, self.method_metadata = format_data(raw, self.weight)
        self.raw_cfs_data = normalize_cf_entries(self.raw_cfs_data)
        self.cfs_number = len(self.raw_cfs_data)

        # Extract parameters or scenarios from method file if not already provided
        if not self.parameters:
            self.parameters = raw.get("scenarios", raw.get("parameters", {}))

        # Fallback to default scenario
        if self.scenario and self.scenario not in self.parameters:
            raise ValueError(
                f"Scenario '{self.scenario}' not found in available parameters: {list(self.parameters)}"
            )

        grouping_mode = self.detect_cf_grouping_mode()
        self.cfs_lookup = preprocess_cfs(self.raw_cfs_data, by=grouping_mode)

        self.required_supplier_fields = {
            k
            for cf in self.raw_cfs_data
            for k in cf["supplier"].keys()
            if k not in {"matrix", "operator", "weight", "position"}
        }

        self.cf_index = build_cf_index(self.raw_cfs_data, self.required_supplier_fields)

    def initialize_weights(self):
        if self.weights is not None:
            return

        if not self.raw_cfs_data:
            self.weights = {}
            return

        self.weights = {}
        for cf in self.raw_cfs_data:
            consumer = cf.get("consumer", {})
            location = consumer.get("location")
            weight = cf.get("weight")
            if location and weight is not None:
                loc_str = get_str(location)
                self.weights[loc_str] = weight

        if hasattr(self, "_geo") and self._geo is not None:
            self._geo._cached_lookup.cache_clear()

    def get_candidate_supplier_keys(self):
        if hasattr(self, "_cached_supplier_keys"):
            return self._cached_supplier_keys

        grouping_mode = self.detect_cf_grouping_mode()
        cfs_lookup = preprocess_cfs(self.raw_cfs_data, by=grouping_mode)

        keys = set()
        for cf_list in cfs_lookup.values():
            for cf in cf_list:
                filtered = {
                    k: cf["supplier"].get(k)
                    for k in self.required_supplier_fields
                    if cf["supplier"].get(k) is not None
                }

                # Normalize classification field
                if "classifications" in filtered:
                    c = filtered["classifications"]
                    if isinstance(c, dict):
                        filtered["classifications"] = tuple(
                            (scheme, tuple(vals)) for scheme, vals in sorted(c.items())
                        )
                    elif isinstance(c, list):
                        filtered["classifications"] = tuple(c)

                keys.add(make_hashable(filtered))

        self._cached_supplier_keys = keys
        return keys

    def detect_cf_grouping_mode(self):
        has_consumer_locations = any(
            "location" in cf.get("consumer", {}) for cf in self.raw_cfs_data
        )
        has_supplier_locations = any(
            "location" in cf.get("supplier", {}) for cf in self.raw_cfs_data
        )
        if has_consumer_locations and not has_supplier_locations:
            return "consumer"
        elif has_supplier_locations and not has_consumer_locations:
            return "supplier"
        else:
            return "both"

    def sample_cf_distribution(self, cf: dict, n: int) -> np.ndarray:
        """
        Generate n random CF values from the distribution info in the 'uncertainty' key.
        Falls back to a constant value if no uncertainty. If 'negative' == 1, samples are negated.
        """
        if not self.use_distributions or cf.get("uncertainty") is None:
            # If value is a string (expression), evaluate once
            value = cf["value"]
            if isinstance(value, str):
                value = safe_eval(
                    expr=value,
                    parameters=self.parameters,
                    scenario_idx=0,
                    SAFE_GLOBALS=self.SAFE_GLOBALS,
                )
            return np.full(n, value, dtype=float)

        unc = cf["uncertainty"]
        dist_name = unc["distribution"]
        params = unc["parameters"]
        negative = unc.get("negative", 0)

        if dist_name == "discrete_empirical":
            values = np.array(params["values"])
            weights = np.array(params["weights"])
            if weights.sum() != 0:
                weights = weights / weights.sum()
            samples = self.random_state.choice(values, size=n, p=weights)

        elif dist_name == "uniform":
            samples = self.random_state.uniform(
                params["minimum"], params["maximum"], size=n
            )

        elif dist_name == "triang":
            left = params["minimum"]
            mode = params["loc"]
            right = params["maximum"]
            samples = self.random_state.triangular(left, mode, right, size=n)

        elif dist_name == "normal":
            samples = self.random_state.normal(
                loc=params["loc"], scale=params["scale"], size=n
            )
            samples = np.clip(samples, params["minimum"], params["maximum"])

        elif dist_name == "lognorm":
            s = params["shape_a"]
            loc = params["loc"]
            scale = params["scale"]
            samples = stats.lognorm.rvs(
                s=s, loc=loc, scale=scale, size=n, random_state=self.random_state
            )
            samples = np.clip(samples, params["minimum"], params["maximum"])

        elif dist_name == "beta":
            a = params["shape_a"]
            b = params["shape_b"]
            x = self.random_state.beta(a, b, size=n)
            samples = params["loc"] + x * params["scale"]
            samples = np.clip(samples, params["minimum"], params["maximum"])

        elif dist_name == "gamma":
            samples = (
                self.random_state.gamma(params["shape_a"], params["scale"], size=n)
                + params["loc"]
            )
            samples = np.clip(samples, params["minimum"], params["maximum"])

        elif dist_name == "weibull_min":
            c = params["shape_a"]
            loc = params["loc"]
            scale = params["scale"]
            samples = stats.weibull_min.rvs(
                c=c, loc=loc, scale=scale, size=n, random_state=self.random_state
            )
            samples = np.clip(samples, params["minimum"], params["maximum"])

        else:
            samples = np.full(n, cf["value"], dtype=float)

        return -samples if negative else samples

    def resolve_param(self, param, key):
        if isinstance(param, dict):
            return param.get(key, list(param.values())[-1])  # fallback to last value
        return param

    def resolve_parameters_for_scenario(
        self, scenario_idx: int, scenario_name: Optional[str] = None
    ) -> dict:
        scenario_name = scenario_name or self.scenario

        param_set = self.parameters.get(scenario_name)

        resolved = {}
        if param_set is not None:
            for k, v in param_set.items():
                if isinstance(v, dict):
                    resolved[k] = v.get(str(scenario_idx), list(v.values())[-1])
                else:
                    resolved[k] = v
        return resolved

    def evaluate_cfs(self, scenario_idx=0, scenario=None):
        if self.use_distributions and self.iterations > 1:
            coords_i, coords_j, coords_k = [], [], []
            data = []

            for cf in self.cfs_mapping:
                samples = self.sample_cf_distribution(cf, n=self.iterations)
                for i, j in cf["positions"]:
                    for k in range(self.iterations):
                        coords_i.append(i)
                        coords_j.append(j)
                        coords_k.append(k)
                        data.append(samples[k])

            matrix_type = (
                "biosphere" if len(self.biosphere_edges) > 0 else "technosphere"
            )
            n_rows, n_cols = (
                self.lca.inventory.shape
                if matrix_type == "biosphere"
                else self.lca.technosphere_matrix.shape
            )

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
            if self.scenario is not None:
                scenario_name = self.scenario
            if scenario is not None:
                scenario_name = scenario
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

                matrix_type = (
                    "biosphere" if len(self.biosphere_edges) > 0 else "technosphere"
                )
                self.characterization_matrix = initialize_lcia_matrix(
                    self.lca, matrix_type=matrix_type
                )

                for cf in self.scenario_cfs:
                    for i, j in cf["positions"]:
                        self.characterization_matrix[i, j] = cf["value"]

                self.characterization_matrix = self.characterization_matrix.tocsr()

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

        print(
            f"Processed edges: {len(self.processed_biosphere_edges) + len(self.processed_technosphere_edges)}"
        )

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

    def equality_supplier_signature(self, supplier_info: dict) -> tuple:
        """
        Return a hashable supplier signature based on required supplier fields.
        If no fields are required, return an empty tuple (matches any supplier).
        """
        if not self.required_supplier_fields:
            return ()

        filtered = {
            k: supplier_info.get(k)
            for k in self.required_supplier_fields
            if supplier_info.get(k) is not None
        }

        # Normalize classifications
        if "classifications" in filtered:
            classifications = filtered["classifications"]

            # Brightway format: list or tuple of (scheme, code)
            if isinstance(classifications, (list, tuple)):
                try:
                    filtered["classifications"] = tuple(
                        sorted((str(s), str(c)) for s, c in classifications)
                    )
                except Exception:
                    self.logger.warning(
                        f"Malformed classification tuples: {classifications}"
                    )
                    filtered["classifications"] = ()

            # CF format: dict of scheme -> list of codes
            elif isinstance(classifications, dict):
                filtered["classifications"] = tuple(
                    (scheme, tuple(sorted(map(str, codes))))
                    for scheme, codes in sorted(classifications.items())
                )
            else:
                self.logger.warning(
                    f"Unexpected classifications format: {classifications}"
                )
                filtered["classifications"] = ()

        return make_hashable(filtered)

    def map_aggregate_locations(self) -> None:
        self.initialize_weights()
        print("Handling static regions...")

        if not hasattr(self, "_unmatched_hashes"):
            self._unmatched_hashes = set()

        cf_operators = {
            cf["supplier"].get("operator", "equals") for cf in self.raw_cfs_data
        }

        candidate_supplier_keys = self._cached_supplier_keys

        for direction in ["biosphere-technosphere", "technosphere-technosphere"]:
            if direction == "biosphere-technosphere":
                unprocessed_edges = self.unprocessed_biosphere_edges
                processed_flows = self.processed_biosphere_edges
            else:
                unprocessed_edges = self.unprocessed_technosphere_edges
                processed_flows = self.processed_technosphere_edges

            edges_index = defaultdict(list)
            processed_flows = set(processed_flows)
            for supplier_idx, consumer_idx in unprocessed_edges:
                if (supplier_idx, consumer_idx) in processed_flows:
                    continue

                consumer_loc = dict(self.reversed_consumer_lookup[consumer_idx]).get(
                    "location"
                )
                if not consumer_loc:
                    flow_info = self.position_to_technosphere_flows_lookup.get(
                        consumer_idx, {}
                    )
                    consumer_loc = flow_info.get("location")

                if not consumer_loc:
                    continue

                edges_index[consumer_loc].append((supplier_idx, consumer_idx))

            prefiltered_groups = defaultdict(list)
            remaining_edges = []

            location_to_candidates = self.geo.batch(
                locations=list(edges_index.keys()),
                containing=True,
            )

            for location, edges in edges_index.items():
                if location in ["RoW", "RoE"]:
                    continue

                try:
                    subregions = [
                        g
                        for g in self.geo.resolve(location, containing=True)
                        if g in self.weights
                    ]

                    if len(subregions) == 0:
                        continue
                except KeyError:
                    self.logger.warning(f"Geometry lookup failed for: {location}")
                    continue

                candidate_locations = subregions

                if not candidate_locations:
                    continue

                for supplier_idx, consumer_idx in edges:
                    if (supplier_idx, consumer_idx) in processed_flows:
                        continue

                    supplier_info = dict(self.reversed_supplier_lookup[supplier_idx])
                    consumer_info = self.complete_consumer_info(consumer_idx)

                    if "location" not in consumer_info:
                        fallback = self.position_to_technosphere_flows_lookup.get(
                            consumer_idx, {}
                        )
                        if fallback and "location" in fallback:
                            consumer_info["location"] = fallback["location"]

                    sig = self.equality_supplier_signature(supplier_info)

                    if sig in candidate_supplier_keys:
                        prefiltered_groups[sig].append(
                            (
                                supplier_idx,
                                consumer_idx,
                                supplier_info,
                                consumer_info,
                                candidate_locations,
                            )
                        )
                    else:
                        if any(x in cf_operators for x in ["contains", "startswith"]):
                            remaining_edges.append(
                                (
                                    supplier_idx,
                                    consumer_idx,
                                    supplier_info,
                                    consumer_info,
                                    candidate_locations,
                                )
                            )
                        else:
                            self.logger.info(
                                f"No match or fallback for edge ({supplier_idx}, {consumer_idx})"
                            )

            # Pass 1
            for sig, group_edges in tqdm(
                prefiltered_groups.items(), desc="Processing static groups (pass 1)"
            ):
                supplier_info = group_edges[0][2]
                consumer_info = group_edges[0][3]
                candidate_locations = group_edges[0][-1]

                # Determine whether to assign candidate locations to supplier or consumer
                candidate_suppliers = (
                    candidate_locations
                    if (
                        "location" in self.required_supplier_fields
                        and self.geo.resolve(
                            supplier_info.get("location"), containing=True
                        )
                    )
                    else []
                )

                candidate_consumers = (
                    candidate_locations
                    if (
                        "location" in self.required_consumer_fields
                        and self.geo.resolve(
                            consumer_info.get("location"), containing=True
                        )
                    )
                    else []
                )

                new_cf, matched_cf_obj = compute_average_cf(
                    candidate_suppliers=candidate_suppliers,
                    candidate_consumers=candidate_consumers,
                    supplier_info=supplier_info,
                    consumer_info=consumer_info,
                    weight=self.weights,
                    required_supplier_fields=self.required_supplier_fields,
                    required_consumer_fields=self.required_consumer_fields,
                    cf_index=self.cf_index,
                    logger=self.logger,
                )

                if new_cf != 0:
                    for (
                        supplier_idx,
                        consumer_idx,
                        supplier_info,
                        consumer_info,
                        _,
                    ) in group_edges:
                        add_cf_entry(
                            cfs_mapping=self.cfs_mapping,
                            supplier_info=supplier_info,
                            consumer_info=consumer_info,
                            direction=direction,
                            indices=[(supplier_idx, consumer_idx)],
                            value=new_cf,
                            uncertainty=None,
                        )

            # Pass 2
            @lru_cache(maxsize=None)
            def compute_cf_memoized(
                supplier_key,
                consumer_key,
                candidate_suppliers_key,
                candidate_consumers_key,
            ):
                supplier_info = dict(supplier_key)
                consumer_info = dict(consumer_key)
                candidate_suppliers = list(candidate_suppliers_key)
                candidate_consumers = list(candidate_consumers_key)

                return compute_average_cf(
                    candidate_suppliers=candidate_suppliers,
                    candidate_consumers=candidate_consumers,
                    supplier_info=supplier_info,
                    consumer_info=consumer_info,
                    weight=self.weights,
                    cf_index=self.cf_index,
                    required_supplier_fields=self.required_supplier_fields,
                    required_consumer_fields=self.required_consumer_fields,
                    logger=self.logger,
                )

            # Group remaining edges by shared (supplier, consumer, candidate locations) signature
            grouped_edges = defaultdict(list)

            for (
                supplier_idx,
                consumer_idx,
                supplier_info,
                consumer_info,
                candidate_locations,
            ) in remaining_edges:
                # 1. Strip irrelevant keys
                filtered_supplier = {
                    k: supplier_info[k]
                    for k in self.required_supplier_fields
                    if k in supplier_info
                }
                filtered_consumer = {
                    k: consumer_info[k]
                    for k in self.required_consumer_fields
                    if k in consumer_info
                }

                # 2. Normalize classifications
                if "classifications" in filtered_supplier:
                    filtered_supplier["classifications"] = tuple(
                        sorted(filtered_supplier["classifications"])
                    )
                if "classifications" in filtered_consumer:
                    filtered_consumer["classifications"] = tuple(
                        sorted(filtered_consumer["classifications"])
                    )

                # 3. Sorted location keys
                # Determine if either side has an aggregate location
                supplier_loc = supplier_info.get("location")
                consumer_loc = consumer_info.get("location")

                # Get subregions (i.e. decomposed countries) for each if needed
                supplier_subregions = []
                consumer_subregions = []

                if "location" in self.required_supplier_fields:
                    try:
                        supplier_subregions = [
                            g
                            for g in self.geo.resolve(supplier_loc, containing=True)
                            if g in self.weights
                        ]
                    except KeyError:
                        pass

                if "location" in self.required_consumer_fields:
                    try:
                        consumer_subregions = [
                            g
                            for g in self.geo.resolve(consumer_loc, containing=True)
                            if g in self.weights
                        ]
                    except KeyError:
                        pass

                # Only assign if they are meaningful (i.e., they represent aggregate regions)
                candidate_suppliers = supplier_subregions if supplier_subregions else []
                candidate_consumers = consumer_subregions if consumer_subregions else []

                # The location key must remain hashable for grouping
                loc_key = (
                    tuple(sorted(set(candidate_suppliers))),
                    tuple(sorted(set(candidate_consumers))),
                )

                # 4. Make hashable keys
                s_key = make_hashable(filtered_supplier)
                c_key = make_hashable(filtered_consumer)

                grouped_edges[(s_key, c_key, loc_key)].append(
                    (
                        supplier_idx,
                        consumer_idx,
                        candidate_suppliers,
                        candidate_consumers,
                    )
                )

            for (s_key, c_key, loc_key), edge_group in tqdm(
                grouped_edges.items(),
                desc="Processing static groups (pass 2)",
            ):
                # Use candidates from first edge in group (they're grouped by identical keys)
                _, _, candidate_suppliers, candidate_consumers = edge_group[0]

                new_cf, matched_cf_obj = compute_cf_memoized(
                    s_key, c_key, tuple(candidate_suppliers), tuple(candidate_consumers)
                )
                if new_cf != 0:
                    for supplier_idx, consumer_idx, *_ in edge_group:
                        add_cf_entry(
                            cfs_mapping=self.cfs_mapping,
                            supplier_info=dict(s_key),
                            consumer_info=dict(c_key),
                            direction=direction,
                            indices=[(supplier_idx, consumer_idx)],
                            value=new_cf,
                            uncertainty=None,
                        )

        self.update_unprocessed_edges()

    def map_dynamic_locations(self) -> None:
        """
        Handle dynamic regions (RoW and RoE) and update CF data, pre-filtering edges
        by supplier signature (using all required supplier fields except location). In Pass 1,
        edges whose supplier info exactly matches a candidate signature are grouped and processed.
        In Pass 2, edges that did not pass prefiltering (and whose operator is not "equals") are
        processed individually using full matching logic.
        """
        self.initialize_weights()

        cf_operators = {
            cf["supplier"].get("operator", "equals") for cf in self.raw_cfs_data
        }

        # Fetch candidate supplier signatures
        candidate_supplier_keys = self._cached_supplier_keys

        for flow in self.technosphere_flows:
            key = (flow["name"], flow.get("reference product"))
            self.technosphere_flows_lookup[key].append(flow["location"])

        raw_exclusion_locs = set()
        for locs in self.technosphere_flows_lookup.values():
            raw_exclusion_locs.update(loc for loc in locs if loc not in ["RoW", "RoE"])

        decomposed_exclusions = self.geo.batch(
            locations=list(raw_exclusion_locs), containing=True
        )

        print("Handling dynamic regions...")

        def consumer_act_signature(consumer_idx) -> tuple:
            consumer_act = self.position_to_technosphere_flows_lookup.get(
                consumer_idx, {}
            )
            return (consumer_act.get("name"), consumer_act.get("reference product"))

        for direction in ["biosphere-technosphere", "technosphere-technosphere"]:
            prefiltered_groups = defaultdict(list)
            remaining_edges = []
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

            processed_flows = set(processed_flows)
            for supplier_idx, consumer_idx in unprocessed_edges:
                if (supplier_idx, consumer_idx) in processed_flows:
                    continue

                consumer_info = self.complete_consumer_info(consumer_idx)
                location = consumer_info.get("location")
                if location not in ["RoW", "RoE"]:
                    continue

                supplier_info = dict(self.reversed_supplier_lookup[supplier_idx])
                sig = self.equality_supplier_signature(supplier_info)
                cons_act_sig = consumer_act_signature(consumer_idx)

                if sig in candidate_supplier_keys:
                    prefiltered_groups[(sig, cons_act_sig)].append(
                        (supplier_idx, consumer_idx, supplier_info, consumer_info)
                    )
                else:
                    if any(x in cf_operators for x in ["contains", "startswith"]):
                        remaining_edges.append(
                            (
                                supplier_idx,
                                consumer_idx,
                                supplier_info,
                                consumer_info,
                                cons_act_sig,
                            )
                        )

            if prefiltered_groups:
                for (supp_sig, cons_act_sig), group_edges in tqdm(
                    prefiltered_groups.items(),
                    desc="Processing dynamic groups (pass 1)",
                ):
                    rep_supplier = group_edges[0][2]
                    name, reference_product = cons_act_sig

                    other_than_RoW_RoE = [
                        loc
                        for loc in self.technosphere_flows_lookup.get(
                            (name, reference_product), []
                        )
                        if loc not in ["RoW", "RoE"] and loc in self.weights
                    ]

                    other_than_RoW_RoE_decomposed = []
                    for loc in other_than_RoW_RoE:
                        locs = decomposed_exclusions.get(loc)
                        if locs:
                            other_than_RoW_RoE_decomposed.extend(locs)
                        else:
                            other_than_RoW_RoE_decomposed.append(loc)

                    candidate_suppliers = []
                    candidate_consumers = self.geo.resolve(
                        location="GLO",
                        containing=True,
                        exceptions=other_than_RoW_RoE_decomposed,
                    )

                    new_cf, matched_cf_obj = compute_average_cf(
                        candidate_suppliers=candidate_suppliers,
                        candidate_consumers=candidate_consumers,
                        supplier_info=rep_supplier,
                        consumer_info=group_edges[0][3],
                        weight=self.weights,
                        cf_index=self.cf_index,
                        required_supplier_fields=self.required_supplier_fields,
                        required_consumer_fields=self.required_consumer_fields,
                        logger=self.logger,
                    )
                    if new_cf:
                        for (
                            supplier_idx,
                            consumer_idx,
                            supplier_info,
                            consumer_info,
                        ) in group_edges:
                            add_cf_entry(
                                cfs_mapping=self.cfs_mapping,
                                supplier_info=supplier_info,
                                consumer_info=consumer_info,
                                direction=direction,
                                indices=[(supplier_idx, consumer_idx)],
                                value=new_cf,
                                uncertainty=None,
                            )

            if remaining_edges:

                @lru_cache(maxsize=None)
                def compute_cf_memoized(s_key, c_key, candidate_supp, candidate_cons):
                    return compute_average_cf(
                        candidate_suppliers=list(candidate_supp),
                        candidate_consumers=list(candidate_cons),
                        supplier_info=dict(s_key),
                        consumer_info=dict(c_key),
                        weight=self.weights,
                        cf_index=self.cf_index,
                        required_supplier_fields=self.required_supplier_fields,
                        required_consumer_fields=self.required_consumer_fields,
                        logger=self.logger,
                    )

                grouped_edges = defaultdict(list)

                for (
                    supplier_idx,
                    consumer_idx,
                    supplier_info,
                    consumer_info,
                    cons_act_sig,
                ) in remaining_edges:
                    name, reference_product = cons_act_sig

                    other_than_RoW_RoE = [
                        loc
                        for loc in self.technosphere_flows_lookup.get(
                            (name, reference_product), []
                        )
                        if loc not in ["RoW", "RoE"] and loc in self.weights
                    ]

                    other_than_RoW_RoE_decomposed = []
                    for loc in other_than_RoW_RoE:
                        locs = decomposed_exclusions.get(loc)
                        if locs:
                            other_than_RoW_RoE_decomposed.extend(locs)
                        else:
                            other_than_RoW_RoE_decomposed.append(loc)

                    candidate_suppliers = []
                    candidate_consumers = self.geo.resolve(
                        location="GLO",
                        containing=True,
                        exceptions=tuple(other_than_RoW_RoE_decomposed),
                    )

                    filtered_supplier = {
                        k: supplier_info[k]
                        for k in self.required_supplier_fields
                        if k in supplier_info
                    }
                    filtered_consumer = {
                        k: consumer_info[k]
                        for k in self.required_consumer_fields
                        if k in consumer_info
                    }

                    if "classifications" in filtered_supplier:
                        filtered_supplier["classifications"] = tuple(
                            sorted(filtered_supplier["classifications"])
                        )
                    if "classifications" in filtered_consumer:
                        filtered_consumer["classifications"] = tuple(
                            sorted(filtered_consumer["classifications"])
                        )

                    s_key = make_hashable(filtered_supplier)
                    c_key = make_hashable(filtered_consumer)

                    grouped_edges[
                        (
                            s_key,
                            c_key,
                            tuple(candidate_suppliers),
                            tuple(candidate_consumers),
                        )
                    ].append((supplier_idx, consumer_idx))

                for (s_key, c_key, supp_locs, cons_locs), edge_group in tqdm(
                    grouped_edges.items(),
                    desc="Processing remaining dynamic edges (batched pass 2)",
                ):
                    new_cf, matched_cf_obj = compute_cf_memoized(
                        s_key, c_key, supp_locs, cons_locs
                    )
                    if new_cf:
                        for supplier_idx, consumer_idx in edge_group:
                            add_cf_entry(
                                cfs_mapping=self.cfs_mapping,
                                supplier_info=dict(s_key),
                                consumer_info=dict(c_key),
                                direction=direction,
                                indices=[(supplier_idx, consumer_idx)],
                                value=new_cf,
                                uncertainty=None,
                            )

        self.update_unprocessed_edges()

    def map_contained_locations(self) -> None:
        """
        Handle contained locations (static regions) while pre-filtering unprocessed edges.
        Edges whose supplier info (based on required supplier fields excluding location)
        doesn't appear in the CF lookup are skipped.
        """
        self.initialize_weights()

        cfs_lookup = preprocess_cfs(
            self.raw_cfs_data, by=self.detect_cf_grouping_mode()
        )

        cf_operators = {
            cf["supplier"].get("operator", "equals") for cf in self.raw_cfs_data
        }

        candidate_supplier_keys = self._cached_supplier_keys

        consumer_locations = set()
        for _, consumer_idx in (
            self.unprocessed_biosphere_edges + self.unprocessed_technosphere_edges
        ):
            info = self.complete_consumer_info(consumer_idx)
            loc = info.get("location")
            if loc and loc not in ("RoW", "RoE"):
                consumer_locations.add(loc)

        contained_by_lookup = self.geo.batch(
            locations=list(consumer_locations), containing=False
        )

        print("Handling contained locations...")

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

            groups_direction = defaultdict(list)
            remaining_edges = []

            processed_flows = set(processed_flows)
            for supplier_idx, consumer_idx in unprocessed_edges:
                if (supplier_idx, consumer_idx) in processed_flows:
                    continue
                supplier_info = dict(self.reversed_supplier_lookup[supplier_idx])
                consumer_info = self.complete_consumer_info(consumer_idx)

                consumer_location = consumer_info.get("location")

                sig = self.equality_supplier_signature(supplier_info)
                if sig in candidate_supplier_keys:
                    groups_direction[(sig, consumer_location)].append(
                        (supplier_idx, consumer_idx, supplier_info, consumer_info)
                    )
                else:
                    if any(x in cf_operators for x in ["contains", "startswith"]):
                        remaining_edges.append(
                            (supplier_idx, consumer_idx, supplier_info, consumer_info)
                        )

            if len(groups_direction) > 0:
                for (supp_sig, cons_loc), group_edges in tqdm(
                    groups_direction.items(),
                    desc="Processing contained groups (pass 1)",
                ):
                    candidate_locations = contained_by_lookup.get(cons_loc, [])
                    candidate_locations = [
                        loc for loc in candidate_locations if loc in self.weights
                    ]
                    if not candidate_locations:
                        continue

                    rep_supplier = group_edges[0][2]
                    new_cf, matched_cf_obj = compute_average_cf(
                        candidate_suppliers=[],
                        candidate_consumers=candidate_locations,
                        supplier_info=rep_supplier,
                        consumer_info=group_edges[0][-1],
                        weight=self.weights,
                        cf_index=self.cf_index,
                        required_supplier_fields=self.required_supplier_fields,
                        required_consumer_fields=self.required_consumer_fields,
                        logger=self.logger,
                    )

                    if new_cf != 0:
                        for (
                            supplier_idx,
                            consumer_idx,
                            supplier_info,
                            consumer_info,
                        ) in group_edges:
                            add_cf_entry(
                                cfs_mapping=self.cfs_mapping,
                                supplier_info=supplier_info,
                                consumer_info=consumer_info,
                                direction=direction,
                                indices=[(supplier_idx, consumer_idx)],
                                value=new_cf,
                                uncertainty=None,
                            )

            @lru_cache(maxsize=None)
            def compute_cf_memoized(
                s_key, c_key, supplier_candidates, consumer_candidates
            ):
                return compute_average_cf(
                    candidate_suppliers=list(supplier_candidates),
                    candidate_consumers=list(consumer_candidates),
                    supplier_info=dict(s_key),
                    consumer_info=dict(c_key),
                    weight=self.weights,
                    cf_index=self.cf_index,
                    required_supplier_fields=self.required_supplier_fields,
                    required_consumer_fields=self.required_consumer_fields,
                    logger=self.logger,
                )

            grouped_edges = defaultdict(list)
            for (
                supplier_idx,
                consumer_idx,
                supplier_info,
                consumer_info,
            ) in remaining_edges:
                consumer_location = consumer_info.get("location")
                candidate_locations = self.geo.resolve(
                    location=consumer_location, containing=False
                )
                candidate_locations = [
                    loc for loc in candidate_locations if loc in self.weights
                ]
                if not candidate_locations:
                    continue

                filtered_supplier = {
                    k: supplier_info[k]
                    for k in self.required_supplier_fields
                    if k in supplier_info
                }
                filtered_consumer = {
                    k: consumer_info[k]
                    for k in self.required_consumer_fields
                    if k in consumer_info
                }

                if "classifications" in filtered_supplier:
                    filtered_supplier["classifications"] = tuple(
                        sorted(filtered_supplier["classifications"])
                    )
                if "classifications" in filtered_consumer:
                    filtered_consumer["classifications"] = tuple(
                        sorted(filtered_consumer["classifications"])
                    )

                s_key = make_hashable(filtered_supplier)
                c_key = make_hashable(filtered_consumer)
                loc_key = tuple(sorted(candidate_locations))

                grouped_edges[(s_key, c_key, loc_key)].append(
                    (supplier_idx, consumer_idx)
                )

            for (s_key, c_key, loc_key), edge_group in tqdm(
                grouped_edges.items(),
                desc="Processing remaining contained edges (batched pass 2)",
            ):
                new_cf, matched_cf_obj = compute_cf_memoized(
                    s_key,
                    c_key,
                    tuple([]),  # Supplier candidates = []
                    loc_key,  # Consumer candidates = derived from containment
                )
                if new_cf != 0:
                    for supplier_idx, consumer_idx in edge_group:
                        add_cf_entry(
                            cfs_mapping=self.cfs_mapping,
                            supplier_info=dict(s_key),
                            consumer_info=dict(c_key),
                            direction=direction,
                            indices=[(supplier_idx, consumer_idx)],
                            value=new_cf,
                            uncertainty=None,
                        )

        self.update_unprocessed_edges()

    def map_remaining_locations_to_global(self) -> None:
        """
        Handle remaining exchanges by assigning global CFs while pre-filtering
        unprocessed edges based on supplier signature.
        """
        self.initialize_weights()

        cf_operators = {
            cf["supplier"].get("operator", "equals") for cf in self.raw_cfs_data
        }

        candidate_supplier_keys = self._cached_supplier_keys

        locations = self.geo.batch(locations=["GLO"], containing=True)
        global_locations = locations.get("GLO", [])

        print("Handling remaining exchanges...")

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

            prefiltered_groups = defaultdict(list)
            remaining_edges = []
            processed_flows = set(processed_flows)

            for supplier_idx, consumer_idx in unprocessed_edges:
                if (supplier_idx, consumer_idx) in processed_flows:
                    continue
                supplier_info = dict(self.reversed_supplier_lookup[supplier_idx])
                consumer_info = self.complete_consumer_info(consumer_idx)
                sig = self.equality_supplier_signature(supplier_info)
                if sig in candidate_supplier_keys:
                    prefiltered_groups[sig].append(
                        (supplier_idx, consumer_idx, supplier_info, consumer_info)
                    )
                else:
                    if any(x in cf_operators for x in ["contains", "startswith"]):
                        remaining_edges.append(
                            (supplier_idx, consumer_idx, supplier_info, consumer_info)
                        )

            @lru_cache(maxsize=None)
            def compute_cf_memoized(s_key, c_key, supp_candidates, cons_candidates):
                return compute_average_cf(
                    candidate_suppliers=list(supp_candidates),
                    candidate_consumers=list(cons_candidates),
                    supplier_info=dict(s_key),
                    consumer_info=dict(c_key),
                    weight=self.weights,
                    cf_index=self.cf_index,
                    required_supplier_fields=self.required_supplier_fields,
                    required_consumer_fields=self.required_consumer_fields,
                    logger=self.logger,
                )

            grouped_edges = defaultdict(list)

            for (
                supplier_idx,
                consumer_idx,
                supplier_info,
                consumer_info,
            ) in remaining_edges:
                candidate_locations = [
                    loc for loc in global_locations if loc in self.weights
                ]
                if not candidate_locations:
                    continue

                filtered_supplier = {
                    k: supplier_info[k]
                    for k in self.required_supplier_fields
                    if k in supplier_info
                }
                filtered_consumer = {
                    k: consumer_info[k]
                    for k in self.required_consumer_fields
                    if k in consumer_info
                }

                if "classifications" in filtered_supplier:
                    filtered_supplier["classifications"] = tuple(
                        sorted(filtered_supplier["classifications"])
                    )
                if "classifications" in filtered_consumer:
                    filtered_consumer["classifications"] = tuple(
                        sorted(filtered_consumer["classifications"])
                    )

                s_key = make_hashable(filtered_supplier)
                c_key = make_hashable(filtered_consumer)
                loc_key = tuple(sorted(candidate_locations))

                grouped_edges[(s_key, c_key, loc_key)].append(
                    (supplier_idx, consumer_idx)
                )

            for (s_key, c_key, loc_key), edge_group in tqdm(
                grouped_edges.items(),
                desc="Processing remaining global edges (batched pass 2)",
            ):
                new_cf, matched_cf_obj = compute_cf_memoized(
                    s_key, c_key, tuple(global_locations), tuple(global_locations)
                )
                if new_cf != 0:
                    for supplier_idx, consumer_idx in edge_group:
                        add_cf_entry(
                            cfs_mapping=self.cfs_mapping,
                            supplier_info=dict(s_key),
                            consumer_info=dict(c_key),
                            direction=direction,
                            indices=[(supplier_idx, consumer_idx)],
                            value=new_cf,
                            uncertainty=(
                                matched_cf_obj.get("uncertainty")
                                if matched_cf_obj
                                else None
                            ),
                        )

        self.update_unprocessed_edges()

    def preprocess_lookups(self):
        """
        Preprocess supplier and consumer flows into lookup dictionaries.
        """

        # Constants for ignored fields
        IGNORED_FIELDS = {"matrix", "operator", "weight", "classifications"}

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

    def complete_consumer_info(self, consumer_idx):
        """
        Returns a dict with consumer flow info, including a fallback for missing fields
        like 'location' from the position_to_technosphere_flows_lookup.
        """
        info = dict(self.reversed_consumer_lookup.get(consumer_idx, {}))
        if "location" not in info:
            fallback = self.position_to_technosphere_flows_lookup.get(consumer_idx, {})
            if fallback and "location" in fallback:
                info["location"] = fallback["location"]
        return info

    def map_exchanges(self):
        self.initialize_weights()
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

        seen_positions = []

        for cf in tqdm(self.raw_cfs_data):
            supplier_criteria = cf["supplier"]
            consumer_criteria = cf["consumer"]

            # --- Supplier matching ---
            if "classifications" in supplier_criteria:
                # Do manual matching for hierarchical classifications
                cf_class = supplier_criteria["classifications"]
                supplier_candidates = [
                    idx
                    for idx in self.reversed_supplier_lookup
                    if matches_classifications(
                        cf_class,
                        dict(self.reversed_supplier_lookup[idx]).get(
                            "classifications", []
                        ),
                    )
                ]
            else:
                cached_match_with_index.index = supplier_index
                cached_match_with_index.lookup_mapping = self.supplier_lookup
                cached_match_with_index.reversed_lookup = self.reversed_supplier_lookup
                supplier_candidates = cached_match_with_index(
                    make_hashable(supplier_criteria),
                    tuple(sorted(self.required_supplier_fields)),
                )

            # --- Consumer matching ---
            if not any(
                k
                for k in consumer_criteria
                if k not in {"matrix", "weight", "position"}
            ):
                consumer_candidates = list(self.consumer_lookup.values())
                consumer_candidates = [
                    pos for sublist in consumer_candidates for pos in sublist
                ]
            else:
                cached_match_with_index.index = consumer_index
                cached_match_with_index.lookup_mapping = self.consumer_lookup
                cached_match_with_index.reversed_lookup = (
                    self.position_to_technosphere_flows_lookup
                )
                consumer_candidates = cached_match_with_index(
                    make_hashable(consumer_criteria),
                    tuple(sorted(self.required_consumer_fields)),
                )

            # --- Combine supplier + consumer ---
            positions = [
                (supplier, consumer)
                for supplier in supplier_candidates
                for consumer in consumer_candidates
                if (supplier, consumer) in edges
            ]

            positions = [pos for pos in positions if pos not in seen_positions]

            if positions:
                cf_entry = {
                    "supplier": supplier_criteria,
                    "consumer": consumer_criteria,
                    "direction": f"{supplier_criteria['matrix']}-{consumer_criteria['matrix']}",
                    "positions": positions,
                    "value": cf["value"],
                    "uncertainty": cf.get("uncertainty"),
                }

                self.cfs_mapping.append(cf_entry)
                seen_positions.extend(positions)

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
        rows.append(["CFs in method", self.cfs_number])
        rows.append(
            [
                "CFs used",
                len([x["value"] for x in self.cfs_mapping if len(x["positions"]) > 0]),
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

    def lcia(self) -> None:
        """
        Calculate the LCIA score.

        - If `characterization_matrix` is 3D (i.e. uncertainty with multiple iterations),
          compute one LCIA score per iteration and return an array.
        - If `characterization_matrix` is 2D (deterministic case), compute a single LCIA score.
        """

        # check that teh sum of processed biosphere and technosphere
        # edges is superior to zero, otherwise, we exit
        if (
            len(self.processed_biosphere_edges) + len(self.processed_technosphere_edges)
            == 0
        ):
            self.score = 0
            return

        is_biosphere = len(self.biosphere_edges) > 0

        if self.use_distributions and self.iterations > 1:
            inventory = (
                self.lca.inventory if is_biosphere else self.technosphere_flow_matrix
            )

            # Convert 2D inventory to sparse.COO
            inventory_coo = sparse.COO.from_scipy_sparse(inventory)

            # Broadcast inventory shape for multiplication
            inv_expanded = inventory_coo[:, :, None]  # (i, j, 1)

            # Element-wise multiply
            characterized = self.characterization_matrix * inv_expanded

            # Sum across dimensions i and j to get 1 value per iteration
            self.characterized_inventory = characterized
            self.score = characterized.sum(axis=(0, 1))

        else:
            inventory = (
                self.lca.inventory if is_biosphere else self.technosphere_flow_matrix
            )
            self.characterized_inventory = self.characterization_matrix.multiply(
                inventory
            )
            self.score = self.characterized_inventory.sum()

    def generate_cf_table(self) -> pd.DataFrame:
        """
        Generate a pandas DataFrame with the evaluated characterization factors
        used in the current scenario. If distributions were used, show summary statistics.
        """

        if not self.scenario_cfs:
            print("You must run evaluate_cfs() first.")
            return

        is_biosphere = True if self.technosphere_flow_matrix is None else False

        inventory = (
            self.lca.inventory if is_biosphere else self.technosphere_flow_matrix
        )
        data = []

        if (
            self.use_distributions
            and hasattr(self, "characterization_matrix")
            and hasattr(self, "iterations")
        ):
            cm = self.characterization_matrix

            for i, j in zip(
                *cm.sum(axis=2).nonzero()
            ):  # Only loop over nonzero entries
                consumer = bw2data.get_activity(self.reversed_activity[j])
                supplier = (
                    bw2data.get_activity(self.reversed_biosphere[i])
                    if is_biosphere
                    else bw2data.get_activity(self.reversed_activity[i])
                )

                samples = np.array(cm[i, j, :].todense()).flatten().astype(float)
                amount = inventory[i, j]
                impact_samples = amount * samples

                # Percentiles
                cf_p = np.percentile(samples, [5, 25, 50, 75, 95])
                impact_p = np.percentile(impact_samples, [5, 25, 50, 75, 95])

                entry = {
                    "supplier name": supplier["name"],
                    "consumer name": consumer["name"],
                    "consumer reference product": consumer.get("reference product"),
                    "consumer location": consumer.get("location"),
                    "amount": amount,
                    "CF (mean)": samples.mean(),
                    "CF (std)": samples.std(),
                    "CF (min)": samples.min(),
                    "CF (5th)": cf_p[0],
                    "CF (25th)": cf_p[1],
                    "CF (50th)": cf_p[2],
                    "CF (75th)": cf_p[3],
                    "CF (95th)": cf_p[4],
                    "CF (max)": samples.max(),
                    "impact (mean)": impact_samples.mean(),
                    "impact (std)": impact_samples.std(),
                    "impact (min)": impact_samples.min(),
                    "impact (5th)": impact_p[0],
                    "impact (25th)": impact_p[1],
                    "impact (50th)": impact_p[2],
                    "impact (75th)": impact_p[3],
                    "impact (95th)": impact_p[4],
                    "impact (max)": impact_samples.max(),
                }

                if is_biosphere:
                    entry["supplier categories"] = supplier.get("categories")
                else:
                    entry["supplier reference product"] = supplier.get(
                        "reference product"
                    )
                    entry["supplier location"] = supplier.get("location")

                data.append(entry)

        else:
            # Deterministic fallback
            for cf in self.scenario_cfs:
                for i, j in cf["positions"]:
                    consumer = bw2data.get_activity(self.reversed_activity[j])
                    supplier = (
                        bw2data.get_activity(self.reversed_biosphere[i])
                        if is_biosphere
                        else bw2data.get_activity(self.reversed_activity[i])
                    )

                    amount = inventory[i, j]
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
                        entry["supplier categories"] = supplier.get("categories")
                    else:
                        entry["supplier reference product"] = supplier.get(
                            "reference product"
                        )
                        entry["supplier location"] = supplier.get("location")

                    data.append(entry)

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Order columns
        preferred_columns = [
            "supplier name",
            "supplier categories",
            "supplier reference product",
            "supplier location",
            "consumer name",
            "consumer reference product",
            "consumer location",
            "amount",
        ]

        # Add CF or CF summary columns
        if self.use_distributions:
            preferred_columns += [
                "CF (mean)",
                "CF (std)",
                "CF (min)",
                "CF (5th)",
                "CF (25th)",
                "CF (50th)",
                "CF (75th)",
                "CF (95th)",
                "CF (max)",
                "impact (mean)",
                "impact (std)",
                "impact (min)",
                "impact (5th)",
                "impact (25th)",
                "impact (50th)",
                "impact (75th)",
                "impact (95th)",
                "impact (max)",
            ]
        else:
            preferred_columns += ["CF", "impact"]

        df = df[[col for col in preferred_columns if col in df.columns]]

        return df

    @property
    def geo(self):
        if getattr(self, "_geo", None) is None:
            self._geo = GeoResolver(self.weights)
        return self._geo

    def reset_geo(self):
        self._geo = None
