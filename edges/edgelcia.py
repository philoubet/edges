"""
Module that implements the base class for country-specific life-cycle
impact assessments, and the AWARE class, which is a subclass of the
LCIA class.
"""

import os
import json
import yaml
import numpy as np
from bw2calc import LCA
import bw2data
from scipy.sparse import lil_matrix, coo_matrix
from collections import defaultdict
from constructive_geometries import Geomatcher
import logging
import pandas as pd

from .filesystem_constants import DATA_DIR

from pathlib import Path
from typing import Iterable, Optional, Union

import bw_processing as bwp
from fsspec import AbstractFileSystem

from bw2calc import prepare_lca_inputs, PYPARDISO
from bw2calc.dictionary_manager import DictionaryManager
from bw2calc.utils import get_datapackage

# initiate the logger
logging.basicConfig(
    filename='edgelcia.log',
    filemode='a',
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


geo = Geomatcher()

def format_method_name(name: str) -> tuple:
    """
    Format the name of the method.
    :param name: The name of the method.
    :return: A tuple with the name of the method.
    """
    return tuple(name.split("_"))

def get_available_methods() -> list:
    """
    Display the available impact assessment methods by reading
     file names under `data` directory
    that ends with ".json" extension.
    :return: A list of available impact assessment methods.
    """
    return sorted([format_method_name(f.replace(".json", "")) for f in os.listdir(DATA_DIR) if f.endswith(".json")])

def check_method(impact_method: str) -> str:
    """
    Check if the impact method is available.
    :param impact_method: The impact method to check.
    :return: The impact method if it is available.
    """
    if impact_method not in get_available_methods():
        raise ValueError(f"Impact method not available: {impact_method}")
    return impact_method

def format_data(data: dict) -> dict:
    """
    Format the data for the LCIA method.
    :param data: The data for the LCIA method.
    :return: The formatted data for the LCIA method.
    """

    for cf in data:
        for category in ["supplier", "consumer"]:
            for field, value in cf.get(category, {}).items():
                if field == "categories":
                    cf[category][field] = tuple(value)

    return add_population_and_gdp_data(data)

def add_population_and_gdp_data(data: dict) -> dict:
    """
    Add population and GDP data to the LCIA method.
    :param data: The data for the LCIA method.
    :return: The data for the LCIA method with population and GDP data.
    """
    # load population data from data/population.yaml
    with open(DATA_DIR / "metadata" / "population.yaml", "r") as f:
        population_data = yaml.safe_load(f)

    # load GDP data from data/gdp.yaml
    with open(DATA_DIR / "metadata" / "gdp.yaml", "r") as f:
        gdp_data = yaml.safe_load(f)

    # add to the data dictionary
    for cf in data:
        for category in ["supplier", "consumer"]:
            if "location" in cf[category]:
                k = cf[category]["location"]
                cf[category]["population"] = population_data.get(k, 0)
                cf[category]["gdp"] = gdp_data.get(k, 0)

    return data

def initialize_lcia_matrix(lca: LCA, type="biosphere") -> lil_matrix:
    """
    Initialize the LCIA matrix. It is a sparse matrix with the
    dimensions of the `inventory` matrix of the LCA object.
    :param lca: The LCA object.
    :return: An empty LCIA matrix with the dimensions of the `inventory` matrix.
    """
    if type=="biosphere":
        return lil_matrix(lca.inventory.shape)
    else:
        return lil_matrix(lca.technosphere_matrix.shape)


def get_flow_matrix_positions(mapping: dict) -> list:
    """
    Retrieve information about the flows in the given matrix.
    :param mapping: The mapping of the flows.
    :return: A list with the positions of the flows.
    """
    flows = []
    for k, v in mapping.items():
        flow = bw2data.get_activity(k)
        flows.append(
            {
                "name": flow["name"],
                "reference product": flow.get("reference product"),
                "categories": flow.get("categories"),
                "unit": flow.get("unit"),
                "location": flow.get("location"),
                "classifications": flow.get("classifications"),
                "type": flow.get("type"),
                "position": v
            }
        )
    return flows


def preprocess_cfs(cfs: list) -> dict:
    """
    Preprocess the characterization factors into a dictionary for faster lookup.
    :param cfs: List of characterization factors.
    :return: A dictionary indexed by consumer location.
    """
    cfs_lookup = defaultdict(list)
    for cf in cfs:
        location = cf["consumer"].get("location")
        if location:
            cfs_lookup[location].append(cf)
    return cfs_lookup


def compute_average_cf(
    constituents: list,
    supplier_info: dict,
    weight: dict,
    cfs_lookup: dict,
    region: str,
):
    """
    Compute the average characterization factors for the region.
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

    # Pre-filter suppliers
    supplier_keys = supplier_info.keys()
    supplier_tuple = tuple(supplier_info[k] for k in supplier_keys)

    # Compute the weighted average CF value
    value = 0
    for loc, share in zip(constituents, shares):
        loc_cfs = cfs_lookup.get(loc, [])
        supplier_filter = tuple((k, supplier_info[k]) for k in supplier_keys)

        filtered_cfs = [
            cf["value"]
            for cf in loc_cfs
            if tuple((k, cf["supplier"].get(k)) for k in supplier_keys) == supplier_filter
        ]
        value += share * sum(filtered_cfs)

    # Log if shares don't sum to 1 due to precision issues
    if not np.isclose(shares.sum(), 1):
        logger.info(f"Shares for {region} do not sum to 1 but {shares.sum()}: {shares}")

    return value


def find_region_constituents(
        region: str,
        supplier_info: dict,
        cfs: dict,
        weight: dict
) -> float:
    """
    Find the constituents of the region.
    :param regions: The regions.
    :param cfs: The characterization factors.
    :param weighting: The weighting to use.
    :return: The new characterization factors for the region
    """

    _ = lambda x: x if isinstance(x, str) else x[-1]

    try:
        constituents = [_(g) for g in geo.contained(region) if _(g) in weight]
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

def preprocess_flows(flows_list, mandatory_fields):
    """
    Preprocess flows into a lookup dictionary.
    :param flows_list: List of flows.
    :param mandatory_fields: Fields that must be included in the lookup key.
    :return: A dictionary for flow lookups.
    """
    lookup = {}
    for flow in flows_list:
        key = tuple(
            (k, v)
            for k, v in flow.items()
            if k in mandatory_fields and v is not None
        )
        lookup.setdefault(key, []).append(flow["position"])
    return lookup

def match_operator(value, target, operator):
    """
    Match a value against a target using the specified operator.
    :param value: The value to match.
    :param target: The target value to compare against.
    :param operator: The matching operator (equals, contains, startswith).
    :return: True if the value matches the target based on the operator.
    """
    if operator == "equals":
        return value == target
    elif operator == "contains":
        return value in target
    elif operator == "startswith":
        return target.startswith(value)
    else:
        raise ValueError(f"Unsupported operator: {operator}")


class EdgeLCIA(LCA):
    """
    Subclass of the `bw2io.lca.LCA` class that implements the calculation
    of the life cycle impact assessment (LCIA) results.
    """

    def __init__(
            self,
            demand: dict,
            # Brightway 2 calling convention
            method: Optional[tuple] = None,
            weighting: Optional[str] = None,
            lcia_weight: Optional[str] = "population",
            normalization: Optional[str] = None,
            # Brightway 2.5 calling convention
            data_objs: Optional[Iterable[Union[Path, AbstractFileSystem, bwp.DatapackageBase]]] = None,
            remapping_dicts: Optional[Iterable[dict]] = None,
            log_config: Optional[dict] = None,
            seed_override: Optional[int] = None,
            use_arrays: Optional[bool] = False,
            use_distributions: Optional[bool] = False,
            selective_use: Optional[dict] = False,
    ):
        """
        Initialize the SpatialLCA class, ensuring `method` is not passed to
        `prepare_lca_inputs` while still being available in the class.
        """

        self.cfs_data = None
        self.biosphere_edges = None
        self.technosphere_flows = None
        self.biosphere_flows = None
        self.characterized_inventory = None
        self.biosphere_characterization_matrix = None
        self.ignored_flows = set()
        self.ignored_locations = set()
        self.method = method  # Store the method argument in the instance
        self.weight = lcia_weight

        if not data_objs:  # If `data_objs` is not provided
            demand, self.packages, remapping_dicts = prepare_lca_inputs(
                demand=demand,
                weighting=weighting,
                normalization=normalization,
            )
            self.method = method
            self.weighting = weighting
            self.normalization = normalization
        else:
            self.packages = [get_datapackage(obj) for obj in data_objs]

        self.dicts = DictionaryManager()
        self.demand = demand
        self.use_arrays = use_arrays
        self.use_distributions = use_distributions
        self.selective_use = selective_use or {}
        self.remapping_dicts = remapping_dicts or {}
        self.seed_override = seed_override


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
        self.technosphere_edges = set(list(zip(*self.technosphere_flow_matrix.nonzero())))
        self.biosphere_edges = set(list(zip(*self.inventory.nonzero())))

        unique_biosphere_flows = set(x[0] for x in self.biosphere_edges)

        self.biosphere_flows = get_flow_matrix_positions({
            k: v for k, v in self.biosphere_dict.items()
            if v in unique_biosphere_flows
        })

        self.technosphere_flows = get_flow_matrix_positions({
            k: v for k, v in self.activity_dict.items()
        })

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
        flow_matrix_sparse = coo_matrix((scaled_data, (rows, cols)), shape=self.technosphere_matrix.shape)

        return flow_matrix_sparse.tocsr()

    def load_lcia_data(self, data_objs=None):
        """
        Load the data for the LCIA method.
        """
        data_file = DATA_DIR / f"{'_'.join(self.method)}.json"

        if not data_file.is_file():
            raise FileNotFoundError(f"Data file not found: {data_file}")

        with open(data_file, "r") as f:
            self.cfs_data = format_data(json.load(f))

    def identify_exchanges(self):
        """
        Based on search criteria under `supplier` and `consumer` keys,
        identify the exchanges in the inventory matrix.
        """

        def match_with_operator(flow, lookup, required_fields):
            """
            Match a flow against a lookup dictionary considering the operator.
            :param flow: The flow to match.
            :param lookup: The lookup dictionary.
            :param required_fields: The required fields for matching.
            :return: A list of matching positions.
            """
            matches = []
            for key, positions in lookup.items():
                if all(
                        match_operator(
                            value=flow.get(k),
                            target=v,
                            operator=flow.get("operator", "equals")
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
                self.biosphere_flows if all(cf["supplier"].get("matrix") == "biosphere" for cf in self.cfs_data)
                else self.technosphere_flows,
                required_supplier_fields
            )

            print(list(supplier_lookup.items())[:5])
            consumer_lookup = preprocess_flows(self.technosphere_flows, required_consumer_fields)

            reversed_supplier_lookup = {
                pos: key for key, positions in supplier_lookup.items() for pos in positions
            }
            reversed_consumer_lookup = {
                pos: key for key, positions in consumer_lookup.items() for pos in positions
            }

            return supplier_lookup, consumer_lookup, reversed_supplier_lookup, reversed_consumer_lookup

        def process_edges(direction, unprocessed_edges, cfs_lookup, unprocessed_locations_cache):
            """
            Process edges for the given direction and update CF data.
            """
            for supplier_idx, consumer_idx in unprocessed_edges:
                supplier_info = dict(reversed_supplier_lookup[supplier_idx])
                consumer_info = dict(reversed_consumer_lookup[consumer_idx])
                location = consumer_info.get("location")

                if location not in ("RoW", "RoE"):
                    # Resolve from cache or compute new CF
                    new_cf = unprocessed_locations_cache.get(location, {}).get(
                        supplier_idx) or find_region_constituents(
                        region=location, supplier_info=supplier_info, cfs=cfs_lookup, weight=weight
                    )
                    if new_cf:
                        unprocessed_locations_cache[location][supplier_idx] = new_cf
                        self.cfs_data.append({
                            "supplier": supplier_info,
                            "consumer": consumer_info,
                            direction: [(supplier_idx, consumer_idx)],
                            "value": new_cf
                        })

        def handle_dynamic_regions(direction, unprocessed_edges, cfs_lookup):
            """
            Handle dynamic regions like RoW and RoE and update CF data.
            """
            for supplier_idx, consumer_idx in unprocessed_edges:
                supplier_info = dict(reversed_supplier_lookup[supplier_idx])
                consumer_info = dict(reversed_consumer_lookup[consumer_idx])
                location = consumer_info.get("location")

                if location in ("RoW", "RoE"):
                    consumer_act = self.position_to_technosphere_flows_lookup[consumer_idx]
                    name, reference_product = consumer_act["name"], consumer_act.get("reference product")

                    # Use preprocessed lookup
                    other_than_RoW_RoE = [
                        loc for loc in self.technosphere_flows_lookup.get((name, reference_product), [])
                        if loc not in ["RoW", "RoE"]
                        and loc in weight
                    ]

                    # constituents are all th candidates in teh World (or in Europe) minus those in other_than_RoW_RoE

                    if location == "RoW":
                        constituents = list(set(list(weight.keys())) - set(other_than_RoW_RoE))
                    else:
                        # RoE
                        # redefine other_than_RoW_RoE to limit to EU candiates
                        other_than_RoW_RoE = [loc for loc in other_than_RoW_RoE if geo.contained("RER")]
                        constituents = list(set(geo.contained("RER")) - set(other_than_RoW_RoE))

                    _ = lambda x: x if isinstance(x, str) else x[-1]

                    extra_constituents = []
                    for constituent in constituents:
                        if constituent not in weight:
                            extras = [
                                _(e) for e in geo_cache.get(constituent, geo.contained(constituent))
                                if _(e) in weight and e != constituent
                            ]
                            extra_constituents.extend(extras)
                            geo_cache[constituent] = extras

                    constituents.extend(extra_constituents)

                    new_cf = compute_average_cf(
                        constituents=constituents,
                        supplier_info=supplier_info,
                        weight=weight,
                        cfs_lookup=cfs_lookup,
                        region=location
                    )

                    logger.info(
                        f"Region: {location}. Activity: {name, reference_product} "
                        f"New CF: {new_cf}. "
                        f"Candidates other than Row/RoE: {other_than_RoW_RoE} "
                        f"Constituents: {constituents}"
                    )

                    if new_cf:
                        self.cfs_data.append({
                            "supplier": supplier_info,
                            "consumer": consumer_info,
                            direction: [(supplier_idx, consumer_idx)],
                            "value": new_cf
                        })
                    else:
                        self.ignored_locations.add(location)

        # Constants for ignored fields
        IGNORED_FIELDS = {"matrix", "population", "gdp", "operator"}

        # Precompute required fields for faster access
        required_supplier_fields = {
            k for cf in self.cfs_data for k in cf["supplier"].keys() if k not in IGNORED_FIELDS
        }
        required_consumer_fields = {
            k for cf in self.cfs_data for k in cf["consumer"].keys() if k not in IGNORED_FIELDS
        }

        # Preprocess flows and lookups
        supplier_lookup, consumer_lookup, reversed_supplier_lookup, reversed_consumer_lookup = preprocess_lookups()

        edges = self.biosphere_edges if all(cf["supplier"].get("matrix") == "biosphere" for cf in self.cfs_data) else self.technosphere_edges

        for cf in self.cfs_data:
            # Generate supplier candidates
            supplier_candidates = match_with_operator(
                cf["supplier"],
                supplier_lookup,
                required_supplier_fields
            )

            # Generate consumer candidates
            consumer_candidates = match_with_operator(
                cf["consumer"],
                consumer_lookup,
                required_consumer_fields
            )

            print(f"cf['supplier']: {cf['supplier']}")
            print(f"cf['consumer']: {cf['consumer']}")

            print(f"Supplier candidates: {supplier_candidates}")
            print(f"Consumer candidates: {consumer_candidates}")

            # Create pairs of supplier and consumer candidates
            cf[f"{cf['supplier']['matrix']}-{cf['consumer']['matrix']}"] = [
                (supplier, consumer)
                for supplier in supplier_candidates
                for consumer in consumer_candidates
                if (supplier, consumer) in edges
            ]

            print(cf[f"{cf['supplier']['matrix']}-{cf['consumer']['matrix']}"])
            print()

        # Preprocess `self.technosphere_flows` once
        if not hasattr(self, "technosphere_flows_lookup"):
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
        supplier_bio_tech_edges = {f[0] for cf in self.cfs_data for f in cf.get("biosphere-technosphere", [])}
        consumer_bio_tech_edges = {f[1] for cf in self.cfs_data for f in cf.get("biosphere-technosphere", [])}
        supplier_tech_tech_edges = {f[0] for cf in self.cfs_data for f in cf.get("technosphere-technosphere", [])}
        consumer_tech_tech_edges = {f[1] for cf in self.cfs_data for f in cf.get("technosphere-technosphere", [])}

        unprocessed_biosphere_edges = [
            edge for edge in self.biosphere_edges
            if edge[0] in supplier_bio_tech_edges and edge[1] not in consumer_bio_tech_edges
        ]
        unprocessed_technosphere_edges = [
            edge for edge in self.technosphere_edges
            if edge[0] in supplier_tech_tech_edges and edge[1] not in consumer_tech_tech_edges
        ]

        weight = {i.get("consumer").get("location"): i.get("consumer").get(self.weight) for i in self.cfs_data if i.get("consumer").get("location")}
        geo_cache = {}

        process_edges("biosphere-technosphere", unprocessed_biosphere_edges, cfs_lookup, defaultdict(dict))
        process_edges("technosphere-technosphere", unprocessed_technosphere_edges, cfs_lookup, defaultdict(dict))

        handle_dynamic_regions("biosphere-technosphere", unprocessed_biosphere_edges, cfs_lookup)
        handle_dynamic_regions("technosphere-technosphere", unprocessed_technosphere_edges, cfs_lookup)

        self.cfs_data = [
            cf for cf in self.cfs_data
            if any([cf.get("biosphere-technosphere"), cf.get("technosphere-technosphere")])
        ]

        # figure out remaining unprocessed edges for information
        processed_biosphere_edges = {f for cf in self.cfs_data for f in cf.get("biosphere-technosphere", [])}
        processed_technosphere_edges = {f for cf in self.cfs_data for f in cf.get("technosphere-technosphere", [])}

        unprocessed_biosphere_edges = set(unprocessed_biosphere_edges) - processed_biosphere_edges
        unprocessed_technosphere_edges = set(unprocessed_technosphere_edges) - processed_technosphere_edges

        print(f"Unprocessed biosphere edges: {len(unprocessed_biosphere_edges)}")
        print(f"Unprocessed technosphere edges: {len(unprocessed_technosphere_edges)}")


    def fill_in_lcia_matrix(self):
        """
        Translate the data to indices in the inventory matrix.
        """

        if all(cf["supplier"].get("matrix") == "biosphere" for cf in self.cfs_data):
            self.characterization_matrix = initialize_lcia_matrix(self)
        else:
            self.characterization_matrix = initialize_lcia_matrix(self, type="technosphere")

        self.identify_exchanges()

        for cf in self.cfs_data:
            for supplier, consumer in cf.get("biosphere-technosphere", []):
                self.characterization_matrix[supplier, consumer] = cf["value"]

            for supplier, consumer in cf.get("technosphere-technosphere", []):
                self.characterization_matrix[supplier, consumer] = cf["value"]

        self.characterization_matrix = self.characterization_matrix.tocsr()

    def lcia_calculation(self):
        """
        Calculate the LCIA score.
        """
        self.load_lcia_data()
        self.fill_in_lcia_matrix()

        try:
            self.characterized_inventory = self.characterization_matrix.multiply(self.inventory)
        except ValueError:
            self.characterized_inventory = self.characterization_matrix.multiply(self.technosphere_flow_matrix)

        if self.ignored_locations:
            print(f"{len(self.ignored_locations)} locations were ignored. Check .ignored_locations attribute.")

    def generate_cf_table(self):
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
                self.reversed_biosphere[i] if is_biosphere else self.reversed_activity[i]
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
                entry.update({
                    "supplier reference product": supplier.get("reference product"),
                    "supplier location": supplier.get("location"),
                })

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
            "impact"
        ]

        # Reorder columns based on the desired order, ignoring missing columns
        df = df[[col for col in column_order if col in df.columns]]

        return df


