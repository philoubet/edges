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
    filename='regionallcia.log',
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

def resolve_dynamic_regions(
        idx: int,
        idx_to_act: dict,
        act_to_locs: dict,
        cfs: dict,
        weighting="population"
):
    """
    Resolve dynamic regions (e.g., RoW and RoE) in the LCIA method.
    We need to figure out the regions, besides the dynamic ones, providing
    a similar product or service, to obtain the regions included in the
    dynamic regions.

    :param idx: The index of the activity.
    :param idx_to_act: A dictionary with the index of the activity.
    :param act_to_locs: A dictionary with the activity and its locations.
    :param cfs: The characterization factors.
    :param weighting: The weighting to use.
    :return: The new characterization factors for the dynamic regions.
    """

    # get the activity
    act = idx_to_act[idx]

    # get all the possible locations of the activity
    locs = act_to_locs[(act["name"], act.get("reference product"))]

    # get the constituents of the dynamic region
    constituents = [loc for loc in locs if loc not in ["RoW", "RoE"] and loc in cfs]

    # get the weighting
    weight = {k: v[weighting] for k, v in cfs.items() if k in constituents}

    new_cf, _ = compute_average_cf(constituents, weight, cfs, act["location"], weighting)

    return new_cf

def compute_average_cf(
        constituents: list,
        supplier_info: dict,
        weight: dict,
        cfs: dict,
        region: str,
        weighting="population"
):
    """
    Compute the average characterization factors for the region.
    :param constituents: The constituents of the region.
    :param weight: The weight of the constituents.
    :param cfs: The characterization factors.
    :param region: The region.
    :param weighting: The weighting to use.
    :return: The new characterization factors for the region.
    """

    # get the weight of the constituents
    constituents = [c for c in constituents if weight.get(c)]
    weight_array = [weight[c] for c in constituents]
    weight_array = [w for w in weight_array if w]

    # if the sum of the weight is zero, return an empty dictionary
    if sum(weight_array) == 0:
        logger.info(
            f"Region: {region}. "
            f"No {weighting} data found when summing {constituents}."
        )
        return 0

    # normalize the weight
    shares = [p / sum(weight_array) for p in weight_array]
    if not np.isclose(sum(shares), 1):
        logger.info(f"Shares for {region} do not sum to 1 but {sum(shares)}: {shares}")

    mix = dict(zip(constituents, shares))

    value = 0
    for loc, share in mix.items():
        for cf in cfs:
            if all(
                    cf["supplier"].get(k) == supplier_info.get(k)
                    for k in supplier_info
            ) and cf["consumer"].get("location") == loc:
                value += share * cf["value"]

    return value


def find_region_constituents(
        region: str,
        supplier_info: dict,
        cfs: dict,
        weighting="population"
) -> float:
    """
    Find the constituents of the region.
    :param regions: The regions.
    :param cfs: The characterization factors.
    :param weighting: The weighting to use.
    :return: The new characterization factors for the region
    """

    weight = {cf["consumer"].get("location"): cf["consumer"].get(weighting, 0) for cf in cfs}

    logger.info(f"Weighting: {weighting}")

    _ = lambda x: x if isinstance(x, str) else x[-1]

    try:
        constituents = [_(g) for g in geo.contained(region) if _(g) in weight]
    except KeyError:
        logger.info(f"Region: {region}. No geometry found.")
        return 0

    new_cfs = compute_average_cf(constituents, supplier_info, weight, cfs, region, weighting)

    return new_cfs

def preprocess_flows(flows_list, mandatory_fields):
    """
    Preprocess flows into a lookup dictionary.
    """
    lookup = {}
    for flow in flows_list:
        # Create a hashable key excluding ignored fields
        key = tuple((k, v) for k, v in flow.items() if k in mandatory_fields and v is not None)
        lookup.setdefault(key, []).append(flow["position"])
    return lookup

class RegionalLCA(LCA):
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
        Initialize the RegionalLCA class, ensuring `method` is not passed to
        `prepare_lca_inputs` while still being available in the class.
        """

        self.cfs_data = None
        self.flows = None
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

        self.biosphere_flows = get_flow_matrix_positions(self.biosphere_dict)
        self.technosphere_flows = get_flow_matrix_positions(self.product_dict)
        self.flows = self.biosphere_flows + self.technosphere_flows

        self.technosphere_edges = set(list(zip(*self.build_technosphere_edges_matrix().nonzero())))
        self.biosphere_edges = set(list(zip(*self.inventory.nonzero())))

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

        return flow_matrix_sparse

    def load_lcia_data(self, data_objs=None):
        """
        Load the data for the LCIA method.
        """
        data_file = DATA_DIR / f"{'_'.join(self.method)}.json"

        if not data_file.is_file():
            raise FileNotFoundError(f"Data file not found: {data_file}")

        with open(data_file, "r") as f:
            self.cfs_data = format_data(json.load(f))

        self.fill_in_lcia_matrix()

    def identify_exchanges(self):
        """
        Based on search criteria under `supplier` and `consumer` keys,
        identify the exchanges in the inventory matrix.
        """

        IGNORED_FIELDS = [
            "matrix",
            "population",
            "gdp"
        ]

        required_supplier_fields = set([
            k for cf in self.cfs_data
            for k in cf["supplier"].keys()
            if k not in IGNORED_FIELDS
        ])

        required_consumer_fields = set([
            k for cf in self.cfs_data
            for k in cf["consumer"].keys()
            if k not in IGNORED_FIELDS
        ])

        # Preprocess flows for faster lookups
        if all([c["supplier"].get("matrix") == "biosphere" for c in self.cfs_data]):
            supplier_lookup = preprocess_flows(self.biosphere_flows, required_supplier_fields)
        else:
            supplier_lookup = preprocess_flows(self.technosphere_flows, required_supplier_fields)
        reversed_supplier_lookup = {x: k for k, v in supplier_lookup.items() for x in v if all(k)}

        consumer_lookup = preprocess_flows(self.technosphere_flows, required_consumer_fields)
        reversed_consumer_lookup = {x: k for k, v in consumer_lookup.items() for x in v if all(k)}

        for cf in self.cfs_data:
            # Generate supplier candidates
            supplier_key = tuple((k, v) for k, v in cf["supplier"].items() if k not in IGNORED_FIELDS)
            supplier_candidates = supplier_lookup.get(supplier_key, [])

            # Generate consumer candidates
            consumer_key = tuple((k, v) for k, v in cf["consumer"].items() if k not in IGNORED_FIELDS)
            consumer_candidates = consumer_lookup.get(consumer_key, [])

            # Create pairs of supplier and consumer candidates
            cf[f"{cf['supplier']['matrix']}-{cf['consumer']['matrix']}"] = [
                (supplier, consumer)
                for supplier in supplier_candidates
                for consumer in consumer_candidates
                if (supplier, consumer) in self.biosphere_edges
            ]

        processed_bio_tech_edges = set(
            [
                (c[0], c[1])
                for cf in self.cfs_data
                for c in cf.get("biosphere-technosphere", [])
             ]
        )

        processed_tech_tech_edges = set(
            [
                (c[0], c[1])
                for cf in self.cfs_data
                for c in cf.get("technosphere-technosphere", [])
             ]
        )

        # make two sets out of the first and second item in the tuple
        supplier_bio_tech_edges, consumer_bio_tech_edges = set([f[0] for f in processed_bio_tech_edges]), set([f[1] for f in processed_bio_tech_edges])
        supplier_tech_tech_edges, consumer_tech_tech_edges = set([f[0] for f in processed_tech_tech_edges]), set([f[1] for f in processed_tech_tech_edges])

        unprocessed_biosphere_edges = [
            (biosphere, technosphere)
            for biosphere, technosphere in self.biosphere_edges
            if biosphere in supplier_bio_tech_edges
            and technosphere not in consumer_bio_tech_edges
        ]

        unprocessed_technosphere_edges = [
            (biosphere, technosphere)
            for biosphere, technosphere in self.technosphere_edges
            if biosphere in supplier_tech_tech_edges
            and technosphere not in consumer_tech_tech_edges
        ]


        for direction, unprocessed_edges in {
            "biosphere-technosphere": unprocessed_biosphere_edges,
            "technosphere-technosphere": unprocessed_technosphere_edges
        }.items():
            unprocessed_locations_cache = {}
            for unprocessed_coordinate in unprocessed_edges:
                supplier_idx, consumer_idx = unprocessed_coordinate
                supplier_info = dict(reversed_supplier_lookup[supplier_idx])
                consumer_info = dict(reversed_consumer_lookup[consumer_idx])

                if consumer_info.get("location") not in ("RoW", "RoE"):
                    new_cf = None

                    if consumer_info.get("location") in unprocessed_locations_cache:
                        if supplier_idx in unprocessed_locations_cache[consumer_info.get("location")]:
                            new_cf = unprocessed_locations_cache[consumer_info.get("location")][supplier_idx]

                    if not new_cf:
                        new_cf = find_region_constituents(consumer_info.get("location"), supplier_info, self.cfs_data, weighting="population")

                    if new_cf:
                        if consumer_info.get("location") not in unprocessed_locations_cache:
                            unprocessed_locations_cache[consumer_info.get("location")] = {supplier_idx: new_cf}
                        else:
                            unprocessed_locations_cache[consumer_info.get("location")][supplier_idx] = new_cf

                        self.cfs_data.append(
                            {
                                "supplier": supplier_info,
                                "consumer": consumer_info,
                                direction: [(supplier_idx, consumer_idx)],
                                "value": new_cf
                            }
                        )
                else:
                    # get all the possible locations of the consumer
                    consumer_act = bw2data.get_activity(self.reversed_activity[consumer_idx])
                    name, reference_product = consumer_act["name"], consumer_act.get("reference product")

                    constituents = [
                        c["location"] for c in self.technosphere_flows
                        if c["name"] == name and c.get("reference product") == reference_product
                        and c["location"] not in ["RoW", "RoE"]
                    ]

                    extra_constituents = []
                    _ = lambda x: x if isinstance(x, str) else x[-1]
                    for constituent in constituents:
                        if len(constituent) > 2:
                            extra_constituents.extend(
                                [_(g) for g in geo.contained(constituent)]
                            )
                    constituents = list(set(constituents + extra_constituents))

                    # get the weighting
                    weight = {cf["consumer"].get("location"): cf["consumer"].get(self.weight, 0) for cf in self.cfs_data}

                    new_cf = compute_average_cf(constituents, supplier_info, weight, self.cfs_data, consumer_info["location"], self.weight)

                    if new_cf:
                        self.cfs_data.append(
                            {
                                "supplier": supplier_info,
                                "consumer": consumer_info,
                                direction: [(supplier_idx, consumer_idx)],
                                "value": new_cf
                            }
                        )
                    else:
                        self.ignored_locations.add(consumer_info.get("location"))

        cfs_data = [
            cf for cf in self.cfs_data
            if any([cf.get("biosphere-technosphere"), cf.get("technosphere-technosphere")])
        ]

        return cfs_data

    def fill_in_lcia_matrix(self):
        """
        Translate the data to indices in the inventory matrix.
        """

        self.biosphere_characterization_matrix = initialize_lcia_matrix(self)
        self.technosphere_characterization_matrix = initialize_lcia_matrix(self)

        self.identify_exchanges()

        for cf in self.cfs_data:
            for supplier, consumer in cf.get("biosphere-technosphere", []):
                self.biosphere_characterization_matrix[supplier, consumer] = cf["value"]

            for supplier, consumer in cf.get("technosphere-technosphere", []):
                self.technosphere_characterization_matrix[supplier, consumer] = cf["value"]

        self.biosphere_characterization_matrix = self.biosphere_characterization_matrix.tocsr()
        self.technosphere_characterization_matrix = self.technosphere_characterization_matrix.tocsr()

    def lcia_calculation(self):
        """
        Calculate the LCIA score.
        """

        self.characterized_inventory = self.biosphere_characterization_matrix.multiply(self.inventory)

        if self.ignored_locations:
            print(f"{len(self.ignored_locations)} locations were ignored. Check .ignored_locations attribute.")

    def generate_cf_table(self):
        """
        Generate a pandas dataframe with the characterization factors,
        from self.characterization_matrix.
        """

        data = []
        non_zero_indices = self.biosphere_characterization_matrix.nonzero()

        for i, j in zip(*non_zero_indices):
            activity = bw2data.get_activity(self.reversed_activity[j])
            biosphere = bw2data.get_activity(self.reversed_biosphere[i])
            name, reference_product, location = activity["name"], activity.get("reference product"), activity.get("location")

            data.append({
                "flow": (biosphere["name"], biosphere["categories"]),
                "name": name,
                "reference product": reference_product,
                "location": location,
                "amount": self.inventory[i, j],
                "CF": self.biosphere_characterization_matrix[i, j],
                "impact": self.characterized_inventory[i, j]
            })
        return pd.DataFrame(data)
