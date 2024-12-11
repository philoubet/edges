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
from scipy.sparse import lil_matrix
from collections import defaultdict
from constructive_geometries import Geomatcher
import logging
import pandas as pd

from .filesystem_constants import DATA_DIR
from .utils import prepare_lca_inputs

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
    reformatted_data = {}
    for k, v in data.items():
        for cf in v:
            cf["categories"] = tuple(cf["categories"])
        reformatted_data[k] = {"flows": v}
    return add_population_and_gdp_data(reformatted_data)

def add_population_and_gdp_data(data: dict) -> dict:
    """
    Add population and GDP data to the LCIA method.
    :param data: The data for the LCIA method.
    :return: The data for the LCIA method with population and GDP data.
    """
    # load population data from data/population.yaml
    with open(DATA_DIR / "population.yaml", "r") as f:
        population_data = yaml.safe_load(f)

    # load GDP data from data/gdp.yaml
    with open(DATA_DIR / "gdp.yaml", "r") as f:
        gdp_data = yaml.safe_load(f)

    # add to the data dictionary
    for k, v in data.items():
        if k in population_data:
            v["population"] = population_data[k]

        if k in gdp_data:
            v["gdp"] = gdp_data[k]

    return data

def initialize_lcia_matrix(lca: LCA) -> lil_matrix:
    """
    Initialize the LCIA matrix. It is a sparse matrix with the
    dimensions of the `inventory` matrix of the LCA object.
    :param lca: The LCA object.
    :return: An empty LCIA matrix with the dimensions of the `inventory` matrix.
    """
    return lil_matrix(lca.inventory.shape)

def get_location_matrix_positions(lca: LCA) -> tuple:
    """
    Get the positions of the country-specific LCIA method in the
    inventory matrix of the LCA object.
    :param lca: The LCA object.
    :return: A tuple with the positions of the country-specific LCIA method.
    """
    idx_to_act_map = {}
    act_to_locations = defaultdict(list)

    for k, v in lca.dicts.activity.items():
        act = bw2data.get_activity(id=k)
        idx_to_act_map[v] = {
            "location": act["location"],
            "name": act["name"],
            "reference product": act.get("reference product")
        }
        act_to_locations[(
            act["name"],
            act.get("reference product"),
        )].append(act["location"])

    # create a dictionary with the unique locations
    # as the key and a list of indices as the value
    locations_to_idx = defaultdict(list)
    for k, v in idx_to_act_map.items():
        if k:
            locations_to_idx[v["location"]].append(k)

    return locations_to_idx, idx_to_act_map, act_to_locations


def get_flow_matrix_positions(lca: LCA) -> tuple:
    """
    Get the positions of the flow-specific LCIA method in the
    inventory matrix of the LCA object.
    :param lca: The LCA object.
    :return: A tuple with the positions of the flow-specific LCIA method.
    """
    flow_indices, reversed_flow_indices = {}, {}
    for k, v in lca.dicts.biosphere.items():
        flow = bw2data.get_activity(id=k)
        flow_indices[(flow["name"], flow["categories"])] = v
        reversed_flow_indices[v] = (flow["name"], flow["categories"])
    return flow_indices, reversed_flow_indices

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
    weight_array = [weight.get(c, 0) for c in constituents]

    # if the sum of the weight is zero, return an empty dictionary
    if sum(weight_array) == 0:
        logger.info(f"Region: {region}. "
                    f"No {weighting} data found when summing {constituents}.")
        return {}, []

    # normalize the weight
    shares = [p / sum(weight_array) for p in weight_array]
    if not np.isclose(sum(shares), 1):
        logger.info(f"Shares for {region} do not sum to 1 but {sum(shares)}: {shares}")

    new_cfs = {}
    for s, share in enumerate(shares):
        constituent = constituents[s]

        for cf in cfs[constituent]["flows"]:
            flow = (cf["name"], cf["categories"])
            if flow in new_cfs:
                new_cfs[flow]["value"] += cf["value"] * share
            else:
                new_cfs[flow] = {
                    "name": cf["name"],
                    "categories": cf["categories"],
                    "value": cf["value"] * share
                }

    return new_cfs, weight_array


def find_region_constituents(
        regions: set,
        cfs: dict,
        weighting="population"
) -> dict:
    """
    Find the constituents of the region.
    :param regions: The regions.
    :param cfs: The characterization factors.
    :param weighting: The weighting to use.
    :return: The new characterization factors for the region
    """

    weight = {k: v[weighting] for k, v in cfs.items()}

    logger.info(f"Weighting: {weighting}")

    for region in regions:
        _ = lambda x: x if isinstance(x, str) else x[-1]

        if region in ["RoW", "RoE"]:
            continue
        try:
            constituents = [_(g) for g in geo.contained(region) if _(g) in weight]
        except KeyError:
            logger.info(f"Region: {region}. No geometry found.")
            continue

        new_cfs, weight_array = compute_average_cf(constituents, weight, cfs, region, weighting)

        if new_cfs:
            cfs[region] = {
                "flows": list(new_cfs.values()),
                weighting: sum(weight_array)
            }
    return cfs

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
        self.reversed_flow_indices = None
        self.act_to_locations = None
        self.idx_to_act_map = None
        self.characterized_inventory = None
        self.characterization_matrix = None
        self.ignored_flows = set()
        self.ignored_locations = set()
        self.flow_indices = None
        self.location_indices = None
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

    def load_lcia_data(self, data_objs=None):
        """
        Load the data for the LCIA method.
        """
        data_file = DATA_DIR / f"{'_'.join(self.method)}.json"

        if not data_file.is_file():
            raise FileNotFoundError(f"Data file not found: {data_file}")

        with open(data_file, "r") as f:
            cfs_data = format_data(json.load(f))

        self.fill_in_lcia_matrix(cfs_data)

    def fill_in_lcia_matrix(self, cfs_data):
        """
        Translate the data to indices in the inventory matrix.
        """

        self.characterization_matrix = initialize_lcia_matrix(self)

        processed_locations = set()

        missing_locations = set([
            loc for loc in self.location_indices.keys()
            if loc not in cfs_data
        ])

        cfs_data = find_region_constituents(
            regions=missing_locations,
            cfs=cfs_data,
            weighting=self.weight
        )

        for loc, rows in self.location_indices.items():

            if loc not in cfs_data:
                continue

            for cf in cfs_data[loc]["flows"]:
                if (cf["name"], cf["categories"]) not in self.flow_indices:
                    self.ignored_flows.add((cf["name"], cf["categories"]))
                    continue
                flow_index = self.flow_indices[(cf["name"], cf["categories"])]
                val = cf["value"]

                # create mask from self.biosphere_matrix, to only write
                # where the flow is present
                mask = self.biosphere_matrix[flow_index, rows].astype(bool)

                # write the value to the characterization matrix, where the mask is True
                if mask.sum() > 0:
                    self.characterization_matrix[flow_index, rows] = val * mask
                    processed_locations.add(loc)


        # we need to deal with RoW and RoE separately
        for k, v in self.location_indices.items():
            if k in ["RoW", "RoE"]:
                processed_locations.add(k)
                for x in v:
                    dynamic_cfs = resolve_dynamic_regions(
                        idx=x,
                        idx_to_act=self.idx_to_act_map,
                        act_to_locs=self.act_to_locations,
                        cfs=cfs_data,
                        weighting=self.weight
                    )

                    for flow, cf in dynamic_cfs.items():
                        if flow not in self.flow_indices:
                            self.ignored_flows.add(flow)
                            continue
                        flow_index = self.flow_indices[flow]
                        val = cf["value"]
                        self.characterization_matrix[flow_index, x] = val

        self.characterization_matrix = self.characterization_matrix.tocsr()

        # print unprocessed locations
        unprocessed_locations = set(self.location_indices.keys()) - processed_locations
        # add to self.ignored_locations
        for loc in unprocessed_locations:
            self.ignored_locations.add(loc)

        if self.ignored_locations:
            print(f"{len(processed_locations)} location-specific factors implemented. "
                  f"{len(self.ignored_locations)} locations ignored. "
                  f"Check self.ignored_locations.")

        if self.ignored_flows:
            print(f"{len(self.ignored_flows)} biosphere flows ignored. "
                  f"Check self.ignored_flows.")


    def lcia_calculation(self):
        """
        Calculate the LCIA score.
        """
        self.location_indices, self.idx_to_act_map, self.act_to_locations = get_location_matrix_positions(self)
        self.flow_indices, self.reversed_flow_indices = get_flow_matrix_positions(self)
        self.load_lcia_data()
        self.characterized_inventory = self.characterization_matrix.multiply(self.inventory)

    def generate_cf_table(self):
        """
        Generate a pandas dataframe with the characterization factors,
        from self.characterization_matrix.
        """

        data = []

        non_zero_indices = self.characterization_matrix.nonzero()

        for i, j in zip(*non_zero_indices):
            name, reference_product, location = self.idx_to_act_map[j]["name"], self.idx_to_act_map[j]["reference product"], self.idx_to_act_map[j]["location"]
            data.append({
                "flow": self.reversed_flow_indices[i],
                "name": name,
                "reference product": reference_product,
                "location": location,
                "value": self.characterization_matrix[i, j]
            })
        return pd.DataFrame(data)
