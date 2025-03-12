"""
Utility functions for the LCIA methods implementation.
"""

import os
from collections import defaultdict
import logging

import bw2data
import yaml
from bw2calc import LCA
from scipy.sparse import lil_matrix

from .filesystem_constants import DATA_DIR


logging.basicConfig(
    filename="edgelcia.log",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


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
    return sorted(
        [
            format_method_name(f.replace(".json", ""))
            for f in os.listdir(DATA_DIR)
            if f.endswith(".json")
        ]
    )


def check_method(impact_method: str) -> str:
    """
    Check if the impact method is available.
    :param impact_method: The impact method to check.
    :return: The impact method if it is available.
    """
    if impact_method not in get_available_methods():
        raise ValueError(f"Impact method not available: {impact_method}")
    return impact_method


def format_data(data: list, weight: str) -> list:
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

    return add_population_and_gdp_data(data, weight)


def add_population_and_gdp_data(data: list, weight: str) -> list:
    """
    Add population and GDP data to the LCIA method.
    :param data: The data for the LCIA method.
    :param weight: the type of weight to include.
    :return: The data for the LCIA method with population and GDP data.
    """
    # load population data from data/population.yaml

    if weight == "population":
        with open(
            DATA_DIR / "metadata" / "population.yaml", "r", encoding="utf-8"
        ) as f:
            weighting_data = yaml.safe_load(f)

    # load GDP data from data/gdp.yaml
    if weight == "gdp":
        with open(DATA_DIR / "metadata" / "gdp.yaml", "r", encoding="utf-8") as f:
            weighting_data = yaml.safe_load(f)

    # add to the data dictionary
    for cf in data:
        for category in ["supplier", "consumer"]:
            if "location" in cf[category]:
                if "weight" not in cf[category]:
                    k = cf[category]["location"]
                    cf[category]["weight"] = weighting_data.get(k, 0)

    return data


def initialize_lcia_matrix(lca: LCA, matrix_type="biosphere") -> lil_matrix:
    """
    Initialize the LCIA matrix. It is a sparse matrix with the
    dimensions of the `inventory` matrix of the LCA object.
    :param lca: The LCA object.
    :return: An empty LCIA matrix with the dimensions of the `inventory` matrix.
    """
    if matrix_type == "biosphere":
        return lil_matrix(lca.inventory.shape)
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
                "position": v,
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


def check_database_references(cfs: list, tech_flows: list, bio_flows: list) -> list:
    """
    Check if all locations in the CFs are available in the database.
    :param cfs: List of characterization factors.
    :param tech_flows: List of technosphere flows.
    :param bio_flows: List of biosphere flows.
    :return: List of CFs with valid locations.

    """

    locations_available = set(x["location"] for x in tech_flows)
    unavailable_locations = []

    for cf in cfs:
        if "location" in cf["consumer"]:
            location = cf["consumer"]["location"]
            if location not in locations_available:
                unavailable_locations.append(location)

    print("Warning: some locations are not found in the database. Check logs.")
    logger.info(
        f"Method locations not found in the database: {set(unavailable_locations)}"
    )

    # remove the cfs with locations not found in the database
    return [cf for cf in cfs if cf["consumer"].get("location") in locations_available]
