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
    Retrieve information about the flows in the given matrix using a batch query.

    :param mapping: A dictionary where keys are activity identifiers (tuples like (database, code)
                    or integers) and values are the desired positions.
    :return: A list of dictionaries containing flow information and their positions.
    """
    # Batch retrieve flows using get_activities() from bw2data.
    keys = list(mapping.keys())
    flows = get_activities(keys)

    # Create a lookup dictionary mapping each key to its flow.
    # Here we assume that each flow contains 'database' and 'code' (if key is a tuple),
    # or an 'id' field (if key is an integer).
    lookup = {}
    for flow in flows:
        try:
            # If available, use
            key_val = flow.get("id")
        except (KeyError, TypeError):
            # Otherwise, fall back to (database, code) as the key.
            key_val = (flow["database"], flow["code"])
        lookup[key_val] = flow

    # Build the list of flows with their associated positions.
    result = []
    for k, pos in mapping.items():
        # Attempt to find the flow using the key as given.
        flow = lookup.get(k)
        if flow is None:
            # Fallback: if the key is a tuple but not found, try matching based on the code
            if isinstance(k, tuple) and len(k) == 2:
                flow = next((f for f in flows if f.get("code") == k[1]), None)
        if flow is None:
            print(f"Flow with key {k} not found.")
            continue

        result.append(
            {
                "name": flow["name"],
                "reference product": flow.get("reference product"),
                "categories": flow.get("categories"),
                "unit": flow.get("unit"),
                "location": flow.get("location"),
                "classifications": flow.get("classifications"),
                "type": flow.get("type"),
                "position": pos,
            }
        )
    return result


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

    logger.info(
        f"Method locations not found in the database: {set(unavailable_locations)}"
    )

    # remove the cfs with locations not found in the database
    return [cf for cf in cfs if cf["consumer"].get("location") in locations_available]


def get_activities(keys, **kwargs):
    """
    Retrieve multiple activity objects in a single SQL query.

    Args:
        keys: A list (or other iterable) of keys. Each key can be either:
              - A tuple (database, code), or
              - An integer (the activity id)
        **kwargs: Additional filtering criteria, if needed.

    Returns:
        A list of activity objects.
    """
    from bw2data.backends import ActivityDataset as AD
    from bw2data.subclass_mapping import NODE_PROCESS_CLASS_MAPPING
    from bw2data import databases
    import numbers

    # Ensure keys is a list
    if not isinstance(keys, (list, tuple, set)):
        raise TypeError("keys must be a list, tuple, or set")

    keys = list(keys)  # Ensure subscriptability
    qs = AD.select()

    # If keys are tuples, assume they are (database, code) pairs.
    if all(isinstance(k, tuple) for k in keys):
        qs = qs.where((AD.database, AD.code).in_(keys))
    # If keys are integers, assume they are activity ids.
    elif all(isinstance(k, numbers.Integral) for k in keys):
        qs = qs.where(AD.id.in_(keys))
    else:
        raise TypeError(
            "All keys must be either tuples (database, code) or integers (ids)."
        )

    # If additional kwargs are provided, add those filters.
    mapping = {
        "id": AD.id,
        "code": AD.code,
        "database": AD.database,
        "location": AD.location,
        "name": AD.name,
        "product": AD.product,
        "type": AD.type,
    }
    for key, value in kwargs.items():
        if key in mapping:
            qs = qs.where(mapping[key] == value)

    # Retrieve all nodes and wrap them in the proper node class.
    nodes = []
    for obj in qs:
        backend = databases[obj.database].get("backend", "sqlite")
        cls = NODE_PROCESS_CLASS_MAPPING[backend]
        nodes.append(cls(obj))

    # Optionally, verify that all requested keys were found
    # (if uniqueness is expected, you might want to raise an error if some keys are missing)
    if len(nodes) != len(keys):
        raise Exception("Not all requested activity objects were found.")

    return nodes
