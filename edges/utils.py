from typing import Dict, List, Union

from bw_processing.datapackage import DatapackageBase
from deprecated import deprecated

from bw2data import (
    Database,
    Method,
    Normalization,
    Weighting,
    databases,
    get_node,
    methods,
    normalizations,
    projects,
    weightings,
)
from bw2data.backends import Node
from bw2data.backends.schema import ActivityDataset as AD
from bw2data.backends.schema import get_id
from bw2data.errors import Brightway2Project, UnknownObject


def prepare_lca_inputs(
    demand=None,
    method=None,
    weighting=None,
    normalization=None,
    demands=None,
    remapping=True,
    demand_database_last=True,
):
    """Prepare LCA input arguments in Brightway 2.5 style."""
    if not projects.dataset.data.get("25"):
        raise Brightway2Project(
            "Please use `projects.migrate_project_25` before calculating using Brightway 2.5"
        )

    databases.clean()
    data_objs = []
    remapping_dicts = None

    if demands:
        demand_database_names = sorted({db_label for dct in demands for db_label in unpack(dct)})
    elif demand:
        demand_database_names = sorted({db_label for db_label in unpack(demand)})
    else:
        demand_database_names = []

    if demand_database_names:
        database_names = set.union(
            *[Database(db_label).find_graph_dependents() for db_label in demand_database_names]
        )

        if demand_database_last:
            database_names = [
                x for x in database_names if x not in demand_database_names
            ] + demand_database_names

        data_objs.extend([Database(obj).datapackage() for obj in database_names])

        if remapping:
            # This is technically wrong - we could have more complicated queries
            # to determine what is truly a product, activity, etc.
            # However, for the default database schema, we know that each node
            # has a unique ID, so this won't produce incorrect responses,
            # just too many values. As the dictionary only exists once, this is
            # not really a problem.
            reversed_mapping = {
                i: (d, c)
                for d, c, i in AD.select(AD.database, AD.code, AD.id)
                .where(AD.database << database_names)
                .tuples()
            }
            remapping_dicts = {
                "activity": reversed_mapping,
                "product": reversed_mapping,
                "biosphere": reversed_mapping,
            }

    if method:
        assert method in methods
        data_objs.append(Method(method).datapackage())
    if weighting:
        assert weighting in weightings
        data_objs.append(Weighting(weighting).datapackage())
    if normalization:
        assert normalization in normalizations
        data_objs.append(Normalization(normalization).datapackage())

    if demands:
        indexed_demand = [{get_id(k): v for k, v in dct.items()} for dct in demands]
    elif demand:
        indexed_demand = {get_id(k): v for k, v in demand.items()}
    else:
        indexed_demand = None

    return indexed_demand, data_objs, remapping_dicts
