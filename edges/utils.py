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
