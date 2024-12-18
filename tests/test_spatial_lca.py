import unittest
from unittest.mock import MagicMock, patch
from collections import defaultdict
import numpy as np
import bw2data, bw2io

# Assuming SpatialLCA and related functions are in a module named spatiallcia
from edges.edgelcia import (
    EdgeLCIA,
    initialize_lcia_matrix,
    preprocess_flows,
    compute_average_cf,
)


class TestSpatialLCA(unittest.TestCase):

    def setUp(self):
        # Set up mock data for testing
        self.demand = {bw2data.Database("Mobility example").random(): 1.0}
        self.method = ("AWARE 1.2c", "Country", "unspecified", "yearly")
        self.weight = "population"
        self.lca = EdgeLCIA(self.demand, method=self.method, lcia_weight=self.weight)

        # Mock data for testing
        self.lca.technosphere_flows = [
            {
                "name": "flow1",
                "reference product": "product1",
                "location": "US",
                "position": 0,
            },
            {
                "name": "flow2",
                "reference product": "product2",
                "location": "EU",
                "position": 1,
            },
        ]
        self.lca.biosphere_flows = [
            {
                "name": "flow3",
                "reference product": "product3",
                "location": "CN",
                "position": 2,
            },
        ]

        self.lca.cfs_data = [
            {
                "supplier": {"matrix": "biosphere", "location": "CN"},
                "consumer": {
                    "matrix": "technosphere",
                    "location": "US",
                    "population": 300,
                },
                "value": 1.5,
            }
        ]

    def test_initialization(self):
        # Test if the SpatialLCA object is initialized correctly
        self.assertEqual(
            np.array(list(self.lca.demand.values())).sum(),
            np.array(list(self.demand.values())).sum(),
        )
        self.assertEqual(self.lca.method, self.method)
        self.assertEqual(self.lca.weight, "population")

    def test_initialize_lcia_matrix(self):
        # Mock LCA inventory matrix for testing
        self.lca.inventory = np.zeros((3, 3))
        matrix = initialize_lcia_matrix(self.lca)
        self.assertEqual(matrix.shape, (3, 3))

    def test_preprocess_flows(self):
        # Test preprocessing of flows into a lookup dictionary
        mandatory_fields = ["name", "reference product"]
        lookup = preprocess_flows(self.lca.technosphere_flows, mandatory_fields)
        self.assertIn((("name", "flow1"), ("reference product", "product1")), lookup)
        self.assertEqual(len(lookup), 2)

    def test_compute_average_cf(self):
        # Test the calculation of average characterization factors
        constituents = ["US", "EU"]
        supplier_info = {"matrix": "technosphere"}
        weight = {"US": 300, "EU": 200}
        cfs_lookup = {
            "US": [{"supplier": {"matrix": "technosphere"}, "value": 1.0}],
            "EU": [{"supplier": {"matrix": "technosphere"}, "value": 2.0}],
        }

        result = compute_average_cf(
            constituents, supplier_info, weight, cfs_lookup, region="global"
        )
        self.assertAlmostEqual(result, 1.4, places=1)


bw2data.projects.set_current("test")

try:
    bw2data.projects.migrate_project_25()
except AssertionError:
    pass

if "Mobility example" not in bw2data.databases:
    bw2io.add_example_database()

if __name__ == "__main__":
    unittest.main()
