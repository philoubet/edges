import pytest
from pathlib import Path
from edges import EdgeLCIA
from bw2data import Database, projects, get_activity

from edges.georesolver import GeoResolver


# Set up once
projects.set_current("EdgeLCIA-Test")
db = Database("lcia-test-db")
activity_A = get_activity(("lcia-test-db", "A"))
activity_B = get_activity(("lcia-test-db", "B"))
activity_C = get_activity(("lcia-test-db", "C"))
activity_D = get_activity(("lcia-test-db", "D"))
activity_E = get_activity(("lcia-test-db", "E"))


@pytest.mark.parametrize("filename, activity, expected", [
    ("technosphere_location.json", activity_A, 50),
    ("technosphere_location.json", activity_B, 0),
    ("technosphere_classifications.json", activity_B, 0),
    ("technosphere_classifications.json", activity_A, 50),
    ("biosphere_name.json", activity_A, 10),
    ("biosphere_categories.json", activity_C, 1.3),
    ("biosphere_categories.json", activity_A, 1.0),
    ("biosphere_categories.json", activity_D, 1.0),
    ("biosphere_name_categories.json", activity_A, 20),
    ("biosphere_name_categories.json", activity_C, 6),
    ("technosphere_name.json", activity_D, 150),
    ("technosphere_name.json", activity_E, 250),
])
def test_cf_mapping(filename, activity, expected):
    GeoResolver._cached_lookup.cache_clear()
    filepath = str(Path("data") / filename)
    lca = EdgeLCIA(
        demand={activity: 1},
        filepath=filepath,
    )
    lca.initialize_weights()
    lca.lci()
    lca.map_exchanges()
    lca.map_aggregate_locations()
    lca.map_dynamic_locations()
    lca.map_contained_locations()
    lca.map_remaining_locations_to_global()
    lca.evaluate_cfs()
    lca.lcia()

    df = lca.generate_cf_table()

    if pytest.approx(lca.score) != expected:
        status = "failed"
    else:
        status = "passed"

    if df is not None:
        df.to_excel(f"test - {filename} {activity['name']} {status}.xlsx", index=False)

    lca._geo = None

    assert pytest.approx(lca.score) == expected
