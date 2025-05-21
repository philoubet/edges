from bw2data import Database, projects, methods, databases, __version__
from bw2io import bw2setup


if __version__ < (4, 0, 0):
    is_bw2 = True
else:
    is_bw2 = False
    try:
        projects.migrate_project_25()
    except:
        pass

print(f"Using bw2: {is_bw2}")


if is_bw2 is True:
    project = "EdgeLCIA-Test"
    projects.set_current(project)
else:
    project = "EdgeLCIA-Test-bw25"
    projects.set_current(project)

# Clean up if exists
if "lcia-test-db" in databases:
    del databases["lcia-test-db"]

if "biosphere" in databases:
    del databases["biosphere"]

# Define biosphere flows
biosphere = Database("biosphere")
biosphere.write(
    {
        ("biosphere", "co2"): {
            "name": "Carbon dioxide, in air",
            "unit": "kilogram",
            "categories": ("air",),
            "type": "emission",
        },
        ("biosphere", "co2_low_pop"): {
            "name": "Carbon dioxide, in air",
            "unit": "kilogram",
            "categories": ("air", "low population"),
            "type": "emission",
        },
    }
)

test_db = Database("lcia-test-db")
test_data = [
    # Technosphere activities
    {
        "name": "A",
        "unit": "kg",
        "location": "RER",
        "reference product": "foo",
        "classifications": [
            ("cpc", "01: crops"),
        ],
        "exchanges": [
            {
                "amount": 1,
                "type": "production",
                "product": "foo",
                "unit": "kg",
                "name": "A",
                "input": ("lcia-test-db", "A"),
            },
            {
                "amount": 0.5,
                "type": "technosphere",
                "input": ("lcia-test-db", "B"),
                "unit": "kg",
            },
            {
                "amount": 0.1,
                "type": "biosphere",
                "input": ("biosphere", "co2"),
                "unit": "kg",
            },
        ],
    },
    {
        "name": "B",
        "unit": "kg",
        "location": "CH",
        "reference product": "bar",
        "classifications": [
            ("cpc", "01.1: cereals"),
        ],
        "exchanges": [
            {
                "amount": 1,
                "type": "production",
                "product": "bar",
                "unit": "kg",
                "name": "B",
                "input": ("lcia-test-db", "B"),
            },
            {
                "amount": 0.2,
                "type": "biosphere",
                "input": ("biosphere", "co2"),
                "unit": "kg",
            },
        ],
    },
    {
        "name": "C",
        "unit": "kg",
        "location": "DE",
        "reference product": "baz",
        "classifications": [
            ("cpc", "01.2: vegetables"),
        ],
        "exchanges": [
            {
                "amount": 1,
                "type": "production",
                "product": "baz",
                "unit": "kg",
                "name": "C",
                "input": ("lcia-test-db", "C"),
            },
            {
                "amount": 0.3,
                "type": "technosphere",
                "input": ("lcia-test-db", "B"),
                "unit": "kg",
            },
            {
                "amount": 0.1,
                "type": "biosphere",
                "input": ("biosphere", "co2_low_pop"),
                "unit": "kg",
            },
        ],
    },
    {
        "name": "D",
        "unit": "kg",
        "location": "IT",
        "reference product": "boz",
        "classifications": [],
        "exchanges": [
            {
                "amount": 1,
                "type": "production",
                "product": "boz",
                "unit": "kg",
                "name": "D",
                "input": ("lcia-test-db", "D"),
            },
            {
                "amount": 1,
                "type": "technosphere",
                "input": ("lcia-test-db", "A"),
                "unit": "kg",
            },  # Will inherit CF for RER
        ],
    },
    {
        "name": "E",
        "unit": "kg",
        "location": "CN",
        "reference product": "dummy",
        "classifications": [],
        "exchanges": [
            {
                "amount": 1,
                "type": "production",
                "product": "dummy",
                "unit": "kg",
                "name": "E",
                "input": ("lcia-test-db", "E"),
            },
            {
                "amount": 1,
                "type": "technosphere",
                "input": ("lcia-test-db", "A"),
                "unit": "kg",
            },  # Will fall back to CF for GLO
        ],
    },
]

# Write the technosphere database
test_db.write({(test_db.name, d["name"]): d for d in test_data})

print(f"The following databases are available: {databases}")
