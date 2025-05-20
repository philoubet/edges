from pathlib import Path
import json

# Output directory
output_dir = Path("data")
output_dir.mkdir(exist_ok=True)

# Define base CF structure
base_cf = {
    "unit": "kg",
    "amount": 1,
    "uncertainty": None,
    "consumer": {"matrix": "technosphere", "location": "RER"},
}

# Technosphere-based discrimination
technosphere_cases = {
    "technosphere_name.json": {
        "supplier": {"name": "B", "matrix": "technosphere"},
    },
    "technosphere_name_refprod.json": {
        "supplier": {"name": "B", "reference product": "bar", "matrix": "technosphere"},
    },
    "technosphere_location.json": {
        "supplier": {"location": "CH", "matrix": "technosphere"},
    },
    "technosphere_classifications.json": {
        "supplier": {
            "classifications": {"cpc": ["01.1"]},
            "matrix": "technosphere",
        },
    },
    "technosphere_name_refprod_location.json": {
        "supplier": {
            "name": "B",
            "reference product": "bar",
            "location": "CH",
            "matrix": "technosphere",
        },
    },
    "technosphere_name_refprod_classifications.json": {
        "supplier": {
            "name": "B",
            "reference product": "bar",
            "classifications": {"cpc": ["01.1"]},
            "matrix": "technosphere",
        },
    },
    "technosphere_refprod_location_classifications.json": {
        "supplier": {
            "reference product": "bar",
            "location": "CH",
            "classifications": {"cpc": ["01.1"]},
            "matrix": "technosphere",
        },
    },
    "technosphere_all_fields.json": {
        "supplier": {
            "name": "B",
            "reference product": "bar",
            "location": "CH",
            "classifications": {"cpc": ["01.1"]},
            "matrix": "technosphere",
        },
    },

}

# Write technosphere files
for filename, content in technosphere_cases.items():
    consumer = content.get("consumer", base_cf["consumer"])
    if "match_location" in content:
        consumer = {"matrix": "technosphere", "location": content["match_location"]}
    exchange = {
        "value": 100,
        "weight": 1.0,
        "supplier": content["supplier"],
        "consumer": consumer,
    }
    data = {
        "name": "Test LCIA Method",
        "version": "1.0",
        "description": f"Test for {filename}",
        "unit": "kg",
        "exchanges": [exchange],
    }
    with open(output_dir / filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

# Biosphere-based discrimination
biosphere_cases = {
    "biosphere_name.json": {
        "supplier": {"name": "Carbon dioxide, in air", "matrix": "biosphere"},
    },
    "biosphere_categories.json": {
        "supplier": {"categories": ("air",), "matrix": "biosphere"},
    },
    "biosphere_name_categories.json": {
        "supplier": {
            "name": "Carbon dioxide, in air",
            "categories": ("air",),
            "matrix": "biosphere",
        },
    },
}

# Write biosphere files
for filename, content in biosphere_cases.items():
    data = {
        "name": "Test LCIA Method",
        "version": "1.0",
        "description": f"Test for {filename}",
        "unit": "kg",
        "exchanges": [
            {
                "value": 50,
                "weight": 1.0,
                "supplier": content["supplier"],
                "consumer": base_cf["consumer"],
            }
        ],
    }
    with open(output_dir / filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
