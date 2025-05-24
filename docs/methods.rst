
Available Methods
=================

This section describes the built-in LCIA methods available in `edges`, including their purpose, implementation, citation, minimal example, and a JSON schema excerpt showing how exchanges are matched.

---

AWARE 2.0
---------

**Name**: `AWARE 2.0_Country_all_yearly.json` (and variants)

**Impact Category**: Water Scarcity

**Description**: AWARE estimates water deprivation potential by measuring the availability of water after human and ecosystem needs are met.

**Usage Example**:

.. code-block:: python

    from edges import EdgeLCIA
    lcia = EdgeLCIA(method="AWARE 2.0_Country_all_yearly.json")
    lcia.lci()
    lcia.map_exchanges()
    lcia.map_aggregate_locations()
    lcia.map_dynamic_locations()
    lcia.map_contained_locations()
    lcia.map_remaining_locations_to_global()
    lcia.evaluate_cfs()
    lcia.lcia()

**Sample CF JSON**:

.. code-block:: json

    {
      "supplier": {
        "name": "Water, lake",
        "categories": ["natural resource", "in water"],
        "matrix": "biosphere"
      },
      "consumer": {
        "location": "AM",
        "matrix": "technosphere",
        "classifications": {"CPC": ["01"]}
      },
      "value": 88.6,
      "weight": 799882000,
      "uncertainty": {
        "distribution": "discrete_empirical",
        "parameters": {
          "values": [84.5, 87.9],
          "weights": [0.031, 0.969]
        }
      }
    }

**Reference**:  
Boulay et al., 2018.  
https://doi.org/10.1007/s11367-017-1333-8

---

GeoPolRisk 1.0
--------------

**Name**: `GeoPolRisk_2024.json`

**Impact Category**: Supply Risk of Minerals

**Usage Example**:

.. code-block:: python

    from edges import EdgeLCIA
    lcia = EdgeLCIA(method="GeoPolRisk_2024.json")
    lcia.lci()
    lcia.map_exchanges()
    lcia.map_aggregate_locations()
    lcia.map_contained_locations()
    lcia.map_remaining_locations_to_global()
    lcia.evaluate_cfs()
    lcia.lcia()

**Sample CF JSON**:

.. code-block:: json

    {
      "supplier": {
        "name": "aluminium production",
        "reference product": "aluminium",
        "location": "AU",
        "operator": "startswith",
        "matrix": "technosphere"
      },
      "consumer": {
        "location": "CA",
        "matrix": "technosphere"
      },
      "value": 1.10e-10
    }

**Reference**:  
Koyamparambath et al., 2024.  
https://doi.org/10.1016/j.resconrec.2024.107801

---

ImpactWorld+ 2.1
----------------

**Name**: `ImpactWorld+ 2.1_<category>_<level>.json`

**Impact Categories**: Acidification, ecotoxicity, eutrophication, land use

**Usage Example**:

.. code-block:: python

    from edges import EdgeLCIA
    lcia = EdgeLCIA(method="ImpactWorld+ 2.1_Freshwater acidification_midpoint.json")
    lcia.lci()
    lcia.map_exchanges()
    lcia.map_aggregate_locations()
    lcia.map_dynamic_locations()
    lcia.map_contained_locations()
    lcia.map_remaining_locations_to_global()
    lcia.evaluate_cfs()
    lcia.lcia()

**Sample CF JSON**:

.. code-block:: json

    {
      "supplier": {
        "name": "Hydrogen chloride, gas",
        "matrix": "biosphere"
      },
      "consumer": {
        "location": "EU",
        "matrix": "technosphere"
      },
      "value": 1.26
    }

**Reference**:  
Bulle et al., 2019.  
https://doi.org/10.1007/s11367-019-01583-0

---

SCP 1.0 (Surplus Cost Potential)
-------------------------------

**Name**: `SCP_1.0.json`

**Impact Category**: Fossil Fuel Resource Scarcity

**Usage Example**:

.. code-block:: python

    from edges import EdgeLCIA
    lcia = EdgeLCIA(method="SCP_1.0.json")
    lcia.lci()
    lcia.map_exchanges()
    lcia.evaluate_cfs(parameters={"MCI_OIL": 0.5, "P_OIL": 400, "d": 0.03})
    lcia.lcia()

**Sample CF JSON**:

.. code-block:: json

    {
      "supplier": {
        "name": "Oil, crude",
        "categories": ["natural resource", "in ground"],
        "matrix": "biosphere"
      },
      "consumer": {
        "matrix": "technosphere"
      },
      "value": "(MCI_OIL * P_OIL / 5) / (1 + d)"
    }

**Reference**:  
Vieira et al., 2017; Luderer et al., 2020.  
https://doi.org/10.1002/jiec.12447

---

Parameterized GWP
-----------------

**Name**: `lcia_parameterized_gwp.json`

**Impact Category**: Global Warming Potential (Dynamic)

**Usage Example**:

.. code-block:: python

    from edges import EdgeLCIA
    lcia = EdgeLCIA(method="lcia_parameterized_gwp.json")
    lcia.lci()
    lcia.map_exchanges()
    lcia.evaluate_cfs(parameters={"C_CH4": 1866, "H": 100})
    lcia.lcia()

**Sample CF JSON**:

.. code-block:: json

    {
      "supplier": {
        "name": "Methane, fossil",
        "matrix": "biosphere",
        "operator": "contains"
      },
      "consumer": {
        "matrix": "technosphere"
      },
      "value": "GWP('CH4', H, C_CH4)"
    }

**Reference**:  
IPCC AR6, 2023.  
https://doi.org/10.1017/9781009157896.017
