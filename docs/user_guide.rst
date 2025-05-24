
User Guide
==========

This guide walks you through practical usage patterns of the `edges` library, covering common workflows such as simple LCIA, regionalized impact assessment, parameterized methods, uncertainty analysis, and scenario-based modeling.

---

Simple LCIA
-----------

For non-regionalized methods with fixed CFs:

.. code-block:: python

    from edges import EdgeLCIA
    lcia = EdgeLCIA(method="lcia_example_1.json")
    lcia.lci()
    lcia.map_exchanges()
    lcia.evaluate_cfs()
    lcia.lcia()
    df = lcia.generate_df_table()
    print(df.head())

---

Regionalized LCIA
-----------------

When using region-specific methods like AWARE or ImpactWorld+:

.. code-block:: python

    lcia = EdgeLCIA(method="AWARE 2.0_Country_all_yearly.json")
    lcia.lci()
    lcia.map_exchanges()
    lcia.map_aggregate_locations()
    lcia.map_dynamic_locations()
    lcia.map_contained_locations()
    lcia.map_remaining_locations_to_global()
    lcia.evaluate_cfs()
    lcia.lcia()

---

Using Built-in Method Files
---------------------------

You can list available method files with:

.. code-block:: python

    from edges import get_available_methods
    print(get_available_methods())

Use the filename in the `method=` argument when instantiating `EdgeLCIA`.

---

Using a Custom Method JSON
--------------------------

Your method file should follow the expected CF JSON schema:

.. code-block:: python

    lcia = EdgeLCIA(method="my_custom_method.json")
    lcia.lci()
    lcia.map_exchanges()
    lcia.evaluate_cfs(parameters={"H": 100, "C_CH4": 1866})
    lcia.lcia()

---

Parameterized CFs
-----------------

If the method uses symbolic expressions, pass parameter values:

.. code-block:: python

    lcia = EdgeLCIA(method="lcia_parameterized_gwp.json")
    lcia.lci()
    lcia.map_exchanges()
    lcia.evaluate_cfs(parameters={"C_CH4": 1800, "H": 100})
    lcia.lcia()

This allows integration with scenario data (e.g., from RCPs or IAMs).

---

Uncertainty-aware LCIA
-----------------------

If CFs include uncertainty (e.g., lognormal, discrete empirical), you can get statistics:

.. code-block:: python

    lcia = EdgeLCIA(method="AWARE 2.0_Country_all_yearly.json")
    lcia.lci()
    lcia.map_exchanges()
    lcia.map_aggregate_locations()
    lcia.map_dynamic_locations()
    lcia.map_contained_locations()
    lcia.map_remaining_locations_to_global()
    lcia.evaluate_cfs(n=1000)  # Monte Carlo with 1000 draws
    lcia.lcia()
    print(lcia.statistics())

---

Working with Technosphere CFs (e.g., GeoPolRisk)
------------------------------------------------

.. code-block:: python

    lcia = EdgeLCIA(method="GeoPolRisk_2024.json")
    lcia.lci()
    lcia.map_exchanges()
    lcia.map_aggregate_locations()
    lcia.map_contained_locations()
    lcia.map_remaining_locations_to_global()
    lcia.evaluate_cfs()
    lcia.lcia()
    df = lcia.generate_df_table()
    df.to_csv("results.csv")

---

Scenario-based Fossil Resource Scarcity
---------------------------------------

Supports expressions depending on extraction volume and discount rate:

.. code-block:: python

    lcia = EdgeLCIA(method="SCP_1.0.json")
    lcia.lci()
    lcia.map_exchanges()
    lcia.evaluate_cfs(parameters={"MCI_OIL": 0.5, "P_OIL": 450, "d": 0.03})
    lcia.lcia()

---

Exporting Results
-----------------

You can inspect or save the detailed contribution table:

.. code-block:: python

    df = lcia.generate_df_table()
    df.to_csv("edge_lcia_detailed_results.csv")
