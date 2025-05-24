
Quick Start
===========

This minimal example walks you through performing a basic LCIA using `edges`.

.. code-block:: python

    from edges import EdgeLCIA

    # Load a built-in method
    lcia = EdgeLCIA(method="AWARE 2.0_Country_all_yearly.json")

    # Step 1: Build the inventory
    lcia.lci()

    # Step 2: Match exchanges to characterization factors
    lcia.map_exchanges()

    # Step 3: Evaluate CFs (e.g., resolve symbolic expressions)
    lcia.evaluate_cfs()

    # Step 4: Compute the LCIA score
    lcia.lcia()

    # Step 5 (optional): Print a summary
    print(lcia.statistics())
